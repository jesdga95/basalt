// src/models/inkling2/kernels/kv_nvfp4_parity.cu - NVFP4 KV writer and decode attention parity against CPU references.
// With --bench, also time the encoder and decode window attention over every cache format at the model's depths.
#include "basalt/core/device_copy.hpp"
#include "prep.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ink = basalt::models::inkling2;

// Encode `rows` head rows of 128 channels (one warp a row) with store_kv_nvfp4: keys Hadamard-rotated, values direct.
__global__ void encode_rows_nvfp4(const float* input, uint8_t* output, int rows, int is_key) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    float v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = input[(size_t) row * ink::kHeadDim + lane * 4 + u];
    ink::prep::store_kv_nvfp4(output + (size_t) row * ink::kKvNvfp4RowBytes, v, 1.f, is_key != 0, lane);
}

static bool check_cuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "kv_nvfp4_parity: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ---- CPU reference ------------------------------------------------------------------------------------------------------
//
// The same arithmetic as the device inlines, in the same order: the butterfly of prep.cuh's hadamard128_warp, the group
// scale of kv4_group_scale and the codes of kv4_group_code. Float arithmetic where the kernel is exact so the bytes match;
// the error accounting in double.

static void hadamard128_cpu(float* x) {
    float v[128];
    for (int l = 0; l < 32; ++l) {
        const float a0 = x[4 * l] + x[4 * l + 1], a1 = x[4 * l] - x[4 * l + 1];
        const float a2 = x[4 * l + 2] + x[4 * l + 3], a3 = x[4 * l + 2] - x[4 * l + 3];
        v[4 * l] = a0 + a2;
        v[4 * l + 2] = a0 - a2;
        v[4 * l + 1] = a1 + a3;
        v[4 * l + 3] = a1 - a3;
    }
    for (int off = 1; off < 32; off <<= 1) {
        float w[128];
        for (int l = 0; l < 32; ++l) {
            const bool hi = (l & off) != 0;
            for (int u = 0; u < 4; ++u) {
                const int d = 4 * l + u;
                const float other = v[4 * (l ^ off) + u];
                w[d] = hi ? other - v[d] : v[d] + other;
            }
        }
        std::memcpy(v, w, sizeof v);
    }
    constexpr float inv = 0.08838834764831845f;
    for (int d = 0; d < 128; ++d) x[d] = v[d] * inv;
}

static uint32_t e4m3_code_cpu(float v) { return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }
static float e4m3_value_cpu(uint32_t q) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
}
static uint32_t e2m1_code_cpu(float v) {
    const float a = std::fabs(v);
    uint32_t m;
    if (a <= 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a < 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return m | ((uint32_t) std::signbit(v) << 3);
}
static double e2m1_value_cpu(uint32_t c) {
    static const double mag[8] = {0., .5, 1., 1.5, 2., 3., 4., 6.};
    return (c & 8u) ? -mag[c & 7u] : mag[c & 7u];
}

// One row's NVFP4 bytes as store_kv_nvfp4 writes them (is_key: the row Hadamard-rotated first); `held` takes the values
// the row holds.
static void put_row_nvfp4(const float* x, bool is_key, uint8_t* row, double* held) {
    float v[128];
    std::memcpy(v, x, sizeof v);
    if (is_key) hadamard128_cpu(v);
    std::memset(row, 0, ink::kKvNvfp4RowBytes);
    for (int grp = 0; grp < 8; ++grp) {
        float amax = 0.f;
        for (int d = grp * 16; d < grp * 16 + 16; ++d) amax = std::max(amax, std::fabs(v[d]));
        const uint32_t sb = e4m3_code_cpu(std::max(amax, 6.0f * 0.001953125f) / 6.0f);
        row[ink::kHeadDim / 2 + grp] = (uint8_t) sb;
        const float s = e4m3_value_cpu(sb);
        for (int d = grp * 16; d < grp * 16 + 16; ++d) {
            const float scaled = std::min(std::max(v[d] / s, -6.0f), 6.0f);
            const uint32_t q = e2m1_code_cpu(scaled);
            if (d % 2 == 0) row[d / 2] = (uint8_t) ((row[d / 2] & 0xf0) | q);
            else row[d / 2] = (uint8_t) ((q << 4) | (row[d / 2] & 0x0f));
            held[d] = e2m1_value_cpu(q) * (double) s;
        }
    }
}

// ---- decode attention parity --------------------------------------------------------------------------------------------

// Repeat random cache records over a prime period so the CPU can fold the far keys by multiplicity.
constexpr int kPeriod = 257;
constexpr int kOutput = ink::kHeads * ink::kHeadDim;
constexpr int kCacheCell = ink::kKvHeads * ink::kKvNvfp4RowBytes;

// Random valid NVFP4 rows: nonzero E4M3 scales and any E2M1 codes, padding zeroed; `exact` the values they hold.
static void random_cache_nvfp4(std::mt19937& rng, std::vector<uint8_t>& bytes, std::vector<double>& exact) {
    std::uniform_int_distribution<int> code(0, 15);
    std::uniform_real_distribution<float> magnitude(0.002f, 0.2f);
    for (int i = 0; i < kPeriod * ink::kKvHeads; ++i) {
        uint8_t* row = bytes.data() + (size_t) i * ink::kKvNvfp4RowBytes;
        std::memset(row, 0, ink::kKvNvfp4RowBytes);
        for (int grp = 0; grp < 8; ++grp) {
            const uint32_t sb = e4m3_code_cpu(magnitude(rng));
            row[ink::kHeadDim / 2 + grp] = (uint8_t) sb;
            const double s = (double) e4m3_value_cpu(sb);
            for (int d = grp * 16; d < grp * 16 + 16; ++d) {
                const uint32_t q = (uint32_t) code(rng);
                if (d % 2 == 0) row[d / 2] = (uint8_t) ((row[d / 2] & 0xf0) | q);
                else row[d / 2] = (uint8_t) ((q << 4) | (row[d / 2] & 0x0f));
                exact[(size_t) i * ink::kHeadDim + d] = e2m1_value_cpu(q) * s;
            }
        }
    }
}

static void cpu_attention(int pos, const float* q, const float* r, const float* sink, const float* gate,
                          const float* table, const std::vector<double>& kc, const std::vector<double>& vc,
                          double scale, double* out, bool windowed = false) {
    const int near_first = std::max(0, pos + 1 - ink::kExtent);
    const int far_end = windowed ? 0 : near_first;   // a windowed read takes only the keys inside the window
    for (int h = 0; h < ink::kHeads; ++h) {
        const int kvh = h / (ink::kHeads / ink::kKvHeads);
        double base[kPeriod], near[ink::kExtent], weights[kPeriod] = {};
        double m = sink[h];
        for (int key = 0; key < kPeriod; ++key) {
            const double* k = kc.data() + ((size_t) key * ink::kKvHeads + kvh) * ink::kHeadDim;
            double score = 0.;
            for (int d = 0; d < ink::kHeadDim; ++d) score += (double) q[h * ink::kHeadDim + d] * k[d];
            base[key] = score * scale;
            if (key < far_end) m = std::max(m, base[key]);
        }
        for (int key = near_first; key <= pos; ++key) {
            const int dist = pos - key;
            double bias = 0.;
            for (int d = 0; d < ink::kRel; ++d)
                bias += (double) r[h * ink::kRel + d] * table[d * ink::kExtent + dist];
            near[dist] = base[key % kPeriod] + bias;
            m = std::max(m, near[dist]);
        }
        for (int key = 0; key < kPeriod; ++key) {
            const int count = far_end / kPeriod + (key < far_end % kPeriod);
            if (count) weights[key] = count * std::exp(base[key] - m);
        }
        for (int key = near_first; key <= pos; ++key)
            weights[key % kPeriod] += std::exp(near[pos - key] - m);
        double denom = std::exp(sink[h] - m), numer[ink::kHeadDim] = {};
        for (int key = 0; key < kPeriod; ++key) {
            const double* v = vc.data() + ((size_t) key * ink::kKvHeads + kvh) * ink::kHeadDim;
            denom += weights[key];
            for (int d = 0; d < ink::kHeadDim; ++d) numer[d] += weights[key] * v[d];
        }
        for (int d = 0; d < ink::kHeadDim; ++d) {
            const int j = h * ink::kHeadDim + d;
            out[j] = numer[d] / denom / (1. + std::exp(-(double) gate[j]));
        }
    }
}

// Full-cache and window-ring decode attention over the periodic cache: against the CPU reference, and the same bits a row
// at a time, under any launch bound and split in two batch segments.
static bool check_attention_nvfp4() {
    constexpr int rows = ink::kDecodeRows, cap = 131072 + rows;
    std::mt19937 rng(0x5f4a11);
    std::normal_distribution<float> normal(0.f, 0.4f);
    std::vector<float> q(rows * kOutput), r(rows * ink::kHeads * ink::kRel), sink(rows * ink::kHeads),
        gate(rows * kOutput), table(ink::kRel * ink::kExtent);
    for (auto* a : {&q, &r, &sink, &gate, &table})
        for (float& x : *a) x = normal(rng);
    std::vector<uint8_t> kp(kPeriod * kCacheCell), vp(kPeriod * kCacheCell);
    std::vector<double> hk(kPeriod * ink::kKvHeads * ink::kHeadDim), hv(hk.size());
    random_cache_nvfp4(rng, kp, hk);
    random_cache_nvfp4(rng, vp, hv);

    int32_t* dp = nullptr;
    float *dq = nullptr, *dr = nullptr, *ds = nullptr, *dg = nullptr, *dt = nullptr, *dout = nullptr;
    uint8_t *dk = nullptr, *dv = nullptr;
    void* scratch = nullptr;
    const auto alloc = [](auto** ptr, size_t bytes) {
        return check_cuda(cudaMalloc((void**) ptr, bytes), "nvfp4 attention parity allocate");
    };
    const auto upload = [&](float** ptr, const std::vector<float>& v) {
        const size_t bytes = v.size() * sizeof(float);
        return alloc(ptr, bytes) && check_cuda(basalt::core::device_copy(*ptr, v.data(), bytes, cudaMemcpyHostToDevice), "nvfp4 attention inputs");
    };
    const auto upload_cache = [&](uint8_t** ptr, const std::vector<uint8_t>& pattern) {
        std::vector<uint8_t> host((size_t) cap * kCacheCell);
        for (size_t i = 0; i < host.size(); i += pattern.size())
            std::memcpy(host.data() + i, pattern.data(), std::min(pattern.size(), host.size() - i));
        return alloc(ptr, host.size()) &&
               check_cuda(basalt::core::device_copy(*ptr, host.data(), host.size(), cudaMemcpyHostToDevice), "nvfp4 attention cache");
    };
    bool ok = upload(&dq, q) && upload(&dr, r) && upload(&ds, sink) && upload(&dg, gate) && upload(&dt, table) &&
              upload_cache(&dk, kp) && upload_cache(&dv, vp) && alloc(&dp, rows * sizeof(int32_t)) &&
              alloc(&dout, rows * kOutput * sizeof(float)) &&
              alloc(&scratch, ink::attention_decode_scratch(rows, cap));
    const int depths[] = {1024, 4096, 32768, 65536, 131072};
    const int widths[] = {1, 2, 3, 6, 8};
    for (const int depth : depths) {
        if (!ok) break;
        int32_t pos[rows];
        for (int j = 0; j < rows; ++j) pos[j] = depth - 4 + j;
        ok = check_cuda(basalt::core::device_copy(dp, pos, sizeof(pos), cudaMemcpyHostToDevice), "nvfp4 attention positions");
        if (!ok) break;
        const auto run = [&](int first, int width, bool window_ring, std::vector<float>& dst, int64_t bound = 0,
                             bool batch = false) {
            ink::AttnArgs a;
            a.nvfp4 = true;
            a.q = dq + (size_t) first * kOutput;
            a.r = dr + (size_t) first * ink::kHeads * ink::kRel;
            a.sink = ds + (size_t) first * ink::kHeads;
            a.gate = dg + (size_t) first * kOutput;
            a.gate_ld = kOutput;
            a.rel_table = dt;
            a.pos = dp + first;
            a.k_cache = dk;
            a.v_cache = dv;
            a.cap = cap;
            a.window = window_ring ? ink::kExtent : 0;
            if (batch) {
                a.seg.at.n = 2;
                a.seg.at.row0[1] = width / 2;
                a.seg.at.row0[2] = width;
                a.seg.c[0].k = a.seg.c[1].k = dk;
                a.seg.c[0].v = a.seg.c[1].v = dv;
                a.k_cache = a.v_cache = nullptr;
            } else a.min_full_keys = window_ring ? 0 : pos[first] + 1;
            const int64_t keys = window_ring ? ink::kExtent : bound ? bound : pos[first + width - 1] + 1;
            dst.assign((size_t) width * kOutput, 0.f);
            ink::attention_decode(a, width, keys, scratch, dout, nullptr);
            return check_cuda(basalt::core::device_copy(dst.data(), dout, width * kOutput * sizeof(float), cudaMemcpyDeviceToHost),
                              "nvfp4 attention output");
        };
        for (const int width : widths) {
            if (!ok) break;
            if (width > rows) continue;
            const int first = (rows - width) / 2;
            for (const bool window_ring : {false, true}) {
                if (!ok) break;
                std::vector<float> got, single, padded, batched;
                ok = run(first, width, window_ring, got) &&
                     run(first, width, window_ring, padded, cap) &&
                     run(first, width, window_ring, batched, 0, width >= 2);
                if (!ok) break;
                // each row alone against its row of the window's output
                bool invariant = true, bound_same = true, batch_same = true, finite = true;
                for (int j = 0; j < width && ok; ++j) {
                    std::vector<float> one;
                    ok = run(first + j, 1, window_ring, one);
                    if (!ok) break;
                    invariant &= std::memcmp(one.data(), got.data() + (size_t) j * kOutput, kOutput * 4) == 0;
                }
                if (!ok) break;
                bound_same = std::memcmp(padded.data(), got.data(), got.size() * 4) == 0;
                if (width >= 2) {
                    // the segments' rows are the same conversations' rows: bitwise the unsegmented output
                    batch_same = std::memcmp(batched.data(), got.data(), got.size() * 4) == 0;
                }
                double max_abs = 0., max_rel = 0.;
                std::vector<double> ref((size_t) width * kOutput);
                for (int j = 0; j < width; ++j) {
                    cpu_attention(pos[first + j], q.data() + (size_t) (first + j) * kOutput, r.data() + (size_t) (first + j) * ink::kHeads * ink::kRel,
                                  sink.data() + (size_t) (first + j) * ink::kHeads, gate.data() + (size_t) (first + j) * kOutput,
                                  table.data(), hk, hv, ink::AttnArgs{}.scale, ref.data() + (size_t) j * kOutput, window_ring);
                    for (int d = 0; d < kOutput; ++d) {
                        const double err = std::abs((double) got[(size_t) j * kOutput + d] - ref[(size_t) j * kOutput + d]);
                        finite &= std::isfinite(got[(size_t) j * kOutput + d]);
                        max_abs = std::max(max_abs, err);
                        const double want = std::abs(ref[(size_t) j * kOutput + d]);
                        if (want >= 1e-2) max_rel = std::max(max_rel, err / want);
                    }
                }
                std::printf("kv nvfp4 parity: depth=%d T=%d %s max_abs=%.7g max_rel(|ref|>=1e-2)=%.7g finite=%d invariant=%d "
                            "launch_bound=%d batch=%d\n", depth, width, window_ring ? "window" : "full  ", max_abs, max_rel,
                            finite, invariant, bound_same, width >= 2 ? batch_same : -1);
                ok &= finite && invariant && bound_same && (width < 2 || batch_same) && max_abs < 0.002 && max_rel < 0.05;
            }
        }
    }
    for (void* p : {(void*) dp, (void*) dq, (void*) dr, (void*) ds, (void*) dg, (void*) dt, (void*) dout, (void*) dk,
                    (void*) dv, scratch})
        if (p) cudaFree(p);
    return ok;
}

// ---- bench ---------------------------------------------------------------------------------------------------------------

static bool bench_attention() {
    struct Fmt { const char* name; bool fp8, int8, nvfp4; };
    const Fmt fmts[] = {{"fp16", false, false, false}, {"fp8", true, false, false}, {"int8", false, true, false},
                        {"nvfp4", false, false, true}};
    const int64_t depths[] = {0, 8192, 32768, 65536, 131072, 262144};
    const int widths[] = {1, 3, 8};
    int dev = 0;
    cudaDeviceProp prop{};
    if (!check_cuda(cudaGetDevice(&dev), "bench device") ||
        !check_cuda(cudaGetDeviceProperties(&prop, dev), "bench device properties")) return false;
    std::printf("kv nvfp4 bench: device=%s SMs=%d\n", prop.name, prop.multiProcessorCount);
    float *y = nullptr, *taps = nullptr, *qs = nullptr, *rs = nullptr, *kr = nullptr, *vr = nullptr, *table = nullptr;
    float *q = nullptr, *r = nullptr, *sink = nullptr;
    int32_t* pos = nullptr;
    uint8_t* q8 = nullptr;
    void* scr = nullptr;
    const auto alloc = [](auto** p, size_t n, const char* what) {
        return check_cuda(cudaMalloc((void**) p, n), what) && check_cuda(cudaMemset(*p, 0, n), what);
    };
    bool ok = alloc(&y, (size_t) ink::kDecodeRows * ink::kProj * 4, "y") && alloc(&taps, 1024 * ink::kConv * 4, "taps") &&
              alloc(&qs, ink::kHeads * ink::kHeadDim * 4, "q scales") && alloc(&rs, ink::kHeads * ink::kRel * 4, "r scales") &&
              alloc(&kr, (size_t) ink::kConvRing * 1024 * 4, "k ring") && alloc(&vr, (size_t) ink::kConvRing * 1024 * 4, "v ring") &&
              alloc(&table, (size_t) ink::kRel * ink::kExtent * 4, "table") &&
              alloc(&q, (size_t) ink::kDecodeRows * ink::kHeads * ink::kHeadDim * 4, "q") &&
              alloc(&r, (size_t) ink::kDecodeRows * ink::kHeads * ink::kRel * 4, "r") && alloc(&sink, (size_t) ink::kDecodeRows * ink::kHeads * 4, "sink") &&
              alloc(&pos, ink::kDecodeRows * 4, "pos") && alloc(&q8, (size_t) ink::kDecodeRows * ink::kHeads * ink::kHeadDim / 32 * 36, "q8") &&
              alloc(&scr, ink::attention_decode_scratch(ink::kDecodeRows, 1 << 20), "scratch");
    cudaEvent_t t0{}, t1{};
    ok = ok && check_cuda(cudaEventCreate(&t0), "event") && check_cuda(cudaEventCreate(&t1), "event");
    for (const Fmt& f : fmts)
        for (const int64_t depth : depths) {
            if (!ok) break;
            const bool window = depth == 0;
            const int64_t cap = window ? ink::kExtent + ink::kDecodeRows : depth;
            const size_t row_bytes = (size_t) ink::kv_cache_row_bytes(f.fp8, f.int8, f.nvfp4);
            const size_t cache = (size_t) cap * ink::kKvHeads * row_bytes;
            uint8_t *kc = nullptr, *vc = nullptr;
            if (!check_cuda(cudaMalloc(&kc, cache), "k cache") || !check_cuda(cudaMalloc(&vc, cache), "v cache")) {
                ok = false;
                break;
            }
            cudaMemset(kc, 0x11, cache);
            cudaMemset(vc, 0x11, cache);
            for (const int rows : widths) {
                const int64_t p0 = (window ? 8192 : depth) - rows;
                std::vector<int32_t> hp(rows);
                for (int i = 0; i < rows; ++i) hp[i] = (int32_t) (p0 + i);
                basalt::core::device_copy(pos, hp.data(), rows * 4, cudaMemcpyHostToDevice);
                ink::AttnPrep pa;
                pa.yq = y; pa.yk = y + ink::kProjQ; pa.yv = y + ink::kProjQ + ink::kProjKV; pa.ldq = pa.ldk = pa.ldv = ink::kProj;
                pa.pos = pos; pa.full = !window; pa.k_conv = taps; pa.v_conv = taps; pa.q_scale = qs; pa.r_scale = rs;
                pa.k_ring = kr; pa.v_ring = vr; pa.k_cache = kc; pa.v_cache = vc; pa.cap = cap;
                pa.fp8 = f.fp8; pa.int8 = f.int8; pa.nvfp4 = f.nvfp4;
                pa.q = q; pa.r = r; pa.sink = sink;
                ink::AttnArgs aa;
                aa.q = q; aa.r = r; aa.sink = sink; aa.rel_table = table; aa.pos = pos; aa.k_cache = kc; aa.v_cache = vc; aa.cap = cap;
                aa.fp8 = f.fp8; aa.int8 = f.int8; aa.nvfp4 = f.nvfp4;
                aa.window = window ? ink::kExtent : 0; aa.gate = y + ink::kOffGate; aa.gate_ld = ink::kProj;
                aa.q8 = q8;
                aa.min_full_keys = window ? 0 : p0 + 1;
                const int64_t keys = window ? ink::kExtent : depth;
                constexpr int reps = 50;
                ink::attention_window(pa, aa, rows, keys, scr, nullptr);
                cudaEventRecord(t0);
                for (int i = 0; i < reps; ++i) ink::attention_window(pa, aa, rows, keys, scr, nullptr);
                cudaEventRecord(t1);
                float ms = 0.f;
                ok = check_cuda(cudaEventSynchronize(t1), "bench") && check_cuda(cudaEventElapsedTime(&ms, t0, t1), "bench");
                const double us = (double) ms * 1000. / reps, kv = 2. * (double) keys * ink::kKvHeads * (double) row_bytes;
                std::printf("kv bench: %s %-6s %d rows: %8.1f us a window's attention, %6.0f GB/s of one row's K and V\n",
                            window ? "window ring" : "full cache ", f.name, rows, us, kv / us / 1000.);
                if (!window) {
                    ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                    cudaEventRecord(t0);
                    for (int i = 0; i < reps; ++i) ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                    cudaEventRecord(t1);
                    float elapsed = 0.f;
                    ok = check_cuda(cudaEventSynchronize(t1), "bench read") &&
                         check_cuda(cudaEventElapsedTime(&elapsed, t0, t1), "bench elapsed") && ok;
                    const double us2 = (double) elapsed * 1000. / reps;
                    std::printf("kv bench: depth %-6lld %-6s %d rows: %8.1f us the read alone,    %6.0f GB/s of one row's K and V\n",
                                (long long) depth, f.name, rows, us2, kv / us2 / 1000.);
                }
            }
            cudaFree(kc);
            cudaFree(vc);
        }
    if (t0) cudaEventDestroy(t0);
    if (t1) cudaEventDestroy(t1);
    for (void* p : {(void*) y, (void*) taps, (void*) qs, (void*) rs, (void*) kr, (void*) vr, (void*) table, (void*) q, (void*) r,
                    (void*) sink, (void*) pos, (void*) q8, scr})
        if (p) cudaFree(p);
    return ok;
}

int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc > 2 || (argc == 2 && !bench)) {
        std::fprintf(stderr, "usage: inkling2_kv_nvfp4_parity [--bench]\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;   // no CUDA device: skip

    constexpr int rows = 257;
    constexpr size_t row_bytes = ink::kKvNvfp4RowBytes;
    std::mt19937 rng(0x4e76);
    std::normal_distribution<float> dist(0.f, 0.35f);
    std::vector<float> input((size_t) rows * ink::kHeadDim);
    for (float& x : input) x = dist(rng);
    for (int c = 0; c < ink::kHeadDim; ++c) input[c] = 0.f;
    for (int c = 0; c < ink::kHeadDim; ++c) input[(size_t) (rows - 2) * ink::kHeadDim + c] = (c % 9 == 0) ? 1.25f : -0.015f;
    std::copy_n(input.data() + (size_t) (rows - 2) * ink::kHeadDim, ink::kHeadDim, input.data() + (size_t) (rows - 1) * ink::kHeadDim);

    float* di = nullptr;
    uint8_t* dout = nullptr;
    if (!check_cuda(cudaMalloc(&di, input.size() * sizeof(float) * 2), "allocate input") ||
        !check_cuda(cudaMalloc(&dout, (size_t) rows * row_bytes * 2), "allocate rows") ||
        !check_cuda(basalt::core::device_copy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input") ||
        !check_cuda(cudaMemset(dout, 0xa5, (size_t) rows * row_bytes * 2), "poison rows")) return 1;

    // keys (Hadamard-rotated) into the first rows, values (unrotated) into the second
    encode_rows_nvfp4<<<rows, 32>>>(di, dout, rows, 1);
    encode_rows_nvfp4<<<rows, 32>>>(di, dout + (size_t) rows * row_bytes, rows, 0);
    if (!check_cuda(cudaGetLastError(), "launch") || !check_cuda(cudaDeviceSynchronize(), "synchronize")) return 1;

    std::vector<uint8_t> got((size_t) rows * row_bytes * 2);
    if (!check_cuda(basalt::core::device_copy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost), "copy rows")) return 1;

    double max_abs = 0., max_rel = 0.;
    int mismatches = 0;
    for (int which = 0; which < 2; ++which)
        for (int r = 0; r < rows; ++r) {
            std::vector<uint8_t> want(row_bytes);
            std::vector<double> held(ink::kHeadDim);
            put_row_nvfp4(input.data() + (size_t) r * ink::kHeadDim, which == 0, want.data(), held.data());
            const uint8_t* row = got.data() + ((size_t) which * rows + r) * row_bytes;
            if (std::memcmp(row, want.data(), row_bytes) != 0) ++mismatches;
            for (size_t b = ink::kHeadDim / 2 + ink::kHeadDim / 16; b < row_bytes; ++b)
                if (row[b] != 0) ++mismatches;
            // the dequantized error against the row's rotated (keys) or raw (values) input
            float rotated[128];
            for (int dd = 0; dd < ink::kHeadDim; ++dd) rotated[dd] = input[(size_t) r * ink::kHeadDim + dd];
            if (which == 0) hadamard128_cpu(rotated);
            for (int d = 0; d < ink::kHeadDim; ++d) {
                const uint32_t q = d % 2 == 0 ? row[d / 2] & 15u : row[d / 2] >> 4;
                const double dequant = e2m1_value_cpu(q) * (double) e4m3_value_cpu(row[ink::kHeadDim / 2 + d / 16]);
                const double abs = std::abs(dequant - (double) rotated[d]);
                max_abs = std::max(max_abs, abs);
                const double ax = std::abs((double) rotated[d]);
                if (ax >= 1e-2) max_rel = std::max(max_rel, abs / ax);
            }
        }
    if (std::memcmp(got.data() + (size_t) (rows - 2) * row_bytes, got.data() + (size_t) (rows - 1) * row_bytes, row_bytes) != 0)
        ++mismatches;
    std::printf("kv_nvfp4_parity: rows=%d width=%d row_bytes=%zu mismatches=%d max_abs=%.7g max_rel(|x|>=1e-2)=%.7g\n",
                rows, ink::kHeadDim, row_bytes, mismatches, max_abs, max_rel);

    bool ok = mismatches == 0 && check_attention_nvfp4();
    if (bench) {
        constexpr int bench_rows = 4096;
        constexpr int reps = 200;
        float* bi = nullptr;
        uint8_t* bo = nullptr;
        cudaEvent_t start{}, stop{};
        ok = ok && check_cuda(cudaMalloc(&bi, (size_t) bench_rows * ink::kHeadDim * sizeof(float)), "allocate bench input");
        ok = ok && check_cuda(cudaMalloc(&bo, (size_t) bench_rows * row_bytes), "allocate bench rows");
        ok = ok && check_cuda(cudaMemset(bi, 0, (size_t) bench_rows * ink::kHeadDim * sizeof(float)), "clear bench input");
        ok = ok && check_cuda(cudaEventCreate(&start), "create start event") && check_cuda(cudaEventCreate(&stop), "create stop event");
        if (ok) {
            encode_rows_nvfp4<<<bench_rows, 32>>>(bi, bo, bench_rows, 1);
            ok = check_cuda(cudaGetLastError(), "bench warm-up") && check_cuda(cudaDeviceSynchronize(), "bench warm-up sync");
        }
        if (ok) ok = check_cuda(cudaEventRecord(start), "record start");
        for (int i = 0; ok && i < reps; ++i) encode_rows_nvfp4<<<bench_rows, 32>>>(bi, bo, bench_rows, 1);
        if (ok) ok = check_cuda(cudaEventRecord(stop), "record stop") && check_cuda(cudaEventSynchronize(stop), "wait stop");
        float ms = 0.f;
        if (ok) ok = check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        if (ok) {
            const double us = (double) ms * 1000. / reps;
            const double bytes = (double) bench_rows * (ink::kHeadDim * sizeof(float) + row_bytes);
            std::printf("kv_nvfp4_parity bench: rows=%d us/call=%.3f GB/s=%.1f\n", bench_rows, us, bytes / us / 1000.);
        }
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (bi) cudaFree(bi);
        if (bo) cudaFree(bo);
        ok = ok && bench_attention();
    }
    cudaFree(di);
    cudaFree(dout);
    return ok ? 0 : 1;
}
