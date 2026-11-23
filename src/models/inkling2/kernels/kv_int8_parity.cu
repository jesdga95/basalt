// src/models/inkling2/kernels/kv_int8_parity.cu - INT8 KV writer and scalar/MMA reader parity against CPU references.
// With --bench, also time decode window attention over every cache format at the model's depths.
#include "prep.cuh"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ink = basalt::models::inkling2;

__global__ void encode_rows(const float* input, uint8_t* output, int rows) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) return;

    float v[4], scale[2];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = input[(size_t) row * ink::kHeadDim + lane * 4 + u];
    ink::prep::int8_scales(v, 1.f, lane, scale);
#pragma unroll
    for (int u = 0; u < 4; ++u)
        ink::prep::store_kv(output + (size_t) row * ink::kKvInt8RowBytes, false, true,
                            (size_t) lane * 4 + u, v[u], scale, lane);
}

static bool check_cuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "kv_int8_parity: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// A decode window of `rows` rows (positions depth - rows ..) over one visit's cache in each format: a sliding-window ring
// (the fused kernel) and a full cache at each depth (prepare + split attention). us a call, and the K and V bytes one row
// reads over that time (the cache's own bytes: every row of a window reads them, from L2 after the first).
static bool bench_attention() {
    struct Fmt { const char* name; bool fp8, int8; };
    const Fmt fmts[] = {{"fp16", false, false}, {"e4m3", true, false}, {"int8", false, true}};
    const int64_t depths[] = {0, 8192, 32768, 65536, ink::kMmaSingletonShallowKeys, 131072, 262144};
    const int widths[] = {1, 2, 3, 5, 8};
    int dev = 0;
    cudaDeviceProp prop{};
    if (!check_cuda(cudaGetDevice(&dev), "bench device") ||
        !check_cuda(cudaGetDeviceProperties(&prop, dev), "bench device properties")) return false;
    std::printf("kv bench: device=%s SMs=%d L2=%d shared_per_SM=%zu\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize, prop.sharedMemPerMultiprocessor);
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
            if ((depth == 32768 || depth == ink::kMmaSingletonShallowKeys) && !f.int8) continue;
            const bool window = depth == 0;
            const int64_t cap = window ? ink::kExtent + ink::kDecodeRows : depth;
            const size_t row_bytes = (size_t) ink::kv_cache_row_bytes(f.fp8, f.int8), cache = (size_t) cap * ink::kKvHeads * row_bytes;
            uint8_t *kc = nullptr, *vc = nullptr;
            if (!ok || !check_cuda(cudaMalloc(&kc, cache), "k cache") || !check_cuda(cudaMalloc(&vc, cache), "v cache")) {
                ok = false;
                break;
            }
            cudaMemset(kc, 0x11, cache);
            cudaMemset(vc, 0x11, cache);
            for (const int rows : widths) {
                const bool compare = f.int8 && depth >= 32768;
                if ((rows == 2 || rows == 5) && !compare) continue;
                const int64_t p0 = (window ? 8192 : depth) - rows;
                std::vector<int32_t> hp(rows);
                for (int i = 0; i < rows; ++i) hp[i] = (int32_t) (p0 + i);
                cudaMemcpy(pos, hp.data(), rows * 4, cudaMemcpyHostToDevice);
                ink::AttnPrep pa;
                pa.yq = y; pa.yk = y + ink::kProjQ; pa.yv = y + ink::kProjQ + ink::kProjKV; pa.ldq = pa.ldk = pa.ldv = ink::kProj;
                pa.pos = pos; pa.full = !window; pa.k_conv = taps; pa.v_conv = taps; pa.q_scale = qs; pa.r_scale = rs;
                pa.k_ring = kr; pa.v_ring = vr; pa.k_cache = kc; pa.v_cache = vc; pa.cap = cap; pa.fp8 = f.fp8; pa.int8 = f.int8;
                pa.q = q; pa.r = r; pa.sink = sink;
                ink::AttnArgs aa;
                aa.q = q; aa.r = r; aa.sink = sink; aa.rel_table = table; aa.pos = pos; aa.k_cache = kc; aa.v_cache = vc; aa.cap = cap;
                aa.fp8 = f.fp8; aa.int8 = f.int8; aa.window = window ? ink::kExtent : 0; aa.gate = y + ink::kOffGate; aa.gate_ld = ink::kProj;
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
                std::printf("kv bench: %s %-7s %d rows: %8.1f us a window's attention, %6.0f GB/s of one row's K and V\n",
                            window ? "window ring" : "full cache ", f.name, rows, us, kv / us / 1000.);
                if (window) continue;
                const auto read_us = [&](bool mma, bool prepared_pdl = false) {
                    const auto launch = [&] {
                        basalt::kernels::PdlScope pdl(prepared_pdl, true);
                        if (prepared_pdl) ink::decode_prepare(pa, rows, nullptr);
                        if (compare) ink::attention_decode_int8_test(aa, rows, keys, scr, nullptr, nullptr, mma);
                        else ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                    };
                    launch();
                    cudaEventRecord(t0);
                    for (int i = 0; i < reps; ++i) launch();
                    cudaEventRecord(t1);
                    float elapsed = 0.f;
                    ok = check_cuda(cudaEventSynchronize(t1), "bench read") &&
                         check_cuda(cudaEventElapsedTime(&elapsed, t0, t1), "bench elapsed") && ok;
                    return (double) elapsed * 1000. / reps;
                };
                double us2 = read_us(false);
                if (compare) {
                    const double first = read_us(true);
                    const double second = read_us(true);
                    us2 = (us2 + read_us(false)) / 2.;
                    const double mma_us = (first + second) / 2.;
                    std::printf("kv bench: depth %-6lld int8    %d rows: %8.1f us MMA read alone, %6.0f GB/s of one row's K and V\n",
                                (long long) depth, rows, mma_us, kv / mma_us / 1000.);
                }
                std::printf("kv bench: depth %-6lld %-7s %d rows: %8.1f us the read alone,    %6.0f GB/s of one row's K and V\n",
                            (long long) depth, f.name, rows, us2, kv / us2 / 1000.);
                if (compare && rows <= 2) {
                    const double scalar_a = read_us(false, true);
                    const double mma_b = read_us(true, true);
                    const double mma_c = read_us(true, true);
                    const double scalar_d = read_us(false, true);
                    std::printf("kv bench: depth %-6lld int8    %d rows: %8.1f -> %8.1f us scalar -> MMA prepare+read (PDL)\n",
                                (long long) depth, rows, (scalar_a + scalar_d) / 2., (mma_b + mma_c) / 2.);
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

// Repeat random cache records over a prime period so the CPU can fold the far keys by multiplicity.
// The GPU still reads every actual key, including unaligned tails and the full relative-bias band.
constexpr int kPeriod = 257;
constexpr int kOutput = ink::kHeads * ink::kHeadDim;
constexpr int kOutputQ8 = kOutput / 32 * 36;
constexpr int kCacheCell = ink::kKvHeads * ink::kKvInt8RowBytes;

static void random_cache(std::mt19937& rng, std::vector<uint8_t>& bytes, std::vector<double>& exact) {
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> magnitude(0.002f, 0.015f);
    for (int i = 0; i < kPeriod * ink::kKvHeads; ++i) {
        auto* row = bytes.data() + (size_t) i * ink::kKvInt8RowBytes;
        for (int group = 0; group < 2; ++group) {
            const __half scale = __float2half_rn(magnitude(rng));
            std::memcpy(row + ink::kHeadDim + group * sizeof(scale), &scale, sizeof(scale));
            for (int d = 64 * group; d < 64 * (group + 1); ++d) {
                const int x = code(rng);
                row[d] = (uint8_t) x;
                exact[(size_t) i * ink::kHeadDim + d] = (double) x * __half2float(scale);
            }
        }
    }
}

static void cpu_attention(int pos, const float* q, const float* r, const float* sink, const float* gate,
                          const float* table, const std::vector<double>& kc, const std::vector<double>& vc,
                          double scale, double* out) {
    const int near_first = std::max(0, pos + 1 - ink::kExtent);
    for (int h = 0; h < ink::kHeads; ++h) {
        const int kvh = h / (ink::kHeads / ink::kKvHeads);
        double base[kPeriod], near[ink::kExtent], weights[kPeriod] = {};
        double m = sink[h];
        for (int key = 0; key < kPeriod; ++key) {
            const double* k = kc.data() + (key * ink::kKvHeads + kvh) * ink::kHeadDim;
            double score = 0.;
            for (int d = 0; d < ink::kHeadDim; ++d) score += (double) q[h * ink::kHeadDim + d] * k[d];
            base[key] = score * scale;
            if (key < near_first) m = std::max(m, base[key]);
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
            const int count = near_first / kPeriod + (key < near_first % kPeriod);
            if (count) weights[key] = count * std::exp(base[key] - m);
        }
        for (int key = near_first; key <= pos; ++key)
            weights[key % kPeriod] += std::exp(near[pos - key] - m);
        double denom = std::exp(sink[h] - m), numer[ink::kHeadDim] = {};
        for (int key = 0; key < kPeriod; ++key) {
            const double* v = vc.data() + (key * ink::kKvHeads + kvh) * ink::kHeadDim;
            denom += weights[key];
            for (int d = 0; d < ink::kHeadDim; ++d) numer[d] += weights[key] * v[d];
        }
        for (int d = 0; d < ink::kHeadDim; ++d) {
            const int j = h * ink::kHeadDim + d;
            out[j] = numer[d] / denom / (1. + std::exp(-(double) gate[j]));
        }
    }
}

struct DecodeResult {
    std::vector<float> fp32 = std::vector<float>(ink::kDecodeRows * kOutput);
    std::vector<uint8_t> q8 = std::vector<uint8_t>(ink::kDecodeRows * kOutputQ8);
};

static bool same_output(const DecodeResult& a, int ar, const DecodeResult& b, int br) {
    return std::memcmp(a.fp32.data() + ar * kOutput, b.fp32.data() + br * kOutput, kOutput * sizeof(float)) == 0 &&
           std::memcmp(a.q8.data() + ar * kOutputQ8, b.q8.data() + br * kOutputQ8, kOutputQ8) == 0;
}

static uint64_t output_hash(const DecodeResult& a, int row) {
    uint64_t hash = 14695981039346656037ull;
    const auto mix = [&](const unsigned char* bytes, size_t n) {
        for (size_t j = 0; j < n; ++j) hash = (hash ^ bytes[j]) * 1099511628211ull;
    };
    mix(reinterpret_cast<const unsigned char*>(a.fp32.data() + row * kOutput), kOutput * sizeof(float));
    mix(a.q8.data() + row * kOutputQ8, kOutputQ8);
    return hash;
}

static bool check_attention_mma() {
    constexpr int rows = ink::kDecodeRows, cap = 262144 + rows;
    std::mt19937 rng(0x4b323062);
    std::normal_distribution<float> normal(0.f, 0.4f);
    std::vector<float> q(rows * kOutput), r(rows * ink::kHeads * ink::kRel), sink(rows * ink::kHeads),
        gate(rows * kOutput), table(ink::kRel * ink::kExtent);
    for (auto* a : {&q, &r, &sink, &gate, &table})
        for (float& x : *a) x = normal(rng);
    std::vector<uint8_t> kp(kPeriod * kCacheCell), vp(kPeriod * kCacheCell);
    std::vector<double> hk(kPeriod * ink::kKvHeads * ink::kHeadDim), hv(hk.size());
    random_cache(rng, kp, hk);
    random_cache(rng, vp, hv);

    int32_t* dp = nullptr;
    float *dq = nullptr, *dr = nullptr, *ds = nullptr, *dg = nullptr, *dt = nullptr, *dout = nullptr;
    uint8_t *dk = nullptr, *dv = nullptr, *dq8 = nullptr;
    void* scratch = nullptr;
    const auto alloc = [](auto** ptr, size_t bytes) {
        return check_cuda(cudaMalloc((void**) ptr, bytes), "attention parity allocate");
    };
    const auto upload = [&](float** ptr, const std::vector<float>& v) {
        const size_t bytes = v.size() * sizeof(float);
        return alloc(ptr, bytes) && check_cuda(cudaMemcpy(*ptr, v.data(), bytes, cudaMemcpyHostToDevice), "attention inputs");
    };
    const auto upload_cache = [&](uint8_t** ptr, const std::vector<uint8_t>& pattern) {
        std::vector<uint8_t> host((size_t) cap * kCacheCell);
        for (size_t i = 0; i < host.size(); i += pattern.size())
            std::memcpy(host.data() + i, pattern.data(), std::min(pattern.size(), host.size() - i));
        return alloc(ptr, host.size()) &&
               check_cuda(cudaMemcpy(*ptr, host.data(), host.size(), cudaMemcpyHostToDevice), "attention cache");
    };
    bool ok = upload(&dq, q) && upload(&dr, r) && upload(&ds, sink) && upload(&dg, gate) && upload(&dt, table) &&
              upload_cache(&dk, kp) && upload_cache(&dv, vp) && alloc(&dp, rows * sizeof(int32_t)) &&
              alloc(&dout, rows * kOutput * sizeof(float)) && alloc(&dq8, rows * kOutputQ8) &&
              alloc(&scratch, ink::attention_decode_scratch(rows, cap));
    for (const int depth : {4096, 32768, 65536, (int) ink::kMmaSingletonShallowKeys, 131072, 262144}) {
        if (!ok) break;
        int32_t pos[rows];
        for (int j = 0; j < rows; ++j) pos[j] = depth - 4 + j;
        ok = check_cuda(cudaMemcpy(dp, pos, sizeof(pos), cudaMemcpyHostToDevice), "attention positions");
        if (!ok) break;
        enum class Reader { Auto, Scalar, Mma };
        const auto run = [&](int first, int width, Reader reader, DecodeResult& dst,
                             bool batch = false, bool swap_kv = false, int key_bound = 0) {
            ink::AttnArgs a;
            a.int8 = true;
            a.q = dq + first * kOutput;
            a.r = dr + first * ink::kHeads * ink::kRel;
            a.sink = ds + first * ink::kHeads;
            a.gate = dg + first * kOutput;
            a.gate_ld = kOutput;
            a.rel_table = dt;
            a.pos = dp + first;
            a.k_cache = swap_kv ? dv : dk;
            a.v_cache = swap_kv ? dk : dv;
            a.cap = cap;
            a.q8 = dq8;
            if (batch) {
                a.seg.at.n = 2;
                a.seg.at.row0[1] = 5;
                a.seg.at.row0[2] = width;
                a.seg.c[0].k = a.seg.c[1].v = dk;
                a.seg.c[0].v = a.seg.c[1].k = dv;
                a.k_cache = a.v_cache = nullptr;
            } else a.min_full_keys = pos[first] + 1;
            const int keys = key_bound ? key_bound : depth + 4;
            if (reader == Reader::Auto) ink::attention_decode(a, width, keys, scratch, dout, nullptr);
            else ink::attention_decode_int8_test(a, width, keys, scratch, dout, nullptr, reader == Reader::Mma);
            return check_cuda(cudaMemcpy(dst.fp32.data(), dout, width * kOutput * sizeof(float), cudaMemcpyDeviceToHost),
                              "attention FP32 output") &&
                   check_cuda(cudaMemcpy(dst.q8.data(), dq8, width * kOutputQ8, cudaMemcpyDeviceToHost), "attention q8_1 output");
        };
        std::vector<double> ref(rows * kOutput);
        DecodeResult singleton[rows];
        for (int j = 0; j < rows && ok; ++j) {
            cpu_attention(pos[j], q.data() + j * kOutput, r.data() + j * ink::kHeads * ink::kRel,
                          sink.data() + j * ink::kHeads, gate.data() + j * kOutput, table.data(), hk, hv,
                          ink::AttnArgs{}.scale, ref.data() + j * kOutput);
            // Also vary the launch bound, which can change both the pipeline plan and CTA grouping.
            ok = run(j, 1, Reader::Mma, singleton[j], false, false, cap);
            if (ok && pos[j] + 1 >= ink::kMmaDeepMinKeys && pos[j] + 1 <= ink::kMmaSingletonShallowKeys) {
                DecodeResult short_bound;
                ok = run(j, 1, Reader::Mma, short_bound, false, false, pos[j] + 1) &&
                     same_output(short_bound, 0, singleton[j], 0);
                if (!ok) std::printf("kv MMA parity: singleton schedule mismatch keys=%d\n", pos[j] + 1);
            }
        }
        for (const int width : {2, 3, 5, 8}) {
            if (!ok) break;
            const int first = 3 - (width - 1) / 2;
            DecodeResult scalar, mma, automatic, segmented, separate;
            ok = run(first, width, Reader::Scalar, scalar) && run(first, width, Reader::Mma, mma) &&
                 run(first, width, Reader::Auto, automatic);
            if (ok && width == rows)
                ok = run(first, width, Reader::Auto, segmented, true) &&
                     run(first + 5, width - 5, Reader::Auto, separate, false, true);
            if (!ok) break;
            for (int j = 0; j < width; ++j) {
                double abs[3] = {}, rel[3] = {};
                bool finite = true;
                for (int d = 0; d < kOutput; ++d) {
                    const double ref_out = ref[(first + j) * kOutput + d];
                    const double old_out = scalar.fp32[j * kOutput + d], mma_out = mma.fp32[j * kOutput + d];
                    finite &= std::isfinite(old_out) && std::isfinite(mma_out);
                    const double actual[] = {mma_out, mma_out, old_out};
                    const double want[] = {old_out, ref_out, ref_out};
                    for (int k = 0; k < 3; ++k) {
                        const double error = std::abs(actual[k] - want[k]);
                        abs[k] = std::max(abs[k], error);
                        if (std::abs(want[k]) >= 1e-2) rel[k] = std::max(rel[k], error / std::abs(want[k]));
                    }
                }
                const bool invariant = same_output(mma, j, singleton[first + j], 0);
                const bool dispatched = same_output(automatic, j, pos[first + j] + 1 >= 32768 ? mma : scalar, j);
                const bool batch_invariant = width != rows ||
                    (j < 5 ? same_output(automatic, j, segmented, j) : same_output(separate, j - 5, segmented, j));
                std::printf("kv MMA parity: depth=%d T=%d row=%d keys=%d max_abs[mma/scalar,mma/cpu,scalar/cpu]=%.7g/%.7g/%.7g "
                            "max_rel(|ref|>=1e-2)=%.7g/%.7g/%.7g finite=%d invariant=%d dispatch=%d batch=%d hash=%016llx\n",
                            depth, width, j, pos[first + j] + 1, abs[0], abs[1], abs[2], rel[0], rel[1], rel[2],
                            finite, invariant, dispatched, batch_invariant, (unsigned long long) output_hash(mma, j));
                ok &= finite && invariant && dispatched && batch_invariant &&
                      *std::max_element(abs, abs + 3) < 0.002 && *std::max_element(rel, rel + 3) < 0.05;
            }
        }
    }
    for (void* p : {(void*) dp, (void*) dq, (void*) dr, (void*) ds, (void*) dg, (void*) dt, (void*) dout, (void*) dq8,
                    (void*) dk, (void*) dv, scratch})
        if (p) cudaFree(p);
    return ok;
}

int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc > 2 || (argc == 2 && !bench)) {
        std::fprintf(stderr, "usage: inkling2_kv_int8_parity [--bench]\n");
        return 2;
    }

    constexpr int rows = 257;
    constexpr size_t row_bytes = ink::kKvInt8RowBytes;
    std::mt19937 rng(0x1a7e);
    std::normal_distribution<float> dist(0.f, 0.35f);
    std::vector<float> input((size_t) rows * ink::kHeadDim);
    for (float& x : input) x = dist(rng);
    for (int c = 0; c < ink::kHeadDim; ++c) input[c] = 0.f;
    for (int c = 0; c < ink::kHeadDim; ++c) input[(size_t) (rows - 2) * ink::kHeadDim + c] = (c % 9 == 0) ? 1.25f : -0.015f;
    std::copy_n(input.data(), ink::kHeadDim, input.data() + (size_t) (rows - 1) * ink::kHeadDim);

    float *di = nullptr;
    uint8_t* dout = nullptr;
    if (!check_cuda(cudaMalloc(&di, input.size() * sizeof(float)), "allocate input") ||
        !check_cuda(cudaMalloc(&dout, (size_t) rows * row_bytes), "allocate rows") ||
        !check_cuda(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input") ||
        !check_cuda(cudaMemset(dout, 0xa5, (size_t) rows * row_bytes), "poison rows")) return 1;

    encode_rows<<<rows, 32>>>(di, dout, rows);
    if (!check_cuda(cudaGetLastError(), "launch") || !check_cuda(cudaDeviceSynchronize(), "synchronize")) return 1;

    std::vector<uint8_t> got((size_t) rows * row_bytes);
    if (!check_cuda(cudaMemcpy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost), "copy rows")) return 1;

    double max_abs = 0., max_rel = 0., fp16_max_abs = 0., fp16_max_rel = 0.;
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        float scales[2]{};
        for (int g = 0; g < 2; ++g) {
            float m = 0.f;
            for (int c = g * 64; c < (g + 1) * 64; ++c)
                m = std::max(m, std::fabs(input[(size_t) r * ink::kHeadDim + c]));
            scales[g] = __half2float(__float2half_rn(m / 127.f));
            __half stored;
            std::memcpy(&stored, got.data() + (size_t) r * row_bytes + ink::kHeadDim + g * sizeof(__half), sizeof(stored));
            if (__half_as_ushort(stored) != __half_as_ushort(__float2half_rn(scales[g]))) ++mismatches;
        }
        for (int c = 0; c < ink::kHeadDim; ++c) {
            const float x = input[(size_t) r * ink::kHeadDim + c];
            const float s = scales[c / 64];
            const int q = s > 0.f ? std::clamp((int) std::nearbyint(x / s), -127, 127) : 0;
            const int got_q = (int) (int8_t) got[(size_t) r * row_bytes + c];
            if (got_q != q) ++mismatches;
            const float dequant = (float) got_q * s;
            const double abs = std::fabs((double) dequant - x);
            max_abs = std::max(max_abs, abs);
            const double ax = std::fabs((double) x);
            if (ax >= 1e-2) max_rel = std::max(max_rel, abs / ax);
            const float h = __half2float(__float2half_rn(x));
            const double h_abs = std::fabs((double) h - x);
            fp16_max_abs = std::max(fp16_max_abs, h_abs);
            if (ax >= 1e-2) fp16_max_rel = std::max(fp16_max_rel, h_abs / ax);
        }
        for (size_t b = ink::kHeadDim + 2 * sizeof(__half); b < row_bytes; ++b)
            if (got[(size_t) r * row_bytes + b] != 0) ++mismatches;
    }
    if (std::memcmp(got.data(), got.data() + (size_t) (rows - 1) * row_bytes, row_bytes) != 0) ++mismatches;
    std::printf("kv_int8_parity: rows=%d width=%d row_bytes=%zu mismatches=%d max_abs=%.7g max_rel(|x|>=1e-2)=%.7g fp16_max_abs=%.7g fp16_max_rel(|x|>=1e-2)=%.7g\n",
                rows, ink::kHeadDim, row_bytes, mismatches, max_abs, max_rel, fp16_max_abs, fp16_max_rel);

    bool ok = mismatches == 0 && check_attention_mma();
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
            encode_rows<<<bench_rows, 32>>>(bi, bo, bench_rows);
            ok = check_cuda(cudaGetLastError(), "bench warm-up") && check_cuda(cudaDeviceSynchronize(), "bench warm-up sync");
        }
        if (ok) ok = check_cuda(cudaEventRecord(start), "record start");
        for (int i = 0; ok && i < reps; ++i) encode_rows<<<bench_rows, 32>>>(bi, bo, bench_rows);
        if (ok) ok = check_cuda(cudaEventRecord(stop), "record stop") && check_cuda(cudaEventSynchronize(stop), "wait stop");
        float ms = 0.f;
        if (ok) ok = check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        if (ok) {
            const double us = (double) ms * 1000. / reps;
            const double bytes = (double) bench_rows * (ink::kHeadDim * sizeof(float) + row_bytes);
            std::printf("kv_int8_parity bench: rows=%d us/call=%.3f GB/s=%.1f\n", bench_rows, us, bytes / us / 1000.);
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
