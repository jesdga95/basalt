// src/models/inkling2/kernels/kv_int8_parity.cu - the Inkling2 INT8 KV row writer against a CPU reference; with --bench
// also the decode window's attention (prepare and read) over each cache format at the model's depths.
#include "prep.cuh"

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
    const int64_t depths[] = {0, 8192, 65536, 131072, 262144};   // 0: the sliding-window ring
    const int widths[] = {1, 3, 8};
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
                ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                cudaEventRecord(t0);
                for (int i = 0; i < reps; ++i) ink::attention_decode(aa, rows, keys, scr, nullptr, nullptr);
                cudaEventRecord(t1);
                ok = ok && check_cuda(cudaEventSynchronize(t1), "bench") && check_cuda(cudaEventElapsedTime(&ms, t0, t1), "bench");
                const double us2 = (double) ms * 1000. / reps;
                std::printf("kv bench: depth %-6lld %-7s %d rows: %8.1f us the read alone,    %6.0f GB/s of one row's K and V\n",
                            (long long) depth, f.name, rows, us2, kv / us2 / 1000.);
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

    bool ok = mismatches == 0;
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
