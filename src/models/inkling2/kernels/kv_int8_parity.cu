// src/models/inkling2/kernels/kv_int8_parity.cu - the Inkling2 INT8 KV row writer against a CPU reference.
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
    }
    cudaFree(di);
    cudaFree(dout);
    return ok ? 0 : 1;
}
