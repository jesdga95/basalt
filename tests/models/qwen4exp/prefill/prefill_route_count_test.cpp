// prefill_route_count_test - the routed-entry histogram kernel against a plain host count, on random routing at the
// model's geometry (512 experts, 10 per token) for chunk widths from one token to a full prompt chunk, with ids outside
// [0, n_expert) mixed in (skipped).  Counts are integers, so the match is exact, and a layer's counts accumulate across
// chunks.  Prints the kernel's time and the id bytes read per second.  Exit 77 without a CUDA device.
#include "basalt/models/qwen4exp/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) return 77;
    constexpr int NE = 512, K = 10;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> id(-2, NE + 1);
    int bad = 0;
    int32_t* d_ids = nullptr;
    uint32_t* d_hist = nullptr;
    cudaMalloc((void**) &d_ids, 8192 * K * 4);
    cudaMalloc((void**) &d_hist, NE * 4);
    cudaMemset(d_hist, 0, NE * 4);
    std::vector<uint32_t> want(NE, 0), got(NE);
    for (int T : {1, 2, 7, 64, 255, 256, 1024, 4096, 8192}) {
        std::vector<int32_t> ids((size_t) T * K);
        for (auto& v : ids) v = id(rng);
        for (int32_t v : ids)
            if (v >= 0 && v < NE) ++want[(size_t) v];
        cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
        basalt::prefill::route_count(d_ids, (int64_t) T * K, d_hist, NE, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), d_hist, NE * 4, cudaMemcpyDeviceToHost);
        int diff = 0;
        for (int e = 0; e < NE; ++e) diff += got[(size_t) e] != want[(size_t) e];
        std::printf("route_count T=%d: %d of %d counts differ\n", T, diff, NE);
        bad += diff;
    }
    const int R = 2000, T = 1024;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < R; ++i) basalt::prefill::route_count(d_ids, (int64_t) T * K, d_hist, NE, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / R;
    std::printf("route_count microbench T=%d: %.2f us per call, %.1f GB/s of ids\n", T, us, (double) T * K * 4 / (us * 1e3));
    cudaFree(d_ids);
    cudaFree(d_hist);
    return bad != 0 ? 1 : 0;
}
