// tests/kernels/router_k22_parity.cpp - the shared router (basalt/kernels/router.hpp) at Nemotron-H Super's geometry:
// 512 experts over 4096 values, a sigmoid gate with a selection bias, 22 chosen, the weights normalised and scaled by 5.
// Against a double reference (the chosen ids as a set, the weights), a token's result alone and in a window of 16,
// bitwise; then a microbench.  Exit 77 without a GPU.
#include "basalt/kernels/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int NE = 512, ND = 4096, K = 22, T = 16;
constexpr float kScale = 5.0f;
}  // namespace

int main() {
    namespace B = basalt::kernels;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> gate((size_t) NE * ND), bias(NE), x((size_t) T * ND);
    for (auto& v : gate) v = 0.02f * nd(rng);
    for (auto& v : bias) v = 0.05f * nd(rng);
    for (auto& v : x) v = nd(rng);
    float *dg, *db, *dx, *dl, *dw;
    int32_t* di;
    cudaMalloc((void**) &dg, gate.size() * 4);
    cudaMalloc((void**) &db, bias.size() * 4);
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMalloc((void**) &dl, (size_t) T * NE * 4);
    cudaMalloc((void**) &dw, (size_t) T * K * 4);
    cudaMalloc((void**) &di, (size_t) T * K * 4);
    cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    B::RouterArgs a;
    a.n_expert = NE;
    a.n_embd = ND;
    a.k = K;
    a.weight_scale = kScale;
    a.gate = B::RouterGate::Sigmoid;
    B::router(dx, dg, db, T, a, dl, di, dw, nullptr);
    cudaDeviceSynchronize();
    std::vector<int32_t> ids((size_t) T * K);
    std::vector<float> w((size_t) T * K);
    cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(w.data(), dw, w.size() * 4, cudaMemcpyDeviceToHost);
    int failures = 0, id_diff = 0, near = 0;
    double wrel = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> s(NE), b(NE);
        for (int e = 0; e < NE; ++e) {
            double acc = 0;
            for (int d = 0; d < ND; ++d) acc += (double) gate[(size_t) e * ND + d] * x[(size_t) t * ND + d];
            s[(size_t) e] = 1.0 / (1.0 + std::exp(-acc));
            b[(size_t) e] = s[(size_t) e] + bias[(size_t) e];
        }
        std::vector<int> idx(NE);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int p, int q) { return b[(size_t) p] > b[(size_t) q]; });
        if (b[(size_t) idx[K - 1]] - b[(size_t) idx[K]] < 1e-6) ++near;   // a near-tie at the boundary may pick either
        double sum = 0;
        for (int k = 0; k < K; ++k) sum += s[(size_t) idx[k]];
        for (int k = 0; k < K; ++k) {
            const int e = ids[(size_t) t * K + k];
            if (e != idx[k]) ++id_diff;
            const double want = s[(size_t) e] / (sum + 1e-20) * kScale;
            wrel = std::max(wrel, std::fabs(w[(size_t) t * K + k] - want) / want);
        }
    }
    // each token alone: bitwise its row of the window
    bool alone = true;
    for (int t = 0; t < T; ++t) {
        B::router(dx + (size_t) t * ND, dg, db, 1, a, dl, di, dw, nullptr);
        cudaDeviceSynchronize();
        int32_t i1[K];
        float w1[K];
        cudaMemcpy(i1, di, K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(w1, dw, K * 4, cudaMemcpyDeviceToHost);
        if (std::memcmp(i1, ids.data() + (size_t) t * K, K * 4) || std::memcmp(w1, w.data() + (size_t) t * K, K * 4)) alone = false;
    }
    const bool ok = id_diff == 0 && wrel < 1e-5 && alone && cudaGetLastError() == cudaSuccess;
    std::printf("router k=22 sigmoid over 512 x 4096, %d tokens: ids vs double rank order %d differ (near-ties %d), weights "
                "max rel %.2e, token alone = in the window %s  %s\n", T, id_diff, near, wrel, alone ? "bitwise" : "DIFFERS",
                ok ? "ok" : "FAIL");
    failures += !ok;
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < 10; ++i) B::router(dx, dg, db, 1, a, dl, di, dw, nullptr);
        cudaEventRecord(e0);
        for (int i = 0; i < 200; ++i) B::router(dx, dg, db, 1, a, dl, di, dw, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / 200;
        std::printf("bench router 1 token: %.1f us (%.0f GB/s of the F32 gate)\n", us, (double) NE * ND * 4 / (us * 1e3));
    }
    for (void* p : {(void*) dg, (void*) db, (void*) dx, (void*) dl, (void*) dw, (void*) di}) cudaFree(p);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
