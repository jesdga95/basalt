// tests/kernels/fp4_swiglu_quant_test.cpp - prefill::fp4::quantize_swiglu (a dense MLP's down input made from gate and up as
// the FP4 quantizer reads them) against kernels::swiglu_scaled then prefill::fp4::quantize: the planes and token scales
// bitwise.  Exit 77 without a GPU.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

namespace fp4 = basalt::prefill::fp4;

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    const int64_t T = 1100, W = 17408;   // a chunk past one 1024-row tile, the 27B's FFN width
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    std::vector<float> g((size_t) T * W), u((size_t) T * W);
    for (auto& v : g) v = 2.0f * nd(rng);
    for (auto& v : u) v = 3.0f * nd(rng);
    const float sc[2] = {0.75f, 1.5f};
    float *dg, *du, *dh, *ds, *gs1, *gs2;
    int32_t *bounds, *tile_off;
    cudaMalloc(&dg, g.size() * 4);
    cudaMalloc(&du, u.size() * 4);
    cudaMalloc(&dh, g.size() * 4);
    cudaMalloc(&ds, 8);
    cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(du, u.data(), u.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sc, 8, cudaMemcpyHostToDevice);
    const int32_t b[2] = {0, (int32_t) T};
    cudaMalloc(&bounds, 8);
    cudaMalloc(&tile_off, 64);
    cudaMemcpy(bounds, b, 8, cudaMemcpyHostToDevice);
    const size_t a4b = fp4::a4_bytes(T, W, 1), gsn = (size_t) fp4::scale_count(T, 1);
    void *a1, *a2;
    cudaMalloc(&a1, a4b);
    cudaMalloc(&a2, a4b);
    cudaMalloc(&gs1, gsn * 4);
    cudaMalloc(&gs2, gsn * 4);
    cudaMemset(a1, 0, a4b);
    cudaMemset(a2, 0, a4b);
    cudaMemset(gs1, 0, gsn * 4);
    cudaMemset(gs2, 0, gsn * 4);
    fp4::plan(bounds, 1, tile_off, nullptr);
    basalt::kernels::swiglu_scaled(dg, du, ds, ds + 1, dh, T, W, W, nullptr);
    fp4::quantize(dh, nullptr, bounds, tile_off, 1, a1, gs1, W, W, T, nullptr);
    fp4::quantize_swiglu(dg, du, ds, ds + 1, bounds, tile_off, 1, a2, gs2, W, W, T, nullptr);
    std::vector<unsigned char> h1(a4b), h2(a4b);
    std::vector<float> s1(gsn), s2(gsn);
    cudaMemcpy(h1.data(), a1, a4b, cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), a2, a4b, cudaMemcpyDeviceToHost);
    cudaMemcpy(s1.data(), gs1, gsn * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s2.data(), gs2, gsn * 4, cudaMemcpyDeviceToHost);
    const bool ok = cudaGetLastError() == cudaSuccess && h1 == h2 && s1 == s2;
    std::printf("quantize_swiglu against swiglu_scaled + quantize: %s\n", ok ? "bitwise" : "DIFFER  FAIL");
    return ok ? 0 : 1;
}
