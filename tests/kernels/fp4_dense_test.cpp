// tests/kernels/fp4_dense_test.cpp - a dense layer's NVFP4 products on the FP4 tensor cores (prefill::fp4::run with one
// matrix): a long chunk (four tiles per block) gives each row the bits a shorter one (two tiles per block) gives it, and
// the SwiGLU product (gate from the blob, up from its own matrix, scales from `tail`) equals silu(s_gate g) (s_up u) of
// the two plain products within the epilogue's approximate exp and reciprocal.  Exit 77 without a GPU.
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fp4 = basalt::prefill::fp4;

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    const int64_t R = 1024, K = 5120, T = 1100, T2 = 1000;
    const size_t mb = (size_t) R * (K / 64) * 36;
    std::mt19937 rng(9);
    auto matrix = [&]() {
        std::vector<uint8_t> w(mb);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t i = 0; i < mb; i += 36)
            for (int k = 0; k < 4; ++k) w[i + k] = (uint8_t) (0x30 + (rng() & 15));
        return w;
    };
    const auto wg = matrix(), wu = matrix();
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) T * K);
    for (auto& v : x) v = nd(rng);
    uint8_t *dg, *du;
    float *dx, *yg, *yu, *yg2, *h, *gs, *tail;
    int32_t *bounds, *toff;
    void* a4;
    cudaMalloc(&dg, mb);
    cudaMalloc(&du, mb);
    cudaMemcpy(dg, wg.data(), mb, cudaMemcpyHostToDevice);
    cudaMemcpy(du, wu.data(), mb, cudaMemcpyHostToDevice);
    cudaMalloc(&dx, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&yg, (size_t) T * R * 4);
    cudaMalloc(&yu, (size_t) T * R * 4);
    cudaMalloc(&yg2, (size_t) T * R * 4);
    cudaMalloc(&h, (size_t) T * R * 4);
    cudaMalloc(&a4, fp4::a4_bytes(T, K, 1));
    cudaMalloc(&gs, fp4::scale_count(T, 1) * 4);
    cudaMalloc(&bounds, 8);
    cudaMalloc(&toff, 4 * (fp4::kMaxExperts + 1));
    const float tl[4] = {0.7f, 1.3f, 1.0f, 0.0f};
    cudaMalloc(&tail, 16);
    cudaMemcpy(tail, tl, 16, cudaMemcpyHostToDevice);
    auto product = [&](const uint8_t* w, const uint8_t* up, int64_t rows, float* dst) {
        fp4::Product p;
        p.blob[0] = w;
        p.w_rows = up ? 2 * R : R;
        p.w_cols = K;
        p.n = 1;
        p.a4 = a4;
        p.gs = gs;
        p.bounds = bounds;
        p.tile_off = toff;
        p.dst = dst;
        p.ld_dst = R;
        p.max_rows = rows;
        if (up) {
            p.swiglu = true;
            p.up = up;
            p.tail = tail;
        }
        fp4::run(p, nullptr);
    };
    auto quant = [&](int64_t rows) {
        const int32_t b[2] = {0, (int32_t) rows};
        cudaMemcpy(bounds, b, 8, cudaMemcpyHostToDevice);
        fp4::plan(bounds, 1, toff, nullptr);
        fp4::quantize(dx, nullptr, bounds, toff, 1, a4, gs, K, K, rows, nullptr);
    };
    quant(T2);
    product(dg, nullptr, T2, yg2);   // two tiles per block
    quant(T);
    product(dg, nullptr, T, yg);     // four
    product(du, nullptr, T, yu);
    product(dg, du, T, h);
    std::vector<float> g((size_t) T * R), u(g.size()), g2(g.size()), hh(g.size());
    cudaMemcpy(g.data(), yg, g.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(u.data(), yu, g.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g2.data(), yg2, (size_t) T2 * R * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hh.data(), h, g.size() * 4, cudaMemcpyDeviceToHost);
    const bool err = cudaGetLastError() != cudaSuccess;
    const bool same = std::memcmp(g.data(), g2.data(), (size_t) T2 * R * 4) == 0;
    double worst = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double gv = 0.7 * g[i], uv = 1.3 * u[i];
        const double ref = gv / (1.0 + std::exp(-gv)) * uv;
        worst = std::max(worst, std::fabs(hh[i] - ref) / (std::fabs(ref) + 1e-3));
    }
    const bool ok = !err && same && worst < 1e-5;
    std::printf("fp4 dense: rows bitwise at 1000 and 1100 rows: %s; SwiGLU product vs the plain ones: max rel %.3g%s\n",
                same ? "yes" : "NO", worst, ok ? "" : "  FAIL");
    return ok ? 0 : 1;
}
