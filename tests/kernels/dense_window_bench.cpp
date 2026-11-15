// tests/kernels/dense_window_bench.cpp - a decode window's products at a dense model's widths: MMVQ (iq_mmvq) at 1 and
// 8 columns against MMQ with an 8-row tile, for Q8_0 and NVFP4.  Prints us per call and GB/s of weights, and whether
// MMQ's rows are bitwise the same whatever the rows given (1..8).  Exit 77 without a GPU.
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = basalt::prefill::mmq;

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    std::mt19937 rng(3);
    struct Shape { int type; int rows, cols; const char* name; };
    const Shape shapes[] = {{40, 17408, 5120, "NVFP4 gate/up 17408x5120"}, {40, 5120, 17408, "NVFP4 down 5120x17408"},
                            {8, 10240, 5120, "Q8_0 GDN qkv 10240x5120"}, {8, 5120, 6144, "Q8_0 out 5120x6144"}};
    mmq::Context ctx;
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (const Shape& sh : shapes) {
        const size_t wb = mmq::matrix_bytes(sh.type, sh.rows, sh.cols);
        std::vector<uint8_t> w(wb);
        for (auto& b : w) b = (uint8_t) rng();
        if (sh.type == 8)   // sane fp16 block scales
            for (size_t i = 0; i < wb; i += 34) { w[i] = 0x00; w[i + 1] = 0x20; }
        else   // UE4M3 scales without the sign bit or NaN
            for (size_t i = 0; i < wb; i += 36) for (int k = 0; k < 4; ++k) w[i + k] = (uint8_t) (0x30 + (rng() & 15));
        void *d_w, *d_xq, *d_xm;
        float *d_x, *d_y, *d_y2;
        int32_t *d_ids, *d_b;
        cudaMalloc(&d_w, wb);
        cudaMemcpy(d_w, w.data(), wb, cudaMemcpyHostToDevice);
        std::vector<float> x((size_t) 8 * sh.cols);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        cudaMalloc(&d_x, x.size() * 4);
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_xq, basalt::kernels::native_q8_1_bytes(sh.cols, 8));
        cudaMalloc(&d_xm, mmq::q8_bytes(8, sh.cols));
        cudaMalloc(&d_y, (size_t) 8 * sh.rows * 4);
        cudaMalloc(&d_y2, (size_t) 8 * sh.rows * 4);
        cudaMalloc(&d_ids, 64 * 4);
        cudaMalloc(&d_b, 8);
        mmq::iota(d_ids, 64, st);
        basalt::kernels::native_quantize_q8_1(d_x, d_xq, sh.cols, 8, st);
        mmq::quantize(d_x, nullptr, d_xm, sh.type, sh.cols, sh.cols, 8, st);
        mmq::load(sh.type, sh.rows);
        auto run_mmq = [&](int T, float* y) {
            const int32_t b[2] = {0, T};
            cudaMemcpyAsync(d_b, b, 8, cudaMemcpyHostToDevice, st);
            mmq::Product p;
            p.w = d_w; p.type = sh.type; p.w_rows = sh.rows; p.w_cols = sh.cols; p.expert_bytes = wb; p.n = 1;
            p.xq = d_xm; p.bounds = d_b; p.ids = d_ids; p.total_rows = 8; p.max_rows = 8; p.dst = y; p.ld_dst = sh.rows;
            p.opt_rows = 8;
            ctx.run(p, st);
        };
        cudaEvent_t a, e;
        cudaEventCreate(&a);
        cudaEventCreate(&e);
        auto time = [&](auto f) {
            for (int i = 0; i < 3; ++i) f();
            cudaEventRecord(a, st);
            for (int i = 0; i < 50; ++i) f();
            cudaEventRecord(e, st);
            cudaEventSynchronize(e);
            float ms;
            cudaEventElapsedTime(&ms, a, e);
            return 1000.0 * ms / 50;
        };
        const double t1 = time([&] { basalt::kernels::iq_mmvq(sh.type, d_w, d_xq, d_y, sh.cols, sh.rows, 1, st); });
        const double t8 = time([&] { basalt::kernels::iq_mmvq(sh.type, d_w, d_xq, d_y, sh.cols, sh.rows, 8, st); });
        const double tq = time([&] { run_mmq(8, d_y); });
        // MMQ rows given 1 vs 8: row 0 bitwise?
        run_mmq(8, d_y);
        run_mmq(1, d_y2);
        cudaStreamSynchronize(st);
        std::vector<float> y8(sh.rows), y1(sh.rows), yv((size_t) sh.rows);
        cudaMemcpy(y8.data(), d_y, sh.rows * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(y1.data(), d_y2, sh.rows * 4, cudaMemcpyDeviceToHost);
        basalt::kernels::iq_mmvq(sh.type, d_w, d_xq, d_y, sh.cols, sh.rows, 1, st);
        cudaStreamSynchronize(st);
        cudaMemcpy(yv.data(), d_y, sh.rows * 4, cudaMemcpyDeviceToHost);
        double md = 0, mv = 0;
        for (int i = 0; i < sh.rows; ++i) { md = std::max(md, (double) std::fabs(y8[i] - y1[i])); mv = std::max(mv, (double) std::fabs(yv[i])); }
        double dq = 0;
        for (int i = 0; i < sh.rows; ++i) dq = std::max(dq, (double) std::fabs(y8[i] - yv[i]));
        std::printf("%-28s mmvq 1 col %7.1f us (%5.0f GB/s)  8 cols %7.1f us  mmq 8 rows %7.1f us (%5.0f GB/s)  "
                    "mmq row 0 at T=1 vs T=8: %s  |mmq-mmvq| %.3g of %.3g\n",
                    sh.name, t1, wb / t1 / 1e3, t8, tq, wb / tq / 1e3, md == 0 ? "bitwise" : "DIFFERS", dq, mv);
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_xm); cudaFree(d_y); cudaFree(d_y2); cudaFree(d_ids); cudaFree(d_b);
    }
    return 0;
}
