// tests/kernels/q80_head_parity.cpp - q80_head_mmvq against native_mmvq (Q8_0, the exact layout), bitwise, on random
// Q8_0 rows and quantized random columns, 1..8 columns, at the head's and smaller row counts; each column against a
// one-column call; the logits against a plain CPU reference (dequantized rows . dequantized columns in double); then
// a microbench of both at the head's shape.
//   q80_head_parity [rows] [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q80_head.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
float h2f(const uint8_t* p) {
    __half h;
    std::memcpy(&h, p, 2);
    return __half2float(h);
}
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    const int n_in = 2560, MT = 8, blk = 34, bpr = n_in / 32, qblk = 36;
    if (!k::q80_head_supported(n_in, 1)) { std::printf("q80_head_parity: shape not supported\n"); return 1; }
    const size_t wbytes = (size_t) rows * bpr * blk;
    std::mt19937 rng(argc > 3 ? (unsigned) std::atoi(argv[3]) : 7u);
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i + 4 <= wbytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(&hw[i], &r, 4);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < (size_t) rows * bpr; ++b) {   // finite block scales
        const __half d = __float2half(std::fabs(nd(rng)) * 0.002f);
        std::memcpy(&hw[b * blk], &d, 2);
    }
    void* w = nullptr;
    ck(cudaMalloc(&w, wbytes), "w");
    ck(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up");
    std::vector<float> hx((size_t) MT * n_in);
    for (auto& v : hx) v = nd(rng);
    float* x = nullptr;
    void* xq = nullptr;
    float *y0 = nullptr, *y1 = nullptr, *y2 = nullptr;
    ck(cudaMalloc(&x, hx.size() * 4), "x");
    ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    const size_t xqb = k::native_q8_1_bytes(n_in, MT);
    ck(cudaMalloc(&xq, xqb), "xq");
    ck(cudaMalloc(&y0, (size_t) MT * rows * 4), "y0");
    ck(cudaMalloc(&y1, (size_t) MT * rows * 4), "y1");
    ck(cudaMalloc(&y2, (size_t) rows * 4), "y2");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    k::native_quantize_q8_1(x, xq, n_in, MT, s);
    std::vector<uint8_t> hxq(xqb);
    ck(cudaStreamSynchronize(s), "sync");
    ck(cudaMemcpy(hxq.data(), xq, xqb, cudaMemcpyDeviceToHost), "xq down");
    int fail = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int counts[] = {rows, 106299, 40525, 1001, 7};
    for (int rr : counts) {
        if (rr > rows) continue;
        for (int nc = 1; nc <= MT; ++nc) {
            ck(cudaMemset(y0, 0xff, (size_t) MT * rows * 4), "m0");
            ck(cudaMemset(y1, 0xee, (size_t) MT * rows * 4), "m1");
            k::native_mmvq(8, w, xq, y0, n_in, rr, nc, s);
            k::q80_head_mmvq(w, xq, y1, n_in, rr, nc, s);
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<float> a((size_t) nc * rr), b((size_t) nc * rr);
            ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
            size_t bad = 0;
            for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;
            // each column alone (a one-column call) gives the same bits: the window's width and a token's place in it
            // do not change its logits
            size_t bad1 = 0;
            for (int j = 0; j < nc; ++j) {
                k::q80_head_mmvq(w, (const char*) xq + (size_t) j * bpr * qblk, y2, n_in, rr, 1, s);
                ck(cudaStreamSynchronize(s), "sync1");
                std::vector<float> c((size_t) rr);
                ck(cudaMemcpy(c.data(), y2, c.size() * 4, cudaMemcpyDeviceToHost), "c");
                for (int r = 0; r < rr; ++r) bad1 += std::memcmp(&c[r], &b[(size_t) j * rr + r], 4) != 0;
            }
            if (rr == rows || nc == 1 || nc == MT)
                std::printf("rows %6d ncols %d: %s vs native_mmvq (%zu of %zu differ), %s vs one-column calls (%zu differ)\n", rr, nc,
                            bad ? "FAIL" : "bitwise", bad, a.size(), bad1 ? "FAIL" : "bitwise", bad1);
            if (bad || bad1) fail = 1;
        }
    }
    {   // the CPU reference: every 97th row, all columns, dequantized and accumulated in double
        const int nc = MT;
        k::native_mmvq(8, w, xq, y0, n_in, rows, nc, s);
        k::q80_head_mmvq(w, xq, y1, n_in, rows, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) nc * rows), b((size_t) nc * rows);
        ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        double max_abs[2] = {0, 0}, max_rel[2] = {0, 0}, ref_max = 0;
        for (int r = 0; r < rows; r += 97)
            for (int j = 0; j < nc; ++j) {
                double ref = 0;
                for (int kb = 0; kb < bpr; ++kb) {
                    const uint8_t* wb = &hw[((size_t) r * bpr + kb) * blk];
                    const uint8_t* xb = &hxq[((size_t) j * bpr + kb) * qblk];
                    const double dw = h2f(wb), dx = h2f(xb);
                    for (int i = 0; i < 32; ++i) ref += dw * (int8_t) wb[2 + i] * dx * (int8_t) xb[4 + i];
                }
                ref_max = std::max(ref_max, std::fabs(ref));
                const float got[2] = {a[(size_t) j * rows + r], b[(size_t) j * rows + r]};
                for (int q = 0; q < 2; ++q) {
                    const double d = std::fabs((double) got[q] - ref);
                    max_abs[q] = std::max(max_abs[q], d);
                    max_rel[q] = std::max(max_rel[q], d / std::max(std::fabs(ref), 1e-30));
                }
            }
        std::printf("CPU reference (every 97th row, %d columns, |ref| max %.3e): native_mmvq max abs %.3e rel %.3e | "
                    "q80_head max abs %.3e rel %.3e\n", nc, ref_max, max_abs[0], max_rel[0], max_abs[1], max_rel[1]);
        if (max_abs[1] > 1e-3 * std::max(ref_max, 1.0)) fail = 1;
    }
    for (int nc = 1; nc <= MT; ++nc) {
        float ms[2] = {0, 0};
        for (int rep = 0; rep < 2; ++rep) {
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::native_mmvq(8, w, xq, y0, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[0], e0, e1);
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::q80_head_mmvq(w, xq, y1, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[1], e0, e1);
        }
        const double u0 = 1e3 * ms[0] / iters, u1 = 1e3 * ms[1] / iters;
        std::printf("ncols %d, %d rows: native_mmvq %7.1f us %6.0f GB/s | q80_head %7.1f us %6.0f GB/s | %.3fx\n", nc, rows,
                    u0, wbytes / u0 / 1e3, u1, wbytes / u1 / 1e3, u0 / u1);
    }
    std::printf("q80_head_parity: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}
