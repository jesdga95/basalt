// tests/kernels/fp8_gemm_test.cpp - prefill::fp8 (fp8_gemm.hpp): the row quantizer against a CPU E4M3 rounding, the
// product against a double reference on the same codes (a chunk whose rows are not a tile multiple), the Add epilogue,
// each row the same whatever the chunk's length, and a segmented product against separate products.  --bench: TFLOP/s (random data) at the Qwen3.8-27B projection shapes,
// 2048 rows, and the segmented products.  Exit 77 without a GPU.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace fp8 = basalt::prefill::fp8;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fp8_gemm_test: CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
float e4m3(uint8_t b) {
    __nv_fp8_e4m3 v;
    std::memcpy(&v, &b, 1);
    return (float) v;
}
// Random E4M3 weight codes (no NaN) and random activations: constant data draws less power and flatters the clocks.
void fill_w(uint8_t* d, size_t n, uint64_t seed) {
    std::vector<uint8_t> h(n);
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        uint8_t b = (uint8_t) (x >> 24);
        if ((b & 0x7f) == 0x7f) b = 0x31;
        h[i] = b;
    }
    ck(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice), "fill_w");
}
void fill_x(float* d, size_t n, uint64_t seed) {
    std::vector<float> h(n);
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 7;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        h[i] = (float) (int32_t) (x >> 32) * (1.0f / 2147483648.0f);
    }
    ck(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice), "fill_x");
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int64_t shapes[][2] = {{16384, 5120}, {14336, 5120}, {5120, 6144}, {34816, 5120}, {5120, 17408}};
        const int64_t T = 2048;
        for (const auto& sh : shapes) {
            const int64_t M = sh[0], K = sh[1];
            uint8_t* w;
            float *x, *y;
            void* act;
            cudaMalloc(&w, (size_t) M * K);
            fill_w(w, (size_t) M * K, 1);
            cudaMalloc(&x, (size_t) T * K * 4);
            fill_x(x, (size_t) T * K, 2);
            cudaMalloc(&y, (size_t) T * M * 4);
            cudaMalloc(&act, fp8::act_bytes(T, K));
            fp8::quantize(x, T, K, K, act, nullptr);
            fp8::gemm(w, M, K, nullptr, act, T, y, M, fp8::Epilogue::Store, nullptr);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) fp8::gemm(w, M, K, nullptr, act, T, y, M, fp8::Epilogue::Store, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            ms /= 10;
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) fp8::gemm(w, M, K, nullptr, act, T, y, M, fp8::Epilogue::Add, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ma;
            cudaEventElapsedTime(&ma, a, b);
            ma /= 10;
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) fp8::quantize(x, T, K, K, act, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float mq;
            cudaEventElapsedTime(&mq, a, b);
            std::printf("fp8 gemm [%lld x %lld] x %lld rows: %.3f ms (%.0f TFLOP/s), Add %.3f ms (%.0f); quantize %.3f ms\n",
                        (long long) M, (long long) K, (long long) T, ms, 2.0 * M * K * T / ms / 1e9, ma,
                        2.0 * M * K * T / ma / 1e9, mq / 10);
            cudaFree(w); cudaFree(x); cudaFree(y); cudaFree(act);
        }
        {   // the output projection with cold weights and a cold residual: 8 copies of each, rotated
            const int64_t M = 5120, K = 6144, C = 8;
            std::vector<uint8_t*> ws(C);
            std::vector<float*> ys(C);
            float* x;
            void* act;
            for (int i = 0; i < C; ++i) {
                cudaMalloc(&ws[i], (size_t) M * K);
                fill_w(ws[i], (size_t) M * K, 10 + i);
                cudaMalloc(&ys[i], (size_t) T * M * 4);
                cudaMemset(ys[i], 0, (size_t) T * M * 4);
            }
            cudaMalloc(&x, (size_t) T * K * 4);
            fill_x(x, (size_t) T * K, 2);
            cudaMalloc(&act, fp8::act_bytes(T, K));
            fp8::quantize(x, T, K, K, act, nullptr);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int e = 0; e < 2; ++e) {
                cudaEventRecord(a);
                for (int i = 0; i < 4 * C; ++i)
                    fp8::gemm(ws[i % C], M, K, nullptr, act, T, ys[i % C], M, e ? fp8::Epilogue::Add : fp8::Epilogue::Store, nullptr);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms;
                cudaEventElapsedTime(&ms, a, b);
                ms /= 4 * C;
                std::printf("fp8 gemm [%lld x %lld] cold weights and outputs, %s: %.3f ms (%.0f TFLOP/s)\n", (long long) M, (long long) K,
                            e ? "Add" : "Store", ms, 2.0 * M * K * T / ms / 1e9);
            }
        }
        // the segmented products at 2048 rows: GDN qkv + z, attention q + k + v (one launch each), and the Add form
        const int64_t K = 5120, T2 = 2048;
        const int64_t segs[][3] = {{10240, 6144, 0}, {12288, 1024, 1024}};
        for (const auto& sg : segs) {
            const int ns = sg[2] ? 3 : 2;
            uint8_t* w;
            float *x, *y;
            void* act;
            int64_t tot = 0;
            for (int i = 0; i < ns; ++i) tot += sg[i];
            cudaMalloc(&w, (size_t) tot * K);
            fill_w(w, (size_t) tot * K, 3);
            cudaMalloc(&x, (size_t) T2 * K * 4);
            fill_x(x, (size_t) T2 * K, 4);
            cudaMalloc(&y, (size_t) T2 * tot * 4);
            cudaMalloc(&act, fp8::act_bytes(T2, K));
            fp8::quantize(x, T2, K, K, act, nullptr);
            fp8::Segment sv[3];
            int64_t wo = 0;
            for (int i = 0; i < ns; ++i) {
                sv[i] = {w + wo * K, sg[i], nullptr, y + wo * T2, sg[i]};
                wo += sg[i];
            }
            fp8::gemm_segmented(sv, ns, K, act, T2, fp8::Epilogue::Store, nullptr);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) fp8::gemm_segmented(sv, ns, K, act, T2, fp8::Epilogue::Store, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            ms /= 10;
            std::printf("fp8 gemm segmented [%lld + %lld%s x %lld] x %lld rows: %.3f ms (%.0f TFLOP/s)\n", (long long) sg[0],
                        (long long) sg[1], ns == 3 ? " + 1024" : "", (long long) K, (long long) T2, ms,
                        2.0 * tot * K * T2 / ms / 1e9);
            cudaFree(w); cudaFree(x); cudaFree(y); cudaFree(act);
        }
        return 0;
    }
    const int64_t M = 384, K = 1024, T = 300;
    std::vector<uint8_t> w((size_t) M * K);
    for (auto& b : w) {
        b = (uint8_t) rng();
        if ((b & 0x7f) == 0x7f) b = 0x00;   // no NaN codes
    }
    std::vector<float> x((size_t) T * K);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) x[(size_t) t * K + k] = std::ldexp(1.0f, (int) (t % 7) - 3) * nd(rng);
    x[5 * K + 7] = 300.0f;
    uint8_t* dw;
    float *dx, *dy, *dy2, *dws;
    void* act;
    ck(cudaMalloc(&dw, w.size()), "malloc dw");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "copy dw");
    ck(cudaMalloc(&dx, x.size() * 4), "malloc dx");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy dx");
    ck(cudaMalloc(&dy, (size_t) T * M * 4), "malloc dy");
    ck(cudaMalloc(&dy2, (size_t) T * M * 4), "malloc dy2");
    ck(cudaMalloc(&act, fp8::act_bytes(T, K)), "malloc act");
    const float wsc = 0.03f;
    ck(cudaMalloc(&dws, 4), "malloc dws");
    ck(cudaMemcpy(dws, &wsc, 4, cudaMemcpyHostToDevice), "copy dws");
    fp8::quantize(dx, T, K, K, act, nullptr);
    fp8::gemm(dw, M, K, dws, act, T, dy, M, fp8::Epilogue::Store, nullptr);
    ck(cudaDeviceSynchronize(), "sync main gemm");
    std::vector<uint8_t> codes((size_t) T * K);
    std::vector<float> sc(T), y((size_t) T * M);
    ck(cudaMemcpy(codes.data(), act, codes.size(), cudaMemcpyDeviceToHost), "copy codes");
    ck(cudaMemcpy(sc.data(), (const uint8_t*) act + (((size_t) T * K + 15) & ~(size_t) 15), T * 4, cudaMemcpyDeviceToHost), "copy sc");
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "copy y");
    int fails = 0;
    // the quantizer: scale = amax / 448, codes the nearest E4M3 of x / scale
    double qerr = 0;
    for (int64_t t = 0; t < T; ++t) {
        float amax = 0;
        for (int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[(size_t) t * K + k]));
        if (!std::isfinite(sc[t]) || std::fabs(sc[t] - amax / 448.0f) > 1e-6f * amax) ++fails;
        for (int64_t k = 0; k < K; ++k) {
            const float v = x[(size_t) t * K + k] / sc[t], q = e4m3(codes[(size_t) t * K + k]);
            const double err = (double) std::fabs(v - q) / std::max(std::fabs(v), 1.0f / 64);
            if (!std::isfinite(q) || !std::isfinite(err)) ++fails;
            else qerr = std::max(qerr, err);
        }
    }
    std::printf("fp8 quantize: scales %s, max relative rounding %.3g (E4M3: 1/16)\n", fails ? "WRONG" : "exact", qerr);
    if (!std::isfinite(qerr) || qerr > 0.0626) ++fails;
    // the product against double on the same codes
    double perr = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t m = 0; m < M; ++m) {
            double s = 0, sa = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double pr = (double) e4m3(w[(size_t) m * K + k]) * e4m3(codes[(size_t) t * K + k]);
                s += pr;
                sa += std::fabs(pr);
            }
            s *= (double) wsc * sc[t];
            sa *= (double) wsc * sc[t];
            const float got = y[(size_t) t * M + m];
            const double err = std::fabs(got - s) / std::max(sa, 1e-30);
            if (!std::isfinite(got) || !std::isfinite(err)) ++fails;
            else perr = std::max(perr, err);
        }
    std::printf("fp8 gemm vs double on the codes: max |err| / sum|w a| %.3g\n", perr);
    if (!std::isfinite(perr) || perr > 1e-5) ++fails;
    // each row the same at a shorter chunk; Add
    fp8::quantize(dx, 130, K, K, act, nullptr);
    fp8::gemm(dw, M, K, dws, act, 130, dy2, M, fp8::Epilogue::Store, nullptr);
    ck(cudaDeviceSynchronize(), "sync short store");
    std::vector<float> y2((size_t) 130 * M);
    ck(cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost), "copy y2 store");
    const bool same = std::memcmp(y2.data(), y.data(), y2.size() * 4) == 0;
    fp8::gemm(dw, M, K, dws, act, 130, dy2, M, fp8::Epilogue::Add, nullptr);
    ck(cudaDeviceSynchronize(), "sync short add");
    ck(cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost), "copy y2 add");
    int bad = 0;
    for (size_t i = 0; i < y2.size(); ++i)
        bad += !std::isfinite(y2[i]) || !std::isfinite(y[i]) || std::fabs(y2[i] - 2.0 * y[i]) > 1e-6 * (std::fabs(y[i]) + 1e-6);
    std::printf("fp8 gemm: rows bitwise at 130 and 300 rows: %s; Add epilogue: %s\n", same ? "yes" : "NO", bad ? "WRONG" : "ok");
    if (!same || bad) ++fails;
    // a segmented product (three disjoint matrices of 128, 384 and 256 rows with own scales, outputs and strides)
    // against the same matrices as separate products, bitwise, in Store and Add
    {
        fp8::quantize(dx, T, K, K, act, nullptr);
        const int64_t R[3] = {128, 384, 256}, LD[3] = {R[0] + 6, R[1], R[2] + 2};
        const float sv_[3] = {0.5f, 0.03f, 0.007f};
        uint8_t* dw_seg = nullptr;
        const size_t seg_rows = (size_t) (R[0] + R[1] + R[2]);
        ck(cudaMalloc(&dw_seg, seg_rows * (size_t) K), "malloc dw_seg");
        fill_w(dw_seg, seg_rows * (size_t) K, 9);
        uint8_t* wp[3] = {dw_seg, dw_seg + (size_t) R[0] * K, dw_seg + (size_t) (R[0] + R[1]) * K};
        float *sc3, *o1[3], *o2[3];
        ck(cudaMalloc(&sc3, 12), "malloc sc3");
        ck(cudaMemcpy(sc3, sv_, 12, cudaMemcpyHostToDevice), "copy sc3");
        fp8::Segment sg1[3], sg2[3];
        for (int i = 0; i < 3; ++i) {
            ck(cudaMalloc(&o1[i], (size_t) T * LD[i] * 4), "malloc o1");
            ck(cudaMalloc(&o2[i], (size_t) T * LD[i] * 4), "malloc o2");
            ck(cudaMemset(o1[i], 0x3f, (size_t) T * LD[i] * 4), "memset o1");
            ck(cudaMemset(o2[i], 0x3f, (size_t) T * LD[i] * 4), "memset o2");
            sg1[i] = {wp[i], R[i], sc3 + i, o1[i], LD[i]};
            sg2[i] = {wp[i], R[i], sc3 + i, o2[i], LD[i]};
        }
        int segbad = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const fp8::Epilogue e = pass ? fp8::Epilogue::Add : fp8::Epilogue::Store;
            fp8::gemm_segmented(sg1, 3, K, act, T, e, nullptr);
            for (int i = 0; i < 3; ++i) fp8::gemm(sg2[i].w, R[i], K, sc3 + i, act, T, o2[i], LD[i], e, nullptr);
            ck(cudaDeviceSynchronize(), "sync segmented");
            for (int i = 0; i < 3; ++i) {
                std::vector<float> a((size_t) T * LD[i]), b((size_t) T * LD[i]);
                ck(cudaMemcpy(a.data(), o1[i], a.size() * 4, cudaMemcpyDeviceToHost), "copy o1");
                ck(cudaMemcpy(b.data(), o2[i], b.size() * 4, cudaMemcpyDeviceToHost), "copy o2");
                bool all_fin = true;
                for (int64_t t = 0; t < T; ++t)
                    for (int64_t r = 0; r < R[i]; ++r) all_fin &= std::isfinite(a[(size_t) t * LD[i] + r]);
                segbad += !all_fin || std::memcmp(a.data(), b.data(), a.size() * 4) != 0;
            }
        }
        for (int i = 0; i < 3; ++i) { cudaFree(o1[i]); cudaFree(o2[i]); }
        cudaFree(sc3);
        cudaFree(dw_seg);
        std::printf("fp8 gemm segmented (3 matrices, Store and Add) bitwise against separate products: %s\n", segbad ? "NO" : "yes");
        fails += segbad;
    }
    {   // the norm in the quantizer: its normed rows against a host RMSNorm, and its codes those of quantize(normed rows)
        std::vector<float> nw(K);
        for (auto& v : nw) v = 1.0f + 0.1f * nd(rng);
        float *d_nw, *d_xn;
        void* act2;
        ck(cudaMalloc(&d_nw, K * 4), "malloc d_nw");
        ck(cudaMalloc(&d_xn, (size_t) T * K * 4), "malloc d_xn");
        ck(cudaMalloc(&act2, fp8::act_bytes(T, K)), "malloc act2");
        ck(cudaMemcpy(d_nw, nw.data(), K * 4, cudaMemcpyHostToDevice), "copy d_nw");
        fp8::rms_norm_quantize(dx, d_nw, 1e-6f, T, K, act, d_xn, nullptr);
        fp8::quantize(d_xn, T, K, K, act2, nullptr);
        ck(cudaDeviceSynchronize(), "sync rms_norm_quantize");
        std::vector<uint8_t> a1(fp8::act_bytes(T, K)), a2(a1.size());
        std::vector<float> xn((size_t) T * K);
        ck(cudaMemcpy(a1.data(), act, a1.size(), cudaMemcpyDeviceToHost), "copy a1");
        ck(cudaMemcpy(a2.data(), act2, a2.size(), cudaMemcpyDeviceToHost), "copy a2");
        ck(cudaMemcpy(xn.data(), d_xn, xn.size() * 4, cudaMemcpyDeviceToHost), "copy xn");
        double nerr = 0;
        for (int64_t t = 0; t < T; ++t) {
            double ss = 0;
            for (int64_t k = 0; k < K; ++k) ss += (double) x[(size_t) t * K + k] * x[(size_t) t * K + k];
            const double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int64_t k = 0; k < K; ++k) {
                const float got = xn[(size_t) t * K + k];
                const double want = x[(size_t) t * K + k] * inv * nw[k];
                const double err = std::fabs(got - want) / (std::fabs(want) + 1e-3);
                if (!std::isfinite(got) || !std::isfinite(err)) ++fails;
                else nerr = std::max(nerr, err);
            }
        }
        const bool same = a1 == a2;
        std::printf("fp8 rms_norm_quantize: normed rows rel %.3g, codes and scales %s\n", nerr, same ? "bitwise quantize's" : "DIFFER");
        if (!std::isfinite(nerr) || nerr > 1e-5 || !same) ++fails;
        cudaFree(d_nw); cudaFree(d_xn); cudaFree(act2);
    }
    {   // tiny normal row (448 / amax overflows to +inf): must not emit E4M3 NaN (0x7f) or poison MMA outputs
        std::vector<float> tiny((size_t) K, 0.0f);
        tiny[0] = 1e-37f;
        float* d_tiny = nullptr;
        ck(cudaMalloc(&d_tiny, tiny.size() * 4), "malloc d_tiny");
        ck(cudaMemcpy(d_tiny, tiny.data(), tiny.size() * 4, cudaMemcpyHostToDevice), "copy d_tiny");
        fp8::quantize(d_tiny, 1, K, K, act, nullptr);
        fp8::gemm(dw, M, K, dws, act, 1, dy2, M, fp8::Epilogue::Store, nullptr);
        ck(cudaDeviceSynchronize(), "sync tiny");
        std::vector<uint8_t> tc((size_t) K);
        std::vector<float> ty((size_t) M);
        ck(cudaMemcpy(tc.data(), act, tc.size(), cudaMemcpyDeviceToHost), "copy tc");
        ck(cudaMemcpy(ty.data(), dy2, ty.size() * 4, cudaMemcpyDeviceToHost), "copy ty");
        bool tiny_ok = true;
        for (uint8_t b : tc) tiny_ok &= (b & 0x7fu) != 0x7fu;
        for (float v : ty) tiny_ok &= std::isfinite(v);
        std::printf("fp8 tiny normal row (1e-37): finite codes and output %s\n", tiny_ok ? "ok" : "FAIL");
        if (!tiny_ok) ++fails;
        cudaFree(d_tiny);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dy2); cudaFree(dws); cudaFree(act);
    if (cudaGetLastError() != cudaSuccess) ++fails;
    std::printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
