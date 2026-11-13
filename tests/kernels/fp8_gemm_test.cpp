// tests/kernels/fp8_gemm_test.cpp - prefill::fp8 (fp8_gemm.hpp): the row quantizer against a CPU E4M3 rounding, the
// product against a double reference on the same codes (a chunk whose rows are not a tile multiple), the Add epilogue,
// each row the same whatever the chunk's length, and a segmented product against separate products.  --bench: TFLOP/s (random data) at the Qwen3.8-27B projection shapes,
// 2048 rows, and the segmented products.  Exit 77 without a GPU.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fp8 = basalt::prefill::fp8;

namespace {
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
    cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
}
void fill_x(float* d, size_t n, uint64_t seed) {
    std::vector<float> h(n);
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 7;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        h[i] = (float) (int32_t) (x >> 32) * (1.0f / 2147483648.0f);
    }
    cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice);
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
            for (int i = 0; i < 10; ++i) fp8::quantize(x, T, K, K, act, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float mq;
            cudaEventElapsedTime(&mq, a, b);
            std::printf("fp8 gemm [%lld x %lld] x %lld rows: %.3f ms (%.0f TFLOP/s); quantize %.3f ms\n", (long long) M,
                        (long long) K, (long long) T, ms, 2.0 * M * K * T / ms / 1e9, mq / 10);
            cudaFree(w); cudaFree(x); cudaFree(y); cudaFree(act);
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
    cudaMalloc(&dw, w.size());
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dx, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dy, (size_t) T * M * 4);
    cudaMalloc(&dy2, (size_t) T * M * 4);
    cudaMalloc(&act, fp8::act_bytes(T, K));
    const float wsc = 0.03f;
    cudaMalloc(&dws, 4);
    cudaMemcpy(dws, &wsc, 4, cudaMemcpyHostToDevice);
    fp8::quantize(dx, T, K, K, act, nullptr);
    fp8::gemm(dw, M, K, dws, act, T, dy, M, fp8::Epilogue::Store, nullptr);
    std::vector<uint8_t> codes((size_t) T * K);
    std::vector<float> sc(T), y((size_t) T * M);
    cudaMemcpy(codes.data(), act, codes.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), (const uint8_t*) act + (((size_t) T * K + 15) & ~(size_t) 15), T * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
    int fails = 0;
    // the quantizer: scale = amax / 448, codes the nearest E4M3 of x / scale
    double qerr = 0;
    for (int64_t t = 0; t < T; ++t) {
        float amax = 0;
        for (int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[(size_t) t * K + k]));
        if (std::fabs(sc[t] - amax / 448.0f) > 1e-6f * amax) ++fails;
        for (int64_t k = 0; k < K; ++k) {
            const float v = x[(size_t) t * K + k] / sc[t], q = e4m3(codes[(size_t) t * K + k]);
            qerr = std::max(qerr, (double) std::fabs(v - q) / std::max(std::fabs(v), 1.0f / 64));
        }
    }
    std::printf("fp8 quantize: scales %s, max relative rounding %.3g (E4M3: 1/16)\n", fails ? "WRONG" : "exact", qerr);
    if (qerr > 0.0626) ++fails;
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
            perr = std::max(perr, std::fabs(y[(size_t) t * M + m] - s) / std::max(sa, 1e-30));
        }
    std::printf("fp8 gemm vs double on the codes: max |err| / sum|w a| %.3g\n", perr);
    if (perr > 1e-5) ++fails;
    // each row the same at a shorter chunk; Add
    fp8::quantize(dx, 130, K, K, act, nullptr);
    fp8::gemm(dw, M, K, dws, act, 130, dy2, M, fp8::Epilogue::Store, nullptr);
    std::vector<float> y2((size_t) 130 * M);
    cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost);
    const bool same = std::memcmp(y2.data(), y.data(), y2.size() * 4) == 0;
    fp8::gemm(dw, M, K, dws, act, 130, dy2, M, fp8::Epilogue::Add, nullptr);
    cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t i = 0; i < y2.size(); ++i) bad += std::fabs(y2[i] - 2.0 * y[i]) > 1e-6 * (std::fabs(y[i]) + 1e-6);
    std::printf("fp8 gemm: rows bitwise at 130 and 300 rows: %s; Add epilogue: %s\n", same ? "yes" : "NO", bad ? "WRONG" : "ok");
    if (!same || bad) ++fails;
    // a segmented product (three matrices of 128, 384 and 256 rows over the first 1024 weight rows' codes, own scales,
    // outputs and strides) against the same matrices as separate products, bitwise, in Store and Add
    {
        fp8::quantize(dx, T, K, K, act, nullptr);
        const int64_t R[3] = {128, 384, 256}, LD[3] = {R[0] + 6, R[1], R[2] + 2};
        const float sv_[3] = {0.5f, 0.03f, 0.007f};
        uint8_t* wp[3] = {dw, dw + 128 * K, dw};
        float *sc3, *o1[3], *o2[3];
        cudaMalloc(&sc3, 12);
        cudaMemcpy(sc3, sv_, 12, cudaMemcpyHostToDevice);
        fp8::Segment sg1[3], sg2[3];
        for (int i = 0; i < 3; ++i) {
            cudaMalloc(&o1[i], (size_t) T * LD[i] * 4);
            cudaMalloc(&o2[i], (size_t) T * LD[i] * 4);
            cudaMemset(o1[i], 0x3f, (size_t) T * LD[i] * 4);
            cudaMemset(o2[i], 0x3f, (size_t) T * LD[i] * 4);
            sg1[i] = {wp[i], R[i], sc3 + i, o1[i], LD[i]};
            sg2[i] = {wp[i], R[i], sc3 + i, o2[i], LD[i]};
        }
        int segbad = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const fp8::Epilogue e = pass ? fp8::Epilogue::Add : fp8::Epilogue::Store;
            fp8::gemm_segmented(sg1, 3, K, act, T, e, nullptr);
            for (int i = 0; i < 3; ++i) fp8::gemm(sg2[i].w, R[i], K, sc3 + i, act, T, o2[i], LD[i], e, nullptr);
            for (int i = 0; i < 3; ++i) {
                std::vector<float> a((size_t) T * LD[i]), b((size_t) T * LD[i]);
                cudaMemcpy(a.data(), o1[i], a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), o2[i], b.size() * 4, cudaMemcpyDeviceToHost);
                segbad += std::memcmp(a.data(), b.data(), a.size() * 4) != 0;
            }
        }
        std::printf("fp8 gemm segmented (3 matrices, Store and Add) bitwise against separate products: %s\n", segbad ? "NO" : "yes");
        fails += segbad;
    }
    {   // the norm in the quantizer: its normed rows against a host RMSNorm, and its codes those of quantize(normed rows)
        std::vector<float> nw(K);
        for (auto& v : nw) v = 1.0f + 0.1f * nd(rng);
        float *d_nw, *d_xn;
        void* act2;
        cudaMalloc(&d_nw, K * 4);
        cudaMalloc(&d_xn, (size_t) T * K * 4);
        cudaMalloc(&act2, fp8::act_bytes(T, K));
        cudaMemcpy(d_nw, nw.data(), K * 4, cudaMemcpyHostToDevice);
        fp8::rms_norm_quantize(dx, d_nw, 1e-6f, T, K, act, d_xn, nullptr);
        fp8::quantize(d_xn, T, K, K, act2, nullptr);
        std::vector<uint8_t> a1(fp8::act_bytes(T, K)), a2(a1.size());
        std::vector<float> xn((size_t) T * K);
        cudaMemcpy(a1.data(), act, a1.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(a2.data(), act2, a2.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xn.data(), d_xn, xn.size() * 4, cudaMemcpyDeviceToHost);
        double nerr = 0;
        for (int64_t t = 0; t < T; ++t) {
            double ss = 0;
            for (int64_t k = 0; k < K; ++k) ss += (double) x[(size_t) t * K + k] * x[(size_t) t * K + k];
            const double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int64_t k = 0; k < K; ++k) {
                const double want = x[(size_t) t * K + k] * inv * nw[k];
                nerr = std::max(nerr, std::fabs(xn[(size_t) t * K + k] - want) / (std::fabs(want) + 1e-3));
            }
        }
        const bool same = a1 == a2;
        std::printf("fp8 rms_norm_quantize: normed rows rel %.3g, codes and scales %s\n", nerr, same ? "bitwise quantize's" : "DIFFER");
        if (nerr > 1e-5 || !same) ++fails;
    }
    if (cudaGetLastError() != cudaSuccess) ++fails;
    std::printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
