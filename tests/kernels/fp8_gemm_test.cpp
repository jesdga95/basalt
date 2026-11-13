// tests/kernels/fp8_gemm_test.cpp - prefill::fp8 (fp8_gemm.hpp): the row quantizer against a CPU E4M3 rounding, the
// product against a double reference on the same codes (a chunk whose rows are not a tile multiple), the Add epilogue,
// and each row the same whatever the chunk's length.  --bench: TFLOP/s at the Qwen3.8-27B projection shapes, 2048
// rows.  Exit 77 without a GPU.
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
            cudaMemset(w, 0x38, (size_t) M * K);
            cudaMalloc(&x, (size_t) T * K * 4);
            cudaMemset(x, 0x3c, (size_t) T * K * 4);
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
    if (cudaGetLastError() != cudaSuccess) ++fails;
    std::printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
