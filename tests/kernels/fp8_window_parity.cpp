// tests/kernels/fp8_window_parity.cpp - a decode window's products against FP8 E4M3 matrices (fp8_window.hpp) at
// Qwen3.8-27B projection widths: against a double-precision reference on the fp16-packed activations, each row of a
// window bitwise the same whatever the window's rows (1..8) and whatever the other rows hold, and the Add epilogue.
// --bench: us per call and GB/s of weights, the weights rotated through copies larger than L2.  Exit 77 without a GPU.
#include "basalt/kernels/fp8_window.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace w8 = basalt::kernels::fp8w;

namespace {

std::mt19937 rng(7);
int fails = 0;

float e4m3(uint8_t b) {
    __nv_fp8_e4m3 v;
    std::memcpy(&v, &b, 1);
    return (float) v;
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
std::vector<uint8_t> codes(size_t n) {
    std::vector<uint8_t> w(n);
    for (auto& b : w) {
        b = (uint8_t) rng();
        if ((b & 0x7f) == 0x7f) b = 0;   // no NaN
    }
    return w;
}

struct Shape {
    int64_t rows, K;
    const char* name;
};
const Shape kShapes[] = {{10240, 5120, "GDN qkv 10240x5120"}, {6144, 5120, "GDN z 6144x5120"},
                         {5120, 6144, "out 5120x6144"},       {12288, 5120, "attn q 12288x5120"},
                         {1024, 5120, "attn k 1024x5120"}};

w8::Workspace make_ws() {
    w8::Workspace ws;
    for (const Shape& sh : kShapes) {
        size_t p, t;
        w8::workspace_need(sh.rows, sh.K, p, t);
        ws.partial_floats = std::max(ws.partial_floats, p);
        ws.tickets_n = std::max(ws.tickets_n, t);
    }
    cudaMalloc(&ws.partial, std::max<size_t>(ws.partial_floats, 1) * 4);
    cudaMalloc(&ws.tickets, ws.tickets_n * 4);
    cudaMemset(ws.tickets, 0, ws.tickets_n * 4);
    return ws;
}

void parity(const Shape& sh, const w8::Workspace& ws) {
    const auto w = codes((size_t) sh.rows * sh.K);
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) 8 * sh.K);
    for (int t = 0; t < 8; ++t)
        for (int64_t k = 0; k < sh.K; ++k) x[(size_t) t * sh.K + k] = std::ldexp(1.0f, 3 * t - 10) * nd(rng);
    x[(size_t) 3 * sh.K + 17] = 3.0e4f;
    uint8_t* d_w = up(w);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, w8::act_bytes(sh.K));
    float *d_y, *d_y2, *d_sc;
    cudaMalloc(&d_y, (size_t) 8 * sh.rows * 4);
    cudaMalloc(&d_y2, (size_t) 8 * sh.rows * 4);
    const std::vector<float> sc = {0.03f};
    d_sc = up(sc);
    w8::pack(d_x, sh.K, sh.K, 8, act, nullptr);
    w8::gemm(d_w, sh.rows, sh.K, act, d_sc, d_y, sh.rows, w8::Epilogue::Store, 8, ws, nullptr);
    const auto y8 = down(d_y, (size_t) 8 * sh.rows);
    const auto hx = down((const __half*) act, (size_t) 8 * sh.K);
    const auto fac = down((const float*) ((const uint8_t*) act + (size_t) 8 * sh.K * 2), 8);
    double worst = 0;
    for (int t = 0; t < 8; ++t)
        for (int64_t r = 0; r < sh.rows; r += (sh.rows > 4096 ? 7 : 1)) {
            double s = 0, sa = 0;
            for (int64_t k = 0; k < sh.K; ++k) {
                const double p = (double) e4m3(w[(size_t) r * sh.K + k]) * (double) __half2float(hx[(size_t) t * sh.K + k]);
                s += p;
                sa += std::fabs(p);
            }
            s *= fac[t] * 0.03;
            sa *= fac[t] * 0.03;
            worst = std::max(worst, std::fabs(y8[(size_t) t * sh.rows + r] - s) / std::max(sa, 1e-30));
        }
    const bool ok = worst < 1e-5;
    std::printf("%-22s vs double: max |err| / sum|w x| %.3g%s\n", sh.name, worst, ok ? "" : "  FAIL");
    if (!ok) ++fails;
    int diff = 0;
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> x2 = x;
        for (int t = T; t < 8; ++t)
            for (int64_t k = 0; k < sh.K; ++k) x2[(size_t) t * sh.K + k] = 1e3f * nd(rng);
        float* d_x2 = up(x2);
        w8::pack(d_x2, sh.K, sh.K, T, act, nullptr);
        cudaMemset(d_y2, 0xFF, (size_t) 8 * sh.rows * 4);
        w8::gemm(d_w, sh.rows, sh.K, act, d_sc, d_y2, sh.rows, w8::Epilogue::Store, T, ws, nullptr);
        const auto yt = down(d_y2, (size_t) 8 * sh.rows);
        diff += std::memcmp(yt.data(), y8.data(), (size_t) T * sh.rows * 4) != 0;
        for (size_t i = (size_t) T * sh.rows; i < yt.size(); ++i) diff += !std::isnan(yt[i]);   // rows past T untouched
        cudaFree(d_x2);
    }
    std::printf("%-22s rows bitwise at T = 1..8: %s\n", sh.name, diff ? "DIFFER  FAIL" : "yes");
    if (diff) ++fails;
    const std::vector<float> base((size_t) 8 * sh.rows, 0.5f);
    cudaMemcpy(d_y2, base.data(), base.size() * 4, cudaMemcpyHostToDevice);
    w8::pack(d_x, sh.K, sh.K, 8, act, nullptr);
    w8::gemm(d_w, sh.rows, sh.K, act, d_sc, d_y2, sh.rows, w8::Epilogue::Add, 8, ws, nullptr);
    const auto ya = down(d_y2, (size_t) 8 * sh.rows);
    int bad = 0;
    for (size_t i = 0; i < ya.size(); ++i) bad += std::fabs(ya[i] - (0.5 + (double) y8[i])) > 1e-6 * (1.0 + std::fabs((double) y8[i]));
    std::printf("%-22s Add epilogue: %s\n", sh.name, bad ? "WRONG  FAIL" : "ok");
    if (bad) ++fails;
    cudaFree(d_w); cudaFree(d_x); cudaFree(act); cudaFree(d_y); cudaFree(d_y2); cudaFree(d_sc);
}

void bench(const Shape& sh, const w8::Workspace& ws) {
    const size_t wb = (size_t) sh.rows * sh.K;
    const int copies = (int) std::max<size_t>(2, (512u << 20) / wb + 1);
    std::vector<uint8_t*> d_w(copies);
    for (auto& p : d_w) {
        cudaMalloc(&p, wb);
        cudaMemset(p, 0x38, wb);
    }
    std::vector<float> x((size_t) 8 * sh.K, 0.5f);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, w8::act_bytes(sh.K));
    float* d_y;
    cudaMalloc(&d_y, (size_t) 8 * sh.rows * 4);
    w8::pack(d_x, sh.K, sh.K, 8, act, nullptr);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int T : {1, 8}) {
        int it = 0;
        for (int i = 0; i < copies; ++i) w8::gemm(d_w[it++ % copies], sh.rows, sh.K, act, nullptr, d_y, sh.rows, w8::Epilogue::Store, T, ws, nullptr);
        const int n = 6 * copies;
        cudaEventRecord(e0);
        for (int i = 0; i < n; ++i) w8::gemm(d_w[it++ % copies], sh.rows, sh.K, act, nullptr, d_y, sh.rows, w8::Epilogue::Store, T, ws, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / n;
        std::printf("%-22s %6.1f MB  T=%d %7.1f us (%5.0f GB/s)\n", sh.name, wb / 1e6, T, us, wb / us / 1e3);
    }
    for (auto p : d_w) cudaFree(p);
    cudaFree(d_x); cudaFree(act); cudaFree(d_y);
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    const w8::Workspace ws = make_ws();
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (const Shape& sh : kShapes) bench(sh, ws);
        return 0;
    }
    for (const Shape& sh : kShapes) parity(sh, ws);
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
