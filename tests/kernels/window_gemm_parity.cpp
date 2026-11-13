// tests/kernels/window_gemm_parity.cpp - window_gemm against a double-precision reference on the same quantized
// activations, for Q8_0 and NVFP4 at a dense 27B's widths; a row's bits whatever T and the other rows hold; and a bench
// line per shape (us per call, GB/s of weights).  Exit 77 without a GPU.
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

double ue4m3_half(uint8_t x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) return 0.0;
    const int e = x >> 3, m = x & 7;
    return 0.5 * (e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7));
}
const int kE2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// row r of W, dequantized
void dequant_row(int type, const uint8_t* w, int64_t K, int64_t r, std::vector<double>& out) {
    out.resize((size_t) K);
    if (type == 8) {
        const uint8_t* p = w + r * (K / 32 * 34);
        for (int64_t b = 0; b < K / 32; ++b, p += 34) {
            __half h;
            std::memcpy(&h, p, 2);
            const double d = (double) __half2float(h);
            for (int i = 0; i < 32; ++i) out[(size_t) (b * 32 + i)] = d * (double) (int8_t) p[2 + i];
        }
    } else {
        const uint8_t* p = w + r * (K / 64 * 36);
        for (int64_t b = 0; b < K / 64; ++b, p += 36)
            for (int s = 0; s < 4; ++s) {
                const double d = ue4m3_half(p[s]);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t byte = p[4 + 8 * s + (j & 7)];
                    const int code = j < 8 ? (byte & 15) : (byte >> 4);
                    out[(size_t) (b * 64 + s * 16 + j)] = d * kE2m1x2[code];
                }
            }
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    struct Shape { int type; int64_t rows, cols; const char* name; };
    const Shape shapes[] = {{8, 12288, 5120, "Q8_0 attn q 12288x5120"},     {8, 1024, 5120, "Q8_0 attn k 1024x5120"},
                            {8, 10240, 5120, "Q8_0 GDN qkv 10240x5120"},    {8, 6144, 5120, "Q8_0 GDN z 6144x5120"},
                            {8, 5120, 6144, "Q8_0 out 5120x6144"},          {40, 17408, 5120, "NVFP4 gate 17408x5120"},
                            {40, 5120, 17408, "NVFP4 down 5120x17408"},     {40, 32768, 5120, "NVFP4 head part 32768x5120"}};
    std::mt19937 rng(7);
    cudaStream_t st;
    cudaStreamCreate(&st);
    bool ok = true;
    for (const Shape& sh : shapes) {
        const int64_t K = sh.cols, R = sh.rows;
        const size_t wb = (size_t) R * (sh.type == 8 ? K / 32 * 34 : K / 64 * 36);
        std::vector<uint8_t> w(wb);
        for (auto& b : w) b = (uint8_t) rng();
        if (sh.type == 8)
            for (size_t i = 0; i < wb; i += 34) { const __half h = __float2half(0.001f + 0.01f * (rng() % 100) / 100.0f); std::memcpy(&w[i], &h, 2); }
        else
            for (size_t i = 0; i < wb; i += 36) for (int s = 0; s < 4; ++s) w[i + s] = (uint8_t) (0x28 + (rng() % 24));
        std::vector<float> x((size_t) 8 * K), x2;
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        void *d_w, *d_a, *d_a2;
        float *d_x, *d_y, *d_y2;
        const size_t ab = k::window_act_bytes(K);
        cudaMalloc(&d_w, wb);
        cudaMalloc(&d_a, ab);
        cudaMalloc(&d_a2, ab);
        cudaMalloc(&d_x, x.size() * 4);
        cudaMalloc(&d_y, (size_t) 8 * R * 4);
        cudaMalloc(&d_y2, (size_t) 8 * R * 4);
        cudaMemcpy(d_w, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        if (!k::window_gemm_supported(sh.type, R, K, d_w)) { std::printf("FAIL %s not supported\n", sh.name); ok = false; continue; }
        k::window_quantize(d_x, K, K, d_a, st);
        k::window_gemm(sh.type, d_w, R, K, d_a, d_y, R, 8, st);
        cudaStreamSynchronize(st);
        std::vector<int8_t> q((size_t) 8 * K);
        std::vector<float> dq((size_t) 8 * (K / 32)), y((size_t) 8 * R);
        cudaMemcpy(q.data(), d_a, q.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(dq.data(), (uint8_t*) d_a + (size_t) 8 * K, dq.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost);
        // reference on a sample of rows, every column
        double max_err = 0, max_ref = 0, max_qerr = 0;
        std::vector<double> row;
        for (int t = 0; t < 8; ++t)
            for (int64_t i = 0; i < K; ++i) {
                const double back = (double) q[(size_t) (t * K + i)] * dq[(size_t) (t * (K / 32) + i / 32)];
                max_qerr = std::max(max_qerr, std::fabs(back - x[(size_t) (t * K + i)]) / (dq[(size_t) (t * (K / 32) + i / 32)] + 1e-30));
            }
        for (int64_t r = 0; r < R; r += std::max<int64_t>(1, R / 64)) {
            dequant_row(sh.type, w.data(), K, r, row);
            for (int t = 0; t < 8; ++t) {
                double s = 0;
                for (int64_t i = 0; i < K; ++i)
                    s += row[(size_t) i] * (double) q[(size_t) (t * K + i)] * dq[(size_t) (t * (K / 32) + i / 32)];
                max_err = std::max(max_err, std::fabs(s - (double) y[(size_t) (t * R + r)]));
                max_ref = std::max(max_ref, std::fabs(s));
            }
        }
        // a row's bits: T = 1..8, the other rows of x changed
        x2 = x;
        for (size_t i = (size_t) K; i < x2.size(); ++i) x2[i] = nd(rng) * 3.0f;
        cudaMemcpy(d_x, x2.data(), x2.size() * 4, cudaMemcpyHostToDevice);
        k::window_quantize(d_x, K, K, d_a2, st);
        bool bitwise = true;
        std::vector<float> y2((size_t) R);
        for (int T = 1; T <= 8; ++T) {
            cudaMemsetAsync(d_y2, 0xFF, (size_t) 8 * R * 4, st);
            k::window_gemm(sh.type, d_w, R, K, d_a2, d_y2, R, T, st);
            cudaStreamSynchronize(st);
            cudaMemcpy(y2.data(), d_y2, (size_t) R * 4, cudaMemcpyDeviceToHost);
            bitwise = bitwise && std::memcmp(y2.data(), y.data(), (size_t) R * 4) == 0;
        }
        cudaEvent_t a, e;
        cudaEventCreate(&a);
        cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) k::window_gemm(sh.type, d_w, R, K, d_a, d_y, R, 8, st);
        cudaEventRecord(a, st);
        for (int i = 0; i < 50; ++i) k::window_gemm(sh.type, d_w, R, K, d_a, d_y, R, 8, st);
        cudaEventRecord(e, st);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        const double us = 1000.0 * ms / 50;
        const double rel = max_err / (max_ref + 1e-30);
        const bool pass = rel < 1e-5 && bitwise && max_qerr <= 0.5 + 1e-6;
        ok = ok && pass;
        std::printf("%s %-28s rel err %.2e (max |ref| %.3g), quantize err %.3f step, row 0 bitwise for T 1-8 and other rows: %s;"
                    " bench %7.1f us, %5.0f GB/s\n",
                    pass ? "ok  " : "FAIL", sh.name, rel, max_ref, max_qerr, bitwise ? "yes" : "NO", us, wb / us / 1e3);
        cudaFree(d_w); cudaFree(d_a); cudaFree(d_a2); cudaFree(d_x); cudaFree(d_y); cudaFree(d_y2);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
