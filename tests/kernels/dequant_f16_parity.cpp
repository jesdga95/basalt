// tests/kernels/dequant_f16_parity.cpp - the device dequantizers' FP16 output on random blocks of the dense weight
// types: bitwise against the FP32 path rounded to half, against the artifact's CPU dequantizers, and a microbench
// line (us per call, GB/s over the bytes read and written).
//
//     dequant_f16_parity [rows cols iters]
#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/dequant_bf16.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Ty {
    int type, be, bb;
    const char* name;
    void (*cpu)(const uint8_t*, float*);
};

void cpu_q8_0(const uint8_t* b, float* o) { basalt::dequantize_q8_0(b, o); }
void cpu_iq4_xs(const uint8_t* b, float* o) { basalt::dequantize_iq4_xs(b, o); }
void cpu_q5_k(const uint8_t* b, float* o) { basalt::dequantize_q5_K(b, o); }
void cpu_q6_k(const uint8_t* b, float* o) { basalt::dequantize_q6_K(b, o); }

}  // namespace

int main(int argc, char** argv) {
    const int64_t rows = argc > 1 ? std::atoll(argv[1]) : 10240, cols = argc > 2 ? std::atoll(argv[2]) : 2560;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 50;
    const Ty tys[] = {{23, 256, 136, "IQ4_XS", cpu_iq4_xs}, {8, 32, 34, "Q8_0", cpu_q8_0},
                      {13, 256, 176, "Q5_K", cpu_q5_k}, {14, 256, 210, "Q6_K", cpu_q6_k}};
    int fails = 0;
    std::mt19937 rng(7);
    for (const Ty& t : tys) {
        const int64_t row_bytes = cols / t.be * t.bb, n = rows * cols;
        std::vector<uint8_t> blk((size_t) (rows * row_bytes));
        for (auto& x : blk) x = (uint8_t) rng();
        // a normal scale in every block (random bytes make NaN/inf scales that the comparison cannot order)
        for (int64_t b = 0; b < rows * cols / t.be; ++b) {
            uint8_t* p = blk.data() + (size_t) b * t.bb;
            const int off = (t.type == 14) ? 208 : 0;
            const __half h = __float2half(0.001f + (float) (rng() % 1000) * 1e-4f);
            std::memcpy(p + off, &h, 2);
            if (t.type == 12 || t.type == 13) { const __half m = __float2half(0.01f); std::memcpy(p + 2, &m, 2); }
        }
        uint8_t* d_b = nullptr;
        uint16_t *d_h = nullptr, *d_h2 = nullptr;
        float* d_f = nullptr;
        cudaMalloc(&d_b, blk.size());
        cudaMalloc(&d_h, (size_t) n * 2 + 16);
        cudaMalloc(&d_h2, (size_t) n * 2 + 16);
        cudaMalloc(&d_f, (size_t) n * 4);
        cudaMemcpy(d_b, blk.data(), blk.size(), cudaMemcpyHostToDevice);
        basalt::kernels::dequant_f16(t.type, d_b, 0, rows, cols, d_h, nullptr);
        basalt::kernels::dequant_f32(t.type, d_b, 0, rows, cols, d_f, nullptr);
        // the same output through the unaligned (scalar-store) instance
        basalt::kernels::dequant_f16(t.type, d_b, 0, rows, cols, d_h2 + 1, nullptr);
        std::vector<uint16_t> h((size_t) n), h2((size_t) n);
        std::vector<float> f((size_t) n);
        cudaMemcpy(h.data(), d_h, (size_t) n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2.data(), d_h2 + 1, (size_t) n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(f.data(), d_f, (size_t) n * 4, cudaMemcpyDeviceToHost);
        int64_t bad_f32 = 0, bad_scalar = 0, bad_cpu = 0;
        std::vector<float> ref(t.be);
        for (int64_t i = 0; i < n; ++i) {
            if (__half_as_ushort(__float2half_rn(f[(size_t) i])) != h[(size_t) i]) ++bad_f32;
            if (h[(size_t) i] != h2[(size_t) i]) ++bad_scalar;
        }
        const int64_t check_rows = std::min<int64_t>(rows, 64);
        for (int64_t r = 0; r < check_rows; ++r)
            for (int64_t b = 0; b < cols / t.be; ++b) {
                t.cpu(blk.data() + (size_t) (r * row_bytes + b * t.bb), ref.data());
                for (int k = 0; k < t.be; ++k) {
                    const float v = ref[(size_t) k];
                    const __half hv = __ushort_as_half(h[(size_t) (r * cols + b * t.be + k)]);
                    const double e = std::fabs((double) __half2float(hv) - v);
                    if (e > 1e-3 * std::fabs((double) v) + 1e-6) ++bad_cpu;
                }
            }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < 5; ++i) basalt::kernels::dequant_f16(t.type, d_b, 0, rows, cols, d_h, nullptr);
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) basalt::kernels::dequant_f16(t.type, d_b, 0, rows, cols, d_h, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / iters, gb = ((double) blk.size() + (double) n * 2) / (us * 1e3);
        const bool ok = bad_f32 == 0 && bad_scalar == 0 && bad_cpu == 0;
        if (!ok) ++fails;
        std::printf("%-7s %lld x %lld: vs f32 path %lld off, vs scalar-store %lld off, vs CPU %lld off (%s); %.1f us, %.0f GB/s\n",
                    t.name, (long long) rows, (long long) cols, (long long) bad_f32, (long long) bad_scalar,
                    (long long) bad_cpu, ok ? "ok" : "FAIL", us, gb);
        cudaFree(d_b); cudaFree(d_h); cudaFree(d_h2); cudaFree(d_f);
    }
    std::printf("dequant_f16_parity: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
