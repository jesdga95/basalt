// tests/kernels/e4m3_producers_test.cpp - the prompt chunk's producers that round their rows to E4M3 for an F8 output
// projection themselves (kernels::attn_gate_e4m3, gdn_recurrence's y8) give prefill::fp8::quantize's codes and scales
// of the rows they would have written, bitwise.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fp8 = basalt::prefill::fp8;
namespace k = basalt::kernels;

namespace {
template <class T> T* dev(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<uint8_t> host(const void* d, size_t n) {
    std::vector<uint8_t> h(n);
    cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
    return h;
}
std::vector<float> randn(size_t n, std::mt19937& rng, float sd, float mean = 0.0f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    std::mt19937 rng(11);
    bool ok = true;
    {   // attention: 24 heads of 256, the gate in q_full's second half
        const int64_t T = 333;
        const int nh = 24, cols = nh * 256;
        float* attn = dev(randn((size_t) T * cols, rng, 1.0f));
        float* qf = dev(randn((size_t) T * nh * 512, rng, 2.0f));
        float* out;
        cudaMalloc(&out, (size_t) T * cols * 4);
        void *a1, *a2;
        cudaMalloc(&a1, fp8::act_bytes(T, cols));
        cudaMalloc(&a2, fp8::act_bytes(T, cols));
        cudaMemset(a1, 0, fp8::act_bytes(T, cols));
        cudaMemset(a2, 0, fp8::act_bytes(T, cols));
        k::attn_gate(attn, qf, out, T, nh, nullptr);
        fp8::quantize(out, T, cols, cols, a1, nullptr);
        k::attn_gate_e4m3(attn, qf, T, nh, (uint8_t*) a2, fp8::act_scales(a2, T, cols), nullptr);
        const bool same = cudaDeviceSynchronize() == cudaSuccess &&
                          host(a1, fp8::act_bytes(T, cols)) == host(a2, fp8::act_bytes(T, cols));
        std::printf("attn_gate_e4m3: %s\n", same ? "bitwise attn_gate + quantize" : "DIFFER");
        ok = ok && same;
    }
    {   // GDN: 16 key heads, 48 value heads of 128; the recurrence twice from the same state, closing norm to yn and to y8
        const int64_t T = 300, HV = 48, S = 128, C = 2 * 16 * S + HV * S, VD = HV * S;
        if (!k::gdn_prompt_supported((int) S, 16, (int) HV, 4)) return 77;
        const std::vector<float> st0 = randn((size_t) S * HV * S, rng, 0.05f);
        float* h = dev(randn((size_t) T * C, rng, 0.5f));
        std::vector<float> gate((size_t) T * HV), beta((size_t) T * HV);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t i = 0; i < gate.size(); ++i) {
            gate[i] = -0.5f * u(rng);
            beta[i] = u(rng);
        }
        float *dg = dev(gate), *db = dev(beta), *z = dev(randn((size_t) T * VD, rng, 1.0f)), *gm = dev(randn(S, rng, 0.2f, 1.0f));
        float *s1 = dev(st0), *s2 = dev(st0), *y, *yn;
        void* work;
        cudaMalloc(&y, (size_t) T * VD * 4);
        cudaMalloc(&yn, (size_t) T * VD * 4);
        cudaMalloc(&work, k::gdn_work_bytes(T));
        void *a1, *a2;
        cudaMalloc(&a1, fp8::act_bytes(T, VD));
        cudaMalloc(&a2, fp8::act_bytes(T, VD));
        cudaMemset(a1, 0, fp8::act_bytes(T, VD));
        cudaMemset(a2, 0, fp8::act_bytes(T, VD));
        bool same = true;
        for (k::GdnGate act : {k::GdnGate::Sigmoid, k::GdnGate::Silu}) {
            cudaMemcpy(s1, s2, st0.size() * 4, cudaMemcpyDeviceToDevice);
            k::gdn_recurrence(s1, h, dg, db, z, gm, 1e-6f, work, y, nullptr, T, nullptr, act, yn);
            fp8::quantize(yn, T, VD, VD, a1, nullptr);
            cudaMemcpy(s1, s2, st0.size() * 4, cudaMemcpyDeviceToDevice);
            k::gdn_recurrence(s1, h, dg, db, z, gm, 1e-6f, work, y, nullptr, T, nullptr, act, nullptr, (uint8_t*) a2,
                              fp8::act_scales(a2, T, VD));
            same = same && cudaDeviceSynchronize() == cudaSuccess && host(a1, fp8::act_bytes(T, VD)) == host(a2, fp8::act_bytes(T, VD));
        }
        std::printf("gdn_recurrence y8: %s\n", same ? "bitwise yn + quantize (sigmoid and silu gates)" : "DIFFER");
        ok = ok && same;
    }
    std::printf("%s\n", ok ? "e4m3 producers ok" : "FAIL");
    return ok ? 0 : 1;
}
