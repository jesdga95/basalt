// tests/models/glm5next/kernels/kda_parity.cpp - the KDA kernel against a CPU oracle, at the model's head geometry.
//
// The oracle is the recurrence from kda.hpp written in plain float in the same operation order the kernel uses (the
// shared products q*g / k*g, the ascending dot(q,k), the fmaf update), so agreement is exact up to fmaf contraction.
// Two steps in sequence, so a carry of the state into a second step is checked too. Random inputs from a fixed seed.
#include "models/glm5next/kernels/kda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

struct Lcg {
    unsigned long long s = 88172645463325252ull;
    float operator()() {  // [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = s >> 11;
        return (float) (double) r / 9007199254740992.0f * 2.0f - 1.0f;
    }
};

// the oracle step, in place on `state`, writing `out`; one head at a time, the kernel's operation order
void oracle_step(std::vector<float>& state, const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v, const std::vector<float>& g, const std::vector<float>& beta,
                 std::vector<float>& out, int heads, int dk, int dv) {
    for (int h = 0; h < heads; ++h) {
        float dotqk = 0.0f;
        for (int j = 0; j < dk; ++j) dotqk = fmaf(q[(size_t) h * dk + j], k[(size_t) h * dk + j], dotqk);
        for (int i = 0; i < dv; ++i) {
            float* row = state.data() + ((size_t) h * dv + i) * dk;
            float acc_o = 0.0f, acc_v = 0.0f;
            for (int j = 0; j < dk; ++j) {
                acc_o = fmaf(row[j], q[(size_t) h * dk + j] * g[(size_t) h * dk + j], acc_o);
                acc_v = fmaf(row[j], k[(size_t) h * dk + j] * g[(size_t) h * dk + j], acc_v);
            }
            const float u = beta[h] * (v[(size_t) h * dv + i] - acc_v);
            for (int j = 0; j < dk; ++j) row[j] = fmaf(k[(size_t) h * dk + j], u, row[j] * g[(size_t) h * dk + j]);
            out[(size_t) h * dv + i] = acc_o + u * dotqk;
        }
    }
}

float* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        std::exit(1);
    }
    return (float*) p;
}

}  // namespace

int main() {
    // the engine kernels build for sm_120 only; pick the first card that runs them (a peer Ada card would refuse
    // the launch with cudaErrorNoKernelImageForDevice)
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    bool have_device = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12) {
            if (cudaSetDevice(d) != cudaSuccess) continue;
            have_device = true;
            break;
        }
    }
    if (!have_device) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }
    // the model's geometry is 64 heads of 128x128 (the head and value dims are equal: ssm_conv1d_v is 8192 = 64*128).
    // Run it at 8 heads, and once with dv != dk so a state-layout mix-up cannot hide behind a square state.
    constexpr int kHeads = 8, kSteps = 2;
    const int kGeoms[][2] = {{128, 128}, {128, 192}};
    for (const auto& geo : kGeoms) {
        const int kDk = geo[0], kDv = geo[1];
        const size_t n_state = (size_t) kHeads * kDv * kDk, n_q = (size_t) kHeads * kDk, n_v = (size_t) kHeads * kDv;

        Lcg rng;
        std::vector<float> state(n_state), q(kSteps * n_q), k(kSteps * n_q), v(kSteps * n_v), g(kSteps * n_q),
            beta(kSteps * kHeads), out_gpu(n_v), out_cpu(n_v), state_cpu(n_state);
        for (auto& x : state) x = 0.2f * rng();
        state_cpu = state;
        for (auto& x : q) x = rng();
        for (auto& x : k) x = rng();
        for (auto& x : v) x = rng();
        for (auto& x : g) x = 0.05f + 0.9f / (1.0f + std::exp(-1.5f * rng()));  // a decay factor in (0.05, 0.95)
        for (auto& x : beta) x = 1.0f / (1.0f + std::exp(-2.0f * rng()));

        float *d_state = d_alloc(n_state * 4), *d_q = d_alloc(n_q * 4), *d_k = d_alloc(n_q * 4),
              *d_v = d_alloc(n_v * 4), *d_g = d_alloc(n_q * 4), *d_beta = d_alloc(kHeads * 4), *d_out = d_alloc(n_v * 4);

        // per-step slices, the state carried between steps in both the device and the oracle copy
        float worst = 0.0f;
        std::vector<float> sq(n_q), sk(n_q), sv(n_v), sg(n_q), sb(kHeads);
        for (int step = 0; step < kSteps; ++step) {
            for (size_t i = 0; i < n_q; ++i) { sq[i] = q[step * n_q + i]; sk[i] = k[step * n_q + i]; sg[i] = g[step * n_q + i]; }
            for (size_t i = 0; i < n_v; ++i) sv[i] = v[step * n_v + i];
            for (int h = 0; h < kHeads; ++h) sb[h] = beta[step * kHeads + h];
            cudaMemcpy(d_state, state.data(), n_state * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, sq.data(), n_q * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_k, sk.data(), n_q * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, sv.data(), n_v * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_g, sg.data(), n_q * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_beta, sb.data(), kHeads * 4, cudaMemcpyHostToDevice);
            basalt::models::glm5next::kda_step(d_state, d_q, d_k, d_v, d_g, d_beta, d_out, kHeads, kDk, kDv, 0);
            check(cudaGetLastError() == cudaSuccess, "kernel launches");
            check(cudaDeviceSynchronize() == cudaSuccess, "kernel runs");
            cudaMemcpy(out_gpu.data(), d_out, n_v * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(state.data(), d_state, n_state * 4, cudaMemcpyDeviceToHost);
            oracle_step(state_cpu, sq, sk, sv, sg, sb, out_cpu, kHeads, kDk, kDv);
            for (size_t i = 0; i < n_v; ++i) worst = std::fmax(worst, std::fabs(out_gpu[i] - out_cpu[i]));
            for (size_t i = 0; i < n_state; ++i) worst = std::fmax(worst, std::fabs(state[i] - state_cpu[i]));
        }
        std::printf("dk=%d dv=%d: max |gpu - oracle| = %.3e\n", kDk, kDv, worst);
        if (worst >= 1e-4f) {
            std::printf("out gpu[0..3]: %.5f %.5f %.5f %.5f\n", out_gpu[0], out_gpu[1], out_gpu[2], out_gpu[3]);
            std::printf("out cpu[0..3]: %.5f %.5f %.5f %.5f\n", out_cpu[0], out_cpu[1], out_cpu[2], out_cpu[3]);
        }
        char what[96];
        std::snprintf(what, sizeof what, "kernel matches the oracle within 1e-4 at %dx%d", kDk, kDv);
        check(worst < 1e-4f, what);
        check(std::isfinite(worst), "no NaN/Inf");

        cudaFree(d_state); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_beta); cudaFree(d_out);
    }
    if (g_failures != 0) {
        std::printf("KDA PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("KDA PARITY: ok\n");
    return 0;
}
