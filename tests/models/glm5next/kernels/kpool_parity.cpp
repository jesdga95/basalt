// tests/models/glm5next/kernels/kpool_parity.cpp - the kpool key pooling against a CPU oracle that mirrors the
// kernel's operation order: the slot-max scan, exp(sc - max) accumulated in slot order, and the fmaf of the weighted
// key. The model's pool is 4 slots x 128 channels; 3 slots x 96 and 8 x 128 run as asymmetric canaries so a
// slot/channel stride swap cannot pass.
#include "models/glm5next/kernels/kpool.hpp"

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
    unsigned long long s = 709111211ull;
    float operator()() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = s >> 11;
        return (float) (double) r / 9007199254740992.0f * 2.0f - 1.0f;
    }
};

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
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    bool have = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) { have = true; break; }
    }
    if (!have) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }

    constexpr int kPools = 5;
    struct Geom { int kpool, hd; };
    const Geom geoms[] = {{4, 128}, {3, 96}, {8, 128}};
    for (const Geom& geom : geoms) {
        const int kpool = geom.kpool, hd = geom.hd;
        const size_t n_pk = (size_t) kPools * kpool * hd;
        Lcg rng;
        std::vector<float> key(n_pk), gate(n_pk), ape((size_t) kpool * hd), out_gpu(n_pk), out_cpu(n_pk);
        for (auto& v : key) v = rng();
        for (auto& v : gate) v = 2.0f * rng();
        for (auto& v : ape) v = rng();

        float* d_key = d_alloc(n_pk * 4);
        float* d_gate = d_alloc(n_pk * 4);
        float* d_ape = d_alloc((size_t) kpool * hd * 4);
        float* d_out = d_alloc(n_pk * 4);
        cudaMemcpy(d_key, key.data(), n_pk * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate, gate.data(), n_pk * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ape, ape.data(), (size_t) kpool * hd * 4, cudaMemcpyHostToDevice);

        basalt::models::glm5next::kpool_compress(d_key, d_gate, d_ape, d_out, kPools, kpool, hd, 0);
        check(cudaGetLastError() == cudaSuccess, "kpool launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "kpool runs");
        cudaMemcpy(out_gpu.data(), d_out, n_pk * 4, cudaMemcpyDeviceToHost);

        // the oracle, in the kernel's exact order
        for (int p = 0; p < kPools; ++p) {
            for (int d = 0; d < hd; ++d) {
                const size_t base = (size_t) p * kpool * hd + d;
                float sc[8];
                float m = -3.402823466e+38f;
                for (int s = 0; s < kpool; ++s) {
                    sc[s] = gate[base + (size_t) s * hd] + ape[(size_t) s * hd + d];
                    m = std::fmax(m, sc[s]);
                }
                float den = 0.0f, acc = 0.0f;
                for (int s = 0; s < kpool; ++s) {
                    const float e = std::expf(sc[s] - m);
                    den += e;
                    acc = std::fmaf(e, key[base + (size_t) s * hd], acc);
                }
                out_cpu[(size_t) p * hd + d] = acc / den;
            }
        }

        float worst = 0.0f;
        for (size_t i = 0; i < n_pk; ++i) worst = std::fmax(worst, std::fabs(out_gpu[i] - out_cpu[i]));
        std::printf("kpool=%d head=%d: max |gpu - oracle| = %.3e\n", kpool, hd, worst);
        char what[96];
        std::snprintf(what, sizeof what, "kpool matches the oracle within 1e-4 at kpool=%d hd=%d", kpool, hd);
        check(worst < 1e-4f, what);
        check(std::isfinite(worst), "no NaN/Inf");
        cudaFree(d_key); cudaFree(d_gate); cudaFree(d_ape); cudaFree(d_out);
    }
    if (g_failures != 0) {
        std::printf("KPOOL PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("KPOOL PARITY: ok\n");
    return 0;
}
