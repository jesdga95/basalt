// tests/models/glm5next/kernels/kda_norm_parity.cpp - the KDA gated output norm against an operation-order-exact CPU
// oracle: per lane the q-major fmaf chain of squares, the shfl_down butterfly, the fp32 scale and gate. The model's
// head is 128 wide (ssm_norm[128], eps attention.layer_norm_rms_epsilon); 192 runs as the asymmetric canary, 16 as
// the under-a-warp one - the launcher must still start for a row narrower than 32 channels.
#include "models/glm5next/kernels/kda_norm.hpp"

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
    float operator()() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = s >> 11;
        return (float) (double) r / 9007199254740992.0f * 2.0f - 1.0f;
    }
};

// the shfl_down butterfly fold, emulated exactly: each level adds the partner off lanes up (lane 0's chain)
float butterfly_fold(float v[32]) {
    for (int off = 16; off > 0; off >>= 1)
        for (int i = 0; i < off; ++i) v[i] = v[i] + v[i + off];
    return v[0];
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

    constexpr int kRows = 8;  // 2 tokens x 4 heads
    const float kEps = 1e-5f;  // attention.layer_norm_rms_epsilon
    for (const int hd : {128, 192, 16}) {
        const size_t n = (size_t) kRows * hd;
        Lcg rng;
        std::vector<float> o(n), z(n), w(hd), out_gpu(n), out_cpu(n);
        for (auto& v : o) v = rng();
        for (auto& v : z) v = rng();
        for (auto& v : w) v = 0.8f + 0.4f * rng();

        float *d_o = d_alloc(n * 4), *d_z = d_alloc(n * 4), *d_w = d_alloc(hd * 4), *d_out = d_alloc(n * 4);
        cudaMemcpy(d_o, o.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), hd * 4, cudaMemcpyHostToDevice);

        basalt::models::glm5next::kda_norm(d_o, d_z, d_w, d_out, kRows, hd, kEps, 0);
        check(cudaGetLastError() == cudaSuccess, "norm launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "norm runs");
        cudaMemcpy(out_gpu.data(), d_out, n * 4, cudaMemcpyDeviceToHost);

        // the oracle, in the kernel's exact order: lane q-chains, then the butterfly, then scale·w·sigmoid. A width
        // under 32 lanes still runs one quarter, and the lanes past the row contribute nothing, as in the kernel.
        const int nq = (hd + 31) / 32;
        for (int r = 0; r < kRows; ++r) {
            float part[32];
            for (int lane = 0; lane < 32; ++lane) {
                float ss = 0.0f;
                for (int q = 0; q < nq; ++q) {
                    const int d = q * 32 + lane;
                    const float ov = d < hd ? o[(size_t) r * hd + d] : 0.0f;
                    ss = fmaf(ov, ov, ss);
                }
                part[lane] = ss;
            }
            const float scale = 1.0f / std::sqrt(butterfly_fold(part) / (float) hd + kEps);
            for (int d = 0; d < hd; ++d) {
                const size_t i = (size_t) r * hd + d;
                out_cpu[i] = o[i] * scale * w[d] / (1.0f + std::expf(-z[i]));
            }
        }

        float worst = 0.0f, worst_rel = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float d = std::fabs(out_gpu[i] - out_cpu[i]);
            worst = std::fmax(worst, d);
            const float m = std::fabs(out_cpu[i]);
            if (m > 1e-3f) worst_rel = std::fmax(worst_rel, d / m);
        }
        std::printf("hd=%d: max |gpu - oracle| = %.3e, max rel = %.3e\n", hd, worst, worst_rel);
        char what[96];
        std::snprintf(what, sizeof what, "norm matches the oracle within 1e-4 at hd=%d", hd);
        check(worst < 1e-4f, what);
        check(std::isfinite(worst), "no NaN/Inf");

        // A row's answer is its own: the same row normed on its own must give the same floats as inside a window, so
        // the kernel's answer cannot depend on how many rows the window carried.
        float alone = 0.0f;
        float* d_one = d_alloc((size_t) hd * 4);
        std::vector<float> one(hd);
        for (int r = 0; r < kRows; ++r) {
            basalt::models::glm5next::kda_norm(d_o + (size_t) r * hd, d_z + (size_t) r * hd, d_w, d_one, 1, hd, kEps,
                                               0);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                check(false, "a lone row runs");
                break;
            }
            cudaMemcpy(one.data(), d_one, (size_t) hd * 4, cudaMemcpyDeviceToHost);
            for (int d = 0; d < hd; ++d)
                alone = std::fmax(alone, std::fabs(one[d] - out_gpu[(size_t) r * hd + d]));
        }
        std::printf("hd=%d: lone row against its window row: %.3e\n", hd, alone);
        check(alone < 1e-6f, "a row normed alone matches its row in the window");
        cudaFree(d_one);
        cudaFree(d_o); cudaFree(d_z); cudaFree(d_w); cudaFree(d_out);
    }
    if (g_failures != 0) {
        std::printf("KDA NORM PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("KDA NORM PARITY: ok\n");
    return 0;
}
