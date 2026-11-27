// tests/models/glm5next/kernels/mla_sparse_parity.cpp - nope sparse-MQA against a double-precision full-softmax
// reference at the real geometry (64 heads x 512 latent, one shared latent per cell = key = value). The reference
// softmaxes over exactly the cells the kernel gathers (ids in [0, cache_len), -1 and out-of-range skipped), so it checks
// the selection semantics and the attention math, not the chunked fp32 accumulation order; the chunk stage and full
// softmax agree to a few 1e-6. The same tolerance style the DeepSeek-V4.1 sparse-MQA parity uses.
#include "models/glm5next/kernels/mla_sparse.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL: %s\n", what); ++g_failures; } }

struct Lcg {
    unsigned long long s = 1442695040888963407ull;
    unsigned int next() { s = s * 6364136223846793005ull + 1; return (unsigned int) (s >> 33); }
    float unit() { return ((float) (next() >> 8) / 8388608.0f) - 1.0f; }  // [-1,1)
};

void* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { std::printf("FAIL: cudaMalloc\n"); std::exit(1); }
    return p;
}

// double reference for one token/head over the valid cells
void ref_head(const float* qh, const std::vector<float>& latent, const int32_t* ids, int n_sel, int cl, double scale,
              float* out) {
    std::vector<const float*> keys;
    for (int i = 0; i < n_sel; ++i) {
        const int id = ids[i];
        if (id >= 0 && id < cl) keys.push_back(latent.data() + (size_t) id * 512);
    }
    if (keys.empty()) { for (int e = 0; e < 512; ++e) out[e] = 0.f; return; }
    std::vector<double> s(keys.size());
    double m = -INFINITY;
    for (size_t k = 0; k < keys.size(); ++k) {
        double d = 0.0;
        for (int e = 0; e < 512; ++e) d += (double) qh[e] * keys[k][e];
        s[k] = d * scale;
        m = std::max(m, s[k]);
    }
    double den = 0.0;
    for (size_t k = 0; k < keys.size(); ++k) den += std::exp(s[k] - m);
    std::vector<double> num(512, 0.0);
    for (size_t k = 0; k < keys.size(); ++k) {
        const double w = std::exp(s[k] - m) / den;
        for (int e = 0; e < 512; ++e) num[e] += w * keys[k][e];
    }
    for (int e = 0; e < 512; ++e) out[e] = (float) num[e];
}

void run_case(const char* name, int n_tokens, int n_head, int n_sel, int n_cells_max, Lcg& rng) {
    const float scale = 1.0f / std::sqrt(256.0f);
    std::vector<float> q_abs((size_t) n_tokens * n_head * 512), latent((size_t) n_cells_max * 512);
    std::vector<int32_t> ids((size_t) n_tokens * n_sel), cache_len(n_tokens);
    for (auto& v : q_abs) v = rng.unit();
    for (auto& v : latent) v = rng.unit();
    for (int t = 0; t < n_tokens; ++t) {
        cache_len[t] = 1 + (int) (rng.next() % n_cells_max);
        int placed = 0;
        for (int i = 0; i < n_sel; ++i) {
            const unsigned int r = rng.next() % 10;
            if (r == 0) ids[(size_t) t * n_sel + i] = -1;                      // hole
            else if (r == 1) ids[(size_t) t * n_sel + i] = cache_len[t] + (int) (rng.next() % 4);  // out of range
            else { ids[(size_t) t * n_sel + i] = (int) (rng.next() % cache_len[t]); ++placed; }
        }
        if (!placed) ids[(size_t) t * n_sel] = (int) (rng.next() % cache_len[t]);  // ensure >= 1 valid
    }
    const size_t ws = basalt::models::glm5next::mla_sparse_workspace_floats(n_tokens, n_head, n_sel);
    std::vector<float> gpu((size_t) n_tokens * n_head * 512), cpu((size_t) n_tokens * n_head * 512);

    float* d_q = (float*) d_alloc(q_abs.size() * 4);
    float* d_lat = (float*) d_alloc(latent.size() * 4);
    int32_t* d_ids = (int32_t*) d_alloc(ids.size() * 4);
    int32_t* d_cl = (int32_t*) d_alloc(cache_len.size() * 4);
    float* d_out = (float*) d_alloc(gpu.size() * 4);
    float* d_ws = (float*) d_alloc(ws * 4);
    cudaMemcpy(d_q, q_abs.data(), q_abs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, latent.data(), latent.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cl, cache_len.data(), cache_len.size() * 4, cudaMemcpyHostToDevice);

    basalt::models::glm5next::mla_sparse(d_q, d_lat, d_ids, d_cl, d_out, scale, n_tokens, n_head, n_sel, d_ws, 0);
    check(cudaGetLastError() == cudaSuccess, "mla_sparse launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "mla_sparse runs");
    cudaMemcpy(gpu.data(), d_out, gpu.size() * 4, cudaMemcpyDeviceToHost);

    for (int t = 0; t < n_tokens; ++t)
        for (int h = 0; h < n_head; ++h)
            ref_head(q_abs.data() + ((size_t) t * n_head + h) * 512, latent, ids.data() + (size_t) t * n_sel, n_sel,
                     cache_len[t], scale, cpu.data() + ((size_t) t * n_head + h) * 512);

    double se = 0.0, sr = 0.0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const double d = std::fabs((double) gpu[i] - (double) cpu[i]);
        se += d * d;
        sr += (double) cpu[i] * cpu[i];
        max_abs = std::max(max_abs, d);
        const double m = std::fabs((double) cpu[i]);
        if (m > 1e-3) max_rel = std::max(max_rel, d / m);
    }
    const double rms_rel = std::sqrt(se / std::max(1e-30, sr));
    std::printf("%-14s T=%d heads=%2d sel=%3d: max_abs %.2e  max_rel %.2e  rms_rel %.2e\n", name, n_tokens, n_head,
                n_sel, max_abs, max_rel, rms_rel);
    check(rms_rel < 1e-5, "attention matches the double reference (rms_rel < 1e-5)");

    // One token attended on its own must answer as it does inside the window: the answer belongs to the token and the
    // ids it was given, not to how many tokens the window happened to carry.
    float* d_one = (float*) d_alloc((size_t) n_head * 512 * 4);
    double alone = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        basalt::models::glm5next::mla_sparse(d_q + (size_t) t * n_head * 512, d_lat, d_ids + (size_t) t * n_sel,
                                             d_cl + t, d_one, scale, 1, n_head, n_sel, d_ws, 0);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            check(false, "a lone token runs");
            break;
        }
        std::vector<float> one((size_t) n_head * 512);
        cudaMemcpy(one.data(), d_one, one.size() * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < one.size(); ++i)
            alone = std::max(alone, std::fabs((double) one[i] - (double) gpu[(size_t) t * n_head * 512 + i]));
    }
    std::printf("%-14s one token alone against its window row: max_abs %.2e\n", name, alone);
    check(alone < 1e-6, "a token attended alone matches its row in the window");
    cudaFree(d_one);

    cudaFree(d_q); cudaFree(d_lat); cudaFree(d_ids); cudaFree(d_cl); cudaFree(d_out); cudaFree(d_ws);
}

}  // namespace

int main() {
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) { std::printf("SKIP: no CUDA device\n"); return 77; }
    bool have = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) { have = true; break; }
    }
    if (!have) { std::printf("SKIP: no sm_120 device\n"); return 77; }

    Lcg rng;
    run_case("model64x64", 2, 64, 64, 300, rng);      // exact chunk, model heads
    run_case("model64x212", 3, 64, 212, 900, rng);    // partial chunk + heavy masking
    run_case("model64x512", 2, 64, 512, 1500, rng);   // eight full chunks (model top_k/pool cells)
    run_case("canary16x100", 4, 16, 100, 400, rng);   // asymmetric heads, partial chunk

    if (g_failures != 0) { std::printf("MLA_SPARSE PARITY: %d FAILURES\n", g_failures); return 1; }
    std::printf("MLA_SPARSE PARITY: ok\n");
    return 0;
}
