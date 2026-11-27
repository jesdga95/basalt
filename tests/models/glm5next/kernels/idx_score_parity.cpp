// tests/models/glm5next/kernels/idx_score_parity.cpp - the DSA indexer pool scores against a CPU oracle that mirrors
// the kernel's operation order: each head's dot accumulated over head_dim ascending, then the relu-weighted sum over
// heads ascending (fmax(acc,0)*(w[h]*w_scale) added in h order). Visibility (p < (pos+1)/pool) is checked with an
// explicit -INFINITY. Model geometry 32 heads x 128, pool 4; 5 heads x 64 pool 3 as an asymmetric canary.
#include "models/glm5next/kernels/idx_score.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
    unsigned long long s = 2862933555777941757ull;
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

    struct Geom { int n_head, head_dim, pool; };
    const Geom geoms[] = {{32, 128, 4}, {5, 64, 3}};
    const int n_tokens = 2;
    const int n_pool = 37;
    const int32_t pos[2] = {19, 41};

    for (const Geom& g : geoms) {
        const float w_scale = 1.0f / std::sqrt((float) (g.head_dim * g.n_head));
        Lcg rng;
        std::vector<float> iq((size_t) n_tokens * g.n_head * g.head_dim), w((size_t) n_tokens * g.n_head);
        std::vector<float> pooled((size_t) n_pool * g.head_dim);
        for (auto& v : iq) v = rng();
        for (auto& v : w) v = rng();
        for (auto& v : pooled) v = rng();
        std::vector<float> gpu((size_t) n_tokens * n_pool), cpu((size_t) n_tokens * n_pool);

        float* d_iq = d_alloc(iq.size() * 4);
        float* d_w = d_alloc(w.size() * 4);
        float* d_pooled = d_alloc(pooled.size() * 4);
        int32_t* d_pos = (int32_t*) d_alloc(sizeof(int32_t) * n_tokens);
        float* d_scores = d_alloc(gpu.size() * 4);
        cudaMemcpy(d_iq, iq.data(), iq.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, pos, sizeof(int32_t) * n_tokens, cudaMemcpyHostToDevice);

        basalt::models::glm5next::idx_score(d_iq, d_w, d_pooled, d_pos, d_scores, n_tokens, n_pool, g.n_head,
                                            g.head_dim, g.pool, w_scale, 0);
        check(cudaGetLastError() == cudaSuccess, "idx_score launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "idx_score runs");
        cudaMemcpy(gpu.data(), d_scores, gpu.size() * 4, cudaMemcpyDeviceToHost);

        const float inf = std::numeric_limits<float>::infinity();
        for (int t = 0; t < n_tokens; ++t) {
            const int nv = (int) ((pos[t] + 1) / g.pool);
            for (int p = 0; p < n_pool; ++p) {
                if (p >= nv) {
                    cpu[(size_t) t * n_pool + p] = -inf;
                    continue;
                }
                const float* prow = pooled.data() + (size_t) p * g.head_dim;
                double s = 0.0;
                for (int h = 0; h < g.n_head; ++h) {
                    const float* qrow = iq.data() + ((size_t) t * g.n_head + h) * g.head_dim;
                    double acc = 0.0;
                    for (int d = 0; d < g.head_dim; ++d) acc += (double) qrow[d] * (double) prow[d];
                    s += std::fmax(acc, 0.0) * (double) (w[(size_t) t * g.n_head + h] * w_scale);
                }
                cpu[(size_t) t * n_pool + p] = (float) s;
            }
        }

        float worst = 0.0f, worst_rel = 0.0f;
        bool vis_bad = false;
        for (int t = 0; t < n_tokens; ++t) {
            for (int p = 0; p < n_pool; ++p) {
                const float a = gpu[(size_t) t * n_pool + p], b = cpu[(size_t) t * n_pool + p];
                const bool ai = std::isinf(a) && a < 0, bi = std::isinf(b) && b < 0;
                if (ai != bi) vis_bad = true;
                else if (!ai) {
                    const float d = std::fabs(a - b);
                    worst = std::fmax(worst, d);
                    if (std::fabs(b) > 1e-3f) worst_rel = std::fmax(worst_rel, d / std::fabs(b));
                }
            }
        }
        std::printf("heads=%d dim=%d pool=%d: max |gpu - oracle| = %.3e, max rel = %.3e\n", g.n_head, g.head_dim,
                    g.pool, worst, worst_rel);
        check(!vis_bad, "visibility (-inf) matches the oracle");
        char what[96];
        std::snprintf(what, sizeof what, "scores match the oracle within 1e-4 at heads=%d dim=%d", g.n_head,
                      g.head_dim);
        check(worst < 1e-4f, what);

        // A token scored on its own, at its own position, must get the scores it got inside the window.
        float* d_one = d_alloc(n_pool * 4);
        std::vector<float> one(n_pool);
        float alone = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            basalt::models::glm5next::idx_score(d_iq + (size_t) t * g.n_head * g.head_dim,
                                                d_w + (size_t) t * g.n_head, d_pooled, d_pos + t, d_one, 1, n_pool,
                                                g.n_head, g.head_dim, g.pool, w_scale, 0);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                check(false, "a lone token runs");
                break;
            }
            cudaMemcpy(one.data(), d_one, n_pool * 4, cudaMemcpyDeviceToHost);
            for (int p = 0; p < n_pool; ++p)
                if (!std::isinf(one[p]))
                    alone = std::fmax(alone, std::fabs(one[p] - gpu[(size_t) t * n_pool + p]));
        }
        std::printf("heads=%d dim=%d pool=%d: lone token against its window row: %.3e\n", g.n_head, g.head_dim,
                    g.pool, alone);
        check(alone < 1e-6f, "a token scored alone matches its row in the window");
        cudaFree(d_one);
        cudaFree(d_iq); cudaFree(d_w); cudaFree(d_pooled); cudaFree(d_pos); cudaFree(d_scores);
    }
    if (g_failures != 0) {
        std::printf("IDX_SCORE PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("IDX_SCORE PARITY: ok\n");
    return 0;
}
