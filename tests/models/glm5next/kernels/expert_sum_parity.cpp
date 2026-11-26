// tests/models/glm5next/kernels/expert_sum_parity.cpp - the last launch of an expert block: the weighted sum of a
// token's expert rows, read from wherever the tier keeps each one. The oracle accumulates the same picks in double
// precision, so what is reported is the float sum's distance from it, in absolute and relative terms. `put_words` is
// checked against the copy it stands in for: the plan words must be in the device buffer before the next launch reads
// them.
#include "models/glm5next/kernels/expert_sum.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // the engine kernels are built for the engine's own architecture: take the first card that runs them, and skip
    // where there is none rather than fail a build that has no such card attached
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
        std::printf("SKIP: no device the engine kernels are built for\n");
        return 77;
    }

    // The model's width, the most picks a token routes, and a window narrower than the tier's own: what matters here
    // is that each pick is read from its own address and weighted in ascending order.
    struct Geom { int n_tokens, k, n; };
    const Geom geoms[] = {{3, 8, 4096}, {2, 3, 4096}, {5, 1, 1024}};
    for (const Geom& g : geoms) {
        Lcg rng;
        const int nk = g.n_tokens * g.k;
        std::vector<float> rows_h((size_t) nk * g.n);
        for (auto& v : rows_h) v = rng();
        std::vector<float> w_h(nk);
        for (auto& v : w_h) v = rng();

        // Each pick gets its own device buffer: the kernel must follow a pointer per pick, not one stride.
        std::vector<float*> d_row(nk);
        for (int e = 0; e < nk; ++e) {
            d_row[e] = d_alloc((size_t) g.n * sizeof(float));
            cudaMemcpy(d_row[e], rows_h.data() + (size_t) e * g.n, (size_t) g.n * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        basalt::models::glm5next::RowPtrs rows;
        for (int e = 0; e < nk; ++e) rows.p[e] = d_row[e];
        for (int e = nk; e < basalt::models::glm5next::kMaxSumRows; ++e) rows.p[e] = d_row[0];

        float* d_w = d_alloc(nk * sizeof(float));
        cudaMemcpy(d_w, w_h.data(), nk * sizeof(float), cudaMemcpyHostToDevice);
        const size_t out_n = (size_t) g.n_tokens * g.n;
        float* d_out = d_alloc(out_n * sizeof(float));
        basalt::models::glm5next::weighted_sum_block(rows, d_w, g.n_tokens, g.k, g.n, d_out, 0);
        check(cudaGetLastError() == cudaSuccess, "weighted_sum_block launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "weighted_sum_block runs");
        std::vector<float> out(out_n);
        cudaMemcpy(out.data(), d_out, out_n * sizeof(float), cudaMemcpyDeviceToHost);

        double worst_abs = 0.0, worst_rel = 0.0;
        for (int t = 0; t < g.n_tokens; ++t) {
            for (int i = 0; i < g.n; ++i) {
                double acc = 0.0;
                for (int kk = 0; kk < g.k; ++kk) {
                    const int e = t * g.k + kk;
                    acc += (double) w_h[e] * (double) rows_h[(size_t) e * g.n + i];
                }
                const double got = out[(size_t) t * g.n + i];
                const double abs = std::fabs(got - acc);
                worst_abs = std::fmax(worst_abs, abs);
                if (std::fabs(acc) > 1e-6) worst_rel = std::fmax(worst_rel, abs / std::fabs(acc));
            }
        }
        std::printf("tokens=%d k=%d n=%d: max |abs| %.3e  max |rel| %.3e\n", g.n_tokens, g.k, g.n, worst_abs,
                    worst_rel);
        check(worst_abs < 1e-2, "the weighted sum matches the double reference");
        bool finite = true;
        for (float v : out) finite = finite && std::isfinite(v);
        check(finite, "no NaN/Inf");

        // Two tokens with the same picks and the same weights must get the same row: the result belongs to the token,
        // not to its place in the window.
        const int other = g.n_tokens - 1;
        bool same_row = other != 0;
        if (same_row) {
            for (int kk = 0; kk < g.k; ++kk) {
                cudaMemcpy(d_row[other * g.k + kk], rows_h.data() + (size_t) kk * g.n, (size_t) g.n * sizeof(float),
                           cudaMemcpyHostToDevice);
                w_h[other * g.k + kk] = w_h[kk];
            }
            cudaMemcpy(d_w, w_h.data(), nk * sizeof(float), cudaMemcpyHostToDevice);
            basalt::models::glm5next::weighted_sum_block(rows, d_w, g.n_tokens, g.k, g.n, d_out, 0);
            check(cudaDeviceSynchronize() == cudaSuccess, "the rerun runs");
            cudaMemcpy(out.data(), d_out, out_n * sizeof(float), cudaMemcpyDeviceToHost);
            same_row = std::memcmp(out.data(), out.data() + (size_t) other * g.n, g.n * sizeof(float)) == 0;
        }
        check(same_row, "one token's row does not depend on its position in the window");

        for (float* p : d_row) cudaFree(p);
        cudaFree(d_w);
        cudaFree(d_out);
    }

    // put_words: the plan words reach the device buffer, in stream order, before the next launch reads them - so a
    // kernel-free read-back right after the call must see them.
    {
        Lcg plan_rng;
        std::vector<int> host(basalt::models::glm5next::kPlanWords);
        for (int& v : host) v = (int) (plan_rng() * 100000.f);
        std::vector<int> back(host.size(), 0);
        void* d_words = d_alloc(host.size() * sizeof(int));
        basalt::models::glm5next::put_words(host.data(), d_words, (int) host.size(), 0);
        check(cudaGetLastError() == cudaSuccess, "put_words launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "put_words runs");
        cudaMemcpy(back.data(), d_words, host.size() * sizeof(int), cudaMemcpyDeviceToHost);
        check(std::memcmp(back.data(), host.data(), host.size() * sizeof(int)) == 0, "put_words lands the plan words");
        cudaFree(d_words);
    }

    if (g_failures != 0) {
        std::printf("EXPERT SUM PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("EXPERT SUM PARITY: ok\n");
    return 0;
}
