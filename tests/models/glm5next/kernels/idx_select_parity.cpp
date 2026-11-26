// tests/models/glm5next/kernels/idx_select_parity.cpp - top-K pool selection against a CPU oracle that mirrors the
// output contract only: the K highest finite scores (ties to the lower index), pool ids ascending, -1 padded, out_n the
// count. The oracle is order-independent of the kernel's internals (it just sorts), so this checks the semantics, not
// the radix/bitonic machinery. Rows mix finite and -INFINITY scores and repeat exact values to force ties.
#include "models/glm5next/kernels/idx_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
    unsigned long long s = 88172645463325252ull;
    unsigned int next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (unsigned int) (s >> 33);
    }
    float unit() { return (float) (next() >> 8) / 16777216.0f; }
};

float* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        std::exit(1);
    }
    return (float*) p;
}

// Oracle: the K highest finite scores of one row, ties to the lower index, ids ascending, -1 padded; returns count.
int oracle_row(const float* s, int n, int K, int32_t* out) {
    const float inf = std::numeric_limits<float>::infinity();
    struct E { float v; int i; };
    std::vector<E> f;
    for (int i = 0; i < n; ++i)
        if (s[i] > -inf) f.push_back({s[i], i});
    std::stable_sort(f.begin(), f.end(), [](const E& a, const E& b) {
        if (a.v != b.v) return a.v > b.v;  // higher score first
        return a.i < b.i;                  // tie -> lower index first
    });
    const int keep = std::min((int) f.size(), K);
    std::vector<int> sel;
    for (int k = 0; k < keep; ++k) sel.push_back(f[k].i);
    std::sort(sel.begin(), sel.end());  // ids ascending
    for (int i = 0; i < K; ++i) out[i] = i < keep ? sel[i] : -1;
    return keep;
}

void run_case(int n_pool, int K, int n_tokens, int value_span, Lcg& rng) {
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> scores((size_t) n_tokens * n_pool);
    for (auto& v : scores) {
        if (rng.next() % 7 == 0) v = -inf;  // some invisible pools
        else v = std::floor(rng.unit() * value_span) - value_span / 2.0f;  // coarse -> forced ties
    }
    std::vector<int32_t> gpu((size_t) n_tokens * K), gpu_n(n_tokens), cpu((size_t) n_tokens * K), cpu_n(n_tokens);

    float* d_scores = d_alloc(scores.size() * 4);
    int32_t* d_ids = (int32_t*) d_alloc(gpu.size() * 4);
    int32_t* d_n = (int32_t*) d_alloc(n_tokens * 4);
    cudaMemcpy(d_scores, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice);

    basalt::models::glm5next::idx_select(d_scores, d_ids, d_n, n_tokens, n_pool, K, 0);
    check(cudaGetLastError() == cudaSuccess, "idx_select launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "idx_select runs");
    cudaMemcpy(gpu.data(), d_ids, gpu.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_n.data(), d_n, n_tokens * 4, cudaMemcpyDeviceToHost);

    int mismatch = 0, ncount_bad = 0;
    for (int t = 0; t < n_tokens; ++t) {
        cpu_n[t] = oracle_row(scores.data() + (size_t) t * n_pool, n_pool, K, cpu.data() + (size_t) t * K);
        if (gpu_n[t] != cpu_n[t]) ++ncount_bad;
        for (int i = 0; i < K; ++i)
            if (gpu[(size_t) t * K + i] != cpu[(size_t) t * K + i]) ++mismatch;
    }
    std::printf("n_pool=%d K=%d ntok=%d: mismatched ids=%d, bad out_n=%d\n", n_pool, K, n_tokens, mismatch,
                ncount_bad);
    check(mismatch == 0, "selected pool ids match the oracle (set, ties, ascending, padding)");
    check(ncount_bad == 0, "out_n matches the count kept");

    // The choice belongs to the row: the same scores given on their own must pick the same pools, in the same order.
    int32_t* d_one_ids = (int32_t*) d_alloc(K * 4);
    int32_t* d_one_n = (int32_t*) d_alloc(4);
    std::vector<int32_t> one_ids(K), one_n(1);
    bool alone = true;
    for (int t = 0; t < n_tokens && alone; ++t) {
        basalt::models::glm5next::idx_select(d_scores + (size_t) t * n_pool, d_one_ids, d_one_n, 1, n_pool, K, 0);
        alone = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        if (!alone) break;
        cudaMemcpy(one_ids.data(), d_one_ids, K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(one_n.data(), d_one_n, 4, cudaMemcpyDeviceToHost);
        alone = one_n[0] == gpu_n[t];
        for (int i = 0; i < K && alone; ++i) alone = one_ids[i] == gpu[(size_t) t * K + i];
    }
    check(alone, "a row selected alone matches its row in the window");
    cudaFree(d_one_ids); cudaFree(d_one_n);

    cudaFree(d_scores); cudaFree(d_ids); cudaFree(d_n);
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

    Lcg rng;
    run_case(37, 8, 3, 40, rng);     // small, radix path, many forced ties
    run_case(40, 512, 2, 30, rng);   // all path (nvalid <= K after -inf), padding dominates
    run_case(1100, 512, 4, 2000, rng);  // larger n_pool, grid-stride loops in the block
    run_case(1100, 64, 3, 12, rng);  // large n_pool, coarse values -> heavy tie handling at the threshold

    if (g_failures != 0) {
        std::printf("IDX_SELECT PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("IDX_SELECT PARITY: ok\n");
    return 0;
}
