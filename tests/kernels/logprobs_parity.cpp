// tests/kernels/logprobs_parity.cpp - row_logprobs (basalt/kernels/logprobs.hpp) against a double CPU reference.
//
// Random rows at the real head widths (Flash-Next 151,936; DeepSeek 129,280): normal rows, peaked rows, rows with ties
// and a masked (-inf) logit. The pick's log probability and each alternative's must be within 2e-6 of the reference
// and the ids exactly its order (larger first, ties to the lower id). A row's output must not depend on the batch width
// or its place in it: every row is also run alone and compared bit for bit. Then a microbench: us per call and GB/s
// over the bytes the rows hold (one pass: the chunks are read once into shared memory).
#include "basalt/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using namespace basalt::kernels;

namespace {

int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)
#define CU(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } \
    } while (0)

uint64_t g_rng = 11;
uint64_t next_u64() {   // splitmix64
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
double unit() { return (double) (next_u64() >> 11) * (1.0 / 9007199254740992.0); }
float normal(float sd) { return (float) (sd * std::sqrt(-2.0 * std::log(unit() + 1e-300)) * std::cos(6.283185307179586 * unit())); }

struct Ref { double pick; std::vector<int> ids; std::vector<double> lps; };

Ref reference(const float* row, int n, int pick, int top) {
    double mx = -INFINITY, sum = 0;
    for (int i = 0; i < n; ++i) mx = std::max(mx, (double) row[i]);
    for (int i = 0; i < n; ++i) sum += std::exp((double) row[i] - mx);
    const double lse = mx + std::log(sum);
    std::vector<int> ord((size_t) n);
    std::iota(ord.begin(), ord.end(), 0);
    std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), [&](int a, int b) {
        return row[a] > row[b] || (row[a] == row[b] && a < b);
    });
    Ref r;
    r.pick = (double) row[pick] - lse;
    for (int k = 0; k < top; ++k) { r.ids.push_back(ord[(size_t) k]); r.lps.push_back((double) row[ord[(size_t) k]] - lse); }
    return r;
}

int run_width(int n_vocab) {
    constexpr int R = 8;
    const int top = kLogprobMaxTop;
    std::vector<float> h((size_t) R * n_vocab);
    std::vector<int32_t> picks(R);
    for (int r = 0; r < R; ++r) {
        float* row = h.data() + (size_t) r * n_vocab;
        for (int i = 0; i < n_vocab; ++i) row[i] = normal(r % 2 ? 2.0f : 5.0f);
        if (r == 1) row[next_u64() % (uint64_t) n_vocab] = 40.0f;                                   // peaked
        if (r == 2) for (int k = 0; k < 40; ++k) row[next_u64() % (uint64_t) n_vocab] = 30.0f;     // ties at the top
        if (r == 3) row[17] = -INFINITY;                                                            // a masked logit
        picks[(size_t) r] = r == 3 ? 17 : (int32_t) (next_u64() % (uint64_t) n_vocab);
    }
    float* d_logits = nullptr;
    float* d_out = nullptr;
    float* d_scratch = nullptr;
    int32_t* d_picks = nullptr;
    CU(cudaMalloc((void**) &d_logits, h.size() * sizeof(float)));
    CU(cudaMalloc((void**) &d_out, (size_t) R * kLogprobStride * sizeof(float)));
    CU(cudaMalloc((void**) &d_picks, R * sizeof(int32_t)));
    CU(cudaMalloc((void**) &d_scratch, row_logprobs_scratch_floats(R, n_vocab) * sizeof(float)));
    CU(cudaMemcpy(d_logits, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_picks, picks.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice));
    row_logprobs(d_logits, R, n_vocab, d_picks, top, d_out, d_scratch, nullptr);
    std::vector<float> out((size_t) R * kLogprobStride), alone((size_t) kLogprobStride);
    CU(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0, max_rel = 0;
    for (int r = 0; r < R; ++r) {
        const float* o = out.data() + (size_t) r * kLogprobStride;
        const Ref ref = reference(h.data() + (size_t) r * n_vocab, n_vocab, picks[(size_t) r], top);
        if (std::isinf(ref.pick)) CHECK(std::isinf(o[0]) && o[0] < 0);
        else {
            max_abs = std::max(max_abs, std::fabs(o[0] - ref.pick));
            max_rel = std::max(max_rel, std::fabs(o[0] - ref.pick) / std::max(1e-12, std::fabs(ref.pick)));
        }
        for (int k = 0; k < top; ++k) {
            int32_t id = 0;
            std::memcpy(&id, &o[2 + 2 * k], sizeof id);
            CHECK(id == ref.ids[(size_t) k]);
            const double err = std::fabs(o[3 + 2 * k] - ref.lps[(size_t) k]);
            max_abs = std::max(max_abs, err);
            max_rel = std::max(max_rel, err / std::max(1e-12, std::fabs(ref.lps[(size_t) k])));
        }
        // the same row alone, at another place: the same bits
        row_logprobs(d_logits + (size_t) r * n_vocab, 1, n_vocab, d_picks + r, top, d_out, d_scratch, nullptr);
        CU(cudaMemcpy(alone.data(), d_out, alone.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(alone.data(), o, (size_t) (2 + 2 * top) * sizeof(float)) == 0);
    }
    CHECK(max_abs < 2e-6);
    std::printf("row_logprobs n_vocab=%d: vs the double reference max abs %.3g, max rel %.3g; ids exact; width-independent\n",
                n_vocab, max_abs, max_rel);

    // microbench: one decode window's rows
    for (int rows : {1, 3, 8})
        for (int t : {0, 5, 20}) {
            cudaEvent_t e0, e1;
            CU(cudaEventCreate(&e0));
            CU(cudaEventCreate(&e1));
            for (int i = 0; i < 5; ++i) row_logprobs(d_logits, rows, n_vocab, d_picks, t, d_out, d_scratch, nullptr);
            constexpr int N = 200;
            CU(cudaEventRecord(e0));
            for (int i = 0; i < N; ++i) row_logprobs(d_logits, rows, n_vocab, d_picks, t, d_out, d_scratch, nullptr);
            CU(cudaEventRecord(e1));
            CU(cudaEventSynchronize(e1));
            float ms = 0;
            CU(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / N;
            const double bytes = (double) rows * n_vocab * sizeof(float);
            std::printf("  rows %d top %2d: %7.1f us/call, %6.1f GB/s over %.1f MB read\n", rows, t, us, bytes / us / 1e3,
                        bytes / 1e6);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    cudaFree(d_logits);
    cudaFree(d_out);
    cudaFree(d_picks);
    cudaFree(d_scratch);
    return 0;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::printf("logprobs_parity: no GPU, skipped\n"); return 0; }
    if (run_width(151936) || run_width(129280)) return 1;
    if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
    std::printf("logprobs_parity: ok\n");
    return 0;
}
