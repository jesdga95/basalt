// tests/kernels/route_tail_skip_parity.cpp - the route tail skip against a plain CPU rule.
//
//     build/route_tail_skip_parity [--bench]
//
// Random windows of 1..8 tokens at each MoE model's geometry (512 experts top 10, 384 top 6, 512 top 16: distinct within
// a token, shared between tokens), random residency on 1..4 tables, ranks 1..k-1, weights sorted as the router leaves
// them and shuffled. An entry must be dropped (id -1, weight 0) exactly when no table holds its expert and every entry
// naming it ranks R or lower by weight in its token; every other id and weight must be untouched, and the counters
// must match the reference.
//
// --bench: one launch of the window kernel at the largest window, timed over many launches.
#include <cstddef>

#include "basalt/kernels/route_lossy.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {
constexpr int kMaxExperts = 512, kMaxT = 8;
struct Geometry {
    int experts, top;
};
constexpr Geometry kGeometries[] = {{512, 10}, {384, 6}, {512, 16}};

// the entry's rank by weight in its token: heavier entries first, equal weights in position order
int weight_rank(const std::vector<float>& w, int i, int top) {
    const int base = i - i % top;
    int r = 0;
    for (int j = 0; j < top; ++j) r += w[base + j] > w[i] || (w[base + j] == w[i] && base + j < i);
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) {
        cudaGetLastError();
        std::printf("route_tail_skip_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(20261030);
    int32_t *d_ids = nullptr, *d_tab[ResidentTabs::kMax] = {};
    float* d_w = nullptr;
    unsigned long long* d_cnt = nullptr;
    cudaMalloc(&d_ids, kRouteTailSkipMax * 4);
    cudaMalloc(&d_w, kRouteTailSkipMax * 4);
    cudaMalloc(&d_cnt, 3 * 8);
    for (auto& t : d_tab) cudaMalloc(&t, kMaxExperts * 4);
    cudaMemset(d_cnt, 0, 3 * 8);
    long bad = 0, dropped = 0, total = 0, missed = 0;
    for (int it = 0; it < 6000; ++it) {
        const int kExperts = kGeometries[it % 3].experts, kTop = kGeometries[it % 3].top;
        const int nt = 1 + (int) (rng() % kMaxT), n = nt * kTop, R = 1 + (int) (rng() % (kTop - 1));
        const int ntab = 1 + (int) (rng() % ResidentTabs::kMax);
        const int pool = kTop + 2 + (int) (rng() % 60);   // a small pool of experts so the tokens share them
        std::vector<std::vector<int32_t>> tab(ntab, std::vector<int32_t>(kExperts));
        for (auto& t : tab)
            for (int e = 0; e < kExperts; ++e) t[e] = rng() % (3 + ntab) == 0 ? e : -1;
        std::vector<int32_t> ids(n);
        std::vector<float> w(n);
        std::vector<int32_t> p(kExperts);
        std::iota(p.begin(), p.end(), 0);
        for (int t = 0; t < nt; ++t) {
            std::vector<int32_t> q(pool);
            for (int j = 0; j < pool; ++j) q[j] = (int32_t) (j * 13 % kExperts);
            std::shuffle(q.begin(), q.end(), rng);
            std::vector<float> ww(kTop);
            for (auto& x : ww) x = 0.01f + (float) (rng() % 1000) / 1000.0f;
            if (it % 2 == 0) std::sort(ww.begin(), ww.end(), std::greater<float>());   // the router's order
            if (it % 7 == 0) ww[3] = ww[4];                                            // an exact tie
            for (int j = 0; j < kTop; ++j) { ids[t * kTop + j] = q[j]; w[t * kTop + j] = ww[j]; }
        }
        std::vector<int32_t> want = ids;
        std::vector<float> wantw = w;
        long want_miss = 0, want_drop = 0;
        for (int i = 0; i < n; ++i) {
            const int e = ids[i];
            bool held = false;
            for (const auto& t : tab) held |= t[e] >= 0;
            if (held) continue;
            ++want_miss;
            bool keep = false;
            for (int j = 0; j < n && !keep; ++j) keep = ids[j] == e && weight_rank(w, j, kTop) < R;
            if (!keep) { want[i] = -1; wantw[i] = 0.0f; ++want_drop; }
        }
        ResidentTabs held;
        for (int q = 0; q < ntab; ++q) {
            cudaMemcpy(d_tab[q], tab[q].data(), kExperts * 4, cudaMemcpyHostToDevice);
            held.add(d_tab[q]);
        }
        cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_cnt, 0, 3 * 8);
        route_tail_skip(d_ids, d_w, held, n, kTop, R, kExperts, d_cnt, nullptr);
        std::vector<int32_t> gi(n);
        std::vector<float> gw(n);
        unsigned long long c[3] = {};
        cudaMemcpy(gi.data(), d_ids, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gw.data(), d_w, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_cnt, sizeof c, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) bad += gi[i] != want[i] || std::memcmp(&gw[i], &wantw[i], 4) != 0;
        bad += (long) c[0] != n || (long) c[1] != want_miss || (long) c[2] != want_drop;
        total += n;
        missed += want_miss;
        dropped += want_drop;
    }
    std::printf("route_tail_skip_parity: %ld of %ld entries dropped (%ld held by no GPU), %ld mismatches\n", dropped, total,
                missed, bad);
    if (bench) {
        ResidentTabs held;
        for (auto* t : d_tab) held.add(t);
        constexpr int kTop = 16;
        std::vector<int32_t> ids(kMaxT * kTop);
        std::vector<float> w(kMaxT * kTop, 0.1f);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) (rng() % kMaxExperts);
        cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t s;
        cudaStreamCreate(&s);
        constexpr int kRuns = 2000;
        for (int i = 0; i < 100; ++i) route_tail_skip(d_ids, d_w, held, kMaxT * kTop, kTop, 7, kMaxExperts, d_cnt, s);
        cudaStreamSynchronize(s);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int i = 0; i < kRuns; ++i) route_tail_skip(d_ids, d_w, held, kMaxT * kTop, kTop, 7, kMaxExperts, d_cnt, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("route_tail_skip_parity: bench %d-token window, top 16, 4 tables: %.2f us per launch\n", kMaxT, 1000.0 * ms / kRuns);
    }
    return bad == 0 && dropped > 0 ? 0 : 1;
}
