// tests/kernels/route_prior_parity.cpp - the residency-biased router against the native router and a CPU rule.
//
//     build/route_prior_parity [--bench]
//
// Random logits for windows of 1..kVerifyMaxT tokens (512 experts, top 10), random residency on 1..4 tables.
// lambda 0 must equal native_router_top10_multi bit for bit. For lambda > 0 the picked experts must be the ten
// highest probability * exp(lambda) (held experts) in a double-precision reference, in that order, and the weights the
// unbiased probabilities of the picks renormalised over the ten (to 1e-5); windows whose tenth and eleventh scores
// are closer than 1e-4 are skipped as ties. The counters must match the reference.
//
// --bench: one launch at the largest window, timed over many launches, beside the native router.
#include <cstddef>

#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) {
        cudaGetLastError();
        std::printf("route_prior_parity: no CUDA device, skipped\n");
        return 77;
    }
    constexpr int NE = 512, K = 10, MAXT = kVerifyMaxT;
    std::mt19937 rng(20261030);
    cudaStream_t s;
    cudaStreamCreate(&s);
    float *d_l, *d_w, *d_w0;
    int32_t *d_ids, *d_ids0, *d_tab[ResidentTabs::kMax];
    unsigned long long* d_cnt;
    cudaMalloc(&d_l, MAXT * NE * 4);
    cudaMalloc(&d_w, MAXT * K * 4);
    cudaMalloc(&d_w0, MAXT * K * 4);
    cudaMalloc(&d_ids, MAXT * K * 4);
    cudaMalloc(&d_ids0, MAXT * K * 4);
    cudaMalloc(&d_cnt, 8 * 8);
    for (auto& t : d_tab) cudaMalloc(&t, NE * 4);
    long bad = 0, windows = 0, skipped = 0, swapped = 0, entries = 0;
    for (int it = 0; it < 3000; ++it) {
        const int nt = 1 + (int) (rng() % MAXT), ntab = 1 + (int) (rng() % ResidentTabs::kMax);
        const float lambda = it % 5 == 0 ? 0.0f : 0.25f + (float) (rng() % 400) / 100.0f;
        std::vector<float> lg((size_t) nt * NE);
        std::normal_distribution<float> nrm(0.0f, 2.0f);
        for (auto& x : lg) x = nrm(rng);
        std::vector<std::vector<int32_t>> tab(ntab, std::vector<int32_t>(NE));
        for (auto& t : tab)
            for (int e = 0; e < NE; ++e) t[e] = rng() % 3 == 0 ? e : -1;
        ResidentTabs held;
        for (int q = 0; q < ntab; ++q) {
            cudaMemcpy(d_tab[q], tab[q].data(), NE * 4, cudaMemcpyHostToDevice);
            held.add(d_tab[q]);
        }
        cudaMemcpy(d_l, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_cnt, 0, 8 * 8);
        route_prior_top10(d_l, held, lambda, d_ids, d_w, nt, d_cnt, s);
        native_router_top10_multi(d_l, d_ids0, d_w0, nt, s);
        cudaStreamSynchronize(s);
        std::vector<int32_t> gi(nt * K), gi0(nt * K);
        std::vector<float> gw(nt * K), gw0(nt * K);
        unsigned long long c[8] = {};
        cudaMemcpy(gi.data(), d_ids, nt * K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gw.data(), d_w, nt * K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gi0.data(), d_ids0, nt * K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gw0.data(), d_w0, nt * K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_cnt, sizeof c, cudaMemcpyDeviceToHost);
        ++windows;
        if (lambda == 0.0f) {
            bad += std::memcmp(gi.data(), gi0.data(), nt * K * 4) != 0 || std::memcmp(gw.data(), gw0.data(), nt * K * 4) != 0;
            continue;
        }
        long want_swap = 0, want_out = 0;
        bool tie = false;
        for (int t = 0; t < nt && !tie; ++t) {
            const float* l = lg.data() + (size_t) t * NE;
            double mx = *std::max_element(l, l + NE), sum = 0;
            std::vector<double> p(NE), sc(NE);
            for (int e = 0; e < NE; ++e) { p[e] = std::exp((double) l[e] - mx); sum += p[e]; }
            std::vector<int> ord(NE), uord(NE);
            for (int e = 0; e < NE; ++e) {
                p[e] /= sum;
                bool in = false;
                for (const auto& tb : tab) in |= tb[e] >= 0;
                sc[e] = p[e] * (in ? std::exp((double) lambda) : 1.0);
                ord[e] = uord[e] = e;
            }
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sc[a] > sc[b]; });
            std::stable_sort(uord.begin(), uord.end(), [&](int a, int b) { return p[a] > p[b]; });
            if (sc[ord[K - 1]] - sc[ord[K]] < 1e-4 * sc[ord[K - 1]] || p[uord[K - 1]] - p[uord[K]] < 1e-4 * p[uord[K - 1]]) tie = true;
            double ws = 0;
            for (int j = 0; j < K; ++j) ws += p[ord[j]];
            for (int j = 0; j < K && !tie; ++j) {
                bad += gi[t * K + j] != ord[j];
                bad += std::fabs(gw[t * K + j] - (float) (p[ord[j]] / ws)) > 1e-5f;
                want_swap += p[ord[j]] < p[uord[K - 1]];
                bool in = false;
                for (const auto& tb : tab) in |= tb[ord[j]] >= 0;
                want_out += !in;
            }
        }
        if (tie) { ++skipped; continue; }
        swapped += want_swap;
        entries += nt * K;
        bad += (long) c[3] != nt * K || (long) c[4] != want_swap || (long) c[5] != want_out;
    }
    std::printf("route_prior_parity: %ld windows (%ld skipped as ties), %ld of %ld entries picked by the bias, %ld mismatches\n",
                windows, skipped, swapped, entries, bad);
    if (bench) {
        ResidentTabs held;
        for (auto* t : d_tab) held.add(t);
        std::vector<float> lg((size_t) MAXT * NE);
        std::normal_distribution<float> nrm(0.0f, 2.0f);
        for (auto& x : lg) x = nrm(rng);
        cudaMemcpy(d_l, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice);
        constexpr int kRuns = 2000;
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < 100; ++i) pass ? route_prior_top10(d_l, held, 1.5f, d_ids, d_w, MAXT, d_cnt, s) : native_router_top10_multi(d_l, d_ids, d_w, MAXT, s);
            cudaStreamSynchronize(s);
            cudaEventRecord(a, s);
            for (int i = 0; i < kRuns; ++i) pass ? route_prior_top10(d_l, held, 1.5f, d_ids, d_w, MAXT, d_cnt, s) : native_router_top10_multi(d_l, d_ids, d_w, MAXT, s);
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("route_prior_parity: bench %d tokens, %s: %.2f us per launch\n", MAXT, pass ? "route_prior_top10 (4 tables, counters)" : "native_router_top10_multi", 1000.0 * ms / kRuns);
        }
    }
    return bad == 0 && swapped > 0 ? 0 : 1;
}
