// tests/kernels/qsa_scores_parity.cpp - the decode indexer (block scores, then the cluster top-k): the scores bitwise against
// the per-block reference kernel and within double-precision error of a CPU reference, at the real indexer geometry (tails
// with and without cells, windows of 1..8 rows), for the single-sequence and the segment entry points; the selection
// identical to the reference top-k on continuous, tied and equal scores; then a microbench line per kernel with the keys
// warm in L2 and flushed from it.
//   qsa_scores_parity [iters]
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
const k::QsaShapes S_ = k::qsa_real_shapes();
const int IQ = (int) S_.idx_n_head, ID = (int) S_.idx_dim, R = (int) S_.idx_block;
int g_fail = 0;
const int64_t MAXB = 16384;   // 65,536 cells

double time_us(cudaStream_t st, int iters, char* flush, size_t flush_n, const std::function<void()>& call) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double tot = 0;
    for (int it = 0; it < iters; ++it) {
        if (flush) ck(cudaMemsetAsync(flush, it, flush_n, st), "flush");
        cudaEventRecord(e0, st);
        call();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        tot += ms;
    }
    return tot / iters * 1000.0;
}
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> pooled((size_t) MAXB * ID), dead(ID);
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    float *d_pooled = dalloc<float>(pooled.size()), *d_dead = dalloc<float>(dead.size());
    up(d_pooled, pooled);
    up(d_dead, dead);
    float* d_q = dalloc<float>((size_t) 16 * IQ * ID);
    int32_t* d_steps = dalloc<int32_t>(16 * k::kStepCount);
    float *d_a = dalloc<float>((size_t) 16 * MAXB), *d_b = dalloc<float>((size_t) 16 * MAXB);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    struct Case { const char* label; int nq; std::vector<int> nbid; std::vector<int> nkv; };
    const std::vector<Case> cases = {
        {"1 row, 12.5k blocks, tail", 1, {12500}, {50003}},
        {"3 rows, tail and none", 3, {12500, 12501, 12501}, {50003, 50004, 50006}},
        {"8 rows, mixed lengths", 8, {16000, 100, 31, 32, 33, 8000, 12500, 1}, {64001, 401, 126, 129, 134, 32002, 50000, 4}},
        {"2 rows, n_bid 0 and 1", 2, {0, 1}, {1, 5}},
        {"5 rows, 63..65 blocks", 5, {63, 64, 65, 95, 96}, {253, 257, 261, 383, 386}},
    };
    for (const Case& c : cases) {
        std::vector<float> q((size_t) c.nq * IQ * ID);
        for (auto& x : q) x = nd(rng);
        std::vector<int32_t> steps((size_t) c.nq * k::kStepCount, 0);
        for (int i = 0; i < c.nq; ++i) {
            steps[i * k::kStepCount + k::kStepNKv] = c.nkv[i];
            steps[i * k::kStepCount + k::kStepNBid] = c.nbid[i];
            steps[i * k::kStepCount + k::kStepWidth] = 1;   // not the identity: scores are written
        }
        up(d_q, q);
        up(d_steps, steps);
        const int64_t active = *std::max_element(c.nbid.begin(), c.nbid.end()) + 1;
        ck(cudaMemset(d_a, 0, (size_t) c.nq * MAXB * 4), "z");
        ck(cudaMemset(d_b, 0, (size_t) c.nq * MAXB * 4), "z");
        k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, c.nq, MAXB, S_, d_a, st, -1, true);        // the decode entry
        k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, c.nq, MAXB, S_, d_b, st, active, false);   // the reference kernel
        ck(cudaStreamSynchronize(st), "sync");
        const auto got = down(d_a, (size_t) c.nq * MAXB), ref = down(d_b, (size_t) c.nq * MAXB);
        const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
        // CPU reference in double over the blocks each query reaches
        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < c.nq; ++i)
            for (int b = 0; b <= c.nbid[i]; ++b) {
                const float* key = b == c.nbid[i] ? dead.data() : pooled.data() + (size_t) b * ID;
                double sc = 0;
                for (int h = 0; h < IQ; ++h) {
                    double d = 0;
                    for (int j = 0; j < ID; ++j) d += (double) key[j] * q[((size_t) i * IQ + h) * ID + j];
                    sc += d > 0 ? d : 0;
                }
                if (b == c.nbid[i] && c.nkv[i] % R != 0) sc += 1e9;
                const double g = got[(size_t) i * MAXB + b];
                max_abs = std::max(max_abs, std::fabs(g - sc));
                max_rel = std::max(max_rel, std::fabs(g - sc) / std::max(1.0, std::fabs(sc)));
            }
        const bool ok = same && max_rel < 1e-5;
        g_fail += !ok;
        std::printf("%-30s decode vs reference: %s; vs CPU double: max |d| %.3g, max rel %.3g  %s\n", c.label,
                    same ? "bitwise" : "DIFFERS", max_abs, max_rel, ok ? "ok" : "FAIL");
    }

    {   // the segment entry: rows on two slots with different keys and lengths, against the reference kernel per slot
        const int nq = 6;
        std::vector<float> pooled2((size_t) MAXB * ID), dead2(ID);
        for (auto& x : pooled2) x = nd(rng);
        for (auto& x : dead2) x = nd(rng);
        float *d_p2 = dalloc<float>(pooled2.size()), *d_dd2 = dalloc<float>(dead2.size());
        up(d_p2, pooled2);
        up(d_dd2, dead2);
        k::QsaSegSlot slots[2];
        slots[0].pooled = d_pooled; slots[0].dead = d_dead;
        slots[1].pooled = d_p2; slots[1].dead = d_dd2;
        k::QsaSegSlot* d_slots = dalloc<k::QsaSegSlot>(2);
        ck(cudaMemcpy(d_slots, slots, sizeof slots, cudaMemcpyHostToDevice), "slots");
        std::vector<k::QsaSeg> rows(16);
        for (int i = 0; i < 16; ++i) { rows[i].slot = i < 3 ? 0 : 1; rows[i].first = 0; rows[i].len = 16; }
        k::QsaSeg* d_rows = dalloc<k::QsaSeg>(16);
        up(d_rows, rows);
        k::QsaSegTable tab;
        tab.slots = d_slots; tab.rows = d_rows; tab.segs = d_rows; tab.n_rows = 16; tab.n_seg = 2;
        const int nkv[nq] = {50003, 50004, 50005, 31001, 31002, 31003};
        std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
        for (int i = 0; i < nq; ++i) {
            steps[i * k::kStepCount + k::kStepNKv] = nkv[i];
            steps[i * k::kStepCount + k::kStepNBid] = nkv[i] / R;
            steps[i * k::kStepCount + k::kStepWidth] = 1;
        }
        std::vector<float> qq((size_t) nq * IQ * ID);
        for (auto& x : qq) x = nd(rng);
        up(d_q, qq);
        up(d_steps, steps);
        ck(cudaMemset(d_a, 0, (size_t) nq * MAXB * 4), "z");
        k::qsa_block_scores_seg(tab, d_q, d_steps, nq, MAXB, S_, d_a, st, true);
        ck(cudaStreamSynchronize(st), "sync");
        const auto got = down(d_a, (size_t) nq * MAXB);
        bool same = true;
        for (int i = 0; i < nq; ++i) {   // one reference call per row: its slot's keys, its own count
            ck(cudaMemset(d_b, 0, (size_t) MAXB * 4), "z");
            k::qsa_block_scores(slots[rows[i].slot].pooled, slots[rows[i].slot].dead, d_q + (size_t) i * IQ * ID,
                                d_steps + i * k::kStepCount, 1, MAXB, S_, d_b, st, nkv[i] / R + 1, false);
            ck(cudaStreamSynchronize(st), "sync");
            const auto ref = down(d_b, (size_t) MAXB);
            same = same && std::memcmp(got.data() + (size_t) i * MAXB, ref.data(), (size_t) MAXB * 4) == 0;
        }
        g_fail += !same;
        std::printf("%-30s segment vs reference: %s  %s\n", "6 rows on 2 slots", same ? "bitwise" : "DIFFERS", same ? "ok" : "FAIL");
        {
            const size_t flush_n = 256u << 20;
            char* flush = dalloc<char>(flush_n);
            for (int cold = 0; cold < 2; ++cold) {
                const double us = time_us(st, iters, cold ? flush : nullptr, flush_n,
                                          [&] { k::qsa_block_scores_seg(tab, d_q, d_steps, nq, MAXB, S_, d_a, st, true); });
                std::printf("bench segment scores keys %-5s: %7.2f us/call\n", cold ? "cold" : "warm", us);
            }
        }
    }

    {   // the cluster top-k against the reference top-k
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, S_);
        int64_t same_n = 0, total = 0;
        float* d_sc = dalloc<float>((size_t) 16 * MAXB);
        int32_t* d_st = dalloc<int32_t>(16 * k::kStepCount);
        int32_t *d_ids = dalloc<int32_t>((size_t) 16 * cap), *d_ref = dalloc<int32_t>((size_t) 16 * cap);
        for (int64_t ctx : {2100, 2500, 9001, 33000, 50003, 65002})
            for (int nq : {1, 3, 8, 16})
                for (int pat = 0; pat < 3; ++pat) {
                    std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
                    std::vector<float> sc((size_t) nq * MAXB);
                    for (int i = 0; i < nq; ++i) {
                        const int64_t n_kv = ctx - i * 3;
                        steps[i * k::kStepCount + k::kStepNKv] = (int) n_kv;
                        steps[i * k::kStepCount + k::kStepNBid] = (int) (n_kv / R);
                        steps[i * k::kStepCount + k::kStepWidth] = (int) k::qsa_selection_width(n_kv, S_);
                        for (int64_t b = 0; b < MAXB; ++b) {
                            float v = std::fabs(nd(rng)) * 40.0f + 3.0f;
                            if (pat == 1) v = std::floor(v * 0.25f) * 4.0f;   // many ties at the threshold
                            if (pat == 2) v = 17.0f;                          // all equal
                            sc[(size_t) i * MAXB + b] = v;
                        }
                        if (n_kv % R != 0) sc[(size_t) i * MAXB + n_kv / R] += 1e9f;
                    }
                    up(d_sc, sc);
                    up(d_st, steps);
                    ck(cudaMemset(d_ids, 0xff, (size_t) nq * cap * 4), "m");
                    ck(cudaMemset(d_ref, 0xff, (size_t) nq * cap * 4), "m");
                    k::qsa_block_topk(d_sc, d_st, nq, MAXB, cap, S_, d_ids, st);
                    k::qsa_block_topk_ref(d_sc, d_st, nq, MAXB, cap, S_, d_ref, st);
                    ck(cudaStreamSynchronize(st), "sync");
                    const auto h = down(d_ids, (size_t) nq * cap), r = down(d_ref, (size_t) nq * cap);
                    for (int i = 0; i < nq; ++i) {
                        const int w = steps[i * k::kStepCount + k::kStepWidth];
                        ++total;
                        same_n += std::equal(h.begin() + (size_t) i * cap, h.begin() + (size_t) i * cap + w, r.begin() + (size_t) i * cap);
                    }
                }
        g_fail += same_n != total;
        std::printf("top-k vs reference: identical selections %lld / %lld (6 contexts x 4 widths x continuous, tied, equal)  %s\n",
                    (long long) same_n, (long long) total, same_n == total ? "ok" : "FAIL");
    }

    {   // microbench: 3 rows at 50k context, keys warm and flushed
        std::vector<float> q((size_t) 3 * IQ * ID);
        for (auto& x : q) x = nd(rng);
        std::vector<int32_t> steps(3 * k::kStepCount, 0);
        for (int i = 0; i < 3; ++i) {
            steps[i * k::kStepCount + k::kStepNKv] = 50003 + i;
            steps[i * k::kStepCount + k::kStepNBid] = (50003 + i) / R;
            steps[i * k::kStepCount + k::kStepWidth] = (int) k::qsa_selection_width(50003 + i, S_);
        }
        up(d_q, q);
        up(d_steps, steps);
        const size_t flush_n = 256u << 20;
        char* flush = dalloc<char>(flush_n);
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, S_);
        int32_t* d_ids = dalloc<int32_t>((size_t) 3 * cap);
        const double bytes = 12501.0 * ID * 4;
        for (int cold = 0; cold < 2; ++cold) {
            const double us = time_us(st, iters, cold ? flush : nullptr, flush_n,
                                      [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, 3, MAXB, S_, d_a, st, -1, true); });
            std::printf("bench scores keys %-5s: %7.2f us/call, keys %.1f MB = %.0f GB/s\n", cold ? "cold" : "warm", us, bytes / 1e6, bytes / us / 1e3);
        }
        const double empty = [&] {
            std::vector<int32_t> id(3 * k::kStepCount, 0);
            for (int i = 0; i < 3; ++i) { id[i * k::kStepCount + k::kStepNKv] = 10; id[i * k::kStepCount + k::kStepWidth] = 10; }
            up(d_steps, id);
            return time_us(st, iters, nullptr, 0, [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, 3, MAXB, S_, d_a, st, -1, true); });
        }();
        std::printf("bench floor (empty launch, event timing): %.2f us\n", empty);
        up(d_steps, steps);
        k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, 3, MAXB, S_, d_a, st, -1, true);
        for (int nq : {1, 3}) {
            const double us = time_us(st, iters, nullptr, 0, [&] { k::qsa_block_topk(d_a, d_steps, nq, MAXB, cap, S_, d_ids, st); });
            std::printf("bench top-k nq=%d: %7.2f us/call\n", nq, us);
        }
    }
    std::printf("qsa_scores_parity: %s\n", g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
