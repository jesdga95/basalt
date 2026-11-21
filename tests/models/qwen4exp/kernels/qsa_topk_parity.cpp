// tests/models/qwen4exp/kernels/qsa_topk_parity.cpp - the QSA top-k the engine dispatches (qsa_block_topk) against the original kernel
// (qsa_block_topk_ref): the selected cells must be identical (GPU, synthetic, no model).
//
// A case is a context under a capacity (the engine's --max-context; max_blocks = capacity / 4 + 2, and without an
// active-block count the dispatch follows it), with consecutive queries at the context's end so that the tail block
// holds 0 to 3 cells.  Three score sets per case: continuous, 64 levels (many ties at the threshold) and all equal
// (the rule's ties go to the lowest index).  The dispatch is called as a decode window calls it (no block count) and as
// the prompt path does (with the call's largest block count).  The reference and the dispatched kernel are also timed.
//
// Usage: qsa_topk_parity --selftest
//        qsa_topk_parity CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] [COUNT=1]   (COUNT=0: no block count)
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

// Independent CPU selection: rank blocks by score, prefer the lower ID on ties, then emit cells in index order.
std::vector<int32_t> cpu_cells(const float* scores, const int32_t* step, int64_t block_size) {
    const int32_t n_cells = step[k::kStepNKv], width = step[k::kStepWidth];
    const int32_t n_blocks = (int32_t) ((n_cells + block_size - 1) / block_size);
    std::vector<int32_t> blocks((size_t) n_blocks);
    for (int32_t i = 0; i < n_blocks; ++i) blocks[(size_t) i] = i;
    if (n_cells > width) {
        std::sort(blocks.begin(), blocks.end(), [&](int32_t a, int32_t b) {
            if (scores[a] == scores[b]) return a < b;
            return scores[a] > scores[b];
        });
    }
    std::vector<int32_t> selected;
    selected.reserve((size_t) width);
    for (const int32_t block : blocks) {
        const int64_t first = block * block_size;
        for (int64_t at = first; at < std::min<int64_t>(n_cells, first + block_size) && selected.size() < (size_t) width; ++at)
            selected.push_back((int32_t) at);
        if (selected.size() == (size_t) width) break;
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

bool run_case(int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t) (ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t) (nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
        // the tail block as the scorers leave it: far above the others when it holds cells
        if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t) (i * max_blocks + b)] = row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f;   // ~64 levels
    }
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMalloc(&d_scores, scores.size() * 4), "malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    const int64_t active = counted ? steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    auto same = [&](const std::vector<float>& sc) -> int64_t {
        ck(cudaMemcpy(d_scores, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemset(ids_ref, 0xff, a.size() * 4), "memset");
        ck(cudaMemset(ids_new, 0xff, a.size() * 4), "memset");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr);
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaMemcpy(a.data(), ids_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(b.data(), ids_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* step = steps.data() + i * k::kStepCount;
            const int64_t w = step[k::kStepWidth];
            n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
            if (i == 0 || i == nq / 2 || i == nq - 1) {
                const auto expected = cpu_cells(sc.data() + i * max_blocks, step, s.idx_block);
                if (expected.size() != (size_t) w || !std::equal(expected.begin(), expected.end(), b.begin() + i * cap)) {
                    std::fprintf(stderr, "CPU Top-k mismatch: context %lld, query %lld\n", (long long) ctx, (long long) i);
                    return -1;
                }
            }
        }
        const int64_t middle = nq / 2;
        const int32_t* middle_step = steps.data() + middle * k::kStepCount;
        const int64_t width = middle_step[k::kStepWidth];
        const int64_t one_active = counted ? middle_step[k::kStepNBid] + 1 : -1;
        k::qsa_block_topk(d_scores + middle * max_blocks, d_steps + middle * k::kStepCount,
                          1, max_blocks, cap, s, ids_new, nullptr, one_active);
        std::vector<int32_t> single((size_t) width);
        ck(cudaMemcpy(single.data(), ids_new, single.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download single");
        if (!std::equal(single.begin(), single.end(), b.begin() + middle * cap)) {
            std::fprintf(stderr, "Top-k changed with batch width or position: context %lld\n", (long long) ctx);
            return -1;
        }
        return n;
    };
    const int64_t same_tied = same(tied), same_equal = same(equal), same_cont = same(scores);   // d_scores: continuous
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto f) {
        ck(cudaEventRecord(e0), "record");
        for (int r = 0; r < reps; ++r) f();
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms / (float) reps;
    };
    const float t_ref = timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr); });
    const float t_new = timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active); });
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_steps); cudaFree(d_scores); cudaFree(ids_ref); cudaFree(ids_new);
    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld\n", ok ? "PASS" : "FAIL", (long long) ctx, (long long) capacity, (long long) max_blocks,
                (long long) nq, counted ? "with their block count" : "without a block count", t_ref, t_new, t_ref / t_new,
                (long long) same_cont, (long long) nq, (long long) same_tied,
                (long long) nq, (long long) same_equal, (long long) nq);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        // {context, capacity}: both sides of the register kernel's reach (33,792 blocks = 135,168 cells), short
        // contexts under a long capacity (fewer blocks than the kernel's threads; a prompt batch below the dispatch's
        // 4,608 blocks), and the long contexts themselves
        const int64_t cases[][2] = {{9000, 9000},    {135160, 135160}, {135176, 135176}, {2100, 262144},
                                    {9000, 262144},  {32768, 262144},  {262144, 262144}, {300001, 524288},
                                    {524288, 524288}};
        bool ok = true;
        for (const auto& c : cases) {
            ok = run_case(c[0], 8, 3, c[1], false) && ok;     // a decode window's call
            ok = run_case(c[0], 64, 3, c[1], true) && ok;     // the prompt path's
        }
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
