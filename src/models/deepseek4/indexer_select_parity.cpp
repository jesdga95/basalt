#include "basalt/models/deepseek4/indexer.hpp"
#include "basalt/models/deepseek4/compress.hpp"
#include "parity_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::models::deepseek4;
using ds4test::Dev;

namespace {

struct ScoredRow { double score; int32_t row; };

void reference(const std::vector<float>& q, const std::vector<float>& weights, const std::vector<float>& keys,
               int n_comp, int start_pos, int n_tokens, std::vector<int32_t>& out, std::vector<int32_t>& counts) {
    for (int t = 0; t < n_tokens; ++t) {
        const int visible = (start_pos + t + 1) / kIndexerRatio;
        const int count = std::min(kIndexerTopK, visible);
        counts[(size_t) t] = count;
        std::vector<ScoredRow> scored((size_t) visible);
        for (int row = 0; row < visible; ++row) {
            double score = 0.0;
            for (int h = 0; h < kIndexerHeads; ++h) {
                double dot = 0.0;
                for (int d = 0; d < kIndexerDim; ++d)
                    dot += (double) q[((size_t) t * kIndexerHeads + h) * kIndexerDim + d] *
                           keys[(size_t) row * kIndexerDim + d];
                score += std::max(dot, 0.0) * weights[(size_t) t * kIndexerHeads + h];
            }
            scored[(size_t) row] = {score, row};
        }
        std::sort(scored.begin(), scored.end(), [](const ScoredRow& a, const ScoredRow& b) {
            return a.score == b.score ? a.row < b.row : a.score > b.score;
        });
        for (int i = 0; i < kIndexerTopK; ++i)
            out[(size_t) t * kIndexerTopK + i] = i < count ? scored[(size_t) i].row : -1;
    }
    (void) n_comp;
}

bool run() {
    constexpr int T = 8;
    constexpr int Start = 2800;
    constexpr int N = (Start + T + kIndexerRatio - 1) / kIndexerRatio;
    auto q = ds4test::randn((size_t) T * kIndexerHeads * kIndexerDim, 0x495851u, .25f);
    auto weights = ds4test::randn((size_t) T * kIndexerHeads, 0x495857u, .01f);
    auto keys = ds4test::randn((size_t) N * kIndexerDim, 0x49584bu, .25f);
    std::vector<int32_t> expected((size_t) T * kIndexerTopK, -1), expected_n(T, 0);
    reference(q, weights, keys, N, Start, T, expected, expected_n);

    Dev<float> dq(q), dw(weights), dk(keys);
    Dev<int32_t> got((size_t) T * kIndexerTopK), got_n(T);
    const size_t workspace_bytes = indexer_select_workspace_bytes(N);
    Dev<uint8_t> workspace(workspace_bytes);
    indexer_select(dq.p, dw.p, dk.p, N, Start, T, got.p, got_n.p, workspace.p, workspace_bytes, nullptr);
    const auto ids = got.down();
    const auto counts = got_n.down();
    const bool ok = ids == expected && counts == expected_n;
    if (!ok) {
        size_t mismatch = 0;
        while (mismatch < ids.size() && ids[mismatch] == expected[mismatch]) ++mismatch;
        std::printf("DS4 indexer select: mismatch at %zu, got %d expected %d; row counts %s\n", mismatch,
                    mismatch < ids.size() ? ids[mismatch] : -1, mismatch < ids.size() ? expected[mismatch] : -1,
                    counts == expected_n ? "match" : "MISMATCH");
    } else {
        std::printf("DS4 indexer select T=%d rows=%d top-k=%d: ids and causal counts match double reference\n",
                    T, N, kIndexerTopK);
    }

    std::vector<int32_t> chunk_ids((size_t) T * kIndexerTopK, -1), chunk_counts(T, 0);
    Dev<int32_t> chunk_out((size_t) T * kIndexerTopK), chunk_n(T);
    indexer_select(dq.p, dw.p, dk.p, N, Start, 4, chunk_out.p, chunk_n.p, workspace.p, workspace_bytes, nullptr);
    auto first_ids = chunk_out.down();
    auto first_counts = chunk_n.down();
    std::copy_n(first_ids.begin(), (size_t) 4 * kIndexerTopK, chunk_ids.begin());
    std::copy_n(first_counts.begin(), 4, chunk_counts.begin());
    indexer_select(dq.p + (size_t) 4 * kIndexerHeads * kIndexerDim, dw.p + (size_t) 4 * kIndexerHeads,
                   dk.p, N, Start + 4, 4, chunk_out.p, chunk_n.p, workspace.p, workspace_bytes, nullptr);
    const auto second_ids = chunk_out.down();
    const auto second_counts = chunk_n.down();
    std::copy_n(second_ids.begin(), (size_t) 4 * kIndexerTopK,
                chunk_ids.begin() + (size_t) 4 * kIndexerTopK);
    std::copy_n(second_counts.begin(), 4, chunk_counts.begin() + 4);
    const bool chunk_ok = chunk_ids == ids && chunk_counts == counts;
    std::printf("DS4 indexer select chunking: %s\n", chunk_ok ? "match" : "MISMATCH");

    Dev<int32_t> edge_out((size_t) 4 * kIndexerTopK), edge_n(4);
    const size_t edge_workspace_bytes = indexer_select_workspace_bytes(1);
    Dev<uint8_t> edge_workspace(edge_workspace_bytes);
    indexer_select(dq.p, dw.p, dk.p, 1, 0, 4, edge_out.p, edge_n.p, edge_workspace.p, edge_workspace_bytes, nullptr);
    const auto edge_ids = edge_out.down();
    const auto edge_counts = edge_n.down();
    const bool edge_ok = edge_counts == std::vector<int32_t>({0, 0, 0, 1}) &&
                         std::all_of(edge_ids.begin(), edge_ids.begin() + 3 * kIndexerTopK,
                                     [](int32_t id) { return id == -1; }) &&
                         edge_ids[(size_t) 3 * kIndexerTopK] == 0 &&
                         std::all_of(edge_ids.begin() + 3 * kIndexerTopK + 1, edge_ids.end(),
                                     [](int32_t id) { return id == -1; });
    std::printf("DS4 indexer select empty/single-row edge: %s\n", edge_ok ? "match" : "MISMATCH");

    const double us = ds4test::bench_stream_us([&](void* stream) {
        indexer_select(dq.p, dw.p, dk.p, N, Start, T, got.p, got_n.p, workspace.p, workspace_bytes, stream);
    }, 25);
    const double bytes = (double) T * (kIndexerHeads * kIndexerDim + kIndexerHeads) * sizeof(float) +
                         (double) N * kIndexerDim * sizeof(float);
    ds4test::bench_line("DS4 indexer score + top-512", us, bytes);
    return ok && chunk_ok && edge_ok;
}

}  // namespace

int main() { return run() ? 0 : 1; }
