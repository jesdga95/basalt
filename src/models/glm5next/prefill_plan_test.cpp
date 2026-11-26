// src/models/glm5next/prefill_plan_test.cpp - the prefill plan's arithmetic (no GPU).
//
// The one property the session's refusal depends on: a chunk the plan recommends actually fits in the memory it says
// it fits in. The rest are the shape of the answer - monotone in the free memory and in the chunk, clamped at both
// ends, and dominated by the term this architecture has and a dense model does not: the indexer's score row, which is
// one row per pool the context can fill, so it grows with the context rather than the chunk.
#include "models/glm5next/prefill_plan.hpp"

#include <cstdio>

using namespace basalt::models::glm5next;
using namespace basalt::models::glm5next::prefillp;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

/// The tiny fixture's geometry, as the pack's meta describes it.
Geometry tiny() {
    Geometry g;
    g.n_layers = 4; g.nextn = 1; g.leading_dense = 1;
    g.n_embd = 128; g.n_vocab = 3; g.n_head = 8;
    g.qk_dim = 16; g.v_dim = 16; g.q_lora = 16; g.kv_lora = 512;
    g.kda_head_dim = 16; g.conv = 4;
    g.idx_heads = 2; g.idx_dim = 16; g.idx_top_k = 2; g.kpool = 2;
    g.n_expert = 32; g.n_used = 2; g.n_shared = 1; g.n_ff_exp = 16; g.n_ff = 32;
    return g;
}

/// The real GLM 5.3 Flash geometry (46 blocks, 4096 wide, 64 KDA heads of 128, a 2048-pool indexer over pools of 4,
/// 288 experts with top-8 plus one shared), for the sizes the plan is actually meant to answer.
Geometry real() {
    Geometry g = tiny();
    g.n_layers = 46; g.nextn = 1; g.leading_dense = 3;
    g.n_embd = 4096; g.n_vocab = 154880; g.n_head = 64;
    g.qk_dim = 512; g.v_dim = 512; g.q_lora = 1536; g.kv_lora = 512;
    g.kda_head_dim = 128; g.conv = 4;
    g.idx_heads = 32; g.idx_dim = 128; g.idx_top_k = 2048; g.kpool = 4;
    g.n_expert = 288; g.n_used = 8; g.n_shared = 1; g.n_ff_exp = 2048; g.n_ff = 12288;
    return g;
}

}  // namespace

int main() {
    const Geometry g = tiny();
    const uint64_t ctx = 1024;

    // the cost of a window rises with its length, and a longer context costs more at the same length because the
    // indexer scores every pool the sequence can fill
    check(window_units(g, ctx, 1) < window_units(g, ctx, 2), "a longer window costs more");
    check(window_units(g, ctx, 4) < window_units(g, ctx, 8), "and keeps rising");
    check(window_units(g, 8 * ctx, 1) > window_units(g, ctx, 1), "a longer context costs more at one token");
    const uint64_t per_ctx = window_units(g, 8 * ctx, 1) - window_units(g, ctx, 1);
    check(window_units(g, 8 * ctx, 8) - window_units(g, ctx, 8) > per_ctx,
          "the context term is paid per token, not once per window");
    check(window_bytes(g, ctx, 1) == 4ull * window_units(g, ctx, 1), "the byte count is the unit count");

    // a recommended chunk fits, by the plan's own arithmetic, over a range of free memory
    bool all_fit = true, some = false;
    for (uint64_t free_mb = 1536; free_mb <= 1ull << 14; free_mb *= 2) {
        const uint64_t free = free_mb << 20;
        const int cap = max_chunk(g, ctx, free);
        if (cap <= 0) continue;
        some = true;
        if (window_bytes(g, ctx, cap) > free - kMargin) all_fit = false;
        if (cap < kMaxChunk && cap < (int) ctx && max_chunk(g, ctx, free * 2) < cap) all_fit = false;
    }
    check(some, "a chunk is recommended for a card's free memory");
    check(all_fit, "what the plan recommends fits in the memory it says it fits in");

    // both ends of the answer are clamped, and nothing at all is promised when there is no room
    check(max_chunk(g, ctx, kMargin) == 0, "no chunk is promised with nothing to spare");
    // a window cannot be longer than the context it lives in, so the cap only shows above that
    check(max_chunk(g, 4 * (uint64_t) kMaxChunk, 1ull << 40) == kMaxChunk, "the longest chunk is capped");
    check(max_chunk(g, 64, 1ull << 40) == 64, "and capped again by the context");
    check(max_chunk(g, ctx, 2ull << 30) >= 1, "a couple of GiB carries a chunk");
    // the release pack's situation: a card whose free memory is around a gigabyte carries a window once kMargin is
    // slack for what the plan does not count rather than a reservation standing in for the sparse attention's
    // workspace - which a caller now prices per token and hands in. A megabyte a token is that workspace's order.
    check(max_chunk(g, ctx, 1ull << 30) >= 1, "a gigabyte to spare carries a chunk");
    check(max_chunk(g, ctx, 2ull << 30, 0) == max_chunk(g, ctx, 2ull << 30), "and nothing changes when there is none");

    // the release pack's own situation, at the context its first run asked for: a gigabyte free and the sparse
    // attention's workspace priced per token still leaves a window. It refused every prompt before, at any length.
    // The same free memory with a shorter answer is the point of pricing the extra: at the tiny geometry both ends
    // clamp to the context, so this is shown where the numbers are real. A megabyte a token is the workspace's order.
    const Geometry r = real();
    check(max_chunk(r, 512, 1ull << 30, 1ull << 18) >= 1, "a gigabyte free carries a window at the real widths");
    check(max_chunk(r, 8 * 1024, 1ull << 30, 1ull << 18) < max_chunk(r, 8 * 1024, 1ull << 30),
          "pricing a per-token cost the plan cannot see shortens the answer");

    // the sizes the plan exists to answer: the real geometry, at the contexts the model is asked for
    check(window_bytes(r, 4096, 512) > window_bytes(r, 4096, 256), "the real geometry scales with the chunk");
    const int at_128k = max_chunk(r, 128 * 1024, 8ull << 30);
    check(at_128k >= 1, "the real geometry carries a chunk in 8 GiB at a 128k context");
    check(at_128k <= max_chunk(r, 8 * 1024, 8ull << 30),
          "a longer context asks for a shorter chunk: the indexer's score rows grow with it");
    // the release pack's own situation, at the context its first run asked for: a gigabyte free and the sparse
    // attention's workspace priced per token still leaves a window. It refused every prompt before, at any length.
    check(max_chunk(r, 512, 1ull << 30, 1ull << 18) >= 1, "a gigabyte free carries a window at the real widths");

    if (g_failures != 0) {
        std::printf("PREFILL PLAN: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("PREFILL PLAN: ok\n");
    return 0;
}
