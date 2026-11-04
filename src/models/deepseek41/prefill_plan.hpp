// src/models/deepseek41/prefill_plan.hpp - how big a prompt chunk the free device memory can carry, and when the
// compressed caches give memory back. Free of CUDA so a CPU test can drive it.
#pragma once

#include "models/deepseek41/context_policy.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41::prefillp {

constexpr int kMaxChunk = 16384;
constexpr int kHomeChunk = 8192;                   // the chunk a prompt keeps at the full context beside a tier's homes
constexpr int kMinChunk = 256;
constexpr uint64_t kMargin = 1ull << 30;           // left free after the workspaces (kernel modules load lazily)
constexpr int64_t kKeepPositions = 32768;          // a new sequence keeps the caches of this many positions
constexpr uint64_t kTrimBytes = 256ull << 20;      // ... and gives back the rest only when it is at least this much
constexpr int64_t kStageRows = 16384;              // (token, slot) rows one expert launch handles
constexpr int kStageExperts = 32;                  // experts staged per launch, two launches in flight
constexpr int kAheadSlots = 6;                     // staged groups of experts in flight in a long chunk
constexpr int64_t kAheadPairs = 4096;              // (token, slot) pairs from which a chunk stages its groups ahead
constexpr int kShortPrompt = 640;                  // a prompt this short takes no staged groups and reads beside a tier's copies
constexpr int kShortChunk = 1024;                  // ... in a chunk of this many tokens

/// The model sizes the prompt workspaces depend on.
struct Dims {
    int64_t n_embd = 0, n_used = 0, n_expert = 0, n_ff_exp = 0;
    int64_t hc_coef = 24, hc_splits = 8, engram_cols = 0;
    uint64_t expert_blob_bytes = 0;   // the largest expert
};

/// Device bytes that grow with the chunk: streams and scratch rows, router outputs, the expert inputs.
inline uint64_t per_token_bytes(const Dims& d) {
    const uint64_t n = (uint64_t) d.n_embd;
    return 4 * (19 * n + (uint64_t) d.hc_coef * 3 + (uint64_t) d.n_expert + 16 + (uint64_t) d.engram_cols +
                (uint64_t) d.n_used * n + (uint64_t) d.hc_splits * (uint64_t) (d.hc_coef + 1));
}

/// Device bytes that do not: the expert launch buffers and the staged blobs.
inline uint64_t fixed_bytes(const Dims& d) {
    const uint64_t R = (uint64_t) kStageRows;
    const uint64_t q8 = (uint64_t) d.n_embd * R * 36 / 32 + (uint64_t) d.n_ff_exp * R * 36 / 32;
    return q8 + R * (uint64_t) d.n_ff_exp * 12 + R * 4 + 2 * (d.expert_blob_bytes * kStageExperts + R * 8);
}

inline uint64_t workspace_bytes(const Dims& d, int chunk) { return fixed_bytes(d) + per_token_bytes(d) * (uint64_t) chunk; }

/// Device bytes of the long chunks' staged groups of experts, ahead of their products.
inline uint64_t ahead_bytes(const Dims& d) { return kAheadSlots * d.expert_blob_bytes * kStageExperts; }

struct Fit {
    int chunk = 0;       ///< tokens per chunk; 0 when even the least does not fit
    std::string error;
};

/// What a chunk of `chunk` tokens takes on the device for a prompt of `tokens`: its workspaces and, in a long one, the staged groups.
inline uint64_t chunk_bytes(const Dims& d, int chunk, int tokens) {
    const int64_t pairs = (int64_t) (chunk < tokens ? chunk : tokens) * d.n_used;
    return workspace_bytes(d, chunk) + (pairs >= kAheadPairs ? ahead_bytes(d) : 0);
}

/// The largest chunk a prompt of `tokens` can use (halving from `max_chunk`, never below `min_chunk`) that leaves `margin` free.
inline Fit plan_chunk(const Dims& d, uint64_t free_bytes, uint64_t margin, int tokens, int max_chunk = kMaxChunk, int min_chunk = kMinChunk) {
    Fit f;
    for (int c = max_chunk; c >= min_chunk; c /= 2) {
        if (c / 2 >= tokens && c / 2 >= min_chunk) continue;   // half of it holds the prompt
        if (chunk_bytes(d, c, tokens) + margin <= free_bytes) {
            f.chunk = c;
            return f;
        }
    }
    f.error = "not enough free device memory to read this prompt: " + std::to_string(free_bytes >> 20) +
              " MiB free, the prompt workspaces need " + std::to_string((chunk_bytes(d, min_chunk, tokens) + margin) >> 20) +
              " MiB at the smallest chunk (a smaller --max-context or a smaller expert tier leaves more)";
    return f;
}

/// Positions the caches keep when a new sequence starts with `backed` positions mapped: all of them when giving the
/// rest back would free little, else `kKeepPositions`.
inline int64_t keep_positions(int64_t backed, const std::vector<int>& ratios) {
    if (backed <= kKeepPositions) return backed;
    return ctxp::cache_bytes(backed, ratios) - ctxp::cache_bytes(kKeepPositions, ratios) >= kTrimBytes ? kKeepPositions : backed;
}

}  // namespace basalt::models::deepseek41::prefillp
