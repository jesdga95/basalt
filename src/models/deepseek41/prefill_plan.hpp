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
constexpr uint64_t kMargin = 1ull << 30;           // left free after a long prompt's workspaces
constexpr uint64_t kShortMargin = 1ull << 28;      // ... and after a short one's, whose memory the tier plans for
constexpr int64_t kKeepPositions = 32768;          // a new sequence keeps the caches of this many positions
constexpr uint64_t kTrimBytes = 256ull << 20;      // ... and gives back the rest only when it is at least this much
constexpr int64_t kStageRows = 16384;              // (token, slot) rows one expert launch handles
constexpr int kStageExperts = 32;                  // experts staged per launch, two launches in flight
constexpr int kRing = 5;                           // layers of file experts read ahead
constexpr int kRingPieces = kRing * 48;            // most experts the pinned ring of those reads holds
constexpr int kEmbedRows = 512;                    // token embeddings a chunk stages in host RAM at a time
constexpr int kEngramRows = 2048;                  // tokens of Engram rows a chunk dequantizes in host RAM at a time
constexpr int kAheadSlots = 6;                     // staged groups of experts in flight in a long chunk
constexpr int kShareRows = 2048;                   // rows a layer of a short read may compute on the CPU
constexpr int kStepPrompt = 12;                    // a prompt this short reads one decode step at a time, faster than a chunk's bus copies
constexpr int kShortPrompt = 2048;                 // a prompt this short takes no staged groups and reads beside a tier's copies
constexpr int kShortChunk = 1024;                  // ... in chunks of this many tokens
constexpr double kPreRouteCycles = 4.3e6;           // main card SM-cycles a token takes in a layer before its routing (measured)
constexpr int64_t kAheadPairs = 50000;              // (token, slot) pairs from which a chunk stages its groups ahead: past a short chunk

/// The first position that stops a read's rows short of the last layers. Each layer past `skip_layer` looks one window further
/// back through its own window rows, and every row reaches the compressed rows of `skip_layer` anyway.
inline int64_t far_below(int64_t stop, int n_layers, int skip_layer, int window) {
    return stop - (int64_t) (n_layers - skip_layer) * (window - 1) - window;
}
/// Rows from here on hold what a full read would at every layer's window, for any token from here on: the state is saved or
/// resumed only at a read's stop, and one window back from it at most.
inline int64_t whole_from(int64_t stop, int window) { return stop > window ? stop - window : 0; }
inline bool may_save(int64_t pos, int64_t whole) { return pos >= whole; }
/// `held`: the positions read. Forward and the start need nothing behind them.
inline bool may_resume(int64_t pos, int64_t held, int64_t whole) { return pos == 0 || pos >= held || pos >= whole; }

/// The model sizes the prompt workspaces depend on.
struct Dims {
    int64_t n_embd = 0, n_used = 0, n_expert = 0, n_ff_exp = 0;
    int64_t hc_coef = 24, hc_splits = 8, engram_cols = 0;
    uint64_t expert_blob_bytes = 0;   // the largest expert
};

/// Device bytes that grow with the chunk: streams and scratch rows, router outputs, the expert inputs, the experts' rows
/// with their row map.
inline uint64_t per_token_bytes(const Dims& d) {
    const uint64_t n = (uint64_t) d.n_embd;
    return 4 * (19 * n + (uint64_t) d.hc_coef * 3 + (uint64_t) d.n_expert + 16 + (uint64_t) d.engram_cols +
                (uint64_t) d.n_used * (n + 2) + (uint64_t) d.hc_splits * (uint64_t) (d.hc_coef + 1));
}

/// (token, slot) rows one expert launch of a chunk handles.
inline int64_t stage_rows(int64_t n_used, int chunk) {
    const int64_t rows = (int64_t) chunk * n_used, least = (int64_t) kShortChunk * n_used;
    return rows < least ? least : rows > kStageRows ? kStageRows : rows;
}

/// Device bytes that do not: the expert launch buffers and the staged blobs.
inline uint64_t fixed_bytes(const Dims& d, int chunk) {
    const uint64_t R = (uint64_t) stage_rows(d.n_used, chunk);
    const uint64_t q8 = (uint64_t) d.n_embd * R * 36 / 32 + (uint64_t) d.n_ff_exp * R * 36 / 32;
    return q8 + R * (uint64_t) d.n_ff_exp * 12 + 2 * (d.expert_blob_bytes * kStageExperts + R * 4);
}

inline uint64_t workspace_bytes(const Dims& d, int chunk) { return fixed_bytes(d, chunk) + per_token_bytes(d) * (uint64_t) chunk; }

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
inline int64_t keep_positions(int64_t backed, const std::vector<int>& ratios, CacheFormat f) {
    if (backed <= kKeepPositions) return backed;
    return ctxp::cache_bytes(backed, ratios, f) - ctxp::cache_bytes(kKeepPositions, ratios, f) >= kTrimBytes ? kKeepPositions : backed;
}

}  // namespace basalt::models::deepseek41::prefillp
