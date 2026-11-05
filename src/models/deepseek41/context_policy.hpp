// src/models/deepseek41/context_policy.hpp - how long a context the compressed caches can hold, free of CUDA so a CPU
// test can drive it. The caches grow with the position; this sizes the address range they may grow into and refuses
// a context the card cannot back.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41::ctxp {

constexpr uint64_t kLatentRowBytes = 512 * 2;   // one compressed latent, fp16
constexpr uint64_t kIndexRowBytes = 128 * 2;    // one index key, fp16
constexpr int kDecodeTokens = 8;                // tokens one decode call scores at once (indexer workspace rows)
constexpr int64_t kLegacyContext = 8192;        // the maximum of a pack that names no context length
constexpr int64_t kMinContext = 1024;

/// The kv-source layers: the ones that own compressed caches (ratio != 0 and one of the source layer indices).
inline bool owns_cache(int layer, int ratio) { return ratio != 0 && (layer == 2 || layer == 8 || layer == 14 || layer == 20); }

/// Rows a cache holds for `positions` positions at a compression ratio.
inline uint64_t cache_rows(int64_t positions, int ratio) { return (uint64_t) positions / (uint64_t) ratio + 1; }

/// Device bytes of the compressed latents and index keys of every kv-source layer at `positions`.
inline uint64_t cache_bytes(int64_t positions, const std::vector<int>& ratios) {
    uint64_t b = 0;
    for (int r : ratios)
        if (r > 0) b += cache_rows(positions, r) * (kLatentRowBytes + kIndexRowBytes);
    return b;
}

/// Device bytes of the decode indexer's score workspace (scores, block scores, block flags) at `positions`.
inline uint64_t workspace_bytes(int64_t positions) { return (uint64_t) kDecodeTokens * ((uint64_t) positions + 1) * 9; }

/// Everything that grows with the context.
inline uint64_t context_bytes(int64_t positions, const std::vector<int>& ratios) {
    return cache_bytes(positions, ratios) + workspace_bytes(positions);
}

struct Plan {
    int64_t context = 0;   ///< the context the engine runs with; 0 when refused
    std::string error;     ///< set when refused
    std::string note;      ///< a warning to print
};

/// The context from --max-context (`requested`, the shared default when not given), the model's native maximum and the
/// free device memory once the dense weights are resident: below kMinContext or past the native maximum is refused, so
/// are buffers that need more than half the free memory; a quarter or more gives a warning (the expert tier needs the
/// rest).
inline Plan plan_context(int64_t requested, int64_t native, uint64_t free_bytes, const std::vector<int>& ratios) {
    Plan p;
    if (native <= 0) native = kLegacyContext;
    const uint64_t quarter = free_bytes / 4, half = free_bytes / 2;
    if (requested < kMinContext) {
        p.error = "--max-context " + std::to_string(requested) + " is below the least DeepSeek context, " +
                  std::to_string(kMinContext);
        return p;
    }
    if (requested > native) {
        p.error = "--max-context " + std::to_string(requested) + " is past the model's maximum context of " +
                  std::to_string(native) + " tokens";
        return p;
    }
    const uint64_t need = context_bytes(requested, ratios);
    if (need > half) {
        p.error = "--max-context " + std::to_string(requested) + " needs " + std::to_string(need >> 20) +
                  " MiB of compressed caches, more than half of the " + std::to_string(free_bytes >> 20) +
                  " MiB free after the weights (a smaller --max-context makes them smaller)";
        return p;
    }
    if (need > quarter)
        p.note = "--max-context " + std::to_string(requested) + " can take up to " + std::to_string(need >> 20) +
                 " MiB for the caches of a full context, a quarter or more of the free memory; a prompt that fills it leaves the expert tier smaller";
    p.context = requested;
    return p;
}

}  // namespace basalt::models::deepseek41::ctxp
