#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace basalt::models::deepseek4::ctxp {

constexpr int kWindow = 128;
constexpr int64_t kMinContext = 1024;
constexpr int64_t kMaxIndexerContext = 1 << 20;
constexpr uint64_t kFloatBytes = sizeof(float);

struct Footprint {
    uint64_t window = 0;
    uint64_t compressed = 0;
    uint64_t index_keys = 0;
    uint64_t partial_state = 0;
    uint64_t total = 0;
    std::string error;
};

inline bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

inline bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

/// Persistent device cache capacity for `positions` tokens. Compressor outputs are F32 rows; ratio-4 CSA
/// also keeps its independent 128-wide index keys. Partial compressor groups are fixed per compressed layer.
inline Footprint cache_footprint(int64_t positions, const std::vector<int>& ratios) {
    Footprint f;
    if (positions < 0 || ratios.empty()) {
        f.error = "invalid DeepSeek4 cache geometry";
        return f;
    }
    // cache_attention_window uses a position % 128 slot address, so the ring keeps its full fixed capacity even
    // for a short requested context.
    const uint64_t window_rows = kWindow;
    uint64_t per_layer_window = 0;
    if (!multiply(window_rows, 512u * kFloatBytes, per_layer_window) ||
        !multiply(per_layer_window, ratios.size(), f.window)) {
        f.error = "DeepSeek4 cache size overflow";
        return f;
    }

    for (const int ratio : ratios) {
        if (ratio == 0) continue;
        if (ratio != 4 && ratio != 128) {
            f.error = "unsupported DeepSeek4 compression ratio " + std::to_string(ratio);
            return f;
        }
        const uint64_t rows = (uint64_t) positions / (uint64_t) ratio;
        const uint64_t attention_width = ratio == 4 ? 512u : 512u;
        const uint64_t index_width = ratio == 4 ? 128u : 0u;
        uint64_t bytes = 0, index_bytes = 0;
        if (!multiply(rows, attention_width * kFloatBytes, bytes) ||
            !add(f.compressed, bytes, f.compressed) ||
            !multiply(rows, index_width * kFloatBytes, index_bytes) ||
            !add(f.index_keys, index_bytes, f.index_keys)) {
            f.error = "DeepSeek4 cache size overflow";
            return f;
        }

        // CSA has two 512-wide lanes and 2*ratio rolling rows for both KV and score state;
        // its indexer has two 128-wide arrays over 2*4 rows. HCA has 512-wide KV and score
        // state over ratio rows. All are float workspaces resident for the sequence.
        const uint64_t partial_floats = ratio == 4 ? (8u * 1024u * 2u + 8u * 128u * 2u)
                                                   : (128u * 512u * 2u);
        uint64_t partial_bytes = 0;
        if (!multiply(partial_floats, kFloatBytes, partial_bytes) ||
            !add(f.partial_state, partial_bytes, f.partial_state)) {
            f.error = "DeepSeek4 cache size overflow";
            return f;
        }
    }
    if (!add(f.window, f.compressed, f.total) || !add(f.total, f.index_keys, f.total) ||
        !add(f.total, f.partial_state, f.total))
        f.error = "DeepSeek4 cache size overflow";
    return f;
}

struct Plan {
    int64_t context = 0;
    Footprint footprint;
    std::string error;
    std::string note;
};

/// Context refusal is based on persistent cache capacity after dense weights load; transient prefill/index workspaces
/// are planned separately from the remaining device memory.
inline Plan plan_context(int64_t requested, int64_t native, uint64_t free_bytes, const std::vector<int>& ratios) {
    Plan p;
    if (native <= 0 || native > kMaxIndexerContext) {
        p.error = "invalid DeepSeek4 native context limit";
        return p;
    }
    const int64_t context = requested > 0 ? requested : native;
    if (context < kMinContext) {
        p.error = "--max-context " + std::to_string(context) + " is below the least DeepSeek4 context, " +
                  std::to_string(kMinContext);
        return p;
    }
    if (context > native || context > kMaxIndexerContext) {
        p.error = "--max-context " + std::to_string(context) + " exceeds the supported DeepSeek4 context of " +
                  std::to_string(std::min<int64_t>(native, kMaxIndexerContext));
        return p;
    }
    p.footprint = cache_footprint(context, ratios);
    if (!p.footprint.error.empty()) {
        p.error = p.footprint.error;
        return p;
    }
    if (p.footprint.total > free_bytes / 2) {
        p.error = "--max-context " + std::to_string(context) + " needs " +
                  std::to_string(p.footprint.total >> 20) + " MiB of DeepSeek4 caches, more than half of the " +
                  std::to_string(free_bytes >> 20) + " MiB free after the weights";
        return p;
    }
    if (p.footprint.total > free_bytes / 4)
        p.note = "DeepSeek4 caches at --max-context " + std::to_string(context) + " can take " +
                 std::to_string(p.footprint.total >> 20) + " MiB, a quarter or more of free device memory";
    p.context = context;
    return p;
}

}  // namespace basalt::models::deepseek4::ctxp
