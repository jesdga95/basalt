// src/models/hy4flash/context_policy.hpp - how long a context the latent caches can hold, free of CUDA so a CPU test can drive
// it. The sparse layers keep a 16-bit row of every position and a pooled key per 4 positions; the window layers a ring. This sizes
// the address range they may grow into and refuses a context whose first positions the card cannot back.
#pragma once

#include "basalt/core/elastic_kv.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

namespace basalt::models::hy4flash::ctxp {

constexpr int64_t kMinContext = 1024;
constexpr int64_t kLegacyContext = 8192;   ///< the maximum of a pack that names no context length
constexpr int kRowWidth = 576;             ///< a latent row: 512 latent values and 64 rotated
constexpr int kRowBytes = kRowWidth * 2;   ///< fp16
constexpr int kFp8RowBytes = 656;          ///< --kv fp8: 512 E4M3 codes, four fp32 scales, 64 BF16
constexpr int kPoolDim = 128, kPoolSize = 4;
constexpr int kPoolRowBytes = kPoolDim * 2;   ///< a pooled key, fp16

/// Device bytes of the sparse layers' caches for `positions` positions.
inline uint64_t cache_bytes(int64_t positions, int sparse_layers, bool fp8 = false) {
    return (uint64_t) sparse_layers * ((uint64_t) positions * (fp8 ? kFp8RowBytes : kRowBytes) + ((uint64_t) positions / kPoolSize + 1) * kPoolRowBytes);
}
/// What does not grow: the window layers' rings.
inline uint64_t ring_bytes(int window, int swa_layers) { return (uint64_t) swa_layers * (uint64_t) window * kRowBytes; }
/// The decode indexer's score row.
inline uint64_t workspace_bytes(int64_t positions) { return ((uint64_t) positions / kPoolSize + 1) * 4; }
inline uint64_t context_bytes(int64_t positions, int sparse_layers, bool fp8 = false) { return cache_bytes(positions, sparse_layers, fp8) + workspace_bytes(positions); }

struct Plan {
    int64_t context = 0;   ///< the context the engine runs with; 0 when refused
    std::string error;     ///< set when refused
};

/// The context from --max-context (`requested`), the model's native maximum and the free device memory once the dense weights
/// are resident: below kMinContext or past the native maximum is refused. The caches are elastic (core/elastic_kv.hpp): a session
/// backs its first kElasticKvBase positions at start, refused when they need more than half the free memory; the rest comes from
/// the expert tier when a prompt needs it.
inline Plan plan_context(int64_t requested, int64_t native, uint64_t free_bytes, int sparse_layers, bool fp8 = false, int sessions = 1) {
    Plan p;
    if (native <= 0) native = kLegacyContext;
    if (requested < kMinContext) {
        p.error = "--max-context " + std::to_string(requested) + " is below the least Hy4-flash context, " + std::to_string(kMinContext);
        return p;
    }
    if (requested > native) {
        p.error = "--max-context " + std::to_string(requested) + " is past the model's maximum context of " + std::to_string(native) + " tokens";
        return p;
    }
    const int64_t kv_base = sessions > 1 ? core::kElasticKvBaseBatch : core::kElasticKvBase;
    const int64_t base = std::min<int64_t>(requested, kv_base);
    const uint64_t need = context_bytes(base, sparse_layers, fp8);
    if (need > free_bytes / 2) {
        p.error = "--max-context " + std::to_string(requested) + ": the first " + std::to_string(base) + " positions need " +
                  std::to_string(need >> 20) + " MiB of latent caches, more than half of the " + std::to_string(free_bytes >> 20) +
                  " MiB free after the weights";
        return p;
    }
    p.context = requested;
    return p;
}

}  // namespace basalt::models::hy4flash::ctxp
