// src/models/ernie55/context_policy.hpp - KV cache precision & context sizing for ERNIE-5.5-Flash (CPU-only).
#pragma once

#include "basalt/models/ernie55/cache_layout.hpp"

#include <cstdint>
#include <string>

namespace basalt::models::ernie55::ctxp {

enum class CacheFormat { Fp8, Fp16 };

inline bool cache_format(const std::string& s, CacheFormat& out) {
    if (s.empty() || s == "fp8") { out = CacheFormat::Fp8; return true; }
    if (s == "fp16") { out = CacheFormat::Fp16; return true; }
    return false;
}

/// Bytes per context position across all HCA + MLA-DSA layers (excluding the fixed HCA sliding-window rings).
inline uint64_t bytes_per_position(int n_hca, int n_dsa, int hca_ratio, int hca_head_dim, int mla_kv_lora,
                                   int mla_rope, int idx_dim, CacheFormat fmt) {
    const bool fp8 = fmt == CacheFormat::Fp8;
    const uint64_t hca_row = layout::hca_row_bytes(fp8, hca_head_dim);
    const uint64_t hca_per_pos = (uint64_t) n_hca * ((hca_row + (uint64_t) hca_ratio - 1) / (uint64_t) hca_ratio);
    const uint64_t dsa_kv = layout::dsa_row_bytes(fp8, mla_kv_lora, mla_rope);
    const uint64_t idx_k = layout::idx_row_bytes(fp8, idx_dim);
    const uint64_t dsa_per_pos = (uint64_t) n_dsa * (dsa_kv + idx_k);
    return hca_per_pos + dsa_per_pos;
}

struct Plan {
    int64_t context = 0;
    uint64_t kv_bytes = 0;
    std::string error;
};

inline Plan plan_context(int64_t asked, int64_t max_pos, uint64_t free_vram, int n_hca, int n_dsa, int hca_ratio,
                         int hca_head_dim, int mla_kv_lora, int mla_rope, int idx_dim, CacheFormat fmt) {
    Plan p;
    if (asked < 1024 || asked > max_pos) {
        p.error = "--max-context " + std::to_string(asked) + " out of range [1024, " + std::to_string(max_pos) + "]";
        return p;
    }
    const uint64_t per_pos = bytes_per_position(n_hca, n_dsa, hca_ratio, hca_head_dim, mla_kv_lora, mla_rope, idx_dim, fmt);
    p.context = asked;
    p.kv_bytes = (uint64_t) asked * per_pos;
    if (free_vram > 0 && p.kv_bytes > free_vram / 2) {
        p.error = "--max-context " + std::to_string(asked) + " needs " + std::to_string(p.kv_bytes >> 20) +
                  " MiB KV cache, more than half of " + std::to_string(free_vram >> 20) + " MiB free VRAM";
        return p;
    }
    return p;
}

struct TierReserve {
    uint64_t copies = 0;        ///< beside slots that are all copies (a long prompt lends them)
    uint64_t homes = 0;         ///< beside permanent VRAM homes (cannot be lent during a prompt)
    uint64_t beside_homes = 0;  ///< beside homes + ext copies (shrunk during a prompt)
    uint64_t modules = 0;       ///< CUDA / MMQ modules loaded on first prompt
};

constexpr uint64_t kCopiesReserveBytes = 512ull << 20;
constexpr uint64_t kBesideHomesReserveBytes = 512ull << 20;
constexpr uint64_t kModuleRoomBytes = 7ull << 27;

/// Computes ExpertTier VRAM reserves so cards holding permanent VRAM homes (`homes_.main > 0`) always leave
/// enough free VRAM for prompt MMQ staging, the prompt expert ring beside the homes, MMQ stream-k headroom, and
/// first-prompt module loads. `home_ring_bytes` is the double-buffered prefill expert ring the prompt path opens
/// next to the homes (0: no ring, as before).
inline TierReserve plan_tier_reserve(uint64_t home_prompt_bytes, uint64_t mmq_headroom_bytes,
                                     uint64_t home_ring_bytes = 0) {
    TierReserve r;
    r.copies = kCopiesReserveBytes;
    r.beside_homes = kBesideHomesReserveBytes;
    r.modules = kModuleRoomBytes;
    r.homes = home_prompt_bytes + home_ring_bytes + mmq_headroom_bytes + kModuleRoomBytes;
    return r;
}

}  // namespace basalt::models::ernie55::ctxp
