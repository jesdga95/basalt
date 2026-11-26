#pragma once

#include "basalt/kernels/sparse_mqa.hpp"

#include <array>
#include <string>

namespace basalt::models::deepseek4 {

constexpr int kRopePairs = 32;

struct RopeParams {
    int rope_dim = 64;
    float rope_base = 0.f;
    float compressed_base = 0.f;
    float factor = 0.f;
    int original_context = 0;
    float beta_fast = 0.f;
    float beta_slow = 0.f;
};

/// Per-layer RoPE tables. Ratio-zero layers use the unscaled attention base; compressed layers use the checkpoint's
/// compressed base and YaRN schedule. `frequency` is also consumed by the CSA/HCA and indexer compressors.
struct RopeTables {
    kernels::MqaRopeTable attention;
    std::array<float, kRopePairs> frequency{};
};

bool make_rope_tables(const RopeParams& params, int compress_ratio, RopeTables& tables, std::string& err);

}  // namespace basalt::models::deepseek4
