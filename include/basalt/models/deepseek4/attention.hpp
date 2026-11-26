#pragma once

#include "basalt/kernels/sparse_mqa.hpp"

#include <cstdint>

namespace basalt::models::deepseek4 {

/// Applies the source V4 KV boundary: BF16 RMSNorm, forward RoPE, UE8M0 E4M3 simulation for non-RoPE lanes,
/// and BF16 rounding. `projected` and `out` are [n_tokens][512] F32 containers.
void prepare_attention_kv(const float* projected, const float* norm, float rms_eps, const int32_t* positions,
                          const kernels::MqaRopeTable& rope, int n_tokens, float* out, void* stream);

}  // namespace basalt::models::deepseek4
