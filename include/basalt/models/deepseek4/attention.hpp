#pragma once

#include "basalt/kernels/sparse_mqa.hpp"

#include <cstdint>
#include <cstddef>

namespace basalt::models::deepseek4 {

/// Applies the source V4 KV boundary: BF16 RMSNorm, forward RoPE, UE8M0 E4M3 simulation for non-RoPE lanes,
/// and BF16 rounding. `projected` and `out` are [n_tokens][512] F32 containers.
void prepare_attention_kv(const float* projected, const float* norm, float rms_eps, const int32_t* positions,
                          const kernels::MqaRopeTable& rope, int n_tokens, float* out, void* stream);

/// Projects source `wkv` `[512][4096]` with native quantized matvec, then applies the KV boundary.
/// The caller provides Q8_1 input scratch sized by `attention_kv_scratch_bytes` and F32 `[T][512]` workspace.
size_t attention_kv_scratch_bytes(int n_tokens);
void project_attention_kv(const float* x, int wkv_type, const void* wkv, const float* norm, float rms_eps,
                          const int32_t* positions, const kernels::MqaRopeTable& rope, int n_tokens,
                          void* q8_scratch, float* projected, float* out, void* stream);

}  // namespace basalt::models::deepseek4
