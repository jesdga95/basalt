#pragma once

#include <cstdint>

namespace basalt::models::deepseek4 {

/// Applies final RMSNorm and the checkpoint's BF16 output matrix to 1..8 hidden rows.
/// `normalized` and `logits` are caller-owned [T][n_embd] and [T][n_vocab] workspaces.
void logits_step(const float* hidden, const float* norm, const uint16_t* output_weight, int n_tokens,
                 int n_embd, int n_vocab, float rms_eps, float* normalized, float* logits, void* stream);

}  // namespace basalt::models::deepseek4
