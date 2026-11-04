// include/basalt/models/deepseek41/prefill_attn.hpp - tensor-core sparse MQA for prompt chunks.
#pragma once

#include "basalt/models/deepseek41/attn.hpp"

namespace basalt::models::deepseek41 {

/// sparse_mqa with the scores and the weighted sum on fp16 tensor cores (fp32 accumulate, fp32 softmax). Same
/// arguments, workspace and result layout as sparse_mqa; both latents must be fp16 (else sparse_mqa runs).
/// Queries and the softmax weights are rounded to fp16.
void sparse_mqa_tc(const MqaArgs& a, float* workspace, void* stream);

}  // namespace basalt::models::deepseek41
