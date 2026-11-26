// src/models/glm5next/kernels/kda_norm.hpp - see kda_norm.cu: the KDA output norm, the gated RMS norm that closes a
// KDA layer between the recurrence's read and attn_output (modeling_kimi_linear's RMSNormGated, sigmoid-gated).
//
// Per token and head, over the head's hd values:
//
//   y[d] = o[d] * rsqrt( sum(o^2)/hd + eps ) * w[d] * sigmoid(z[d])
//
// with w the shared per-head gain (ssm_norm[hd], the same weights for every head) and z this token's output gate
// (ssm_g_b(ssm_g_a(x)), applied RAW - the sigmoid is the norm's). The reference computes the whole norm in fp32.
#pragma once

#include <cuda_runtime.h>

namespace basalt::models::glm5next {

/// `n_rows` = n_tokens * n_heads rows of hd values each (the model: 64 heads of 128, eps = 1e-5 from
/// attention.layer_norm_rms_epsilon). o and z are [n_rows][hd], w [hd], out may alias o.
void kda_norm(const float* o, const float* z, const float* w, float* out, int n_rows, int hd, float eps,
              cudaStream_t stream);

}  // namespace basalt::models::glm5next
