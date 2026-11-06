// src/models/deepseek41/attn_dense.hpp - small kernels of the batched attention's dense stages.
#pragma once

namespace basalt::models::deepseek41 {

/// x[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w for `rows` rows of n floats, in place.
void rms_norm_rows(float* x, const float* w, int n, float eps, int rows, void* stream);

}  // namespace basalt::models::deepseek41
