// include/basalt/kernels/row_norm.hpp - RMS norms of fp32 rows of up to 8192 values, alone and fused with the hyper-connection
// collapse that feeds them.
//
// A call takes T tokens (any T >= 1) and the result for a token does not depend on T. `stream` may be null (the call then
// synchronises).
#pragma once

namespace basalt::kernels {

/// out[t][i] = x[t][i] * rsqrt(mean(x[t]^2) + eps) * w[i], one fp32 row of n (multiple of 4, at most 8192) per token.
void rms_norm_rows(const float* x, const float* w, int n_tokens, int n, float eps, float* out, void* stream);

/// hc_pre followed by rms_norm_rows in one launch: cur = hc_pre(x, pre) is stored, out = rms_norm(cur) * w.
void hc_pre_rms_norm(const float* x, const float* pre, int pre_stride, const float* w, int n_tokens, int n, float eps,
                     float* cur, float* out, void* stream);

}  // namespace basalt::kernels
