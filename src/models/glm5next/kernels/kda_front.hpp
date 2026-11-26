// src/models/glm5next/kernels/kda_front.hpp - see kda_front.cu: the KDA per-token front, the work in front of the
// recurrence (kda.hpp) on the raw projections of one token.
//
// The model's KDA block projects q, k and v at 64 heads x 128 (8192 channels each, tensor names attn_q/k/v), runs a
// depthwise causal conv1d of width 4 over each stream (ssm_conv1d_q/k/v, weights [4, 1, 8192]) with SiLU, L2-normalizes
// q and k and scales q by 1/sqrt(128), and computes the gates: the per-channel decay
//
//   g_raw = kGateLowerBound * sigmoid(-(ssm_a[head] * (a + ssm_dt_bias)))   (a = ssm_f_b(ssm_f_a(x)), one f32
//   g     = exp(g_raw)                                                      per channel, dt bias per channel)
//
// (the reference's bounded smooth form when the model sets a KDA gate lower bound; ssm_a stores -exp(A_log), so the
// sigmoid sees exp(A_log) * (a + dt)). beta = sigmoid(ssm_beta(x)) per head (ssm_beta.weight, 64 rows). The out
// gate z (ssm_g_a/g_b) and the gated RMS norm belong behind the recurrence, not here.
#pragma once

#include <cuda_runtime.h>

namespace basalt::models::glm5next {

/// One token's front for `n_heads` heads of `hc` channels each (the model: 64 heads x 128).
///   x      [3 * n_heads * hc] f32   this token's raw q projection; `x + s * x_stride` is stream s (q, k, v)
///   x_stride                     the gap between the streams. The projections arrive as one [T][hc*n_heads] slab
///                                  per stream, so it is the window's token count; a single token whose q|k|v sit
///                                  side by side passes n_heads * hc.
///   hist   [3][3 * n_heads * hc] f32  the conv's 3 past positions, oldest first; each row holds that position's
///                                  q|k|v contiguously, n_heads * hc channels per stream (the cache's layout)
///   conv   [3][4][n_heads * hc] f32  ssm_conv1d_q then _k then _v, tap-major within each stream
///   a      [n_heads * hc] f32       the decay projection (ssm_f path) for this token
///   dt     [n_heads * hc] f32       ssm_dt.bias
///   a_neg  [n_heads] f32            ssm_a = -exp(A_log), per head
///   b_raw  [n_heads] f32            the ssm_beta projection for this token
///   qn/kn  [n_heads * hc] f32       the conv'd, L2-normalized (q scaled) streams - the recurrence's q and k
///   vout   [n_heads * hc] f32       the conv'd v stream
///   g      [n_heads * hc] f32       the per-channel decay FACTOR exp(g_raw), in (exp(-5), 1)
///   beta   [n_heads] f32            sigmoid(b_raw)
constexpr float kGateLowerBound = -5.0f;
/// The FLA l2norm epsilon, inside the root: x / sqrt(sum(x^2) + kL2Eps) (modeling_kimi_linear.l2norm).
constexpr float kL2Eps = 1e-6f;

void kda_front(const float* x, size_t x_stride, const float* hist, const float* conv, const float* a, const float* dt,
               const float* a_neg, const float* b_raw, float* qn, float* kn, float* vout, float* g, float* beta,
               int n_heads, int hc, cudaStream_t stream);

}  // namespace basalt::models::glm5next
