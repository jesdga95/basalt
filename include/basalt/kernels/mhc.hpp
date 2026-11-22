// include/basalt/kernels/mhc.hpp - manifold-constrained hyper-connections (mHC): the mixing coefficients of a sublayer
// (projection, sigmoids, softmax and Sinkhorn), the collapse of the streams into a sublayer's input and the expansion of
// its output back into them.  Every model with such connections shares these formulas.
//
// All activations are fp32 and device pointers; a call takes T tokens (any T >= 1) and the result for a token does
// not depend on T or on the token's place in the window. `stream` may be null (the call then synchronises).
//
// Hyper-connection streams are laid out [T][4][n_embd] (stream-major per token). The mixing coefficients of one
// token are 24 floats: pre[4], post[4], comb[16] with comb[src*4 + dst].
#pragma once

#include <cstddef>

namespace basalt::kernels {

constexpr int kHcMult = 4;
constexpr int kHcCoef = 24;           ///< pre(4) + post(4) + comb(16)
constexpr int kHcSplits = 8;          ///< partial sums per mix row (workspace sizing)

struct HcArgs {
    int n_embd = 5120;
    float rms_eps = 1e-20f;           ///< of the stream's rms
    float hc_eps = 1e-6f;             ///< sigmoid offset and Sinkhorn floor
    int sinkhorn_iters = 20;
};

/// Floats of workspace hc_mixes needs for n_tokens.
inline size_t hc_mixes_workspace_floats(int n_tokens) { return (size_t) n_tokens * kHcSplits * (kHcCoef + 1); }

/// coef[T][24] from the streams: mixes = hc_fn[24][4*n_embd] . x_flat * rsqrt(mean(x_flat^2) + rms_eps), then
/// pre = sigmoid(mix*scale[0] + base) + eps, post = 2*sigmoid(mix*scale[1] + base), comb = softmax over dst per
/// src (+eps), a column normalisation, then (iters-1) rounds of row and column normalisation.
void hc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
              const HcArgs& a, float* workspace, float* coef, void* stream);

/// out[T][n_embd] = sum_s pre[t*pre_stride + s] * x[t][s]. `pre` may point into a coef buffer of another sublayer
/// (the single-pass mHC uses the previous sublayer's pre).
void hc_pre(const float* x, const float* pre, int pre_stride, int n_tokens, int n_embd, float* out, void* stream);

/// out[T][dst][d] = post[dst]*f[t][d] + sum_src comb[src*4+dst]*x[t][src][d]. post and comb are read from coef
/// (post at +4, comb at +8, `coef_stride` floats per token). `out` must not alias x or f.
void hc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
             float* out, void* stream);

// ---- identity connections ---------------------------------------------------------------------------------------------
// The identity-connection form: a sublayer has 8 mixes (pre 4, post 4) and no cross-stream mix: stream i keeps its own residual,
// out[i] = x[i] + post[i] * f. The coefficients are written in the layout above with comb the identity, so hc_pre and
// hc_post run on them unchanged. The head has 4 mixes (the pre weights alone).

struct HcPlainArgs {
    int n_embd = 3072;
    float rms_eps = 1e-5f;            ///< of the stream's rms
    float hc_eps = 1e-6f;             ///< offset of both sigmoids
    float magnitude = 2.0f;           ///< post = magnitude * sigmoid + eps
};

/// Floats per token of the coefficients hc_mixes_plain writes for `n_mix` mixes: 24 (pre, post, identity comb) for 8
/// mixes, 4 (pre) for the head's 4.
inline int hc_plain_coef_stride(int n_mix) { return n_mix == 8 ? kHcCoef : n_mix; }

/// Floats of workspace hc_mixes_plain needs for n_tokens.
inline size_t hc_plain_workspace_floats(int n_tokens, int n_mix = 8) { return (size_t) n_tokens * kHcSplits * (n_mix + 1); }

/// coef from the streams: mixes = hc_fn[n_mix][4*n_embd] . x_flat * rsqrt(mean(x_flat^2) + rms_eps), then pre =
/// sigmoid(mix*scale[0] + base) + eps and, with 8 mixes, post = magnitude * sigmoid(mix*scale[1] + base) + eps. The
/// result for a token does not depend on n_tokens.
void hc_mixes_plain(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
                    int n_mix, const HcPlainArgs& a, float* workspace, float* coef, void* stream);

}  // namespace basalt::kernels
