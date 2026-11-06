// include/basalt/models/deepseek41/kernels.hpp - the DeepSeek-V4.1 decode kernels: router, hyper-connections, Engram.
//
// All activations are fp32 and device pointers; a call takes T tokens (any T >= 1) and the result for a token does
// not depend on T or on the token's place in the window. `stream` may be null (the call then synchronises).
//
// Hyper-connection streams are laid out [T][4][n_embd] (stream-major per token). The mixing coefficients of one
// token are 24 floats: pre[4], post[4], comb[16] with comb[src*4 + dst].
#pragma once

#include <cstdint>
#include <vector>

namespace basalt::models::deepseek41 {

constexpr int kHcMult = 4;
constexpr int kHcCoef = 24;           ///< pre(4) + post(4) + comb(16)
constexpr int kHcSplits = 8;          ///< partial sums per mix row (workspace sizing)

// ---- router ---------------------------------------------------------------------------------------------------

struct RouterArgs {
    int n_expert = 384;               ///< multiple of 32, at most 512
    int n_embd = 5120;                ///< multiple of 128
    int k = 6;                        ///< at most 8
    float weight_scale = 1.5f;        ///< multiplies the normalised weights
};

/// logits = gate . x (fp32, gate row-major [n_expert][n_embd]); s = sqrt(softplus(logit)); the top-k of s + bias
/// are chosen (the lower index wins a tie); weights are the unbiased s of those experts, divided by their sum plus
/// 1e-20, times weight_scale. `logits` is a [T][n_expert] output (workspace); ids and weights are [T][k].
void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream);

// ---- hyper-connections ----------------------------------------------------------------------------------------

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

// ---- row norms ------------------------------------------------------------------------------------------------

/// out[t][i] = x[t][i] * rsqrt(mean(x[t]^2) + eps) * w[i], one fp32 row of n (multiple of 4, at most 8192) per token.
void rms_norm_rows(const float* x, const float* w, int n_tokens, int n, float eps, float* out, void* stream);

/// hc_pre followed by rms_norm_rows in one launch: cur = hc_pre(x, pre) is stored, out = rms_norm(cur) * w.
void hc_pre_rms_norm(const float* x, const float* pre, int pre_stride, const float* w, int n_tokens, int n, float eps,
                     float* cur, float* out, void* stream);

// ---- Engram ---------------------------------------------------------------------------------------------------

struct EngramArgs {
    int n_embd = 5120;
    float rms_eps = 1e-20f;
};

/// The post-op on the projected rows. kv[T][(4+1)*n_embd] is wkv . rows (key copies 0..3, then the value);
/// gate_w[4][n_embd] is the per-copy gate weight (row c at c*n_embd). In place on x[T][4][n_embd]:
///   dot_c = sum_d rms(x_c)[d] * rms(key_c)[d] * gate_w[c][d] / sqrt(n_embd)
///   gate_c = sigmoid(copysign(sqrt(max(|dot_c|, 1e-6)), dot_c));  x_c += gate_c * value
/// `keep` ([T], 1 for text tokens, 0 for embedding tokens) may be null.
void engram_apply(const float* kv, const float* gate_w, const float* keep, int n_tokens, const EngramArgs& a,
                  float* x, void* stream);

/// Hash constants of the Engram row lookup (the pack's meta.engram and the GGUF's token map).
struct EngramHashParams {
    int n_gram = 4;                       ///< longest n-gram
    int n_head = 8;                       ///< hash heads per n
    int pad_token_id = 2;
    std::vector<int32_t> token_map;       ///< token id -> compressed id (negative: dead)
    std::vector<uint64_t> mult;           ///< [engram layer][n_gram]
    std::vector<uint64_t> primes;         ///< [engram layer][n_cols]
    std::vector<uint64_t> offsets;        ///< [engram layer][n_cols]
    int n_cols() const { return (n_gram - 1) * n_head; }
};

/// Table row ids for engram layer slot `e` (host). tokens[n] sit at positions pos0..pos0+n-1; `prev` holds the
/// n_gram-1 tokens before tokens[0], oldest first (null when pos0 == 0). out is [n][n_cols].
void engram_hash(const EngramHashParams& p, int e, const int32_t* tokens, int n, int64_t pos0, const int32_t* prev,
                 int32_t* out);

}  // namespace basalt::models::deepseek41
