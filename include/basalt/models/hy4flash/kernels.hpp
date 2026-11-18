// include/basalt/models/hy4flash/kernels.hpp - the Hy4-flash decode kernels: the per-head Q8_0 products of the absorbed
// MLA, its query/key preparation and attention over the shared latent with a learnable sink, the sigmoid router and the
// Engram apply.
//
// One token per call; all activations fp32, all pointers device memory. `stream` may be null (the call then
// synchronises).
#pragma once

#include <cstdint>

namespace basalt::models::hy4flash {

// ---- attention ----------------------------------------------------------------------------------------------------

struct MlaArgs {
    int heads = 32;
    int nope = 192, rope = 64, rank = 512;   ///< a query head is [nope | rope]; a cache row is [latent (rank) | rope]
    float rope_base = 1e7f;                  ///< rotation of consecutive pairs
    float scale = 0.0625f;                   ///< of the scores
    float rms_eps = 1e-5f;                   ///< of the latent's norm
};

/// A product per head over Q8_0 rows: y[h][r] = w[h * rows + r] . x[h] for h < heads, r < rows, over `k` values (k % 32 == 0);
/// x of head h starts at x + h * x_stride. `w` is the tensor's bytes (34 bytes per 32 values, rows of k / 32 blocks).
void q8_0_heads_gemv(const void* w, const float* x, int x_stride, int heads, int rows, int k, float* y, void* stream);

/// The attention inputs of one token at `pos`. q: the query projection [heads][nope + rope]; q_abs: the absorbed
/// nope part [heads][rank]; kv_raw: the key/value projection [rank + rope]. Writes qcur [heads][rank + rope] = q_abs then
/// the rotated rope part, and the cache row [rank + rope] = the latent normed by kv_norm then the rotated rope part.
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos,
                 float* qcur, float* cache_row, void* stream);

/// o_lat[h][c] = sum_j softmax_j(scale * qcur[h] . cache[first + j]) * cache[first + j][c] for c < rank, j < n, where the
/// softmax's denominator also holds exp(sinks[h]). cache [cells][rank + rope]; n <= 4096.
void mla_attend(const float* qcur, const float* cache, int first, int n, const float* sinks, const MlaArgs& a,
                float* o_lat, void* stream);

/// x[i] *= sigmoid(g[i]).
void mul_sigmoid(const float* g, float* x, int n, void* stream);

// ---- MoE ----------------------------------------------------------------------------------------------------------

/// The router of one token: logits = gate . x (gate fp32 [n_expert][n_embd]); s = sigmoid(logits); the top-k of s + bias
/// (a lower index wins a tie); weights the unbiased s of those over their sum, times `scale`. n_expert <= 512, k <= 16.
void router_sigmoid(const float* x, const float* gate, const float* bias, int n_expert, int n_embd, int k, float scale,
                    float* logits, int32_t* ids, float* weights, void* stream);

// ---- hyper-connection streams -------------------------------------------------------------------------------------

/// x[4][n] = 4 copies of e[n].
void fill_streams(const float* e, int n, float* x, void* stream);

// ---- Engram ---------------------------------------------------------------------------------------------------------

struct EngramArgs {
    int n_embd = 3072;
    float eps = 1.1920929e-07f;   ///< of the norms
};

/// The Engram step on the four streams x[4][n], in place. kv: the fused projection [5][n] (four keys, one value) before its
/// bias kv_b [5][n]; gate_w and hc_scale [4][n]; pre_gamma [n]. valid: 0 leaves the streams as they are.
/// gate_i = sigmoid(copysign(sqrt(max(|d_i|, 1e-6)), d_i)), d_i = sum_j rms(x_i)[j] rms(key_i)[j] gate_w[i][j] / sqrt(n);
/// x_i += rms(value) * pre_gamma * (gate_i + hc_scale[i]).
void engram_apply(const float* kv, const float* kv_b, const float* gate_w, const float* pre_gamma, const float* hc_scale,
                  float valid, const EngramArgs& a, float* x, void* stream);

}  // namespace basalt::models::hy4flash
