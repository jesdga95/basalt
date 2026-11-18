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

/// o_lat[h][c] = sum_j softmax_j(scale * qcur[h] . cache[cell_j]) * cache[cell_j][c] for c < rank, j < n, where the
/// softmax's denominator also holds exp(sinks[h]) and cell_j = cells[j] (cells null: first + j, modulo `ring` when it is set). cache [cells][rank + rope];
/// n <= 4096.
void mla_attend(const float* qcur, const float* cache, int first, int n, const float* sinks, const MlaArgs& a,
                float* o_lat, void* stream, const int32_t* cells = nullptr, int ring = 0);

/// x[i] *= sigmoid(g[i]).
void mul_sigmoid(const float* g, float* x, int n, void* stream);

// ---- the indexer of the sparse layers --------------------------------------------------------------------------------

struct IndexerArgs {
    int heads = 16, dim = 128, rope = 64, pool = 4;
    float rope_base = 1e7f;
    float eps = 1e-5f;   ///< of the pooled key's norm
};

/// The pooled key of one group of `pool` positions: values and gates [pool][dim] (the projections of the normed keys), ape
/// [pool][dim]; pooled[d] = sum_j softmax_j(gates[j][d] + ape[j][d]) * values[j][d], normed by norm_w, its last `rope` values
/// rotated at `first_pos`, the group's first position. dim = 128.
void idx_pool(const float* values, const float* gates, const float* ape, const float* norm_w, const IndexerArgs& a,
              int first_pos, float* pooled, void* stream);
/// The last `rope` values of each of the query's heads q[heads][dim], rotated at `pos`, in place.
void idx_rope_heads(float* q, const IndexerArgs& a, int pos, void* stream);
/// scores[p] = sum_h w[h] * relu(q[h] . pooled[p]) for p < n_pool (q [heads][dim], pooled [n_pool][dim]).
void idx_scores(const float* q, const float* w, const float* pooled, int n_pool, const IndexerArgs& a, float* scores,
                void* stream);
/// The cells a query attends: those of the k pools with the largest scores (a lower pool wins a tie), in pool order, then the
/// tail cells tail0 .. tail0 + tail_n - 1. cells[k * pool + tail_n]; k <= n_pool.
void idx_select(const float* scores, int n_pool, int k, int pool, int tail0, int tail_n, int32_t* cells, void* stream);

/// y[r] = scale * w[r] . x for r < rows over `cols` values (w fp32 [rows][cols], cols % 4 == 0).
void f32_gemv(const float* x, const float* w, int rows, int cols, float scale, float* y, void* stream);
/// out[i] = a[i] + b[i].
void add_vectors(const float* a, const float* b, float* out, int n, void* stream);

// ---- MoE ----------------------------------------------------------------------------------------------------------

/// Row pointers of the routed experts' outputs (device or mapped host memory).
struct RowPtrs {
    const float* p[16];
};
/// out[i] = sum_k w[k] * rows.p[k][i] for k < n_rows, in that order.
void weighted_sum_rows(const RowPtrs& rows, const float* w, int n_rows, int n, float* out, void* stream);

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
