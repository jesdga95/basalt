// include/basalt/models/hy4flash/kernels.hpp - the Hy4-flash decode kernels: the per-head Q8_0 products of the absorbed
// MLA, its query/key preparation and attention over the shared latent with a learnable sink, and the Engram apply.
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
    /// Interleaved MRoPE for pictures: when `cell_rope` (device, [cell][3] = t, h, w) is set, pair i of the rotated part of the row at
    /// cell c turns by the h axis of c when i % 3 == 1 and i < 3 * sec_h, by the w axis when i % 3 == 2 and i < 3 * sec_w, by the t
    /// axis otherwise; null: every pair turns by the cell index, as text does.
    const int32_t* cell_rope = nullptr;
    int sec_h = 0, sec_w = 0;
};

/// A product per head over Q8_0 rows, for n_tok tokens: y[t][h][r] = w[h * rows + r] . x[t][h] for h < heads, r < rows, over `k`
/// values (k % 32 == 0); x of token t and head h starts at x + t * x_tok_stride + h * x_head_stride. `w` is the tensor's bytes
/// (34 bytes per 32 values, rows of k / 32 blocks).
void q8_0_heads_gemv(const void* w, const float* x, int x_head_stride, int x_tok_stride, int heads, int rows, int k, int n_tok,
                     float* y, void* stream);

/// The attention inputs of n_tok rows at pos0 + t. q: the query projection [t][heads][nope + rope]; q_abs: the absorbed nope
/// part [t][heads][rank]; kv_raw: the key/value projection [t][rank + rope]. Writes qcur [t][heads][rank + rope] = q_abs then the
/// rotated rope part, and each row's cache row (the latent normed by kv_norm, then the rotated rope part) at cell pos0 + t
/// (modulo `ring` when it is set).
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, float* cache, void* stream);
/// The same with a BF16 cache (round to nearest even).
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint16_t* cache, void* stream);

/// The same with an FP8 cache: rows of kFp8RowBytes (512 E4M3 codes, four fp32 scales, one per 128 codes, amax / 448, then the 64 rope
/// values as BF16), written and read as the flash-MLA kernels' rows are.
constexpr int kFp8RowBytes = 656;
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint8_t* cache, void* stream);

/// The cells each row attends (device arrays of n_tok entries): row t takes counts[t] cells, the cells of row t listed at
/// cells + t * cells_stride, or (cells null) firsts[t] + j (modulo the ring when it is set).
struct RowSpans {
    const int32_t* firsts = nullptr;
    const int32_t* counts = nullptr;
    const int32_t* cells = nullptr;
    int cells_stride = 0;
};

/// o_lat[t][h][c] = sum_j softmax_j(scale * qcur[t][h] . cache[cell_j]) * cache[cell_j][c] for c < rank, j < counts[t], where the
/// softmax's denominator also holds exp(sinks[h]). cache [cells][rank + rope]; counts <= 4096.
void mla_attend(const float* qcur, const float* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream);
/// The same over a BF16 cache.
void mla_attend(const float* qcur, const uint16_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream);
void mla_attend(const float* qcur, const uint8_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream);
/// cells[t][j] = j for j < fill[t]; a row with fill 0 is left alone.
void iota_cells(int32_t* cells, int stride, const int32_t* fill, int n_tok, void* stream);

/// x[i] *= sigmoid(g[i]).
void mul_sigmoid(const float* g, float* x, int n, void* stream);

// ---- the indexer of the sparse layers --------------------------------------------------------------------------------

struct IndexerArgs {
    int heads = 16, dim = 128, rope = 64, pool = 4;
    float rope_base = 1e7f;
    float eps = 1e-5f;   ///< of the pooled key's norm
    /// Interleaved MRoPE for pictures: when `cell_rope` (device, [cell][3] = t, h, w) is set, pair i of the rotated part of the row at
    /// cell c turns by the h axis of c when i % 3 == 1 and i < 3 * sec_h, by the w axis when i % 3 == 2 and i < 3 * sec_w, by the t
    /// axis otherwise; null: every pair turns by the cell index, as text does.
    const int32_t* cell_rope = nullptr;
    int sec_h = 0, sec_w = 0;
};

/// The pooled keys of n_groups consecutive groups of `pool` positions starting at first_pos (a multiple of `pool`): the rows of
/// position p are at cell p % ring of values and gates [ring][dim] (the projections of the normed keys; ring a multiple of
/// `pool`), ape [pool][dim]; pooled[d] = sum_j softmax_j(gates[j][d] + ape[j][d]) * values[j][d], normed by norm_w, its last
/// `rope` values rotated at the group's first position; written as fp16 bits to pooled[(first_pos + g * pool) / pool]. dim = 128.
void idx_pool(const float* values, const float* gates, const float* ape, const float* norm_w, const IndexerArgs& a, int ring,
              int first_pos, int n_groups, uint16_t* pooled, void* stream);
/// The last `rope` values of each head of n_tok queries q[t][heads][dim], rotated at pos0 + t, in place.
void idx_rope_heads(float* q, const IndexerArgs& a, int pos0, int n_tok, void* stream);
/// scores[p] = sum_h w[h] * relu(q[h] . pooled[p]) for p < n_pool (q [heads][dim], pooled [n_pool][dim] fp16 bits).
void idx_scores(const float* q, const float* w, const uint16_t* pooled, int n_pool, const IndexerArgs& a, float* scores,
                void* stream);
/// The cells a query attends: those of the k pools with the largest scores (a lower pool wins a tie), in pool order, then the
/// tail cells tail0 .. tail0 + tail_n - 1. cells[k * pool + tail_n]; k <= n_pool.
void idx_select(const float* scores, int n_pool, int k, int pool, int tail0, int tail_n, int32_t* cells, void* stream);

/// y[t][r] = scale * w[r] . x[t] for r < rows over `cols` values, t < n_tok (w fp32 [rows][cols], cols % 4 == 0).
void f32_gemm_rows(const float* x, const float* w, int rows, int cols, float scale, int n_tok, float* y, void* stream);
/// out[i] = a[i] + b[i].
void add_vectors(const float* a, const float* b, float* out, int n, void* stream);

// ---- MoE ----------------------------------------------------------------------------------------------------------

/// out[t][i] = sum_k w[t * K + k] * rows[t * K + k][i] for k < K, in that order (rows: device array of T * K row pointers, each
/// device or mapped host memory).
void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream);

// ---- hyper-connection streams -------------------------------------------------------------------------------------

/// x[t][4][n] = 4 copies of e[t][n], t < n_tok.
void fill_streams(const float* e, int n, int n_tok, float* x, void* stream);

// ---- Engram ---------------------------------------------------------------------------------------------------------

struct EngramArgs {
    int n_embd = 3072;
    float eps = 1.1920929e-07f;   ///< of the norms
};

/// The Engram step on the four streams x[t][4][n] of n_tok rows, in place. kv: the fused projection [t][5][n] (four keys, one value)
/// before its bias kv_b [5][n]; gate_w and hc_scale [4][n]; pre_gamma [n]. valid[t]: 0 leaves row t as it is.
/// gate_i = sigmoid(copysign(sqrt(max(|d_i|, 1e-6)), d_i)), d_i = sum_j rms(x_i)[j] rms(key_i)[j] gate_w[i][j] / sqrt(n);
/// x_i += rms(value) * pre_gamma * (gate_i + hc_scale[i]).
void engram_apply(const float* kv, const float* kv_b, const float* gate_w, const float* pre_gamma, const float* hc_scale,
                  const float* valid, int n_tok, const EngramArgs& a, float* x, void* stream);

}  // namespace basalt::models::hy4flash
