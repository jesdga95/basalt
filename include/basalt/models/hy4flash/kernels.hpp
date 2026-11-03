// include/basalt/models/hy4flash/kernels.hpp - the Hy4-flash decode kernels: the per-head Q8_0 products of the absorbed
// MLA, its query/key preparation and attention over the shared latent with a learnable sink, and the Engram apply.
//
// One token per call; all activations fp32, all pointers device memory. `stream` may be null (the call then
// synchronises).
#pragma once

#include <cstddef>
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
/// values (the model's two: 512, v_b over the latent, and 192, k_b over the query's nope part); x of token t and head h starts
/// at x + t * x_tok_stride + h * x_head_stride. `w` is the tensor's bytes (34 bytes per 32 values, rows of k / 32 blocks),
/// 16-byte aligned.
void q8_0_heads_gemv(const void* w, const float* x, int x_head_stride, int x_tok_stride, int heads, int rows, int k, int n_tok,
                     float* y, void* stream);

/// The attention inputs of n_tok rows at pos0 + t. q: the query projection [t][heads][nope + rope]; q_abs: the absorbed nope
/// part [t][heads][rank]; kv_raw: the key/value projection [t][rank + rope]. Writes qcur [t][heads][rank + rope] = q_abs then the
/// rotated rope part, and each row's cache row (the latent normed by kv_norm, then the rotated rope part) at cell pos0 + t
/// (modulo `ring` when it is set).
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, float* cache, void* stream);
/// The same with an fp16 cache (round to nearest even).
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint16_t* cache, void* stream);

/// The same with an FP8 cache: vLLM's fp8_ds_mla rows of kFp8RowBytes: 512 E4M3 codes of the BF16-rounded latent, four fp32 scales
/// (one per 128 codes, 2^ceil(log2(max(amax / 448, 1e-4)))), then the 64 rope values as BF16.
constexpr int kFp8RowBytes = 656;
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint8_t* cache, void* stream);

/// The cells each row attends (device arrays of n_tok entries): row t takes counts[t] cells, the cells of row t listed at
/// cells + t * cells_stride, or (cells null, or fill set and fill[t] > 0) firsts[t] + j (modulo the ring when it is set).
struct RowSpans {
    const int32_t* firsts = nullptr;
    const int32_t* counts = nullptr;
    const int32_t* cells = nullptr;
    int cells_stride = 0;
    const int32_t* fill = nullptr;
};

/// o_lat[t][h][c] = sum_j softmax_j(scale * qcur[t][h] . cache[cell_j]) * cache[cell_j][c] for c < rank, j < counts[t], where the
/// softmax's denominator also holds exp(sinks[h]). cache [cells][rank + rope], rank 512 and rope 64; counts <= max_cells. Up to 8
/// rows a launch, a block per 64 cells and 8 or 16 heads (all heads read a cell once), the chunks merged by the last block of a row's
/// head group; a BF16 cache with more rows (a prompt's chunk) takes the tensor-core kernel. `work`: mla_attend_work_bytes for calls of
/// up to max_rows rows, zeroed once before its first use (the launches leave it so).
size_t mla_attend_work_bytes(int heads, int max_cells, int max_rows);
void mla_attend(const float* qcur, const float* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                int max_cells, void* work, float* o_lat, void* stream);
/// The same over an fp16 cache.
void mla_attend(const float* qcur, const uint16_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                int max_cells, void* work, float* o_lat, void* stream);
void mla_attend(const float* qcur, const uint8_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                int max_cells, void* work, float* o_lat, void* stream);

/// mla_prepare for a window of 1 to 8 rows in one launch, with q_abs computed in it: the k_b product (w_kb: Q8_0 [heads][rank][nope],
/// as q8_0_heads_gemv with the query's head stride) goes straight into qcur. Row t is at position pos[t] (device).
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, float* cache, void* stream);
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, uint16_t* cache, void* stream);
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, uint8_t* cache, void* stream);

/// ids[h * rows + t] = t * heads + h: the (token, head) rows of a [token][head][k] array in head-major order; bounds[h] = h * rows
/// for h <= heads (each head an "expert" of an MMQ product).
void head_major_ids(int32_t* ids, int32_t* bounds, int heads, int rows, void* stream);

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

/// The same for `rows` rows at once: row t has q[t][heads][dim], w[t][heads], and meta[t] = {pools, tail0, tail_n} (device); its scores
/// go to scores[t * stride ..] (stride >= max_pool), its cells to cells[t * cells_stride ..]. A row with pools <= k is left alone.
void idx_scores_rows(const float* q, const float* w, const uint16_t* pooled, const int32_t* meta, int max_pool, int rows, const IndexerArgs& a,
                     float* scores, int stride, void* stream);
void idx_select_rows(const float* scores, int stride, const int32_t* meta, int rows, int k, int pool, int32_t* cells, int cells_stride, void* stream);

/// y[t][r] = scale * w[r] . x[t] for r < rows over `cols` values, t < n_tok (w fp32 [rows][cols], cols % 4 == 0).
void f32_gemm_rows(const float* x, const float* w, int rows, int cols, float scale, int n_tok, float* y, void* stream);
/// out[i] = a[i] + b[i].
void add_vectors(const float* a, const float* b, float* out, int n, void* stream);
/// A sliding-window layer's row spans for R <= 32 rows from the first position `*pos0` (device memory), as the host writes
/// them: meta[t] = max(0, pos + 1 - window), meta[arows + t] = pos + 1 - first, meta[2 arows + t] = 0, meta[3 arows + t] = pos.
void swa_meta(const int32_t* pos0, int R, int window, int arows, int32_t* meta, void* stream);
/// A sublayer's end in one launch: f = a + b (the MoE's routed and shared outputs; b null: f = a, the attention's output), its
/// rms_norm_rows by `w`, then hc_post of f into the four streams of `x` by `coef` (n_tokens rows of E = 3072, the model's width):
/// add_vectors, rms_norm_rows and hc_post bit for bit, whatever n_tokens.
void add_norm_hc_post(const float* a, const float* b, const float* w, float eps, const float* x, const float* coef, int coef_stride,
                      int n_tokens, int E, float* out, void* stream);

// ---- MoE ----------------------------------------------------------------------------------------------------------

/// Most words put_words carries: a decode window's expert plan and its row pointers (the kernel's parameters stay under 4 KB).
constexpr int kPutWords = 960;
/// dst[0..n_words) = src[0..n_words) for `src` in host memory, n_words <= kPutWords: the words travel in the kernel's
/// parameters, so a plan reaches the device in stream order without a copy-engine job (which would queue behind the tier's
/// copies) and the host may reuse `src` at once.
void put_words(const void* src, void* dst, int n_words, void* stream);

/// out[t][i] = sum_k w[t * K + k] * rows[t * K + k][i] for k < K, in that order (rows: device array of T * K row pointers, each
/// device or mapped host memory).
void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream);

/// Holds the stream until *word (mapped host memory the host writes) equals `value`, fences, then stores `value` in mark[0]
/// (device memory), which an l2_read_ahead on another stream stops at.
void wait_word_mark(const uint32_t* word, uint32_t value, uint32_t* mark, void* stream);
/// Byte ranges of device memory (16-byte aligned, 16-byte multiples) in the order the kernels after a wait read them; empty
/// ones last.
constexpr int kL2Ranges = 4;
struct L2Ranges {
    const void* p[kL2Ranges] = {};
    uint64_t n[kL2Ranges] = {};
};
/// Requests the ranges into L2, in order, until mark[0] equals `value` (mark: 2 words of device memory; mark[1] is scratch
/// the kernel never writes in practice). Reads only: nothing queued after it waits for it.
void l2_read_ahead(const L2Ranges& r, uint32_t* mark, uint32_t value, void* stream);

/// h[r][k] = silu(min(gu[r][k], limit)) * clamp(gu[r][n_ff + k], -limit, limit) for rows of 2 * n_ff values (the gate rows' products
/// then the up rows').
/// The per-expert FP32 scales an NVFP4 blob ends with ({s_gate, s_up, s_down, 0} at blob + tail_off), for the rows of n experts
/// staged `stride` bytes apart from `blobs`: expert q's rows are [bounds[q], bounds[q + 1]) (device). MMQ's NVFP4 products
/// leave these scales out (DeepSeek's prompt path applies them the same way).
struct ExpertTails {
    const uint8_t* blobs = nullptr;
    size_t stride = 0, tail_off = 0;
    const int32_t* bounds = nullptr;
    int n = 0;
};
/// With `tails`, gate and up are first times row r's expert's s_gate and s_up (before the clamp), and h times its s_down, so
/// the down product's output carries s_down.
void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream,
                          const ExpertTails* tails = nullptr);
/// out[t][i] = sum_k w[t * K + k] * rows[map[t * K + k]][i] for k < K, in that order.
void weighted_sum_tokens(const float* rows, const int32_t* map, const float* w, int n_tok, int K, int n, float* out, void* stream);

// ---- hyper-connection streams -------------------------------------------------------------------------------------

/// x[t][4][n] = 4 copies of e[t][n], t < n_tok.
void fill_streams(const float* e, int n, int n_tok, float* x, void* stream);

// ---- Engram ---------------------------------------------------------------------------------------------------------

struct EngramArgs {
    int n_embd = 3072;
    float eps = 1.1920929e-07f;   ///< of the gate's two norms (the stream and the key) and the value's: FP32's machine epsilon
};

/// The Engram step on the four streams x[t][4][n] of n_tok rows, in place. kv: the fused projection [t][5][n] (four keys, one value)
/// before its bias kv_b [5][n]; gate_w and hc_scale [4][n]; pre_gamma [n]. valid[t]: 0 leaves row t as it is.
/// gate_i = sigmoid(copysign(sqrt(max(|d_i|, 1e-6)), d_i)), d_i = sum_j rms(x_i)[j] rms(key_i)[j] gate_w[i][j] / sqrt(n);
/// x_i += rms(value) * pre_gamma * (gate_i + hc_scale[i]).
void engram_apply(const float* kv, const float* kv_b, const float* gate_w, const float* pre_gamma, const float* hc_scale,
                  const float* valid, int n_tok, const EngramArgs& a, float* x, void* stream);

}  // namespace basalt::models::hy4flash
