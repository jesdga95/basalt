// include/basalt/models/ernie55/kernels.hpp - ERNIE-5.5-Flash CUDA kernel declarations.
#pragma once

#include "basalt/kernels/route_lossy.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::ernie55 {

constexpr int kHcMult = 4;
constexpr int kHcCoef = 24;   // 4 pre + 4 post + 16 comb
constexpr int kHcSplits = 16;

struct RopeTable {
    float step[32];   ///< frequency per rotary pair (rope_dim = 64 -> 32 pairs)
    float mscale = 1.0f;
};

RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig);
RopeTable rope_table_plain(float freq_base);
RopeTable rope_table_yarn(float freq_base, float factor, float beta_fast, float beta_slow, int n_ctx_orig);

// ---- mHC (ERNIE 5.5 exact RMS factor: 1 / (sqrt(mean) + rms_eps)) ---------------------------------------------

size_t mhc_workspace_bytes(int n_tokens);

/// Computes 24 mHC coefficients [pre(4), post(4), comb(16)] per token from x [n_tokens, 4 * n_embd].
void mhc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
               int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, float* workspace, float* coef,
               void* stream);

/// Collapses 4 residual streams x [n_tokens, 4, n_embd] -> out [n_tokens, n_embd] using pre = coef[:, 0:4].
void mhc_pre(const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd, float* out, void* stream);

/// Updates 4 residual streams: out[d] = sum_s comb[s, d] * x[s] + post[d] * f (matching vLLM mhc_post_torch); with f2,
/// f + f2 in place of f (add_f32's sum).
void mhc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
              float* out, void* stream, const float* f2 = nullptr);

// ---- a decode window's fused launches: each the bits of the sequence it names, started as a programmatic dependent
// (basalt/kernels/pdl.hpp) inside a PdlScope ----------------------------------------------------------------------

/// mhc_mixes, mhc_pre and rms_norm_rows (row_norm.hpp; norm_w, norm_eps) into x_norm [n_tokens, n_embd], and with `xq`
/// native_quantize_q8_1 of x_norm into it: one launch, the coefficients into `coef` as mhc_mixes writes them.
/// `counters` (n_tokens, zeroed once) are left zeroed. n_embd a multiple of 1024 up to 8192.
void mhc_pre_norm(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
                  int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, const float* norm_w, float norm_eps,
                  float* workspace, unsigned* counters, float* coef, float* x_norm, void* xq, void* stream);
/// rms_norm_rows (row_norm.hpp) of x [n_tokens, cols] into out (if set) and native_quantize_q8_1 of the result into xq (if set); with
/// `wts`, the rows normalized are weighted_sum_f32(x, wts, top_k)'s. cols a multiple of 1024 up to 8192.
void rms_norm_q8(const float* x, const float* w, int n_tokens, int cols, float eps, float* out, void* xq, void* stream,
                 const float* wts = nullptr, int top_k = 0);
/// situ_glu_f32 then native_quantize_q8_1 of its output into xq.
void situ_glu_q8(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta, void* xq,
                 void* stream);
/// hca_groups_merge_f32 of src [n_groups, n_tokens, gout], sigmoid_gate_mul_f32 by gate [n_tokens, n_groups * gout],
/// then native_quantize_q8_1 into xq (n_groups = 1: the gate and the quantization of src [n_tokens, gout]).
void merge_gate_q8(const float* src, const float* gate, int n_tokens, int n_groups, int gout, void* xq, void* stream);

/// Final mHC stream contraction: hc_head_fn [4, 4*H], hc_head_base [4], hc_head_scale [1] -> out [n_tokens, H].
void mhc_contract(const float* x, const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                  int n_tokens, int n_embd, float rms_eps, float hc_eps, float* workspace, float* coef4, float* out,
                  void* stream);

// ---- Elementwise & ERNIE-specific building blocks -------------------------------------------------------------

/// Broadcasts single-stream x [n_tokens, n_embd] to 4 identical mHC streams out [n_tokens, 4, n_embd].
void broadcast_streams(const float* x, int n_tokens, int n_embd, float* out, void* stream);

/// In-place FP32 LM-head multimax: logits += t0*relu(r0-x) + t1*relu(x-r1) + t2*relu(r2-x)^2 + t3*relu(x-r3)^2.
void multimax_f32(float* logits, const float* ranges, const float* ts, int n_tokens, int n_vocab, void* stream);

/// SiTU-GLU: out[i] = (beta * tanh(g / beta) * sigmoid(g)) * (linear_beta * tanh(u / linear_beta)).
void situ_glu_f32(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta,
                  float* out, void* stream);

/// Up to kL2Spans device ranges (16-byte aligned) to pull into L2 ahead of their use, by one small kernel.
constexpr int kL2Spans = 8;
struct L2Spans {
    const void* ptr[kL2Spans] = {};
    uint32_t bytes[kL2Spans] = {};
    int n = 0;
};
void l2_prefetch(const L2Spans& spans, void* stream);
/// BF16 bits -> F32 (exact), n values.
void bf16_widen_f32(const uint16_t* in, float* out, int64_t n, void* stream);

/// Split-feature MoE router: logits [n_tokens, 2*E] -> scores = sigmoid(l0) + sigmoid(l1),
/// bias-corrected top-K (with optional GPU-residency prior on selection only), optional renorm,
/// multiplied by learnable per-expert scale[id].
void split_sigmoid_route(const float* logits, const float* bias, const float* scale, int n_tokens, int n_expert,
                         int top_k, bool renorm, int32_t* ids, float* weights, void* stream,
                         kernels::ResidentTabs held = {}, float route_prior = 0.0f);

/// VHA PostMix: out[t, h, d] = sum_g mix[h, g] * in[t, g, d] (or in[g, t, d] when head_major is true).
void vha_postmix_f32(const float* in, const float* mix, int n_tokens, int n_heads, int head_dim, float* out,
                     void* stream, bool head_major = false);

/// In-place sigmoid output gating: z[i] *= sigmoid(gate[i]).
void sigmoid_gate_mul_f32(float* z, const float* gate, int64_t n, void* stream);

/// Standard LayerNorm (with weight and bias) over rows [n_tokens, dim].
void layernorm_f32(const float* x, const float* weight, const float* bias, int n_tokens, int dim, float eps,
                   float* out, void* stream);

/// In-place orthonormal Walsh-Hadamard transform (dim = 128, scaled by 1/sqrt(128)) on rows [n_rows, 128].
void hadamard_128_f32(float* x, int n_rows, void* stream);

/// Weighted sum of routed expert outputs: out[t, c] = sum_{k=0}^{top_k-1} w[t, k] * exp_out[t * top_k + k, c].
// ---- decode windows' attention with the positions split across blocks ------------------------------------------

constexpr int kAttnPartFloats = 2 + 512;   ///< a split's running max, sum and unnormalized output (head_dim 512)
constexpr int kAttnSplitSlots = 32;        ///< rows x splits at most (a window of up to 8 rows)
/// The splits of a window of `rows` rows (rows <= 8).
inline int attn_splits(int rows) { return rows <= 2 ? 16 : rows <= 4 ? 8 : 4; }
/// hca_attention for a window of up to 8 rows and head_dim 512, the positions split across blocks: `part` holds
/// kAttnSplitSlots * n_heads * kAttnPartFloats floats.
void hca_attention_split(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                         const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int win_size,
                         int ratio, bool fp8, float* out, float* part, void* stream, int win_cap);
/// dsa_mla_attention for a window of up to 8 rows, the selected positions split across blocks: `part` as above and
/// `q_lat` 8 * n_heads * kv_lora floats. With an FP8 cache and the rows' positions `pos` (device), the first
/// min(pos + 1, sel_width) entries of a row's selection are its positions (dsa_indexer_topk_merge packs them) and they
/// spread evenly over the splits. False (nothing launched) for a geometry it does not cover.
bool dsa_mla_attention_split(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                             const int32_t* sel_idx, const int32_t* pos, int n_tokens, int n_heads, int qk_nope,
                             int qk_rope, int kv_lora, int v_dim, int sel_width, bool fp8, float* out, float* part,
                             float* q_lat, void* stream);

// ---- decode windows' routed experts (moe_dec.cu, models/ernie55/decode_moe.hpp) -------------------------------

/// Doorbell words in mapped host memory, accessed atomically at system scope: the ring's release publishes the rows,
/// the choices and a nonzero visit ticket; the host takes the ticket, clears the ring and computes its pairs, and its
/// release store to `done` publishes those rows with that ticket. `abort` is sticky: a visit that sees it does not
/// publish, and a waiting merge gives up.
struct MoeBell {
    alignas(64) uint32_t ring;
    uint32_t ticket;
    alignas(64) uint32_t done;
    uint32_t abort;
};
constexpr int kMoeDecPairs = 128;   ///< a window's (row, slot) pairs at most
constexpr int kMoeHintK = 16;       ///< the experts a hint names per row, the likeliest first
constexpr int kMoeHintRows = 8;     ///< a decode window's rows at most
/// A layer's routing predicted while the layer before it runs (moe_dec_hint), in mapped host memory: `word` is
/// sequence << 16 | layer << 8 | rows, stored with release after the ids of slot `sequence & 1` (row r's at r * kMoeHintK).
struct MoeHint {
    alignas(64) uint32_t word;
    alignas(64) int32_t ids[2][kMoeHintRows * kMoeHintK];
};

/// A decode window's Q8_0 products (kernels/q8x.cu): matrix i, W [n_out][n_in] in ggml Q8_0 blocks, against the
/// window's `ncols` (1..8) activation columns in Q8_1 at x (column j at block j * n_in / 32), into y (column j at
/// j * n_out). The bits of kernels::native_q8_0_mmvq for each matrix; several matrices of one width share a launch,
/// started as a programmatic dependent (pdl.hpp) with its weights read before it waits. n_in a multiple of 1024 up
/// to 16384.
constexpr int kQ8xMats = 8;
struct Q8xMat {
    const void* w = nullptr;
    const void* x = nullptr;
    float* y = nullptr;
    int n_out = 0;
};
void q8x_mmvq(const Q8xMat* mats, int n, int n_in, int ncols, void* stream);
/// The same products by q8x_mmvq where it is the faster, else native_q8_0_mmvq a matrix at a time (the same bits either
/// way). Called right after a kernel on `stream` (q8x starts as its programmatic dependent).
void q8_mmvq(const Q8xMat* mats, int n, int n_in, int ncols, void* stream);

/// The grouped expert kernel's plan from the router's choices: group g is an expert the main card holds (its blob at
/// tab[e], 0 when it does not) and its entries are the pairs that chose it, in row order; entry j reads token tok[j]
/// and writes row dst[j] = r * k + slot. Pairs off the card and dropped ones (id < 0) are left out. With `log`, the
/// choices are copied there too. n_expert <= 512, rows * k <= kMoeDecPairs.
void moe_dec_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* tab,
                  unsigned long long* ptr, int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok,
                  int32_t* log, void* stream);
/// When any pair's expert is off the card: advances `seq` (device), publishes the choices to ids_m, the rows with such
/// a pair to x_m (width floats each) and the ticket, then raises the ring with `layer` and `rows`.
void moe_dec_ring(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* x, int width,
                  int layer, float* x_m, int32_t* ids_m, MoeBell* bell, uint32_t* seq, void* stream);
/// Pair p off the card takes row p of `host_rows` (mapped) into row p of `out` once the bell's done matches `seq`; a
/// dropped pair's row is zeroed; the card's pairs stay.
/// The predicted choices `ids` (rows * kMoeHintK, device) of `layer` into `hint`, then its word; `hseq` (device) counts
/// the hints.
void moe_dec_hint(const int32_t* ids, int rows, int layer, MoeHint* hint, uint32_t* hseq, void* stream);
void moe_dec_merge(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* host_rows,
                   float* out, int width, MoeBell* bell, const uint32_t* seq, void* stream);

void weighted_sum_f32(const float* exp_out, const float* weights, int n_tokens, int top_k, int dim, float* out,
                      void* stream);

/// Mapped weighted sum for prompt MMQ: out[t, c] = sum_{k=0}^{top_k-1} w[t, k] * pair_rows[pair_map[t * top_k + k], c].
void weighted_sum_mapped_f32(const float* pair_rows, const int32_t* pair_map, const float* weights, int n_tokens,
                             int top_k, int dim, float* out, void* stream);

/// Splits group `grp` out of `[T, n_groups, gin]` into contiguous `[T, gin]`.
void hca_group_split_f32(const float* src, int n_tokens, int n_groups, int grp, int gin, float* dst, void* stream);

/// Every group at once: `[T, n_groups, gin]` into `[n_groups, T, gin]` (gin % 4 == 0).
void hca_groups_split_f32(const float* src, int n_tokens, int n_groups, int gin, float* dst, void* stream);
/// Every group at once: `[n_groups, T, gout]` into `[T, n_groups, gout]` (gout % 4 == 0).
void hca_groups_merge_f32(const float* src, int n_tokens, int n_groups, int gout, float* dst, void* stream);

/// Merges contiguous `[T, gout]` into group `grp` of `[T, n_groups, gout]`.
void hca_group_merge_f32(const float* src, int n_tokens, int n_groups, int grp, int gout, float* dst, void* stream);

/// Residual add: dst[i] += src[i].
void add_f32(float* dst, const float* src, int64_t n, void* stream);

// ---- HCA Attention (37 layers) --------------------------------------------------------------------------------

/// Weightless per-head RMSNorm on q_b [n_tokens, n_heads, head_dim] + forward interleaved RoPE on last 64 dims.
void hca_q_rope(float* q, const int32_t* pos, int n_tokens, int n_heads, int head_dim, float rms_eps,
                const RopeTable& rt, void* stream);

/// RMSNorm on kv [n_tokens, 512] + forward interleaved RoPE on last 64 dims -> kv_rot [n_tokens, 512] (in-place safe).
void hca_kv_norm_rope(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                      float rms_eps, const RopeTable& rt, float* kv_rot, void* stream);

/// Commits the trailing min(n_tokens, win_cap) rows of kv_rot into sliding-window ring [win_cap, 512] (race-free).
/// `win_cap <= 0` defaults to `win_size`. Using `win_cap >= win_size + kMaxDecodeWindow` makes speculative rollback lossless.
void hca_win_commit(const float* kv_rot, const int32_t* pos, int n_tokens, int head_dim, int win_size,
                    uint8_t* win_cache, bool fp8, void* stream, int win_cap = 0);

/// RMSNorm on kv [n_tokens, 512] + forward interleaved RoPE on last 64 dims, writing into sliding-window ring [win, 512].
void hca_kv_store_window(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                         int win_size, float rms_eps, const RopeTable& rt, uint8_t* win_cache, bool fp8, void* stream);

/// Streaming 128:1 HCA compressor with additive APE[pos % ratio]: updates state [ratio, 512] x 2 and writes closed
/// compressed blocks into comp_cache (RMSNorm + YaRN RoPE at block start position). When `pos0_hint >= 0` (the host
/// knows pos[t] = pos0_hint + t), full ratio-aligned blocks run in parallel across SMs with identical output.
void hca_compress(const float* ckv, const float* cscore, const float* ape, const float* norm_w, const int32_t* pos,
                  int n_tokens, int ratio, int head_dim, float rms_eps, const RopeTable& comp_rt,
                  float* state_kv, float* state_score, uint8_t* comp_cache, bool fp8, void* stream,
                  int64_t pos0_hint = -1);

/// HCA multi-query attention over compressed blocks [0 .. pos/ratio) + recent window [max(0, pos-win+1) .. pos],
/// with per-head attention sink `attn_sink[n_heads]`. When `kv_rot != nullptr`, positions >= pos[0] are read from
/// `kv_rot[(j - pos[0]) * head_dim]` so prefill chunks never race on `win_cache`.
/// `win_cap <= 0` defaults to `win_size`.
void hca_attention(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                   const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int head_dim, int win_size,
                   int ratio, bool fp8, float* out, void* stream, int win_cap = 0);

/// Inverse interleaved RoPE on last 64 dims of VHA-mixed attention output [n_tokens, n_heads, head_dim].
void hca_inv_rope(float* o, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                  void* stream);

/// A decode window's hca_q_rope, hca_kv_norm_rope and hca_win_commit in one launch (the commit before the attention:
/// it reads the window's own rows from kv_rot, and win_cap >= win_size + the window's rows keeps the slots it reads
/// apart from the ones written).
void hca_qkv_rope_commit(float* q, const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                         int n_heads, int head_dim, float rms_eps, const RopeTable& rt, float* kv_rot, int win_cap,
                         uint8_t* win_cache, bool fp8, void* stream);
/// vha_postmix_f32, hca_inv_rope, hca_groups_split_f32 (heads_per_group heads a group) and native_quantize_q8_1 of the
/// groups' columns (column g * n_tokens + t) into xq, in one launch. head_dim a multiple of 128.
void hca_post_rope_q8(const float* in, const float* mix, const int32_t* pos, int n_tokens, int n_heads, int head_dim,
                      int heads_per_group, const RopeTable& rt, void* xq, void* stream);

// ---- MLA-DSA Attention + Lightning Indexer (6 layers) ---------------------------------------------------------

/// Applies interleaved RoPE to q[..., nope:nope+rope] and k_pe [n_tokens, rope], and writes normed kv_c + rotated k_pe
/// into DSA KV cache at positions `pos[0..n_tokens)`.
void dsa_qkv_rope_and_store(float* q, const float* kv_a, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                            int n_heads, int qk_nope, int qk_rope, int kv_lora, float rms_eps, const RopeTable& rt,
                            uint8_t* kv_cache, bool fp8, void* stream);

/// Prepares Lightning Indexer queries (partial NeoX RoPE + Hadamard-128) and keys (LayerNorm + partial NeoX RoPE +
/// Hadamard-128), storing keys into indexer K cache (MXFP4 when fp8, else FP16).
void dsa_indexer_prepare(float* iq, float* ik, const float* k_norm_w, const float* k_norm_b, const int32_t* pos,
                         int n_tokens, int idx_heads, int idx_dim, int rope_dim, float eps, const RopeTable& rt,
                         uint8_t* idx_k_cache, bool fp8, void* stream);

/// Scores older tokens [0 .. pos - win) with Lightning Indexer (ReLU(iq . ik) * weights), selects top-K, and merges
/// with the trailing sliding window [max(0, pos - win + 1) .. pos] into `out_idx [n_tokens, top_k + win]`.
/// `scores_buf` (stride `scores_stride >= max(pos) + 1` per token) caches scores for radix top-K selection.
void dsa_indexer_topk_merge(const float* iq, const float* iweights, const uint8_t* idx_k_cache, const int32_t* pos,
                            int n_tokens, int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                            int32_t* out_idx, void* stream, float* scores_buf, int scores_stride);

/// Dequantizes `n_keys` rows of `idx_k_cache` starting at `k0` (MXFP4 when fp8, else FP16) into BF16 `[n_keys, idx_dim]`.
void dsa_dequant_idx_k_bf16(const uint8_t* idx_k_cache, int k0, int n_keys, int idx_dim, bool fp8, uint16_t* k_bf16,
                            void* stream);

/// Reduces one key tile of cuBLAS dot products `dots [n_tokens * idx_heads, tile_k]` (column-major `[tile_k, n_tokens * idx_heads]`)
/// via ReLU(dot) * iweights / sqrt(idx_dim * idx_heads) into `scores_buf[t * scores_stride + k0 + j]`.
void dsa_indexer_reduce_tile(const float* dots, const float* iweights, const int32_t* pos, int n_tokens, int idx_heads,
                             int idx_dim, int top_k, int win_size, int k0, int tile_k, float* scores_buf,
                             int scores_stride, void* stream);

/// Runs exact 256-bin radix top-K + sliding-window merge on precomputed `scores_buf [n_tokens, scores_stride]`.
void dsa_indexer_select_cached(const float* scores_buf, int scores_stride, const int32_t* pos, int n_tokens, int top_k,
                               int win_size, int32_t* out_idx, void* stream);

/// Sparse MLA attention over selected positions `sel_idx [n_tokens, width]` with `attn_sink[n_heads]`,
/// reading `kv_b` (Q8_0 [n_heads * (qk_nope + v_dim), kv_lora]) or pre-expanded per-head K/V, writing
/// `out [n_tokens, n_heads, v_dim]`.
void dsa_mla_attention(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                       const int32_t* sel_idx, int n_tokens, int n_heads, int qk_nope, int qk_rope, int kv_lora,
                       int v_dim, int sel_width, bool fp8, float* out, void* stream);

/// Coalesced Q8_0 -> FP16 dequantization of `n_blocks` 32-element Q8_0 blocks into `f16_dst [n_blocks * 32]`.
void dequant_q8_0_to_f16(const void* q8_src, uint16_t* f16_dst, int64_t n_blocks, void* stream);

/// Packs q_nope slice `[0..qk_nope)` from `q [n_tokens, n_heads, qk_nope + qk_rope]` FP32 into head-major FP16
/// `q_nope_f16 [n_heads, n_tokens, qk_nope]` for batched cuBLAS Tensor-Core W_UK^T absorption.
void dsa_pack_q_nope_f16(const float* q, uint16_t* q_nope_f16, int n_tokens, int n_heads, int qk_nope, int qk_rope,
                         void* stream);

/// Pure pre-absorbed sparse MLA attention (kv_lora == 512, qk_rope == 64): reads pre-absorbed
/// `q_lat [n_heads, n_tokens, 512]` FP32 and `q_pe` from `q_full [n_tokens, n_heads, qk_nope + 64]`, writing
/// head-major FP16 `o_lat_f16 [n_heads, n_tokens, 512]` for batched cuBLAS Tensor-Core W_UV projection.
void dsa_mla_attention_preabsorbed(const float* q_lat, const float* q_full, const uint8_t* kv_cache,
                                   const float* attn_sink, const int32_t* sel_idx, int n_tokens, int n_heads,
                                   int qk_nope, int qk_rope, int kv_lora, int sel_width, bool fp8,
                                   uint16_t* o_lat_f16, void* stream);

// ---- DSpark Speculative Drafter Kernels -----------------------------------------------------------------------

/// Averages the 4 mHC residual streams `res [n_tokens, 4, n_embd]` into `out[t * out_stride + c]`.
void mhc_stream_mean(const float* res, int n_tokens, int n_embd, float* out, int out_stride, void* stream);

/// RMSNorm + plain interleaved RoPE (last 64 dims) on `kv_in [n_tokens, head_dim]` at positions `pos0 + t`,
/// storing FP16 rows into `ctx_ring_f16[((pos0 + t) % win_cap) * head_dim]`.
void dspark_kv_ctx_store(const float* kv_in, const float* kv_norm_w, int64_t pos0, int n_tokens, int head_dim,
                         int win_cap, float rms_eps, const RopeTable& rt, uint16_t* ctx_ring_f16, void* stream);

/// Bidirectional block + sliding-window context attention for DSpark: each of the `n_block` block rows attends to
/// valid context positions `[max(ctx_start, pos0 - win_size) .. pos0 - 1]` in `ctx_ring_f16` (indexed `% win_cap`)
/// PLUS all `n_block` rows in `block_kv` (FP32), with per-head `attn_sink[n_heads]` and scale `1 / sqrt(head_dim)`.
void dspark_swa_block_attn(const float* q, const uint16_t* ctx_ring_f16, const float* block_kv, const float* attn_sink,
                           int64_t pos0, int64_t ctx_start, int n_block, int n_heads, int head_dim, int win_size,
                           int win_cap, float* out, void* stream);

/// Fused Markov bias addition + ERNIE-5.5 multimax (preserving -inf) + GPU argmax over `n_vocab` logits.
/// `scratch_val` and `scratch_idx` need at least 256 elements each.
void dspark_markov_multimax_argmax(const float* base_logits, const float* markov_bias, const float* mmr,
                                   const float* mmt, int n_vocab, float* scratch_val, int32_t* scratch_idx,
                                   int32_t* out_tok, void* stream);

/// Computes DSpark confidence `sigmoid(dot(conf_w[0:n_embd], hcout_row) + dot(conf_w[n_embd:], markov_embed))`.
void dspark_confidence(const float* conf_w, const float* hcout_row, const float* markov_embed, int n_embd, int rank,
                       float* out_conf, void* stream);

}  // namespace basalt::models::ernie55
