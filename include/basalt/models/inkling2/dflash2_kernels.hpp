// include/basalt/models/inkling2/dflash2_kernels.hpp - Inkling 2's DFlash 2 block-diffusion drafter kernels:
// target tap BF16 packing, dynamic two-tap group-16 causal convolutions (including fused post-conv residual + RMSNorm +
// BF16 packing), block QK norm + RoPE + ring commit, fused draft-row RMSNorm + BF16/Q8_1 packing, barrier-minimal exact
// Q8-rescored Top-16 + full-vocab tail LSE extraction, and 16-warp 128-bit vectorized bilinear candidate selection.
#pragma once

#include "basalt/models/inkling2/dspark_kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int kDf2Block = 8;
constexpr int kDf2Drafts = kDf2Block - 1;
constexpr int kDf2Targets = 4;
constexpr int kDf2MaxLayers = 8;
constexpr int kDf2Heads = 32;
constexpr int kDf2KvHeads = 8;
constexpr int kDf2HeadDim = 128;
constexpr int kDf2Q = kDf2Heads * kDf2HeadDim;       // 4096
constexpr int kDf2KV = kDf2KvHeads * kDf2HeadDim;    // 1024
constexpr int kDf2ConvGroup = 16;
constexpr int kDf2ConvGroups = 2304 / kDf2ConvGroup; // 144
constexpr int kDf2Dyn = 4 * kDf2ConvGroups;          // 576
constexpr int kDf2Rank = 256;
constexpr int kDf2TopK = 16;
constexpr int kDf2Rescore = 64;

/// Gathers the four slot-major target feature taps (tap + k * tap_stride + r * kWidth, k = 0..3) of `rows` rows
/// directly into BF16 [rows][4 * kWidth] activations for Tensor Core wgemm.
void dflash2_pack_tap_bf16(const float* tap, int64_t tap_stride, int rows, void* act_bf16, void* stream);

/// Gathers the four slot-major target feature taps of `rows` rows directly into [rows][4 * kWidth] Q8_1 blocks.
void dflash2_pack_tap_q8_1(const float* tap, int64_t tap_stride, int rows, void* act_q8_1, void* stream);

/// Splits fc.weight ([kWidth][4 * kWidth] Q8_0, 288 blocks/row) into four contiguous [kWidth][kWidth] Q8_0 submatrices
/// (72 blocks/row each) so prompt chunks project each tap slot in place without extra concatenation scratch.
void dflash2_split_fc_q8(const void* fc_q8_full, void* fc_q8_parts, void* stream);

/// Computes the target-normed mask embedding row once at load: mask_out = rms_norm(embedding[mask_id], eps) * embed_norm.
void dflash2_make_mask(const uint16_t* embedding, int32_t mask_id, const float* embed_norm, float eps, float* mask_out,
                       void* stream);

/// Fills `count` blocks of `block` input rows from device `meta` ({anchor, pos0} per slot):
/// row 0 = rms_norm(embedding[anchor], eps) * embed_norm, rows 1..block-1 = `mask`.
void dflash2_block_inputs_batch(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps,
                                const float* mask, int block, int count, float* x, void* stream);

/// Pre-attention / pre-MLP dynamic 2-tap group-16 causal convolution over `count` blocks of `block` rows, packed to BF16.
void dflash2_conv2_pack(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block, int count,
                        void* act_bf16, float* y_opt, void* stream);

/// Post-attention / post-MLP dynamic 2-tap group-16 causal convolution added into `x`.
void dflash2_conv2_add(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block, int count,
                       float* x, void* stream);

/// Fused post-conv residual add (`x += conv_post(z)`) + next sublayer's RMSNorm (`xn_out = rms_norm(x, norm_w, eps)`) +
/// BF16 activation pack (`act_bf16`) in one single pass.
void dflash2_conv2_add_norm_pack(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block,
                                 int count, float* x, const float* norm_w, float eps, void* act_bf16, float* xn_out,
                                 void* stream);

/// Per-head RMSNorm + plain NeoX RoPE (32 query heads, 8 KV heads, dim 128) for `count` blocks of `block` rows, reading
/// each slot's start position from `meta[2 * s + 1]` and writing K, V into each slot's ring at (pos0 + t) % cap.
void dflash2_block_qk_prep_batch(const float* q, const float* kb, const float* vb, const float* q_norm, const float* k_norm,
                                 float eps, float rope_base, const int32_t* meta, int block, int count, float* q_out,
                                 const DsBatchKv& kv, int64_t cap, void* stream);

/// RMSNorm of draft rows t = 1..n_drafts (skipping row 0, the anchor) of `count` blocks in `x` ([count][block][kWidth]),
/// packed in a single pass to BF16 `act_bf16` and Q8_1 `hq_q8_1`, with optional fp32 output `hd_opt`.
void dflash2_norm_draft_pack(const float* x, const float* norm_w, float eps, int block, int count, int n_drafts,
                             void* act_bf16, void* hq_q8_1, float* hd_opt, void* stream);

/// Scratch bytes `dflash2_top16` needs for `max_rows` draft rows over `n_vocab` columns (holds chunk candidates + LSE).
size_t dflash2_top16_scratch_bytes(int max_rows, int n_vocab);

/// Extracts the sorted Top-16 (token id, raw logit) pairs per draft row into `cand_out` ([rows][32]: 16 ids then 16 raw
/// logit bit patterns, ties broken to lower token id) and writes each row's softcapped tail log-sum-exp `(m_tail, l_tail)`
/// over the remaining `n_vocab - 16` tokens into `tail_lse_out` ([rows * 2] floats, optional). Row r's logits at
/// logits + r * ld.
void dflash2_top16(const float* logits, const void* head_q8, bool head_planes, const void* hq, int rows, int n_vocab,
                   int64_t ld, float cap, void* scratch, int32_t* cand_out, float* tail_lse_out, void* stream);

/// Walks the bilinear candidate selector across `n_drafts` steps for `count` slots using 16 warps (one warp per candidate,
/// 128-bit uint4 codebook loads). Applies after-correction softcapping `cap * tanh(raw_sc / cap)`; when `tail_lse` is
/// non-null, `conf_out` is the full-vocabulary calibrated probability `exp(top) / (sum_16 exp(capped) + Z_tail)`.
void dflash2_select(const int32_t* meta, int meta_stride, const int32_t* cand, const float* tail_lse, const float* g,
                    const uint16_t* pred, const uint16_t* succ, int count, int n_drafts, float cap, int32_t* ids_out,
                    int ids_stride, float* conf_out, int conf_stride, void* stream);

}  // namespace basalt::models::inkling2
