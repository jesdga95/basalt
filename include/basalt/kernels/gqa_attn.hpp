// include/basalt/kernels/gqa_attn.hpp - causal grouped-query attention over an fp16 KV cache, head width 256.
//
// The cache holds one layer's cells as [cell][n_kv][256] fp16, K and V apart; query heads h and h + 1 ... share KV
// head h / (n_head / n_kv). Row t of a window or chunk sits at cell pos0 + t and sees cells 0 .. pos0 + t.  q is fp32
// [T][n_head][256], already normed and rotated; out fp32 [T][n_head][256]; scale 1/16.
//
//   gqa_decode_attn   a decode window (T <= 8): cells split in chunks over the grid, fp32 dot products, the chunks'
//                     partial softmax sums merged by a second kernel.  A row's output does not depend on T.
//   gqa_prompt_attn   a prompt chunk.  An int8 cache: blocks of 128 positions (64 for narrow launches) of one query
//                     head, a warp per 16 positions for the whole sweep; 64-cell K/V tiles as raw codes, QK^T on int8
//                     tensor cores (q rounded to int8 per 64 values), V decoded to fp16 in registers, P V on fp16
//                     tensor cores summed in fp16 per tile; with split scratch, the cells of a launch whose blocks
//                     would leave SMs idle split over more blocks and merged.  An fp16 cache: blocks of 96 rows
//                     (positions x the KV group's query heads), QK^T on fp16 tensor cores with fp32 accumulation, P V
//                     summed in fp16 per 32-cell tile.  Online softmax and the output in fp32.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kGqaMaxWindow = 16;

/// An int8 K/V cache in place of the fp16 one: codes [cell][n_kv][256], fp16 scales [cell][n_kv][4] (one per 64 values,
/// amax / 127, codes rounded to nearest).  `k` null: the fp16 cache.
struct KvInt8 {
    int8_t* k = nullptr;
    int8_t* v = nullptr;
    uint16_t* ks = nullptr;
    uint16_t* vs = nullptr;
};

/// Scratch floats gqa_decode_attn needs for up to `max_cells` cells and `T` rows (gqa_window_attn: cells its window
/// spans plus the window's rows).
size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim = 256);
/// `pos_dev` (optional): pos0 read from the device instead, the grid sized for `max_cells` (a captured window).
void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                     const int32_t* pos_dev = nullptr, KvInt8 q8 = {}, const float* gate_q = nullptr,
                     uint16_t* gated16 = nullptr);
/// (gate_q, gated16: also each head's output times sigmoid of its gate, q_full [T][n_head][512]'s second half, as bf16
/// rows [T][n_head * 256] into gated16 (bitwise attn_gate, then rounded); `out` may then be null.)
/// gqa_decode_attn for a block drafter's rows: head width 128 or 256, row t at position pos0 + t sees positions
/// [pos0 + t - window + 1, hi) (not causal), position p in cache cell p % cap.
void gqa_window_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv, int head_dim,
                     int64_t pos0, int T, int64_t hi, int64_t window, int64_t cap, float* scratch, int64_t scratch_cells,
                     float* out, void* stream);
/// Scratch floats gqa_prompt_attn needs to split a T-row chunk's cells `splits` ways (an int8 cache).
size_t gqa_prompt_split_floats(int64_t T, int n_head, int splits);
/// `split_scratch` (optional, `split_floats` floats): lets an int8-cache launch split its cells over more blocks.
void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8 = {}, float* split_scratch = nullptr,
                     size_t split_floats = 0);

}  // namespace basalt::kernels
