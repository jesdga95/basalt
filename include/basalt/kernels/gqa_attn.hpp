// include/basalt/kernels/gqa_attn.hpp - causal grouped-query attention over an fp16 KV cache, head width 256.
//
// The cache holds one layer's cells as [cell][n_kv][256] fp16, K and V apart; query heads h and h + 1 ... share KV
// head h / (n_head / n_kv). Row t of a window or chunk sits at cell pos0 + t and sees cells 0 .. pos0 + t.  q is fp32
// [T][n_head][256], already normed and rotated; out fp32 [T][n_head][256]; scale 1/16.
//
//   gqa_decode_attn   a decode window (T <= 8): cells split in chunks over the grid, fp32 dot products, the chunks'
//                     partial softmax sums merged by a second kernel.  A row's output does not depend on T.
//   gqa_prompt_attn   a prompt chunk: 64-query tiles per head on fp16 tensor cores (fp32 accumulation, online softmax).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kGqaMaxWindow = 8;

/// Scratch floats gqa_decode_attn needs for up to `max_cells` cells and `T` rows.
size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head);
void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream);
void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream);

}  // namespace basalt::kernels
