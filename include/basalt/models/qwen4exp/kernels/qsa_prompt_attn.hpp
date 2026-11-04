// include/basalt/models/qwen4exp/kernels/qsa_prompt_attn.hpp - the prompt path's QSA attention on tensor cores.
//
// `qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel: FP32 dot products with a warp
// reduction per head and cell, 64-cell chunks whose partial sums go through global scratch to a second (merge)
// kernel. At long context that is a large share of the prompt, and it is limited by
// instruction issue, not memory.
//
// This kernel keeps the same per-query selection (no masking, no union) and reads the same pools, but:
//   * one block per (query, KV head) walks all its selected cells in chunks of 32 with an online softmax: no
//     split-K scratch and no merge kernel;
//   * q.k and p.v are m16n8k16 FP16 MMAs with FP32 accumulation over the 12 query heads of the KV head (+4 pad rows);
//   * the stored values enter exactly: int8 codes are exact in FP16 and their per-64 scales are applied in FP32
//     (to the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is;
//   * fp16 and Q4_0 KV: q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: the
//     result differs from the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
//   * int8 KV: one FP16 MMA with FP32 accumulation (q and p rounded to FP16, the V scale folded into the FP16 B
//     operand). Each warp owns whole 16-cell tiles with its own online softmax and the four are merged once at the
//     end, so there are no block syncs in the cell loop.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. Q4_0 KV (mode 4): each block's codes enter
// exactly as int8 and its scale (one per 32 values) in FP32, as int8 KV's do (PROMPT_ATTN_Q4=0: the old kernel).
#pragma once

#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace basalt::kernels {

/// Same arguments and output as `qsa_decode_attn_batch` minus the scratch. Returns false (nothing launched) when the
/// geometry is not 24 heads / 2 KV heads / 256: the caller then uses the old kernel.
bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

}  // namespace basalt::kernels
