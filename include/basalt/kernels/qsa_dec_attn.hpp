// include/basalt/kernels/qsa_dec_attn.hpp - decode attention over the selected cells for a few queries (T <= 8),
// with the arithmetic of qsa_decode_attn_batch bit for bit (same 64-cell chunks, same per-chunk sums and merge order).
//
// One block per (chunk, kv head, query) stages the chunk's K and V rows, their scales and the 12 query heads in
// shared memory with asynchronous copies, so the block pays one memory round trip instead of one per cell.  The merge
// walks only the chunks the query's width reaches.  `fuse_gate`: the merge also applies the sigmoid gate and writes
// the gated rows (native_qsa_gate_apply's arithmetic) next to the plain output.
#pragma once

#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace basalt::kernels {

/// The pools this kernel reads: FP16 or INT8 (not Q4_0, not hybrid); false sends the caller to qsa_decode_attn_batch.
bool qsa_dec_attn_supported(const QsaAttnPools& pools, const QsaShapes& s);

/// qsa_decode_attn_batch's contract and scratch (qsa_decode_attn_scratch_floats(cap, s) floats per query).
/// q_full / attn_gated non-null: also attn_gated = attn * sigmoid(gate half of q_full) per head.
/// xq non-null (with the gated rows): the merge also writes the gated rows' q8_1 blocks (native_quantize_q8_1's).
/// merge_after non-null: a cudaEvent_t the merge also waits for (the gate half of q_full written on another stream).
void qsa_dec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                  const float* q_full = nullptr, float* attn_gated = nullptr, void* xq = nullptr,
                  void* merge_after = nullptr);

}  // namespace basalt::kernels
