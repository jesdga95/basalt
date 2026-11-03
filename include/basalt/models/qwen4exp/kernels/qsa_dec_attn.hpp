// include/basalt/models/qwen4exp/kernels/qsa_dec_attn.hpp - decode attention over the selected cells for a few queries (T <= 8),
// with the arithmetic of qsa_decode_attn_batch bit for bit (same 64-cell chunks, same per-chunk sums and merge order).
//
// One block per (chunk, KV head, query) stages K and V in shared memory with asynchronous copies. It reads
// the query heads into registers. The merge walks only the chunks the query's width reaches and can apply
// the sigmoid gate with native_qsa_gate_apply's arithmetic.
#pragma once

#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace basalt::kernels {

/// Supports FP16, INT8 and Q4_0 pools. Use qsa_decode_attn_batch for hybrid pools.
bool qsa_dec_attn_supported(const QsaAttnPools& pools, const QsaShapes& s);

/// qsa_decode_attn_batch's contract and scratch (qsa_decode_attn_scratch_floats(cap, s) floats per query).
/// K and V pool bases must be 16-byte aligned for the asynchronous row copies.
/// q_full / attn_gated non-null: also attn_gated = attn * sigmoid(gate half of q_full) per head.
/// xq non-null (with the gated rows): the merge also writes the gated rows' q8_1 blocks (native_quantize_q8_1's).
/// merge_after non-null: a cudaEvent_t the merge also waits for (the gate half of q_full written on another stream).
void qsa_dec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                  const float* q_full = nullptr, float* attn_gated = nullptr, void* xq = nullptr,
                  void* merge_after = nullptr);

}  // namespace basalt::kernels
