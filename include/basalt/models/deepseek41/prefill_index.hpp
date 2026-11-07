// include/basalt/models/deepseek41/prefill_index.hpp - DeepSeek-V4.1 attention, index side, for prompt chunks of
// hundreds of tokens per call: a staged window, a batched compressor and a batched indexer.
//
// Same maths as attn_index.hpp (fp32 activations, packed caches, bf16 weights); the projections are tensor-core GEMMs
// that keep the activations at fp32 accuracy by splitting them into two bf16 halves.
#pragma once

#include "basalt/models/deepseek41/attn_index.hpp"

namespace basalt::models::deepseek41 {

/// y[t][o] = dot(w[o][0..in), x[t][0..in)) for any T; w is bf16, in a multiple of 32.
void gemm_bf16(const uint16_t* w, const float* x, int n_tokens, int in, int out, float* y, void* stream);

/// Copies the window rows of positions [p_lo, p_lo + count) from the decode ring (slot p % kWindowSlots) to the
/// staging ring (slot p % stage_cap, [stage_cap][win_row_bytes(fmt)]).
void window_stage_copy(const uint8_t* ring, uint8_t* stage, int stage_cap, int64_t p_lo, int count, CacheFormat fmt,
                       void* stream);

/// kv_window for n_tokens consecutive positions: every row goes to the staging ring, and the rows of tokens
/// >= ring_from also go to the decode ring (so the ring ends as a token-by-token run would leave it).
void kv_window_stage(const float* kv_raw, const float* norm_w, const int32_t* pos, int n_tokens, float rms_eps,
                     const float* rope_freq, uint8_t* stage, int stage_cap, uint8_t* ring, int ring_from, CacheFormat fmt,
                     void* stream);

/// Bytes of workspace compress_batch needs for up to max_tokens.
size_t compress_batch_workspace_bytes(int max_tokens);

/// compress() for n_tokens consecutive positions in one launch per stage.
void compress_batch(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace,
                    void* stream);

/// Bytes of workspace indexer_select_batch needs for up to max_tokens tokens and max_comp compressed rows.
size_t indexer_batch_workspace_bytes(int max_tokens, int max_comp);

/// indexer_select() for any number of tokens per call (same arguments and results).
void indexer_select_batch(const IndexerArgs& a, const float* qr, const float* x, const int32_t* pos, int n_tokens,
                          void* workspace, void* stream);

}  // namespace basalt::models::deepseek41
