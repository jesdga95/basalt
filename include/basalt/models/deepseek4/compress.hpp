#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek4 {

enum class CompressKind : int { Csa = 0, Hca = 1 };

/// V4 compressed attention state. Projection rows are [tokens][width], with width 1024 for CSA and
/// 512 for HCA. `ape` is [ratio][width] and is added to score projections before the rows enter state.
/// State rows are [2*ratio][width] for CSA and [ratio][width] for HCA. For each token that closes a
/// group, the normalized and rotated compressed row is written to `out` at `pos/ratio`; `row_of`
/// receives that row index, or -1 for a token that did not close a group.
void compress_projected(CompressKind kind, int ratio, int n_tokens, const float* kv, const float* score,
                        const float* ape, const int32_t* pos, const float* norm_w, float rms_eps,
                        const float* rope_freq, float* state_kv, float* state_score, float* out,
                        int32_t* row_of, void* stream);

/// Projects the layer input with the checkpoint's BF16 KV/gate matrices, then updates the V4 compressed cache.
/// Weights are [width][n_embd], where width is 1024 for CSA and 512 for HCA; `kv_projected` and `score_projected`
/// each hold [n_tokens][width] float workspace. The projection matches the source's FP32 activation path.
void project_and_compress(CompressKind kind, int ratio, int n_embd, int n_tokens, const float* x,
                          const uint16_t* kv_weight, const uint16_t* score_weight, const float* ape,
                          const int32_t* pos, const float* norm_w, float rms_eps, const float* rope_freq,
                          float* kv_projected, float* score_projected, float* state_kv, float* state_score,
                          float* out, int32_t* row_of, void* stream);

/// Zeroes only the rolling partial-group state. Call at the start of a sequence.
void reset_compress_state(CompressKind kind, int ratio, float* state_kv, float* state_score, void* stream);

/// The 128-wide ratio-4 compressor owned by V4's sparse-attention indexer. Its cache uses the source's
/// BF16 -> Hadamard -> FP4 quant/dequant path and does not share the attention compressor's FP8 rows.
constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerRatio = 4;
void indexer_compress_projected(int n_tokens, const float* kv, const float* score, const float* ape,
                                const int32_t* pos, const float* norm_w, float rms_eps, const float* rope_freq,
                                float* state_kv, float* state_score, float* out, int32_t* row_of, void* stream);
void reset_indexer_compress_state(float* state_kv, float* state_score, void* stream);

}  // namespace basalt::models::deepseek4
