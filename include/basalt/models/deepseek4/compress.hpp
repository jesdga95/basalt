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

/// Zeroes only the rolling partial-group state. Call at the start of a sequence.
void reset_compress_state(CompressKind kind, int ratio, float* state_kv, float* state_score, void* stream);

}  // namespace basalt::models::deepseek4
