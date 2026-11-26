// src/models/glm5next/kernels/idx_score.hpp - see idx_score.cu: the DSA indexer scores each pool from the token's
// multi-head query and the pooled keys. For token t and pool p:
//
//   acc[h] = sum_d iq[t][h][d] * pooled[p][d]                     (the head's dot, d in head_dim order)
//   score[t][p] = sum_h fmax(acc[h], 0) * (w[t][h] * w_scale)     (sum over heads, h ascending)
//
// iq = indexer_attn_q_b(q_resid) reshaped [n_head][head_dim] (nope - no rotary), w = indexer_proj(x) (the learned head
// weights), and the model folds 1/sqrt(head_dim * n_head) into w_scale (llama.cpp glm5-next: weights scaled by
// 1/sqrt(n_embd_indexer * n_indexer_head), then the indexer sums relu(q.k)*w). A pool p is visible to token t when
// p < (pos[t] + 1) / pool - the pool of the token's own position is incomplete and not yet scored; a non-visible pool
// scores -INFINITY so the selection drops it. The result for a token does not depend on the other tokens.
//
// All buffers fp32 device pointers. `stream` may be null (the call then synchronises).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {

/// The model's indexer: n_head = 32, head_dim = 128, pool = 4.
///   iq      [n_tokens][n_head][head_dim] the token's query heads
///   w       [n_tokens][n_head]           the head weights (before w_scale)
///   pooled  [n_pool][head_dim]           all pools' pooled keys (from kpool), shared across tokens
///   pos     [n_tokens] int32             each token's position
///   scores  [n_tokens][n_pool]           score, or -INFINITY where the pool is not visible
void idx_score(const float* iq, const float* w, const float* pooled, const int32_t* pos, float* scores, int n_tokens,
               int n_pool, int n_head, int head_dim, int pool, float w_scale, cudaStream_t stream);

}  // namespace basalt::models::glm5next
