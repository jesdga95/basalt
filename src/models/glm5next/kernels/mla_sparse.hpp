// src/models/glm5next/kernels/mla_sparse.hpp - see mla_sparse.cu: the nope sparse-MQA attention core of GLM-5.3-Flash.
//
// It scores the token's already-absorbed queries against the shared latent cache over only the selected cells, softmaxes
// per head, and returns the weighted latent. The value/output projections (wv_b, wo) and every dense projection are
// engine mul_mat - they are not this kernel's concern. All 64 heads read the same latent per cell (MQA); the cache holds
// one row per cell that is simultaneously the key and the value (the architecture is nope, so there is no rope half).
//
// Buffer layouts (fp32, contiguous unless noted):
//   q_abs    [n_tokens][n_head][512]  absorbed queries (engine: rmsnorm, wq_a/wq_b, wk_b absorb), latent-major innermost
//   latent   [n_cells][512]           the KV cache; row c is cell c's latent (key == value), 512 floats stride
//   ids      [n_tokens][n_sel] int32  the cells each token attends (any order); -1 is a hole (masked)
//   cache_len[n_tokens] int32          cells of `latent` valid for that token (causal); id >= cache_len[t] is masked
//   out      [n_tokens][n_head][512]  the weighted latent, pre value-projection
// `scale` is 1/sqrt(n_embd_head_k_mla). `workspace` must hold at least mla_sparse_workspace_floats floats.
//
// `stream` may be null (the call then synchronises). n_head must be 8 or 16 divisible (64 works); n_sel >= 1.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace basalt::models::glm5next {

void mla_sparse(const float* q_abs, const float* latent, const int32_t* ids, const int32_t* cache_len, float* out,
                float scale, int n_tokens, int n_head, int n_sel, float* workspace, cudaStream_t stream);

size_t mla_sparse_workspace_floats(int n_tokens, int n_head, int n_sel);

}  // namespace basalt::models::glm5next
