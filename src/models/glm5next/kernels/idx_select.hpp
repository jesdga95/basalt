// src/models/glm5next/kernels/idx_select.hpp - see idx_select.cu: top-K pool selection for the DSA indexer.
//
// From each token's row of pool scores (produced by idx_score, -INFINITY on pools the token can't see), keep the K
// highest finite scores. Ties go to the lower pool index. The kept pool ids are written in ascending order; the row is
// padded with -1 to K, and out_n[t] is the number kept (never more than K, and only the finite scores count). The
// selection is order-independent as a set - ascending output is what the downstream pool -> token-cell expansion wants.
//
// All buffers fp32/int32 device pointers. `stream` may be null (the call then synchronises).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {

///   scores [n_tokens][n_pool]  pool scores, -INFINITY where not visible
///   ids    [n_tokens][K]       selected pool ids, ascending, -1 after them
///   out_n  [n_tokens]          number of pools kept for each token
/// K must not exceed kIdxTopkMax.
constexpr int kIdxTopkMax = 2048;

void idx_select(const float* scores, int32_t* ids, int32_t* out_n, int n_tokens, int n_pool, int K,
                cudaStream_t stream);

}  // namespace basalt::models::glm5next
