// src/models/glm5next/kernels/cell_expand.hpp - see cell_expand.cu: turn the selected pools into the token-cell list the
// sparse MLA gathers (the compact form of the reference's sel_idx, before it is scattered into a dense mask).
//
// For token t, the top pools (from idx_select, -1 where the token has fewer visible pools than slots) each expand to
// their kpool member cells, then the incomplete tail (kpool - 1 recent cells) is appended. A pool's member cells come
// from pool_idxs (laid out [kpool][n_pool], pool-major: member s of pool p is pool_idxs[s*n_pool + p]); a member that
// the cache padded, a pool slot past the token's visible count, or a missing tail cell carries `sentinel` and becomes a
// -1 (masked) slot. A tail cell whose pool is among the token's selected pools is masked too: that pool already
// contributed the cell, and a second copy would double its weight in the softmax. The order of the slots is
// pool-slot-major, member-minor - the set, not the order, is what the attention's softmax depends on.
//
// Output ids [n_tokens][n_sel] (int32) feeds mla_sparse's ids; n_sel = kpool*n_top_pool + (use_tail ? kpool-1 : 0).
// All buffers device pointers. `stream` may be null (the call then synchronises).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {

void cell_expand(const int32_t* top_pools, const int32_t* pool_idxs, const int32_t* tail_idxs, int32_t sentinel,
                 int32_t* ids_out, int n_tokens, int n_pool, int kpool, int n_top_pool, bool use_tail,
                 cudaStream_t stream);

inline int cell_expand_n_sel(int kpool, int n_top_pool, bool use_tail) {
    return kpool * n_top_pool + (use_tail ? kpool - 1 : 0);
}

}  // namespace basalt::models::glm5next
