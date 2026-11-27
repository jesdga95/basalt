// src/models/glm5next/kernels/cell_expand.cu - see cell_expand.hpp: selected pools -> token-cell ids.
//
// One block per token; a thread per output slot walks n_sel. Each pool slot gathers its member cell from pool_idxs
// (pool-major), mapping the cache's pad sentinel and a dead (past-visible) pool slot to -1; tail slots gather tail_idxs.
// Pure index arithmetic, so a CPU oracle reproduces it exactly.
#include "models/glm5next/kernels/cell_expand.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {
namespace {

__global__ void cell_expand_kernel(const int32_t* __restrict__ top_pools, const int32_t* __restrict__ pool_idxs,
                                   const int32_t* __restrict__ tail_idxs, int32_t sentinel, int32_t* __restrict__ ids_out,
                                   int n_pool, int kpool, int n_top_pool, bool use_tail, int n_sel) {
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) {
        int32_t cell = -1;
        if (j < kpool * n_top_pool) {
            const int ps = j / kpool, s = j % kpool;
            const int32_t pool = top_pools[(size_t) t * n_top_pool + ps];
            if (pool >= 0 && pool < n_pool) {
                const int32_t m = pool_idxs[(size_t) s * n_pool + pool];
                cell = (m != sentinel) ? m : -1;
            }
        } else if (use_tail) {
            const int ts = j - kpool * n_top_pool;
            const int32_t m = tail_idxs[(size_t) ts * gridDim.x + t];
            cell = (m != sentinel) ? m : -1;
            // A tail cell in a pool this token already attends is a duplicate: the pool expansion emits all of that
            // pool's member cells, and the tail's cells always live in the token's own pool. Attending one twice
            // doubles its weight in the softmax - the freshest kpool-1 rows of the context, which is most of a short
            // one and vanishes relative to a long one. A duplicate becomes a masked slot, not a dropped one: n_sel is
            // the window's contract.
            if (cell >= 0) {
                const int32_t pool = cell / kpool;
                for (int ps = 0; ps < n_top_pool; ++ps)
                    if (top_pools[(size_t) t * n_top_pool + ps] == pool) {
                        cell = -1;
                        break;
                    }
            }
        }
        ids_out[(size_t) t * n_sel + j] = cell;
    }
}

}  // namespace

void cell_expand(const int32_t* top_pools, const int32_t* pool_idxs, const int32_t* tail_idxs, int32_t sentinel,
                 int32_t* ids_out, int n_tokens, int n_pool, int kpool, int n_top_pool, bool use_tail,
                 cudaStream_t stream) {
    // No pool is not the same as nothing to write: before a sequence has filled its first pool, the tail is the whole
    // attended set, and ids_out has to say so - the buffer outlives the window, and slots left alone carry the cells of
    // whatever window ran here last. Only kpool and n_tokens can make the work empty.
    if (n_tokens <= 0 || kpool <= 0) return;
    const int n_sel = cell_expand_n_sel(kpool, n_top_pool, use_tail);
    const int threads = 128;
    cell_expand_kernel<<<n_tokens, threads, 0, stream>>>(top_pools, pool_idxs, tail_idxs, sentinel, ids_out, n_pool,
                                                         kpool, n_top_pool, use_tail, n_sel);
}

}  // namespace basalt::models::glm5next
