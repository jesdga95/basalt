// src/kernels/cuda/route_tail_skip.cu - see basalt/kernels/route_lossy.hpp.
#include "basalt/kernels/route_lossy.hpp"

#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <cstdlib>

namespace basalt::kernels {
namespace {

// One block, one thread per window entry.
__global__ void __launch_bounds__(kRouteTailSkipMax) route_tail_skip_kernel(int32_t* __restrict__ ids, float* __restrict__ w,
                                                                           ResidentTabs held, int n, int k, int R,
                                                                           int n_expert, unsigned long long* counts) {
    __shared__ int32_t s_ids[kRouteTailSkipMax];
    __shared__ uint8_t s_tail[kRouteTailSkipMax];   // the entry's weight rank in its token is R or lower
    const int tid = threadIdx.x;
    bool miss = false;
    if (tid < n) {
        const int32_t e = ids[tid];
        s_ids[tid] = e;
        s_tail[tid] = route_weight_rank(w, tid, k) >= R;
        miss = e >= 0 && e < n_expert && !held.held(e);
    }
    __syncthreads();
    const bool drop = tid < n && route_tail_drop(s_ids, s_tail, n, tid, miss);
    const int n_miss = __syncthreads_count(miss), n_drop = __syncthreads_count(drop);
    if (drop) {
        ids[tid] = -1;
        w[tid] = 0.0f;
    }
    if (counts != nullptr && tid == 0) {
        atomicAdd(&counts[0], (unsigned long long) n);
        atomicAdd(&counts[1], (unsigned long long) n_miss);
        atomicAdd(&counts[2], (unsigned long long) n_drop);
    }
}

}  // namespace

void route_tail_skip(int32_t* ids, float* weights, const ResidentTabs& held, int n_entries, int k, int rank, int n_expert,
                     unsigned long long* counts, void* stream) {
    if (rank < 1 || rank >= k || n_entries < 1 || n_entries > kRouteTailSkipMax || k < 1) return;
    route_tail_skip_kernel<<<1, kRouteTailSkipMax, 0, (cudaStream_t) stream>>>(ids, weights, held, n_entries, k, rank,
                                                                               n_expert, counts);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "route_tail_skip: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
