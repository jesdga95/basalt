// src/models/glm5next/kernels/idx_select.cu - see idx_select.hpp: top-K pool selection.
//
// One block per token. It finds the K-th highest finite score with a four-pass 8-bit radix select over the order-key
// of each score (histogram the high byte, walk from the top to the bin that holds the K-th, refine), collects every id
// whose key is strictly above that threshold, then the lowest-index ids whose key equals it, until K are held. The ids
// are then bitonic-sorted ascending. This is the same algorithm the DSA top-k uses elsewhere in the tree; it is written
// here self-contained (this model does not link another model's kernels) and the parity oracle checks only the output
// contract - the set of K highest scores, ties to the lower index, ids ascending, -1 padded - which is independent of
// how the block got there.
#include "models/glm5next/kernels/idx_select.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t order_key(float f) {
    if (f == 0.f) f = 0.f;  // -0 and +0 tie
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Exclusive scan of a 0/1 flag over the block; `total` is the block sum. wsum >= 32 ints.
__device__ int block_scan(int flag, int& total, int* wsum) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const uint32_t b = __ballot_sync(0xffffffffu, flag);
    const int rank = __popc(b & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) wsum[warp] = __popc(b);
    __syncthreads();
    int base = 0, tot = 0;
    for (int i = 0; i < nw; ++i) {
        if (i < warp) base += wsum[i];
        tot += wsum[i];
    }
    total = tot;
    return base + rank;
}

__device__ void topk_block(const float* __restrict__ s, int n, int K, int32_t* __restrict__ out) {
    __shared__ int ids[kIdxTopkMax];
    __shared__ uint32_t hist[256];
    __shared__ int wsum[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need, s_cnt, s_base;

    int nvalid_local = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) nvalid_local += s[i] > -INFINITY;
    int nvalid;
    {
        __shared__ int red[32];
        const float v = warp_sum((float) nvalid_local);
        __syncthreads();
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = (int) v;
        __syncthreads();
        nvalid = 0;
        for (int i = 0; i < (blockDim.x >> 5); ++i) nvalid += red[i];
    }

    uint32_t thr = 0;
    int need_eq = 0;
    const bool all = nvalid <= K;
    if (!all) {
        if (threadIdx.x == 0) { s_prefix = 0; s_need = K; }
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                if (!(s[i] > -INFINITY)) continue;
                const uint32_t key = order_key(s[i]);
                if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8)))
                    atomicAdd(&hist[(key >> shift) & 255], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int need = s_need, cum = 0, sel = 0;
                for (int b = 255; b >= 0; --b) {
                    if (cum + (int) hist[b] >= need) { sel = b; break; }
                    cum += hist[b];
                }
                s_need = need - cum;
                s_prefix = prefix | ((uint32_t) sel << shift);
            }
            __syncthreads();
        }
        thr = s_prefix;
        need_eq = s_need;
    }
    if (threadIdx.x == 0) { s_cnt = 0; s_base = 0; }
    __syncthreads();

    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const bool ok = i < n && s[i] > -INFINITY && (all || order_key(s[i]) > thr);
        int tot;
        const int r = block_scan(ok ? 1 : 0, tot, wsum);
        if (ok) ids[s_cnt + r] = i;
        __syncthreads();
        if (threadIdx.x == 0) s_cnt += tot;
        __syncthreads();
    }
    if (!all) {
        for (int i0 = 0; i0 < n && s_base < need_eq; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            const bool eq = i < n && s[i] > -INFINITY && order_key(s[i]) == thr;
            int tot;
            const int r = block_scan(eq ? 1 : 0, tot, wsum);
            if (eq && s_base + r < need_eq) ids[s_cnt + s_base + r] = i;
            __syncthreads();
            if (threadIdx.x == 0) s_base += min(tot, need_eq - s_base);
            __syncthreads();
        }
        if (threadIdx.x == 0) s_cnt += s_base;
        __syncthreads();
    }
    const int cnt = s_cnt;
    int N = 1;
    while (N < cnt) N <<= 1;
    for (int i = cnt + threadIdx.x; i < N; i += blockDim.x) ids[i] = INT32_MAX;
    __syncthreads();
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int ix = i ^ j;
                if (ix > i) {
                    const bool up = (i & k) == 0;
                    const int a = ids[i], b = ids[ix];
                    if ((a > b) == up) { ids[i] = b; ids[ix] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) out[i] = i < cnt ? ids[i] : -1;
}

__global__ void __launch_bounds__(kThreads) idx_select_kernel(const float* __restrict__ scores, int n_pool, int K,
                                                              int32_t* __restrict__ ids, int32_t* __restrict__ out_n) {
    const int t = blockIdx.x;
    topk_block(scores + (size_t) t * n_pool, n_pool, K, ids + (size_t) t * K);
    if (threadIdx.x == 0) {
        int cnt = 0;
        while (cnt < K && ids[(size_t) t * K + cnt] >= 0) ++cnt;
        out_n[t] = cnt;
    }
}

}  // namespace

void idx_select(const float* scores, int32_t* ids, int32_t* out_n, int n_tokens, int n_pool, int K,
                cudaStream_t stream) {
    if (n_tokens <= 0 || n_pool <= 0 || K <= 0 || K > kIdxTopkMax) return;
    idx_select_kernel<<<n_tokens, kThreads, 0, stream>>>(scores, n_pool, K, ids, out_n);
}

}  // namespace basalt::models::glm5next
