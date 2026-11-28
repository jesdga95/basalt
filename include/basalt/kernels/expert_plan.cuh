// include/basalt/kernels/expert_plan.cuh - the device step moe_plan (expert_plan.hpp) and the models' fused router plans share:
// each routed pair's entry in its expert's group.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// The entries of the pairs sid[0..n) whose expert has a blob: pair i takes first[x] plus the pairs before it on the same
/// expert x (dst = i, tok = i / k). NT pairs at a time (NT the block's threads): a warp finds its pairs on one expert with
/// __match_any_sync and adds the counts of the warps before it (wcnt [NT / 32][n_expert], zero on entry and on return);
/// first[] advances past each chunk.
template <int NT>
__device__ __forceinline__ void plan_entries(const int32_t* sid, int n, int k, int n_expert, const unsigned long long* blob,
                                             int32_t* first, int32_t* wcnt, int32_t* __restrict__ dst, int32_t* __restrict__ tok) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    for (int base = 0; base < n; base += NT) {
        const int i = base + tid;
        int x = i < n ? sid[i] : -1;
        if (x >= 0 && !blob[x]) x = -1;
        const unsigned m = __match_any_sync(0xffffffffu, x);
        const int lower = __popc(m & ((1u << lane) - 1u));
        if (x >= 0 && lower == 0) wcnt[w * n_expert + x] = __popc(m);
        __syncthreads();
        if (x >= 0) {
            int p = first[x] + lower;
            for (int v = 0; v < w; ++v) p += wcnt[v * n_expert + x];
            dst[p] = i;
            tok[p] = i / k;
        }
        __syncthreads();
        for (int e = tid; e < n_expert; e += NT) {
            int t = 0;
            for (int v = 0; v < NT / 32; ++v) {
                t += wcnt[v * n_expert + e];
                wcnt[v * n_expert + e] = 0;
            }
            first[e] += t;
        }
        __syncthreads();
    }
}

}  // namespace basalt::kernels
