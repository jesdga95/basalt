// src/kernels/cuda/expert_plan.cu - moe_plan (basalt/kernels/expert_plan.hpp).
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::kernels {
namespace {

void check(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// One block of kPlanThreads, a thread an expert: the (row, slot) pairs counted by expert (the experts on the card); every
// warp scans its experts' counts at once and adds the warps' before it, giving each used expert its group and first entry;
// each pair then takes its place in its group from the pairs before it that chose the same expert (the rows' order). The
// blob table (constant) is staged in shared memory before pdl_wait, the pairs' choices after it.
constexpr int kPlanThreads = 256;
__global__ void __launch_bounds__(kPlanThreads) moe_plan_kernel(const int32_t* __restrict__ ids, int n, int n_expert,
                                                                 const unsigned long long* __restrict__ blobs, unsigned long long* __restrict__ ptr,
                                                                 int32_t* __restrict__ start, int32_t* __restrict__ n_groups,
                                                                 int32_t* __restrict__ dst, int32_t* __restrict__ tok, int k,
                                                                 uint32_t* __restrict__ tally, unsigned long long* __restrict__ rec_blobs,
                                                                 int32_t* __restrict__ rec_n) {
    extern __shared__ unsigned long long smb[];
    unsigned long long* blob = smb;                                // [n_expert]
    unsigned long long* gblob = smb + n_expert;                    // [n_expert]: group g's blob
    int32_t* count = reinterpret_cast<int32_t*>(smb + 2 * n_expert);   // [n_expert]
    int32_t* first = count + n_expert;                             // [n_expert]: the expert's first entry
    int32_t* sid = first + n_expert;                               // [n]: the pairs' experts
    __shared__ int wsum[kPlanThreads / 32][2];                     // each warp's entries and groups
    __shared__ int groups;
    const int e = threadIdx.x, w = e >> 5, lane = e & 31;
    if (e < n_expert) {
        blob[e] = blobs[e];
        count[e] = 0;
    }
    pdl_wait();
    pdl_launch_dependents();
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = ids[i];
        sid[i] = x;
        if (blob[x]) atomicAdd(&count[x], 1);
    }
    __syncthreads();
    const int c = e < n_expert ? count[e] : 0;
    int s = c, used = c > 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, s, o), u = __shfl_up_sync(0xffffffffu, used, o);
        if (lane >= o) {
            s += t;
            used += u;
        }
    }
    if (lane == 31) {
        wsum[w][0] = s;
        wsum[w][1] = used;
    }
    __syncthreads();
    int at = 0, g = 0;
    for (int j = 0; j < w; ++j) {
        at += wsum[j][0];
        g += wsum[j][1];
    }
    if (e < n_expert) first[e] = at + s - c;
    if (c > 0) {
        gblob[g + used - 1] = blob[e];
        ptr[g + used - 1] = blob[e];
        start[g + used - 1] = at + s - c;
    }
    if (e == kPlanThreads - 1) {   // the totals
        start[g + used] = at + s;
        start[g + used + 1] = at + s;
        *n_groups = g + used;
        groups = g + used;
    }
    __syncthreads();
    if (rec_blobs && e < kPlanAhead) {   // the window's first groups, for the next window's read-ahead
        const int ng = groups;
        rec_blobs[e] = e < ng ? gblob[e] : 0ull;
        if (e == 0) *rec_n = ng < kPlanAhead ? ng : kPlanAhead;
    }
    if (tally && c)   // a reduction (its result unused), so no thread waits on the load a plain add would make
        atomicAdd(&tally[e], (uint32_t) c);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = sid[i];
        if (!blob[x]) continue;
        int before = 0;
        for (int j = 0; j < i; ++j) before += sid[j] == x;
        const int p = first[x] + before;
        dst[p] = i;
        tok[p] = i / k;
    }
}

}  // namespace

void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally,
              unsigned long long* rec_blobs, int32_t* rec_n) {
    const int n = rows * k;
    if (n <= 0) return;
    if (n > 1024) throw std::runtime_error("moe_plan: more than 1024 pairs");
    if (n_expert > kPlanThreads) throw std::runtime_error("moe_plan: more experts than a block's threads");
    launch_pdl(moe_plan_kernel, dim3(1), dim3(kPlanThreads),
               (size_t) n_expert * (2 * sizeof(unsigned long long) + 2 * sizeof(int32_t)) + (size_t) n * sizeof(int32_t),
               (cudaStream_t) stream, ids, n, n_expert, blobs, ptr, start, n_groups, dst, tok, k, tally, rec_blobs, rec_n);
    check("moe_plan");
}

}  // namespace basalt::kernels
