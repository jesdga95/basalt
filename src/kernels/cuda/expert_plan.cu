// src/kernels/cuda/expert_plan.cu - moe_plan and moe_sort (basalt/kernels/expert_plan.hpp).
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/expert_plan.cuh"
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
// each pair then takes its place in its group after the pairs before it that chose the same expert (plan_entries). The
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
    int32_t* wcnt = first + n_expert;                              // [kPlanThreads / 32][n_expert]: plan_entries' counts
    int32_t* sid = wcnt + kPlanThreads / 32 * n_expert;            // [n]: the pairs' experts
    __shared__ int wsum[kPlanThreads / 32][2];                     // each warp's entries and groups
    __shared__ int groups;
    const int e = threadIdx.x, w = e >> 5, lane = e & 31;
    if (e < n_expert) {
        blob[e] = blobs[e];
        count[e] = 0;
    }
    for (int j = e; j < kPlanThreads / 32 * n_expert; j += kPlanThreads) wcnt[j] = 0;
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
    plan_entries<kPlanThreads>(sid, n, k, n_expert, blob, first, wcnt, dst, tok);
}

// ---- moe_sort: a stable counting sort of a prompt chunk's (row, slot) pairs by expert
constexpr int kSortPairs = 1024;   // pairs a counting block takes
constexpr int kMaxExpert = 512;
int sort_blocks(int pairs) { return (pairs + kSortPairs - 1) / kSortPairs; }

// Block b counts the experts of pairs [b kSortPairs, (b + 1) kSortPairs).
__global__ void __launch_bounds__(256) sort_count_kernel(const int32_t* __restrict__ ids, int pairs, int n_expert,
                                                         int32_t* __restrict__ counts) {
    __shared__ int32_t hist[kMaxExpert];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x)
        if (ids[i] >= 0) atomicAdd(&hist[ids[i]], 1);   // a pair another card computes has no expert here
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) counts[(size_t) blockIdx.x * n_expert + e] = hist[e];
}

// One block: each expert's first entry (bounds) and each counting block's first entry of each expert (counts, in place).
__global__ void __launch_bounds__(kMaxExpert) sort_scan_kernel(int32_t* __restrict__ counts, int n_blocks, int n_expert,
                                                               int pairs, int32_t* __restrict__ bounds) {
    __shared__ int32_t total[kMaxExpert];
    __shared__ int32_t all;
    const int e = threadIdx.x;
    int run = 0;
    if (e < n_expert)
        for (int b = 0; b < n_blocks; ++b) {
            const int c = counts[(size_t) b * n_expert + e];
            counts[(size_t) b * n_expert + e] = run;
            run += c;
        }
    total[e] = e < n_expert ? run : 0;
    __syncthreads();
    if (e == 0) {
        int at = 0;
        for (int x = 0; x < n_expert; ++x) {
            const int t = total[x];
            total[x] = at;
            at += t;
        }
        all = at;
    }
    __syncthreads();
    if (e < n_expert) {
        bounds[e] = total[e];
        for (int b = 0; b < n_blocks; ++b) counts[(size_t) b * n_expert + e] += total[e];
    }
    if (e == 0) bounds[n_expert] = all;
}

// One warp a counting block: its pairs in order, 32 at a time; a pair's place is its expert's next entry plus the lanes
// before it with the same expert, so the entries of an expert keep the pairs' order.
__global__ void __launch_bounds__(32) sort_scatter_kernel(const int32_t* __restrict__ ids, int pairs, int k, int n_expert,
                                                          const int32_t* __restrict__ counts, int32_t* __restrict__ tok,
                                                          int32_t* __restrict__ dst, int32_t* __restrict__ exp) {
    __shared__ int32_t next[kMaxExpert];
    const int lane = threadIdx.x;
    for (int e = lane; e < n_expert; e += 32) next[e] = counts[(size_t) blockIdx.x * n_expert + e];
    __syncwarp();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;
        const int e = i < hi ? ids[i] : -1;
        const unsigned same = __match_any_sync(0xffffffffu, e);
        const int rank = __popc(same & ((1u << lane) - 1u));
        if (e >= 0) {
            const int at = next[e] + rank;
            tok[at] = i / k;
            dst[at] = i;
            exp[at] = e;
        }
        __syncwarp();
        if (e >= 0 && rank == 0) next[e] += __popc(same);
        __syncwarp();
    }
}

}  // namespace

void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally,
              unsigned long long* rec_blobs, int32_t* rec_n) {
    const int n = rows * k;
    if (n <= 0) return;
    if (n > kPlanMaxPairs) throw std::runtime_error("moe_plan: more pairs than kPlanMaxPairs");
    if (n_expert > kPlanThreads) throw std::runtime_error("moe_plan: more experts than a block's threads");
    launch_pdl(moe_plan_kernel, dim3(1), dim3(kPlanThreads),
               (size_t) n_expert * (2 * sizeof(unsigned long long) + (2 + kPlanThreads / 32) * sizeof(int32_t)) + (size_t) n * sizeof(int32_t),
               (cudaStream_t) stream, ids, n, n_expert, blobs, ptr, start, n_groups, dst, tok, k, tally, rec_blobs, rec_n);
    check("moe_plan");
}

size_t moe_sort_scratch(int pairs, int n_expert) { return (size_t) sort_blocks(pairs) * (size_t) n_expert * sizeof(int32_t); }

void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream) {
    const int pairs = rows * k;
    if (pairs <= 0) return;
    if (n_expert > kMaxExpert) throw std::runtime_error("moe_sort: more than 512 experts");
    cudaStream_t s = (cudaStream_t) stream;
    int32_t* counts = static_cast<int32_t*>(scratch);
    const int nb = sort_blocks(pairs);
    sort_count_kernel<<<nb, 256, 0, s>>>(ids, pairs, n_expert, counts);
    check("moe_sort/count");
    sort_scan_kernel<<<1, kMaxExpert, 0, s>>>(counts, nb, n_expert, pairs, bounds);
    check("moe_sort/scan");
    sort_scatter_kernel<<<nb, 32, 0, s>>>(ids, pairs, k, n_expert, counts, tok, dst, exp);
    check("moe_sort/scatter");
}

}  // namespace basalt::kernels
