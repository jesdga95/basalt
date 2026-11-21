// src/models/inkling2/kernels/prompt_ops.cu - the prompt path's small kernels: the experts' plan as a stable counting sort
// of a chunk's (row, slot) pairs by expert, and the staged sliding-window keys moved into the window rings.
#include "basalt/models/inkling2/prompt_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

constexpr int kSortPairs = 1024;   // pairs a counting block takes
constexpr int kMaxExpert = 512;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

int sort_blocks(int pairs) { return (pairs + kSortPairs - 1) / kSortPairs; }

// Block b counts the experts of pairs [b kSortPairs, (b + 1) kSortPairs).
__global__ void __launch_bounds__(256) sort_count_kernel(const int32_t* __restrict__ ids, int pairs, int n_expert,
                                                         int32_t* __restrict__ counts) {
    __shared__ int32_t hist[kMaxExpert];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) atomicAdd(&hist[ids[i]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) counts[(size_t) blockIdx.x * n_expert + e] = hist[e];
}

// One block: each expert's first entry (bounds) and each counting block's first entry of each expert (counts, in place).
__global__ void __launch_bounds__(kMaxExpert) sort_scan_kernel(int32_t* __restrict__ counts, int n_blocks, int n_expert,
                                                               int pairs, int32_t* __restrict__ bounds) {
    __shared__ int32_t total[kMaxExpert];
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
    }
    __syncthreads();
    if (e < n_expert) {
        bounds[e] = total[e];
        for (int b = 0; b < n_blocks; ++b) counts[(size_t) b * n_expert + e] += total[e];
    }
    if (e == 0) bounds[n_expert] = pairs;
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

// One 16-byte piece of one position's cell (its key or value rows of every head): x = position, y = (value).
__global__ void stage_to_ring_kernel(const uint4* __restrict__ k_stage, const uint4* __restrict__ v_stage, int64_t stage_cap,
                                     uint4* __restrict__ k_ring, uint4* __restrict__ v_ring, int64_t ring_cap, int64_t first,
                                     int pieces) {
    const int64_t p = first + blockIdx.x;
    const uint4* src = (blockIdx.y ? v_stage : k_stage) + (size_t) (p % stage_cap) * pieces;
    uint4* dst = (blockIdx.y ? v_ring : k_ring) + (size_t) (p % ring_cap) * pieces;
    for (int i = threadIdx.x; i < pieces; i += blockDim.x) dst[i] = src[i];
}

__global__ void prompt_rows_kernel(int32_t* __restrict__ pos, int64_t pos0, int rows, int32_t* __restrict__ bounds) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) pos[i] = (int32_t) (pos0 + i);
    if (i == 0) {
        bounds[0] = 0;
        bounds[1] = rows;
    }
}

}  // namespace

void prompt_rows(int32_t* pos, int64_t pos0, int rows, int32_t* bounds, void* stream) {
    if (rows <= 0) return;
    prompt_rows_kernel<<<(rows + 255) / 256, 256, 0, (cudaStream_t) stream>>>(pos, pos0, rows, bounds);
    check("prompt_rows");
}

size_t moe_sort_scratch(int pairs, int n_expert) { return (size_t) sort_blocks(pairs) * (size_t) n_expert * sizeof(int32_t); }

void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream) {
    const int pairs = rows * k;
    if (pairs <= 0) return;
    if (n_expert > kMaxExpert) throw std::runtime_error("inkling2 moe_sort: more than 512 experts");
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

namespace {
__global__ void remap_ids_kernel(int32_t* __restrict__ ids, int n, const int32_t* __restrict__ map) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = map[ids[i]];
}
}  // namespace

void remap_ids(int32_t* ids, int n, const int32_t* map, void* stream) {
    if (n <= 0) return;
    remap_ids_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(ids, n, map);
    check("remap_ids");
}

void stage_to_ring(const void* k_stage, const void* v_stage, int64_t stage_cap, void* k_ring, void* v_ring, int64_t ring_cap,
                   int64_t pos0, int rows, int keep, int elem, void* stream) {
    const int64_t end = pos0 + rows, first = end - keep > pos0 ? end - keep : pos0;
    const int n = (int) (end - first);
    if (n <= 0) return;
    const int pieces = kKvHeads * kHeadDim * elem / 16;
    stage_to_ring_kernel<<<dim3((unsigned) n, 2), 128, 0, (cudaStream_t) stream>>>(
        static_cast<const uint4*>(k_stage), static_cast<const uint4*>(v_stage), stage_cap, static_cast<uint4*>(k_ring),
        static_cast<uint4*>(v_ring), ring_cap, first, pieces);
    check("stage_to_ring");
}

}  // namespace basalt::models::inkling2
