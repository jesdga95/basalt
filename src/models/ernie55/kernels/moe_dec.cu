// src/models/ernie55/kernels/moe_dec.cu - the decode windows' routed experts without a host round trip: the plan of
// the main card's experts from the router's choices and a device table of blobs, and the doorbell hand-off of the other
// pairs to the host (the CPU pool and the second card).
#include "basalt/kernels/doorbell.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::ernie55 {
namespace {

namespace bk = basalt::kernels;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ernie55 ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kPlanThreads = 512;

// One block, a thread an expert: the pairs on the main card counted by expert, each used expert's group and first
// entry from a block scan of the counts, then each pair's place in its group from the pairs before it with the same
// expert (row order). A dropped pair (id < 0) is in no group.
__global__ void __launch_bounds__(kPlanThreads) moe_dec_plan_kernel(const int32_t* __restrict__ ids, int n, int n_expert,
                                                                    const unsigned long long* __restrict__ tab,
                                                                    unsigned long long* __restrict__ ptr,
                                                                    int32_t* __restrict__ start, int32_t* __restrict__ n_groups,
                                                                    int32_t* __restrict__ dst, int32_t* __restrict__ tok, int k,
                                                                    int32_t* __restrict__ log) {
    bk::pdl_wait();
    __shared__ int32_t count[kPlanThreads], first[kPlanThreads], sid[kMoeDecPairs];
    __shared__ int wsum[kPlanThreads / 32][2];
    const int e = threadIdx.x, w = e >> 5, lane = e & 31;
    const unsigned long long blob = e < n_expert ? tab[e] : 0ull;
    count[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = ids[i];
        sid[i] = x;
        if (log) log[i] = x;
        if (x >= 0 && tab[x]) atomicAdd(&count[x], 1);
    }
    __syncthreads();
    const int c = count[e];
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
    first[e] = at + s - c;
    if (c > 0) {
        ptr[g + used - 1] = blob;
        start[g + used - 1] = at + s - c;
    }
    if (e == kPlanThreads - 1) {
        start[g + used] = at + s;
        *n_groups = g + used;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = sid[i];
        if (x < 0 || !tab[x]) continue;
        int before = 0;
        for (int j = 0; j < i; ++j) before += sid[j] == x;
        const int p = first[x] + before;
        dst[p] = i;
        tok[p] = i / k;
    }
}

constexpr int kRingThreads = 256;

// One block: the rows with a pair off the main card go to mapped memory with every choice, then the ticket and the
// ring. A window without such a pair neither rings nor takes a ticket.
__global__ void __launch_bounds__(kRingThreads) moe_dec_ring_kernel(const int32_t* __restrict__ ids, int rows, int k,
                                                                    const unsigned long long* __restrict__ tab,
                                                                    const float* __restrict__ x, int width, int layer,
                                                                    float* x_m, int32_t* ids_m, MoeBell* bell, uint32_t* seq) {
    bk::pdl_wait();
    __shared__ uint32_t off_rows, stopped;
    const int n = rows * k;
    if (threadIdx.x == 0) off_rows = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += kRingThreads) {
        const int e = ids[i];
        if (e >= 0 && !tab[e]) atomicOr(&off_rows, 1u << (i / k));
    }
    __syncthreads();
    const uint32_t m = off_rows;
    if (!m) return;
    uint32_t ticket = 0;
    if (threadIdx.x == 0) {
        ticket = *seq + 1;
        if (!ticket) ticket = 1;
        *seq = ticket;
        stopped = bk::doorbell_load_acquire(&bell->abort);
    }
    __syncthreads();
    if (stopped) return;
    for (int i = threadIdx.x; i < n; i += kRingThreads) ids_m[i] = ids[i];
    const int vecs = width / 4;
    for (int r = 0; r < rows; ++r)
        if (m >> r & 1)
            for (int c = threadIdx.x; c < vecs; c += kRingThreads)
                reinterpret_cast<float4*>(x_m)[(size_t) r * vecs + c] = reinterpret_cast<const float4*>(x)[(size_t) r * vecs + c];
    if (threadIdx.x == 0) bk::doorbell_store_relaxed(&bell->ticket, ticket);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) bk::doorbell_store_release(&bell->ring, 1u | (uint32_t) layer << 8 | (uint32_t) rows << 16);
}

constexpr int kHintThreads = 128;

// One block: the ids into the slot of the next sequence number, then the word that names it.
__global__ void __launch_bounds__(kHintThreads) moe_dec_hint_kernel(const int32_t* __restrict__ ids, int rows, int layer,
                                                                    MoeHint* hint, uint32_t* hseq) {
    __shared__ uint32_t s;
    if (threadIdx.x == 0) {
        s = (*hseq + 1) & 0xffffu;
        *hseq = s;
    }
    __syncthreads();
    int32_t* dst = hint->ids[s & 1];
    for (int i = threadIdx.x; i < rows * kMoeHintK; i += kHintThreads) dst[i] = ids[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) bk::doorbell_store_release(&hint->word, s << 16 | (uint32_t) layer << 8 | (uint32_t) rows);
}

constexpr int kMergeThreads = 256;

// Block p: a pair off the main card takes row p of the host's rows once its ticket is done; a dropped pair takes zeros
// (its weight is zero, and the row must not hold what an earlier window left). The main card's pairs stay.
__global__ void __launch_bounds__(kMergeThreads) moe_dec_merge_kernel(const int32_t* __restrict__ ids,
                                                                      const unsigned long long* __restrict__ tab,
                                                                      const float* host_rows, float* __restrict__ out,
                                                                      int width, MoeBell* bell, const uint32_t* seq) {
    const int p = blockIdx.x, e = ids[p];
    float* o = out + (size_t) p * width;
    if (e < 0) {
        for (int c = threadIdx.x; c < width; c += kMergeThreads) o[c] = 0.0f;
        return;
    }
    if (tab[e]) return;
    __shared__ uint32_t taken;
    if (threadIdx.x == 0) taken = bk::doorbell_spin(&bell->done, *seq, &bell->abort, 8);
    __syncthreads();
    if (!taken) return;
    for (int c = threadIdx.x; c < width; c += kMergeThreads) o[c] = __ldcv(host_rows + (size_t) p * width + c);
}

}  // namespace

void moe_dec_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* tab,
                  unsigned long long* ptr, int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok,
                  int32_t* log, void* stream) {
    if (rows <= 0) return;
    if (n_expert > kPlanThreads || rows * k > kMoeDecPairs) throw std::runtime_error("ernie55 moe_dec_plan: geometry");
    bk::launch_pdl(moe_dec_plan_kernel, dim3(1), dim3(kPlanThreads), 0, (cudaStream_t) stream, ids, rows * k, n_expert, tab, ptr,
                   start, n_groups, dst, tok, k, log);
    check("moe_dec_plan");
}

void moe_dec_ring(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* x, int width,
                  int layer, float* x_m, int32_t* ids_m, MoeBell* bell, uint32_t* seq, void* stream) {
    if (rows <= 0) return;
    if (rows > 32 || rows * k > kMoeDecPairs || layer < 0 || layer > 0xff || width % 4)
        throw std::runtime_error("ernie55 moe_dec_ring: rows, pairs, layer or width");
    bk::launch_pdl(moe_dec_ring_kernel, dim3(1), dim3(kRingThreads), 0, (cudaStream_t) stream, ids, rows, k, tab, x, width, layer,
                   x_m, ids_m, bell, seq);
    check("moe_dec_ring");
}

void moe_dec_hint(const int32_t* ids, int rows, int layer, MoeHint* hint, uint32_t* hseq, void* stream) {
    if (rows <= 0) return;
    if (rows > kMoeHintRows || layer < 0 || layer > 0xff) throw std::runtime_error("ernie55 moe_dec_hint: rows or layer");
    moe_dec_hint_kernel<<<1, kHintThreads, 0, (cudaStream_t) stream>>>(ids, rows, layer, hint, hseq);
    check("moe_dec_hint");
}

void moe_dec_merge(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* host_rows,
                   float* out, int width, MoeBell* bell, const uint32_t* seq, void* stream) {
    if (rows <= 0) return;
    moe_dec_merge_kernel<<<rows * k, kMergeThreads, 0, (cudaStream_t) stream>>>(ids, tab, host_rows, out, width, bell, seq);
    check("moe_dec_merge");
}

}  // namespace basalt::models::ernie55
