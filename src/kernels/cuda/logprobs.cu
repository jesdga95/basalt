// src/kernels/cuda/logprobs.cu - a pick's log probability and its row's likeliest tokens (see the header).
#include "basalt/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPart = 4 + 2 * kLogprobMaxTop;   // a chunk's scratch: max, (pad), exp sum (double), top ids, top values
constexpr int kMaxVocab = 1 << 20;              // the widest head the merge's shared memory holds

// the better of two (value, id) entries: the larger value, ties to the lower id
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }
// ranked after (pv, pi): what the next argmax round may take
__device__ __forceinline__ bool after(float v, int i, float pv, int pi) { return v < pv || (v == pv && i > pi); }

// The block's best (value, id) of every thread's own `bv`/`bi`: within each warp, then across the warps through shared
// memory. Every thread gets the result.
__device__ void block_best(float& bv, int& bi, float* s_v, int* s_i, int tid) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if ((tid & 31) == 0) { s_v[tid >> 5] = bv; s_i[tid >> 5] = bi; }
    __syncthreads();
    bv = s_v[0];
    bi = s_i[0];
    for (int w = 1; w < kWarps; ++w)
        if (better(s_v[w], s_i[w], bv, bi)) { bv = s_v[w]; bi = s_i[w]; }
    __syncthreads();
}

// chunk `chunk` of row `row`: its max, the sum of exp(x - max) and its `top` likeliest entries (raw logits)
__device__ void chunk_stage(const float* __restrict__ logits, int n_vocab, int top, float* __restrict__ scratch, int chunk,
                            int row, int tid) {
    __shared__ float s_x[kLogprobChunk];
    __shared__ float s_v[kWarps];
    __shared__ int s_i[kWarps];
    __shared__ double s_d[kWarps];
    const int i0 = chunk * kLogprobChunk;
    const int n = min(kLogprobChunk, n_vocab - i0);
    const float* src = logits + (size_t) row * (size_t) n_vocab + i0;
    float mx = -FLT_MAX;
    for (int i = tid; i < n; i += kThreads) { s_x[i] = src[i]; mx = fmaxf(mx, s_x[i]); }
    float bv = mx;
    int bi = 0;
    block_best(bv, bi, s_v, s_i, tid);   // (the max; the ids are not read)
    mx = bv;
    double sum = 0.0;
    for (int i = tid; i < n; i += kThreads) sum += (double) expf(s_x[i] - mx);
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((tid & 31) == 0) s_d[tid >> 5] = sum;
    __syncthreads();
    float* part = scratch + ((size_t) row * row_logprobs_chunks(n_vocab) + chunk) * kPart;
    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < kWarps; ++w) total += s_d[w];
        part[0] = mx;
        reinterpret_cast<double*>(part)[1] = total;
    }
    float pv = FLT_MAX;
    int pi = -1;
    for (int j = 0; j < top; ++j) {
        bv = -INFINITY;
        bi = 0x7fffffff;
        for (int i = tid; i < n; i += kThreads)
            if (after(s_x[i], i0 + i, pv, pi) && better(s_x[i], i0 + i, bv, bi)) { bv = s_x[i]; bi = i0 + i; }
        block_best(bv, bi, s_v, s_i, tid);
        if (tid == 0) {
            reinterpret_cast<int*>(part)[4 + j] = bi;
            part[4 + kLogprobMaxTop + j] = bv;
        }
        pv = bv;
        pi = bi;
    }
}

// row `row`: the chunks merged in order, then the top entries among their candidates
__device__ void merge_stage(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ picks, int top,
                            float* __restrict__ out, const float* __restrict__ scratch, int row, int tid) {
    __shared__ float s_v[kWarps];
    __shared__ int s_i[kWarps];
    __shared__ double s_lse;
    __shared__ double s_c[(kMaxVocab + kLogprobChunk - 1) / kLogprobChunk];
    const int nc = row_logprobs_chunks(n_vocab);
    const float* parts = scratch + (size_t) row * nc * kPart;
    float* o = out + (size_t) row * kLogprobStride;
    float bv = -FLT_MAX;
    int bi = 0;
    for (int c = tid; c < nc; c += kThreads) bv = fmaxf(bv, parts[(size_t) c * kPart]);
    block_best(bv, bi, s_v, s_i, tid);   // (the max; the ids are not read)
    const float mx = bv;
    for (int c = tid; c < nc; c += kThreads)   // each chunk's sum rescaled to the row's max, one thread a chunk
        s_c[c] = reinterpret_cast<const double*>(parts + (size_t) c * kPart)[1] * exp((double) parts[(size_t) c * kPart] - mx);
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int c = 0; c < nc; ++c) sum += s_c[c];   // in chunk order
        const double lse = (double) mx + log(sum);
        s_lse = lse;
        const int pick = picks[row];
        o[0] = pick >= 0 && pick < n_vocab ? (float) ((double) logits[(size_t) row * n_vocab + pick] - lse) : -INFINITY;
        o[1] = (float) lse;
    }
    __syncthreads();
    const double lse = s_lse;
    const int nk = nc * top;   // candidates: `top` per chunk (a short last chunk pads with -inf)
    float pv = FLT_MAX;
    int pi = -1;
    for (int j = 0; j < top; ++j) {
        bv = -INFINITY;
        bi = 0x7fffffff;
        for (int k = tid; k < nk; k += kThreads) {
            const float* part = parts + (size_t) (k / top) * kPart;
            const int id = reinterpret_cast<const int*>(part)[4 + k % top];
            const float v = part[4 + kLogprobMaxTop + k % top];
            if (id < n_vocab && after(v, id, pv, pi) && better(v, id, bv, bi)) { bv = v; bi = id; }
        }
        block_best(bv, bi, s_v, s_i, tid);
        if (tid == 0) {
            const bool real = bi < n_vocab;
            reinterpret_cast<int*>(o)[2 + 2 * j] = real ? bi : -1;
            o[3 + 2 * j] = real ? (float) ((double) bv - lse) : -INFINITY;
        }
        pv = bv;
        pi = bi;
    }
}

__global__ void __launch_bounds__(kThreads) chunk_kernel(const float* __restrict__ logits, int n_vocab, int top,
                                                         float* __restrict__ scratch) {
    chunk_stage(logits, n_vocab, top, scratch, (int) blockIdx.x, (int) blockIdx.y, (int) threadIdx.x);
}

__global__ void __launch_bounds__(kThreads) merge_kernel(const float* __restrict__ logits, int n_vocab,
                                                         const int32_t* __restrict__ picks, int top,
                                                         float* __restrict__ out, const float* __restrict__ scratch) {
    merge_stage(logits, n_vocab, picks, top, out, scratch, (int) blockIdx.x, (int) threadIdx.x);
}

}  // namespace

void row_logprobs(const float* logits, int n_rows, int n_vocab, const int32_t* picks, int top, float* out, float* scratch,
                  void* stream) {
    if (n_rows <= 0 || n_vocab <= 0 || n_vocab > kMaxVocab) return;
    top = top < 0 ? 0 : top > kLogprobMaxTop ? kLogprobMaxTop : top;
    const cudaStream_t st = (cudaStream_t) stream;
    chunk_kernel<<<dim3((unsigned) row_logprobs_chunks(n_vocab), (unsigned) n_rows), kThreads, 0, st>>>(logits, n_vocab, top,
                                                                                                        scratch);
    merge_kernel<<<n_rows, kThreads, 0, st>>>(logits, n_vocab, picks, top, out, scratch);
}

}  // namespace basalt::kernels
