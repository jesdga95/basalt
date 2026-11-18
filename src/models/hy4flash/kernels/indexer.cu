// src/models/hy4flash/kernels/indexer.cu - the indexer of Hy4-flash's sparse layers for one token: the gated pooling of
// the keys of 4 positions, the rotation of the indexer's heads, the relu score of every pooled key and the selection of
// the best pools.
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Rotation of consecutive pairs of the last `n_rot` values of a row by pos * base^(-2i / n_rot).
__device__ __forceinline__ void rot_pair(float* row, int i, int n_rot, float base, int pos) {
    const float theta = (float) pos * powf(base, -(float) (2 * i) / (float) n_rot);
    float s, c;
    sincosf(theta, &s, &c);
    const float x0 = row[2 * i], x1 = row[2 * i + 1];
    row[2 * i] = x0 * c - x1 * s;
    row[2 * i + 1] = x0 * s + x1 * c;
}

// The position pair i of the rotated part of the row at cell `cell` turns by (see MlaArgs::cell_rope).
template <typename A>
__device__ __forceinline__ int axis_pos(const A& a, int cell, int i) {
    if (a.cell_rope == nullptr) return cell;
    const int32_t* p = a.cell_rope + (size_t) cell * 3;
    if (i % 3 == 1 && i < 3 * a.sec_h) return p[1];
    if (i % 3 == 2 && i < 3 * a.sec_w) return p[2];
    return p[0];
}

// One block of `dim` threads: the pooled key of a group.
__global__ void pool_kernel(const float* __restrict__ values, const float* __restrict__ gates, const float* __restrict__ ape,
                            const float* __restrict__ norm_w, IndexerArgs a, int ring, int first_pos, __half* __restrict__ pooled) {
    __shared__ float row[128];
    __shared__ float red[4];
    const int d = threadIdx.x;
    const int group_pos = first_pos + blockIdx.x * a.pool;   // the group's first position
    const int slot0 = group_pos % ring;
    values += (size_t) slot0 * a.dim;
    gates += (size_t) slot0 * a.dim;
    float lg[8], mx = -INFINITY;
    for (int j = 0; j < a.pool; ++j) {
        lg[j] = gates[(size_t) j * a.dim + d] + ape[(size_t) j * a.dim + d];
        mx = fmaxf(mx, lg[j]);
    }
    float sum = 0.f, acc = 0.f;
    for (int j = 0; j < a.pool; ++j) {
        lg[j] = expf(lg[j] - mx);
        sum += lg[j];
    }
    for (int j = 0; j < a.pool; ++j) acc = fmaf(lg[j] / sum, values[(size_t) j * a.dim + d], acc);
    // RMS norm over the row
    float ss = warp_sum(acc * acc);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    ss = red[0] + red[1] + red[2] + red[3];
    row[d] = acc * rsqrtf(ss / (float) a.dim + a.eps) * norm_w[d];
    __syncthreads();
    if (d < a.rope / 2) rot_pair(row + a.dim - a.rope, d, a.rope, a.rope_base, axis_pos(a, group_pos, d));
    __syncthreads();
    pooled[(size_t) (group_pos / a.pool) * a.dim + d] = __float2half_rn(row[d]);
}

__global__ void rope_heads_kernel(float* __restrict__ q, IndexerArgs a, int pos0) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    if (i < a.rope / 2) rot_pair(q + ((size_t) t * a.heads + h) * a.dim + a.dim - a.rope, i, a.rope, a.rope_base, axis_pos(a, pos0 + t, i));
}

// A thread per pooled key.
__global__ void scores_kernel(const float* __restrict__ q, const float* __restrict__ w, const __half* __restrict__ pooled,
                              int n_pool, IndexerArgs a, float* __restrict__ scores) {
    extern __shared__ float qs[];   // [heads][dim] and [heads]
    for (int i = threadIdx.x; i < a.heads * a.dim; i += blockDim.x) qs[i] = q[i];
    for (int i = threadIdx.x; i < a.heads; i += blockDim.x) qs[a.heads * a.dim + i] = w[i];
    __syncthreads();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pool) return;
    const __half2* k = reinterpret_cast<const __half2*>(pooled + (size_t) p * a.dim);
    float total = 0.f;
    for (int h = 0; h < a.heads; ++h) {
        const float2* qh = reinterpret_cast<const float2*>(qs + (size_t) h * a.dim);
        float dot = 0.f;
        for (int i = 0; i < a.dim / 2; ++i) {
            const float2 kv = __half22float2(k[i]), qv = qh[i];
            dot = fmaf(qv.x, kv.x, dot); dot = fmaf(qv.y, kv.y, dot);
        }
        total = fmaf(qs[a.heads * a.dim + h], fmaxf(dot, 0.f), total);
    }
    scores[p] = total;
}

constexpr int kSelThreads = 1024;

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}

// One block. The k-th largest score by a search over its order key's bits, then the pools above it and the lowest-indexed
// ties, written in pool order as their cells, then the tail's.
__global__ void select_kernel(const float* __restrict__ scores, int n, int k, int pool, int tail0, int tail_n,
                              int32_t* __restrict__ cells) {
    __shared__ uint32_t red[kSelThreads / 32];
    __shared__ uint32_t sh_thr;
    __shared__ int sh_count[kSelThreads + 1], sh_eq[kSelThreads + 1], sh_need;
    const int t = threadIdx.x;
    const int chunk = (n + kSelThreads - 1) / kSelThreads, lo = min(n, t * chunk), hi = min(n, lo + chunk);
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        int c = 0;
        for (int i = lo; i < hi; ++i) c += order_key(scores[i]) >= cand;
        for (int m = 16; m; m >>= 1) c += __shfl_xor_sync(0xffffffffu, c, m);
        if ((t & 31) == 0) red[t >> 5] = c;
        __syncthreads();
        int total = 0;
        for (int i = 0; i < kSelThreads / 32; ++i) total += red[i];
        __syncthreads();
        if (total >= k) thr = cand;
    }
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(scores[i]);
        gt += key > thr;
        eq += key == thr;
    }
    sh_count[t] = gt;
    sh_eq[t] = eq;
    __syncthreads();
    if (t == 0) {   // exclusive scans
        int a = 0, b = 0;
        for (int i = 0; i < kSelThreads; ++i) {
            const int x = sh_count[i], y = sh_eq[i];
            sh_count[i] = a;
            sh_eq[i] = b;
            a += x;
            b += y;
        }
        sh_need = k - a;   // ties to take
    }
    __syncthreads();
    // how many of this thread's pools are taken, then the output slots in pool order
    int tie_before = sh_eq[t];
    const int need = sh_need;
    int mine = gt;
    {
        const int take = max(0, min(eq, need - tie_before));
        mine += take;
    }
    __shared__ int sh_taken[kSelThreads];
    sh_taken[t] = mine;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < kSelThreads; ++i) {
            const int x = sh_taken[i];
            sh_taken[i] = a;
            a += x;
        }
    }
    __syncthreads();
    int slot = sh_taken[t];
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(scores[i]);
        bool take = key > thr;
        if (key == thr) take = tie_before++ < need;
        if (take) {
            for (int j = 0; j < pool; ++j) cells[(size_t) slot * pool + j] = i * pool + j;
            ++slot;
        }
    }
    for (int j = t; j < tail_n; j += kSelThreads) cells[(size_t) k * pool + j] = tail0 + j;
}

}  // namespace

void idx_pool(const float* values, const float* gates, const float* ape, const float* norm_w, const IndexerArgs& a, int ring,
              int first_pos, int n_groups, uint16_t* pooled, void* stream) {
    if (a.dim != 128 || a.pool > 8 || ring % a.pool || first_pos % a.pool) throw std::invalid_argument("idx_pool: geometry");
    if (n_groups > 0) pool_kernel<<<n_groups, a.dim, 0, (cudaStream_t) stream>>>(values, gates, ape, norm_w, a, ring, first_pos, reinterpret_cast<__half*>(pooled));
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_rope_heads(float* q, const IndexerArgs& a, int pos0, int n_tok, void* stream) {
    rope_heads_kernel<<<dim3(a.heads, n_tok), 32, 0, (cudaStream_t) stream>>>(q, a, pos0);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_scores(const float* q, const float* w, const uint16_t* pooled, int n_pool, const IndexerArgs& a, float* scores,
                void* stream) {
    const size_t smem = ((size_t) a.heads * a.dim + a.heads) * sizeof(float);
    scores_kernel<<<(n_pool + 127) / 128, 128, smem, (cudaStream_t) stream>>>(q, w, reinterpret_cast<const __half*>(pooled), n_pool, a, scores);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_select(const float* scores, int n_pool, int k, int pool, int tail0, int tail_n, int32_t* cells, void* stream) {
    if (k < 1 || k > n_pool) throw std::invalid_argument("idx_select: k");
    select_kernel<<<1, kSelThreads, 0, (cudaStream_t) stream>>>(scores, n_pool, k, pool, tail0, tail_n, cells);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
