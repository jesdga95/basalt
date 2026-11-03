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
// With `meta` ([row][3] = pools, tail0, tail_n) a row of blockIdx.y: its q and w, its pool count, its scores at row * stride.
__global__ void scores_kernel(const float* __restrict__ q, const float* __restrict__ w, const __half* __restrict__ pooled,
                              int n_pool, IndexerArgs a, float* __restrict__ scores, const int32_t* __restrict__ meta, int stride) {
    if (meta) {
        const int row = blockIdx.y;
        n_pool = meta[row * 3];
        if (blockIdx.x * blockDim.x >= n_pool) return;
        q += (size_t) row * a.heads * a.dim;
        w += (size_t) row * a.heads;
        scores += (size_t) row * stride;
    }
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

// Tensor-core scores of a row (16 heads, dim 128): S = Q K^T per 8 pooled keys on mma.m16n8k16 with q split into an fp16 high and low
// part (the keys are fp16 already), then sum_h w_h relu(S_h). A block of 4 warps takes 512 pools of a row (blockIdx.y), a warp 128.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128) scores_mma_kernel(const float* __restrict__ q, const float* __restrict__ w, const __half* __restrict__ pooled,
                                                         const int32_t* __restrict__ meta, int stride, float* __restrict__ scores) {
    const int row = blockIdx.y, n_pool = meta[row * 3];
    if ((int) blockIdx.x * 512 >= n_pool) return;
    q += (size_t) row * 16 * 128;
    w += (size_t) row * 16;
    scores += (size_t) row * stride;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t qh[8][4], ql[8][4];
    for (int ks = 0; ks < 8; ++ks)
        for (int j = 0; j < 4; ++j) {
            const int r = (lane >> 2) + (j & 1) * 8, c = ks * 16 + (lane & 3) * 2 + (j >> 1) * 8;
            const float2 v = *reinterpret_cast<const float2*>(q + r * 128 + c);
            const __half2 h = __floats2half2_rn(v.x, v.y);
            const float2 hf = __half22float2(h);
            const __half2 l = __floats2half2_rn(v.x - hf.x, v.y - hf.y);
            qh[ks][j] = *reinterpret_cast<const uint32_t*>(&h);
            ql[ks][j] = *reinterpret_cast<const uint32_t*>(&l);
        }
    const float wr0 = w[lane >> 2], wr1 = w[(lane >> 2) + 8];
    const int base = blockIdx.x * 512 + warp * 128;
    for (int nt = 0; nt < 16; ++nt) {
        const int p0 = base + nt * 8;
        if (p0 >= n_pool) break;
        const int pn = min(p0 + (lane >> 2), n_pool - 1);
        const uint32_t* kr = reinterpret_cast<const uint32_t*>(pooled + (size_t) pn * 128);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        for (int ks = 0; ks < 8; ++ks) {
            const uint32_t b0 = kr[ks * 8 + (lane & 3)], b1 = kr[ks * 8 + 4 + (lane & 3)];
            mma_f16(c, qh[ks], b0, b1);
            mma_f16(c, ql[ks], b0, b1);
        }
        float v0 = wr0 * fmaxf(c[0], 0.f) + wr1 * fmaxf(c[2], 0.f);
        float v1 = wr0 * fmaxf(c[1], 0.f) + wr1 * fmaxf(c[3], 0.f);
        for (int o = 4; o < 32; o <<= 1) {
            v0 += __shfl_xor_sync(0xffffffffu, v0, o);
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        }
        if (lane < 4) {
            const int p = p0 + lane * 2;
            if (p < n_pool) scores[p] = v0;
            if (p + 1 < n_pool) scores[p + 1] = v1;
        }
    }
}

constexpr int kSelThreads = 1024;

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}

// One block. The k-th largest score's order key by a radix search over its top 11, next 11 and last 10 bits (a histogram of
// the keys under the prefix found so far each, reads coalesced, one shared-memory add per bin and warp), then the pools above
// it and the lowest-indexed ties, written in pool order as their cells, then the tail's.
__global__ void select_kernel(const float* __restrict__ scores, int n, int k, int pool, int tail0, int tail_n,
                              int32_t* __restrict__ cells, const int32_t* __restrict__ meta, int stride, int cells_stride) {
    if (meta) {   // a row per block: n <= k means the row attends everything and has no selection
        const int row = blockIdx.x;
        n = meta[row * 3];
        if (n <= k) return;
        tail0 = meta[row * 3 + 1];
        tail_n = meta[row * 3 + 2];
        scores += (size_t) row * stride;
        cells += (size_t) row * cells_stride;
    }
    __shared__ int hist[2048];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_rem;
    __shared__ int sh_count[kSelThreads + 1], sh_eq[kSelThreads + 1], sh_need;
    const int t = threadIdx.x, lane = t & 31;
    const int chunk = (n + kSelThreads - 1) / kSelThreads, lo = min(n, t * chunk), hi = min(n, lo + chunk);
    uint32_t prefix = 0, mask = 0;
    int rem = k;   // the rank of the key sought among the keys under the prefix, from the largest
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0, nb = pass == 2 ? 1024 : 2048;
        for (int i = t; i < nb; i += kSelThreads) hist[i] = 0;
        __syncthreads();
        for (int base = 0; base < n; base += kSelThreads) {
            const int i = base + t;
            const uint32_t key = i < n ? order_key(scores[i]) : 0u;
            const int bin = i < n && (key & mask) == prefix ? (int) ((key >> shift) & (uint32_t) (nb - 1)) : -1;
            const unsigned same = __match_any_sync(0xffffffffu, bin);
            if (bin >= 0 && lane == __ffs(same) - 1) atomicAdd(&hist[bin], __popc(same));
        }
        __syncthreads();
        if (t < 32) {   // lane l sums the l-th run of nb / 32 bins from the top; the run where the running count reaches rem holds the key
            const int per = nb / 32, top = nb - 1 - lane * per;
            int s = 0;
            for (int j = 0; j < per; ++j) s += hist[top - j];
            int incl = s;
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int first = __ffs(__ballot_sync(0xffffffffu, incl >= rem)) - 1;
            if (lane == first) {
                int c = incl - s, b = top;
                for (; c + hist[b] < rem; --b) c += hist[b];
                sh_prefix = prefix | ((uint32_t) b << shift);
                sh_rem = rem - c;
            }
        }
        __syncthreads();
        prefix = sh_prefix;
        rem = sh_rem;
        mask |= (uint32_t) (nb - 1) << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;
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
    scores_kernel<<<(n_pool + 127) / 128, 128, smem, (cudaStream_t) stream>>>(q, w, reinterpret_cast<const __half*>(pooled), n_pool, a, scores, nullptr, 0);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_select(const float* scores, int n_pool, int k, int pool, int tail0, int tail_n, int32_t* cells, void* stream) {
    if (k < 1 || k > n_pool) throw std::invalid_argument("idx_select: k");
    select_kernel<<<1, kSelThreads, 0, (cudaStream_t) stream>>>(scores, n_pool, k, pool, tail0, tail_n, cells, nullptr, 0, 0);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_scores_rows(const float* q, const float* w, const uint16_t* pooled, const int32_t* meta, int max_pool, int rows, const IndexerArgs& a,
                     float* scores, int stride, void* stream) {
    if (a.heads == 16 && a.dim == 128) {
        scores_mma_kernel<<<dim3((max_pool + 511) / 512, rows), 128, 0, (cudaStream_t) stream>>>(q, w, reinterpret_cast<const __half*>(pooled), meta, stride, scores);
        if (!stream) cudaStreamSynchronize(nullptr);
        return;
    }
    const size_t smem = ((size_t) a.heads * a.dim + a.heads) * sizeof(float);
    scores_kernel<<<dim3((max_pool + 127) / 128, rows), 128, smem, (cudaStream_t) stream>>>(q, w, reinterpret_cast<const __half*>(pooled), 0, a, scores,
                                                                                             meta, stride);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_select_rows(const float* scores, int stride, const int32_t* meta, int rows, int k, int pool, int32_t* cells, int cells_stride, void* stream) {
    select_kernel<<<rows, kSelThreads, 0, (cudaStream_t) stream>>>(scores, 0, k, pool, 0, 0, cells, meta, stride, cells_stride);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
