// src/models/hy4flash/kernels/mla.cu - the absorbed MLA of Hy4-flash over a window or chunk of rows: the per-head Q8_0
// products, the query and cache-row preparation, and the attention over the shared latent with a per-head sink. A row's
// result does not depend on the other rows or on how many there are.
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

constexpr int kThreads = 512;
constexpr int kMaxCells = 4096;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// Block sum in a fixed order; every thread gets the result.
template <int N>
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < N / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

template <int N>
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
#pragma unroll
    for (int i = 1; i < N / 32; ++i) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}

// A cache row's element type: fp32, or BF16 (the reference's own rounding of the latent rows).
__device__ __forceinline__ float load_row(const float* p) { return *p; }
__device__ __forceinline__ float load_row(const uint16_t* p) { return __uint_as_float((uint32_t) *p << 16); }
__device__ __forceinline__ void store_row(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_row(uint16_t* p, float v) { *p = __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// A warp per output row; z = the token.
__global__ void heads_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int x_head_stride, int x_tok_stride,
                                  int rows, int nb, int heads, float* __restrict__ y) {
    const int h = blockIdx.y, t = blockIdx.z, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const uint8_t* row = w + ((size_t) h * rows + r) * nb * 34;
    const float* xv = x + (size_t) t * x_tok_stride + (size_t) h * x_head_stride;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float q = (float) (int8_t) blk[2 + lane];
        acc = fmaf(d * q, xv[b * 32 + lane], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[((size_t) t * heads + h) * rows + r] = acc;
}

// Rotation of consecutive pairs (2i, 2i + 1) by pos * base^(-2i / n_rot).
__device__ __forceinline__ void rope_pair(float x0, float x1, int i, int n_rot, float base, int pos, float& o0, float& o1) {
    const float theta = (float) pos * powf(base, -(float) (2 * i) / (float) n_rot);
    float s, c;
    sincosf(theta, &s, &c);
    o0 = x0 * c - x1 * s;
    o1 = x0 * s + x1 * c;
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

// Block h < heads: the head's query row; block `heads`: the cache row. y = the token.
template <typename C>
__global__ void prepare_kernel(const float* __restrict__ q, const float* __restrict__ q_abs, const float* __restrict__ kv_raw,
                               const float* __restrict__ kv_norm, MlaArgs a, int pos0, int ring, float* __restrict__ qcur,
                               C* __restrict__ cache) {
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, t = blockIdx.y, W = a.rank + a.rope, Q = a.nope + a.rope, pos = pos0 + t;
    if (h < a.heads) {
        const float* qt = q + (size_t) t * a.heads * Q;
        const float* qa = q_abs + (size_t) t * a.heads * a.rank;
        float* out = qcur + (size_t) t * a.heads * W;
        for (int c = threadIdx.x; c < a.rank; c += blockDim.x) out[(size_t) h * W + c] = qa[(size_t) h * a.rank + c];
        if (threadIdx.x < a.rope / 2) {
            const int i = threadIdx.x;
            const float* p = qt + (size_t) h * Q + a.nope + 2 * i;
            float o0, o1;
            rope_pair(p[0], p[1], i, a.rope, a.rope_base, axis_pos(a, pos, i), o0, o1);
            out[(size_t) h * W + a.rank + 2 * i] = o0;
            out[(size_t) h * W + a.rank + 2 * i + 1] = o1;
        }
        return;
    }
    const float* kr = kv_raw + (size_t) t * W;
    C* row = cache + (size_t) (ring ? pos % ring : pos) * W;
    float ss = 0.f;
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) ss = fmaf(kr[c], kr[c], ss);
    ss = block_sum<kThreads>(ss, red);
    const float rs = rsqrtf(ss / (float) a.rank + a.rms_eps);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) store_row(row + c, kr[c] * rs * kv_norm[c]);
    if (threadIdx.x < a.rope / 2) {
        const int i = threadIdx.x;
        float o0, o1;
        rope_pair(kr[a.rank + 2 * i], kr[a.rank + 2 * i + 1], i, a.rope, a.rope_base, axis_pos(a, pos, i), o0, o1);
        store_row(row + a.rank + 2 * i, o0);
        store_row(row + a.rank + 2 * i + 1, o1);
    }
}

// A block per head and row: the scores (a warp per cell), the softmax with the sink, the weighted sum of the latents.
template <typename C>
__global__ void attend_kernel(const float* __restrict__ qcur, const C* __restrict__ cache, const int32_t* __restrict__ firsts,
                              const int32_t* __restrict__ counts, const int32_t* __restrict__ cells, int cells_stride,
                              const float* __restrict__ sinks, MlaArgs a, int ring, float* __restrict__ o_lat) {
    __shared__ float s[kMaxCells];
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, t = blockIdx.y, W = a.rank + a.rope, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = counts[t], first = firsts[t];
    const int32_t* cl = cells ? cells + (size_t) t * cells_stride : nullptr;
    const float* qv = qcur + ((size_t) t * a.heads + h) * W;
    for (int j = warp; j < n; j += kThreads / 32) {
        const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
        const C* kr = cache + (size_t) cell * W;
        float acc = 0.f;
        for (int c = lane; c < W; c += 32) acc = fmaf(qv[c], load_row(kr + c), acc);
        acc = warp_sum(acc);
        if (lane == 0) s[j] = acc * a.scale;
    }
    __syncthreads();
    const float sink = sinks[h];
    float m = sink;
    for (int j = threadIdx.x; j < n; j += kThreads) m = fmaxf(m, s[j]);
    m = block_max<kThreads>(m, red);
    float den = 0.f;
    for (int j = threadIdx.x; j < n; j += kThreads) {
        const float p = expf(s[j] - m);
        s[j] = p;
        den += p;
    }
    den = block_sum<kThreads>(den, red);
    den += expf(sink - m);
    const float inv = 1.f / den;
    for (int c = threadIdx.x; c < a.rank; c += kThreads) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
            acc = fmaf(s[j], load_row(cache + (size_t) cell * W + c), acc);
        }
        o_lat[((size_t) t * a.heads + h) * a.rank + c] = acc * inv;
    }
}

// cells[t][j] = j for j < fill[t] (rows with fill 0 are left alone).
__global__ void iota_cells_kernel(int32_t* __restrict__ cells, int stride, const int32_t* __restrict__ fill) {
    const int t = blockIdx.x, n = fill[t];
    for (int j = threadIdx.x; j < n; j += blockDim.x) cells[(size_t) t * stride + j] = j;
}

template <typename C>
void prepare_impl(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                  int n_tok, int ring, float* qcur, C* cache, void* stream) {
    prepare_kernel<C><<<dim3(a.heads + 1, n_tok), kThreads, 0, (cudaStream_t) stream>>>(q, q_abs, kv_raw, kv_norm, a, pos0, ring, qcur, cache);
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C>
void attend_impl(const float* qcur, const C* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                 float* o_lat, void* stream) {
    attend_kernel<C><<<dim3(a.heads, n_tok), kThreads, 0, (cudaStream_t) stream>>>(qcur, cache, r.firsts, r.counts, r.cells, r.cells_stride,
                                                                               sinks, a, ring, o_lat);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace

void q8_0_heads_gemv(const void* w, const float* x, int x_head_stride, int x_tok_stride, int heads, int rows, int k, int n_tok,
                     float* y, void* stream) {
    if (k % 32) throw std::invalid_argument("q8_0_heads_gemv: k is not a multiple of 32");
    heads_gemv_kernel<<<dim3((rows + 7) / 8, heads, n_tok), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) w, x, x_head_stride,
                                                                                         x_tok_stride, rows, k / 32, heads, y);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, float* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint16_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_attend(const float* qcur, const float* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, o_lat, stream);
}
void mla_attend(const float* qcur, const uint16_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, o_lat, stream);
}

void iota_cells(int32_t* cells, int stride, const int32_t* fill, int n_tok, void* stream) {
    iota_cells_kernel<<<n_tok, 256, 0, (cudaStream_t) stream>>>(cells, stride, fill);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
