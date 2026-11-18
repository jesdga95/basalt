// src/models/hy4flash/kernels/mla.cu - the absorbed MLA of Hy4-flash for one token: the per-head Q8_0 products,
// the query and cache-row preparation, and the attention over the shared latent with a per-head sink.
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

// A warp per output row.
__global__ void heads_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int x_stride, int rows,
                                  int nb, float* __restrict__ y) {
    const int h = blockIdx.y, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const uint8_t* row = w + ((size_t) h * rows + r) * nb * 34;
    const float* xv = x + (size_t) h * x_stride;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float q = (float) (int8_t) blk[2 + lane];
        acc = fmaf(d * q, xv[b * 32 + lane], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) h * rows + r] = acc;
}

// Rotation of consecutive pairs (2i, 2i + 1) by pos * base^(-2i / n_rot).
__device__ __forceinline__ void rope_pair(float x0, float x1, int i, int n_rot, float base, int pos, float& o0, float& o1) {
    const float theta = (float) pos * powf(base, -(float) (2 * i) / (float) n_rot);
    float s, c;
    sincosf(theta, &s, &c);
    o0 = x0 * c - x1 * s;
    o1 = x0 * s + x1 * c;
}

// Block h < heads: the head's query row; block `heads`: the cache row.
template <typename C>
__global__ void prepare_kernel(const float* __restrict__ q, const float* __restrict__ q_abs, const float* __restrict__ kv_raw,
                               const float* __restrict__ kv_norm, MlaArgs a, int pos, float* __restrict__ qcur,
                               C* __restrict__ cache_row) {
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, W = a.rank + a.rope, Q = a.nope + a.rope;
    if (h < a.heads) {
        for (int c = threadIdx.x; c < a.rank; c += blockDim.x) qcur[(size_t) h * W + c] = q_abs[(size_t) h * a.rank + c];
        if (threadIdx.x < a.rope / 2) {
            const int i = threadIdx.x;
            const float* p = q + (size_t) h * Q + a.nope + 2 * i;
            float o0, o1;
            rope_pair(p[0], p[1], i, a.rope, a.rope_base, pos, o0, o1);
            qcur[(size_t) h * W + a.rank + 2 * i] = o0;
            qcur[(size_t) h * W + a.rank + 2 * i + 1] = o1;
        }
        return;
    }
    float ss = 0.f;
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) ss = fmaf(kv_raw[c], kv_raw[c], ss);
    ss = block_sum<kThreads>(ss, red);
    const float rs = rsqrtf(ss / (float) a.rank + a.rms_eps);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) store_row(cache_row + c, kv_raw[c] * rs * kv_norm[c]);
    if (threadIdx.x < a.rope / 2) {
        const int i = threadIdx.x;
        float o0, o1;
        rope_pair(kv_raw[a.rank + 2 * i], kv_raw[a.rank + 2 * i + 1], i, a.rope, a.rope_base, pos, o0, o1);
        store_row(cache_row + a.rank + 2 * i, o0);
        store_row(cache_row + a.rank + 2 * i + 1, o1);
    }
}

// A block per head: the scores (a warp per cell), the softmax with the sink, the weighted sum of the latents.
template <typename C>
__global__ void attend_kernel(const float* __restrict__ qcur, const C* __restrict__ cache, int first, int n,
                              const float* __restrict__ sinks, MlaArgs a, float* __restrict__ o_lat,
                              const int32_t* __restrict__ cells, int ring) {
    __shared__ float s[kMaxCells];
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, W = a.rank + a.rope, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* qv = qcur + (size_t) h * W;
    for (int j = warp; j < n; j += kThreads / 32) {
        const C* kr = cache + (size_t) (cells ? cells[j] : ring ? (first + j) % ring : first + j) * W;
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
        for (int j = 0; j < n; ++j) acc = fmaf(s[j], load_row(cache + (size_t) (cells ? cells[j] : ring ? (first + j) % ring : first + j) * W + c), acc);
        o_lat[(size_t) h * a.rank + c] = acc * inv;
    }
}

}  // namespace

void q8_0_heads_gemv(const void* w, const float* x, int x_stride, int heads, int rows, int k, float* y, void* stream) {
    if (k % 32) throw std::invalid_argument("q8_0_heads_gemv: k is not a multiple of 32");
    heads_gemv_kernel<<<dim3((rows + 7) / 8, heads), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) w, x, x_stride, rows,
                                                                                    k / 32, y);
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C>
static void prepare_impl(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos,
                         float* qcur, C* cache_row, void* stream) {
    prepare_kernel<C><<<a.heads + 1, kThreads, 0, (cudaStream_t) stream>>>(q, q_abs, kv_raw, kv_norm, a, pos, qcur, cache_row);
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C>
static void attend_impl(const float* qcur, const C* cache, int first, int n, const float* sinks, const MlaArgs& a, float* o_lat,
                        void* stream, const int32_t* cells, int ring) {
    if (n < 1 || n > kMaxCells) throw std::invalid_argument("mla_attend: cells");
    attend_kernel<C><<<a.heads, kThreads, 0, (cudaStream_t) stream>>>(qcur, cache, first, n, sinks, a, o_lat, cells, ring);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos,
                 float* qcur, float* cache_row, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos, qcur, cache_row, stream);
}
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos,
                 float* qcur, uint16_t* cache_row, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos, qcur, cache_row, stream);
}
void mla_attend(const float* qcur, const float* cache, int first, int n, const float* sinks, const MlaArgs& a, float* o_lat,
                void* stream, const int32_t* cells, int ring) {
    attend_impl(qcur, cache, first, n, sinks, a, o_lat, stream, cells, ring);
}
void mla_attend(const float* qcur, const uint16_t* cache, int first, int n, const float* sinks, const MlaArgs& a, float* o_lat,
                void* stream, const int32_t* cells, int ring) {
    attend_impl(qcur, cache, first, n, sinks, a, o_lat, stream, cells, ring);
}

}  // namespace basalt::models::hy4flash
