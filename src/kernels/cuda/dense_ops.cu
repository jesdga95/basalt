// src/kernels/cuda/dense_ops.cu - see include/basalt/kernels/dense_ops.hpp.
#include "basalt/kernels/dense_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int HD = 256;   // attention head width

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

__global__ void rms_norm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                     int64_t cols, int64_t ld_out, float eps) {
    __shared__ float red[32];
    const float* xr = x + (int64_t) blockIdx.x * cols;
    float* yr = out + (int64_t) blockIdx.x * ld_out;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss = fmaf(xr[c], xr[c], ss);
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * inv * w[c];
}

__global__ void add_scaled_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ scale,
                                  int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale ? y[i] * *scale : y[i];
}

__global__ void scale_values_kernel(float* __restrict__ y, const float* __restrict__ scale, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= *scale;
}

__global__ void swiglu_scaled_kernel(const float* __restrict__ g, const float* __restrict__ u, const float* __restrict__ sg,
                                     const float* __restrict__ su, float* __restrict__ h, int64_t rows, int64_t width,
                                     int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const int64_t r = i / width, c = i % width;
    const float gv = g[r * ld + c] * (sg ? *sg : 1.0f), uv = u[r * ld + c] * (su ? *su : 1.0f);
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

// one warp per output (t, n): K values in fp32 against a BF16 row
__global__ void bf16_rows_dot_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y,
                                     int64_t T, int64_t N, int64_t K) {
    const int64_t o = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (o >= T * N) return;
    const int64_t t = o / N, n = o % N;
    const float* xr = x + t * K;
    const uint16_t* wr = w + n * K;
    float s = 0.0f;
    for (int64_t k = lane; k < K; k += 32) s = fmaf(__uint_as_float((uint32_t) wr[k] << 16), xr[k], s);
    s = warp_sum(s);
    if (lane == 0) y[o] = s;
}

// Block (row t, head j): j < n_head a query head, else a key/value head; one thread per head dim.
__global__ void __launch_bounds__(HD) attn_qk_prepare_kernel(const float* __restrict__ q_full, const float* __restrict__ k,
                                                           const float* __restrict__ v, const float* __restrict__ q_norm,
                                                           const float* __restrict__ k_norm, float eps, RopeSpec rope,
                                                           const int32_t* __restrict__ pos, int n_head, int n_kv,
                                                           float* __restrict__ q_out, __half* __restrict__ k_cache,
                                                           __half* __restrict__ v_cache, int64_t cell0) {
    __shared__ float red[HD / 32];
    __shared__ float xs[HD];
    const int64_t t = blockIdx.x;
    const int j = blockIdx.y, d = threadIdx.x;
    const bool is_q = j < n_head;
    const int head = is_q ? j : j - n_head;
    const float x = is_q ? q_full[(t * n_head + head) * 2 * HD + d] : k[(t * n_kv + head) * HD + d];
    const float inv = rsqrtf(block_sum(x * x, red) / (float) HD + eps);
    float y = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    const int half = rope.rot / 2;
    xs[d] = y;
    __syncthreads();
    if (d < rope.rot) {
        const int i = d < half ? d : d - half;
        const float inv_freq = (float) pow((double) rope.theta, -2.0 * (double) i / (double) rope.rot);
        const float ang = (float) pos[t * 3 + rope.axis[i]] * inv_freq;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        y = d < half ? y * cs - xs[d + half] * sn : y * cs + xs[d - half] * sn;
    }
    if (is_q) {
        q_out[(t * n_head + head) * HD + d] = y;
    } else {
        const int64_t at = ((cell0 + t) * n_kv + head) * HD + d;
        k_cache[at] = __float2half_rn(y);
        v_cache[at] = __float2half_rn(v[(t * n_kv + head) * HD + d]);
    }
}

__global__ void attn_gate_kernel(const float* __restrict__ attn, const float* __restrict__ q_full, float* __restrict__ out,
                                 int64_t n, int n_head) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t th = i / HD, d = i % HD;
    const float g = q_full[th * 2 * HD + HD + d];
    out[i] = attn[i] / (1.0f + expf(-g));
}

}  // namespace

void rms_norm_rows(const float* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream,
                   int64_t ld_out) {
    if (rows <= 0) return;
    rms_norm_rows_kernel<<<(unsigned) rows, 512, 0, (cudaStream_t) stream>>>(x, w, out, cols, ld_out > 0 ? ld_out : cols, eps);
    check("rms_norm_rows");
}

void add_scaled(float* x, const float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0) return;
    add_scaled_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, scale, n);
    check("add_scaled");
}

void scale_values(float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0 || scale == nullptr) return;
    scale_values_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(y, scale, n);
    check("scale_values");
}

void swiglu_scaled(const float* g, const float* u, const float* sg, const float* su, float* h, int64_t rows,
                   int64_t width, int64_t ld_gu, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks_for(rows * width), 256, 0, (cudaStream_t) stream>>>(g, u, sg, su, h, rows, width, ld_gu);
    check("swiglu_scaled");
}

void bf16_rows_dot(const float* x, const uint16_t* w, float* y, int64_t T, int64_t N, int64_t K, void* stream) {
    if (T <= 0) return;
    bf16_rows_dot_kernel<<<blocks_for(T * N * 32), 256, 0, (cudaStream_t) stream>>>(x, w, y, T, N, K);
    check("bf16_rows_dot");
}

void attn_qk_prepare(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                     float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv,
                     float* q_out, uint16_t* k_cache, uint16_t* v_cache, int64_t cell0, void* stream) {
    if (T <= 0) return;
    if (rope.rot <= 0 || rope.rot > HD || rope.rot % 2) {
        basalt::log::error("basalt", "attn_qk_prepare: a rotary width of %d", rope.rot);
        std::exit(1);
    }
    attn_qk_prepare_kernel<<<dim3((unsigned) T, (unsigned) (n_head + n_kv)), HD, 0, (cudaStream_t) stream>>>(
        q_full, k, v, q_norm, k_norm, eps, rope, pos, n_head, n_kv, q_out, (__half*) k_cache, (__half*) v_cache, cell0);
    check("attn_qk_prepare");
}

void attn_gate(const float* attn, const float* q_full, float* out, int64_t T, int n_head, void* stream) {
    const int64_t n = T * n_head * HD;
    if (n <= 0) return;
    attn_gate_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out, n, n_head);
    check("attn_gate");
}

}  // namespace basalt::kernels
