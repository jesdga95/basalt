// src/models/hy4flash/kernels/ops.cu - the sigmoid gate, small products, the stream fill and the Engram apply of
// Hy4-flash.
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
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

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

__global__ void mul_sigmoid_kernel(const float* __restrict__ g, float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf(g[i]);
}

// A warp per row.
__global__ void rows_gemv_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_expert, int n_embd,
                                 float scale, float* __restrict__ logits) {
    const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (e >= n_expert) return;
    const float4* w = reinterpret_cast<const float4*>(gate + (size_t) e * n_embd);
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * n_embd);
    float acc = 0.f;
    for (int i = lane; i < n_embd / 4; i += 32) {
        const float4 a = xv[i], b = w[i];
        acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[(size_t) t * n_expert + e] = acc * scale;
}

__global__ void wsum_kernel(const float* const* __restrict__ rows, const float* __restrict__ w, int k, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += w[t * k + e] * rows[t * k + e][i];
    out[(size_t) t * n + i] = acc;
}

struct PutWords {
    int w[kPutWords];
};

__global__ void put_words_kernel(const PutWords s, int* dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s.w[i];
}

template <bool TAILS>
__global__ void swiglu_split_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int n_ff, float limit,
                                    ExpertTails t) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i - r * n_ff;
    float g = gu[r * 2 * n_ff + k], u = gu[r * 2 * n_ff + n_ff + k], sd = 1.f;
    if constexpr (TAILS) {
        int q = 0;   // the row's expert: the last q with bounds[q] <= r
        while (q + 1 < t.n && t.bounds[q + 1] <= r) ++q;
        const float* s = (const float*) (t.blobs + (size_t) q * t.stride + t.tail_off);
        g *= s[0];
        u *= s[1];
        sd = s[2];
    }
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    if constexpr (TAILS) h[i] = g / (1.f + expf(-g)) * u * sd;
    else h[i] = g / (1.f + expf(-g)) * u;
}

__global__ void wsum_tokens_kernel(const float* __restrict__ rows, const int32_t* __restrict__ map, const float* __restrict__ w, int K, int n,
                                   float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += w[t * K + k] * rows[(size_t) map[t * K + k] * n + i];
    out[(size_t) t * n + i] = acc;
}

__global__ void fill_kernel(const float* __restrict__ e, int n, float* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float v = e[(size_t) t * n + i];
    for (int s = 0; s < 4; ++s) x[((size_t) t * 4 + s) * n + i] = v;
}

constexpr int kEngramThreads = 256;

// A block per stream.
__global__ void engram_kernel(const float* __restrict__ kv, const float* __restrict__ kv_b, const float* __restrict__ gate_w,
                              const float* __restrict__ pre_gamma, const float* __restrict__ hc_scale,
                              const float* __restrict__ valid_rows, EngramArgs a, float* __restrict__ x) {
    __shared__ float red[kEngramThreads / 32];
    const int i = blockIdx.x, t = blockIdx.y, n = a.n_embd;
    const float valid = valid_rows[t];
    float* xi = x + ((size_t) t * 4 + i) * n;
    const float* kvt = kv + (size_t) t * 5 * n;
    const float* key = kvt + (size_t) i * n;
    const float* kb = kv_b + (size_t) i * n;
    const float* val = kvt + (size_t) 4 * n;
    const float* vb = kv_b + (size_t) 4 * n;
    float ssx = 0.f, ssk = 0.f, ssv = 0.f;
    for (int j = threadIdx.x; j < n; j += kEngramThreads) {
        const float k = key[j] + kb[j], v = val[j] + vb[j];
        ssx = fmaf(xi[j], xi[j], ssx);
        ssk = fmaf(k, k, ssk);
        ssv = fmaf(v, v, ssv);
    }
    ssx = block_sum<kEngramThreads>(ssx, red);
    ssk = block_sum<kEngramThreads>(ssk, red);
    ssv = block_sum<kEngramThreads>(ssv, red);
    const float rx = rsqrtf(ssx / (float) n + a.gate_eps), rk = rsqrtf(ssk / (float) n + a.gate_eps), rv = rsqrtf(ssv / (float) n + a.eps);
    float dot = 0.f;
    for (int j = threadIdx.x; j < n; j += kEngramThreads)
        dot = fmaf((xi[j] * rx) * ((key[j] + kb[j]) * rk), gate_w[(size_t) i * n + j], dot);
    dot = block_sum<kEngramThreads>(dot, red) * rsqrtf((float) n);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gate = sigmoidf(dot < 0.f ? -mag : mag);
    for (int j = threadIdx.x; j < n; j += kEngramThreads) {
        const float signal = ((val[j] + vb[j]) * rv) * pre_gamma[j];
        xi[j] += signal * (gate + hc_scale[(size_t) i * n + j]) * valid;
    }
}

}  // namespace

void mul_sigmoid(const float* g, float* x, int n, void* stream) {
    mul_sigmoid_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(g, x, n);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void f32_gemm_rows(const float* x, const float* w, int rows, int cols, float scale, int n_tok, float* y, void* stream) {
    if (cols % 4) throw std::invalid_argument("f32_gemm_rows: cols");
    rows_gemv_kernel<<<dim3((rows + 7) / 8, n_tok), 256, 0, (cudaStream_t) stream>>>(x, w, rows, cols, scale, y);
    if (!stream) cudaStreamSynchronize(nullptr);
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// one row a block of 512: f = a + b, its RMS over the row exactly as rms_norm_rows takes it (same threads, same sums), then
// hc_post's four streams from f * inv * w, each value computed as the three kernels compute it, so the result is theirs bit
// for bit
__global__ void add_norm_hc_post_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ w,
                                        float eps, const float* __restrict__ x, const float* __restrict__ coef, int coef_stride,
                                        int E, float* __restrict__ out) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float* ar = a + (size_t) t * E;
    const float* br = b ? b + (size_t) t * E : nullptr;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float f = br ? ar[c] + br[c] : ar[c];
        ss = fmaf(f, f, ss);
    }
    {   // block_sum of rms_norm_rows
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        __syncthreads();
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float s = 0.0f;
        for (int i = 0; i < nw; ++i) s += red[i];
        ss = s;
    }
    const float inv = rsqrtf(ss / (float) E + eps);
    const float* cf = coef + (size_t) t * coef_stride;
    const float* xr = x + (size_t) t * 4 * E;
    float* orow = out + (size_t) t * 4 * E;
    const int n4 = E >> 2;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int c = 4 * i;
        float4 f4 = make_float4(ar[c], ar[c + 1], ar[c + 2], ar[c + 3]);
        if (br) f4 = make_float4(f4.x + br[c], f4.y + br[c + 1], f4.z + br[c + 2], f4.w + br[c + 3]);
        const float4 fv = make_float4(f4.x * inv * w[c], f4.y * inv * w[c + 1], f4.z * inv * w[c + 2], f4.w * inv * w[c + 3]);
        float4 xs[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) xs[s] = reinterpret_cast<const float4*>(xr)[(size_t) s * n4 + i];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float post = cf[4 + d];
            float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float cc = cf[8 + s * 4 + d];
                o.x = fmaf(xs[s].x, cc, o.x); o.y = fmaf(xs[s].y, cc, o.y);
                o.z = fmaf(xs[s].z, cc, o.z); o.w = fmaf(xs[s].w, cc, o.w);
            }
            reinterpret_cast<float4*>(orow)[(size_t) d * n4 + i] = o;
        }
    }
}

void add_norm_hc_post(const float* a, const float* b, const float* w, float eps, const float* x, const float* coef, int coef_stride,
                      int n_tokens, int E, float* out, void* stream) {
    add_norm_hc_post_kernel<<<n_tokens, 512, 0, (cudaStream_t) stream>>>(a, b, w, eps, x, coef, coef_stride, E, out);
}

// a sliding-window layer's row spans from the window's first position, read on the card (so a captured graph replays them for any
// position): firsts, counts, fill and positions, `arows` apart
__global__ void swa_meta_kernel(const int32_t* __restrict__ pos0, int R, int window, int arows, int32_t* __restrict__ meta) {
    const int t = threadIdx.x;
    if (t >= R) return;
    const int pos = *pos0 + t, first = pos + 1 - window > 0 ? pos + 1 - window : 0;
    meta[t] = first;
    meta[arows + t] = pos + 1 - first;
    meta[2 * arows + t] = 0;
    meta[3 * arows + t] = pos;
}

void swa_meta(const int32_t* pos0, int R, int window, int arows, int32_t* meta, void* stream) {
    swa_meta_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(pos0, R, window, arows, meta);
}

void add_vectors(const float* a, const float* b, float* out, int n, void* stream) {
    add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(a, b, out, n);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream) {
    wsum_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(rows, w, K, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void put_words(const void* src, void* dst, int n_words, void* stream) {
    if (n_words < 1 || n_words > kPutWords) throw std::invalid_argument("put_words: n_words");
    PutWords s;
    std::memcpy(s.w, src, (size_t) n_words * 4);
    put_words_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(s, (int*) dst, n_words);
}

void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream, const ExpertTails* tails) {
    if (rows <= 0) return;
    const unsigned blocks = (unsigned) ((rows * n_ff + 255) / 256);
    if (tails && tails->n > 0) swiglu_split_kernel<true><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, limit, *tails);
    else swiglu_split_kernel<false><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, limit, ExpertTails{});
    if (!stream) cudaStreamSynchronize(nullptr);
}

void weighted_sum_tokens(const float* rows, const int32_t* map, const float* w, int n_tok, int K, int n, float* out, void* stream) {
    wsum_tokens_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(rows, map, w, K, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void fill_streams(const float* e, int n, int n_tok, float* x, void* stream) {
    fill_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(e, n, x);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void engram_apply(const float* kv, const float* kv_b, const float* gate_w, const float* pre_gamma, const float* hc_scale,
                  const float* valid, int n_tok, const EngramArgs& a, float* x, void* stream) {
    engram_kernel<<<dim3(4, n_tok), kEngramThreads, 0, (cudaStream_t) stream>>>(kv, kv_b, gate_w, pre_gamma, hc_scale, valid, a, x);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
