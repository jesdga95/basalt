// src/models/deepseek41/ops.cu - small device helpers for the DeepSeek layer loop.
#include "models/deepseek41/ops.hpp"

#include "basalt/kernels/dp4a.hpp"
#include "basalt/models/deepseek41/kernels.hpp"

#include <cuda_runtime.h>

#include <cstring>

namespace basalt::models::deepseek41 {
namespace {

__global__ void k_wait_word(const volatile uint32_t* word, uint32_t value) {
    while (*word != value) spin_pause();
    __threadfence_system();
}

__global__ void k_add(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_swiglu(const float* g, const float* u, int n, float limit, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = fminf(g[i], limit);
    const float uv = fminf(fmaxf(u[i], -limit), limit);
    out[i] = gv / (1.f + __expf(-gv)) * uv;
}

__global__ void k_wsum(const float* rows, const float* w, int n_rows, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int k = 0; k < n_rows; ++k) s += w[k] * rows[(size_t) k * n + i];
    out[i] = s;
}

__global__ void k_wsum_ptrs(RowPtrs r, const float* w, int n_rows, int n, float* out, const float* add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = k < n_rows ? r.p[k][i] : 0.f;   // all loads in flight before the sum
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (k < n_rows) s += w[k] * v[k];
    out[i] = add ? s + add[i] : s;
}

__global__ void k_wsum_window(WindowRowPtrs r, const float* w, int n_rows, int n, float* out, const float* add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = blockIdx.y;
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = k < n_rows ? r.p[t * n_rows + k][i] : 0.f;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (k < n_rows) s += w[t * n_rows + k] * v[k];
    out[(size_t) t * n + i] = add ? s + add[(size_t) t * n + i] : s;
}

__global__ void k_copy_words(const int* src, int* dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

struct PutWords {
    int w[kPutWords];
};

__global__ void k_put_words(const PutWords s, int* dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s.w[i];
}

__global__ void k_copy_rows(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

constexpr int kArgmaxBlocks = 64;
struct ArgmaxWs {
    float val[kArgmaxBlocks];
    int idx[kArgmaxBlocks];
    unsigned done;
};

// First largest value wins on ties (lowest index), as a sequential scan with a strict > does.
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// Each block scans a strided slice; the last block to finish reduces the block results.
__global__ void k_argmax(const float* x, int n, int* out, ArgmaxWs* ws) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ bool last;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (better(x[i], i, bv, bi)) bv = x[i], bi = i;
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) bv = ov, bi = oi;
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = blockDim.x >> 5;
        bv = threadIdx.x < nw ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < nw ? si[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, o);
            const int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        if (threadIdx.x == 0) {
            ws->val[blockIdx.x] = bv;
            ws->idx[blockIdx.x] = bi;
            __threadfence();
            last = atomicInc(&ws->done, gridDim.x - 1) == gridDim.x - 1;
        }
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    bv = -INFINITY;
    bi = 0x7fffffff;
    for (int b = threadIdx.x; b < gridDim.x; b += 32) {
        const float v = *(volatile float*) &ws->val[b];
        const int i = *(volatile int*) &ws->idx[b];
        if (better(v, i, bv, bi)) bv = v, bi = i;
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) bv = ov, bi = oi;
    }
    if (threadIdx.x == 0) *out = bi == 0x7fffffff ? 0 : bi;
}

}  // namespace

size_t argmax_workspace_bytes() { return sizeof(ArgmaxWs); }

void argmax_index(const float* x, int n, int32_t* out, void* ws, void* stream) {
    k_argmax<<<kArgmaxBlocks, 256, 0, (cudaStream_t) stream>>>(x, n, out, (ArgmaxWs*) ws);
}

void weighted_sum_window(const WindowRowPtrs& rows, const float* w, int n_tokens, int n_rows, int n, float* out,
                         void* stream, const float* add) {
    k_wsum_window<<<dim3((n + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(rows, w, n_rows, n, out, add);
}

void argmax_rows(const float* x, int n_rows, int n, int32_t* out, void* ws, void* stream) {
    for (int r = 0; r < n_rows; ++r) argmax_index(x + (size_t) r * n, n, out + r, ws, stream);
}

void wait_word(const uint32_t* word, uint32_t value, void* stream) {
    k_wait_word<<<1, 1, 0, (cudaStream_t) stream>>>(word, value);
}

void copy_words(const void* src, void* dst, int n_words, void* stream) {
    k_copy_words<<<1, 64, 0, (cudaStream_t) stream>>>((const int*) src, (int*) dst, n_words);
}

void put_words(const void* src, void* dst, int n_words, void* stream) {
    PutWords s{};
    std::memcpy(s.w, src, (size_t) n_words * 4);
    k_put_words<<<1, kPutWords, 0, (cudaStream_t) stream>>>(s, (int*) dst, n_words);
}

void copy_rows(const float* src, float* dst, int n, void* stream) {
    const int n4 = n / 4;
    k_copy_rows<<<(n4 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const float4*>(src),
                                                                      reinterpret_cast<float4*>(dst), n4);
}

void weighted_sum_row_ptrs(const RowPtrs& rows, const float* w, int n_rows, int n, float* out, void* stream,
                           const float* add) {
    k_wsum_ptrs<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(rows, w, n_rows, n, out, add);
}

void weighted_sum_rows(const float* rows, const float* w, int n_rows, int n, float* out, void* stream) {
    k_wsum<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(rows, w, n_rows, n, out);
}

void swiglu_clamped(const float* gate, const float* up, int n, float limit, float* out, void* stream) {
    k_swiglu<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(gate, up, n, limit, out);
}

void rms_norm(const float* x, const float* w, int n_tokens, int n, float eps, float* out, void* stream) {
    rms_norm_rows(x, w, n_tokens, n, eps, out, stream);
}

void add_inplace(float* out, const float* a, int n, void* stream) { add(out, a, out, n, stream); }

void add(const float* a, const float* b, float* out, int n, void* stream) {
    k_add<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(a, b, out, n);
}

}  // namespace basalt::models::deepseek41
