// src/models/gemma4/kernels/glue.cu - the glue between the products (basalt/models/gemma4/kernels.hpp): a block per row,
// its sums in a fixed order, so a row's bits do not depend on the rows beside it.
#include "basalt/models/gemma4/kernels.hpp"

#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;

constexpr int kThreads = 256;

void check(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(e));
}

// The block's sum of v (every thread gets it), in a fixed order.
__device__ float block_sum(float v) {
    __shared__ float part[kThreads / 32];
    __shared__ float total;
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < kThreads / 32; ++i) s += part[i];
        total = s;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 1 / rms of a row of `width` values.
__device__ float inv_rms(const float* __restrict__ x, int width, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < width; i += kThreads) ss += x[i] * x[i];
    return 1.0f / sqrtf(block_sum(ss) / (float) width + eps);
}

__global__ void __launch_bounds__(kThreads) embed_kernel(const uint16_t* table, const int32_t* __restrict__ tokens, int width,
                                                          float scale, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x;
    const uint16_t* row = table + (size_t) tokens[r] * width;
    for (int i = threadIdx.x; i < width; i += kThreads)
        out[(size_t) r * width + i] = __uint_as_float((uint32_t) row[i] << 16) * scale;
}

__global__ void __launch_bounds__(kThreads) add_norm_kernel(float* __restrict__ x, const float* __restrict__ y,
                                                             const float* __restrict__ w, float eps, int width) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x;
    const float* yr = y + (size_t) r * width;
    const float inv = inv_rms(yr, width, eps);
    for (int i = threadIdx.x; i < width; i += kThreads) x[(size_t) r * width + i] += yr[i] * inv * w[i];
}

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void __launch_bounds__(kThreads) gelu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, int n,
                                                             int64_t ld_gu, int64_t ld, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * kThreads + threadIdx.x;
    if (i >= ld) return;
    out[r * ld + i] = i < n ? gelu_tanh(g[r * ld_gu + i]) * u[r * ld_gu + i] : 0.f;
}

// One row: the experts' sum (slot order) and the MLP output normed and added, the sum normed, the residual and the scalar.
// `buf` holds the row's experts' sum while the norms read it.
__global__ void __launch_bounds__(kThreads) ffn_end_kernel(FfnEnd f, int width) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    extern __shared__ float sm[];   // [2][width]: the MLP part, then the experts' sum
    const int r = blockIdx.x;
    float* a = sm;
    float* b = sm + width;
    const float* mlp = f.mlp + (size_t) r * width;
    if (f.experts) {
        for (int i = threadIdx.x; i < width; i += kThreads) {
            float s = 0.f;
            for (int k = 0; k < f.k; ++k) s += f.weights[(size_t) r * f.k + k] * f.experts[((size_t) r * f.k + k) * width + i];
            b[i] = s;
        }
        __syncthreads();
        const float i1 = inv_rms(mlp, width, f.eps);
        const float i2 = inv_rms(b, width, f.eps);
        for (int i = threadIdx.x; i < width; i += kThreads) a[i] = mlp[i] * i1 * f.n1[i] + b[i] * i2 * f.n2[i];
    } else {
        for (int i = threadIdx.x; i < width; i += kThreads) a[i] = mlp[i];
    }
    __syncthreads();
    const float ip = inv_rms(a, width, f.eps);
    const float sc = f.scalar[0];
    float* x = f.x + (size_t) r * width;
    for (int i = threadIdx.x; i < width; i += kThreads) x[i] = (x[i] + a[i] * ip * f.n_post[i]) * sc;
}

__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float inv = 1.0f / cap;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = cap * tanhf(x[i] * inv);
}

}  // namespace

void embed_rows(const uint16_t* table, const int32_t* tokens, int rows, int width, float scale, float* out, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(embed_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, table, tokens, width, scale, out);
    check("embed_rows");
}

void add_norm(float* x, const float* y, const float* w, float eps, int rows, int width, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(add_norm_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, x, y, w, eps, width);
    check("add_norm");
}

void gelu_mul(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, float* out, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(gelu_mul_kernel, dim3((unsigned) ((ld + kThreads - 1) / kThreads), (unsigned) rows), dim3(kThreads), 0,
                   (cudaStream_t) stream, g, u, n, ld_gu, ld, out);
    check("gelu_mul");
}

void ffn_end(const FfnEnd& f, int rows, int width, void* stream) {
    if (rows <= 0) return;
    const size_t smem = (size_t) 2 * width * sizeof(float);
    cudaFuncSetAttribute(ffn_end_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    bk::launch_pdl(ffn_end_kernel, dim3((unsigned) rows), dim3(kThreads), smem, (cudaStream_t) stream, f, width);
    check("ffn_end");
}

void softcap(float* x, int64_t n, float cap, void* stream) {
    if (n <= 0) return;
    bk::launch_pdl(softcap_kernel, dim3(1024), dim3(256), 0, (cudaStream_t) stream, x, n, cap);
    check("softcap");
}

}  // namespace basalt::models::gemma4
