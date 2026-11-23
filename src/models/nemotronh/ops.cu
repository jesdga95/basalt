// src/models/nemotronh/ops.cu - see ops.hpp.
#include "models/nemotronh/ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace basalt::models::nemotronh {
namespace {

__global__ void weighted_sum_kernel(const float* const* rows, const float* w, int K, int n, float* out) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc = __fmaf_rn(w[t * K + k], rows[t * K + k][i], acc);
        out[(size_t) t * n + i] = acc;
    }
}

__global__ void relu2_scaled_kernel(float* h, int64_t rows, int n, const int32_t* bounds, int ne, const float* tails) {
    const int64_t r = blockIdx.y;
    if (r >= rows) return;
    int j = 0;
    while (j + 1 < ne && bounds[j + 1] <= r) ++j;
    const float si = tails[4 * j], sd = tails[4 * j + 2];
    float* row = h + r * n;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        const float x = fmaxf(__fmul_rn(row[c], si), 0.0f);
        row[c] = __fmul_rn(__fmul_rn(x, x), sd);
    }
}

__global__ void relu2_kernel(float* x, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float r = fmaxf(x[i], 0.0f);
        x[i] = __fmul_rn(r, r);
    }
}

__global__ void kv_store_kernel(const float* k, const float* v, int64_t n, int64_t base, __half* kc, __half* vc) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        kc[base + i] = __float2half_rn(k[i]);
        vc[base + i] = __float2half_rn(v[i]);
    }
}

__global__ void add2_kernel(float* x, const float* a, const float* b, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = b ? __fadd_rn(x[i], __fadd_rn(a[i], b[i])) : __fadd_rn(x[i], a[i]);
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096); }

}  // namespace

void weighted_sum(const float* const* rows, const float* w, int T, int K, int n, float* out, void* stream) {
    if (T <= 0) return;
    const dim3 grid((unsigned) ((n + 255) / 256), (unsigned) T);
    weighted_sum_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(rows, w, K, n, out);
}

void relu2_scaled_rows(float* h, int64_t rows, int n, const int32_t* bounds, int n_experts, const float* tails, void* stream) {
    if (rows <= 0) return;
    const dim3 grid((unsigned) ((n + 255) / 256), (unsigned) rows);
    relu2_scaled_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(h, rows, n, bounds, n_experts, tails);
}

void relu2(float* x, int64_t n, void* stream) {
    if (n > 0) relu2_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, n);
}

void kv_store(const float* k, const float* v, int T, int64_t pos0, int n_kv, int hd, uint16_t* kc, uint16_t* vc, void* stream) {
    const int64_t n = (int64_t) T * n_kv * hd;
    if (n > 0)
        kv_store_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(k, v, n, pos0 * n_kv * hd, (__half*) kc, (__half*) vc);
}

void add2(float* x, const float* a, const float* b, int64_t n, void* stream) {
    if (n > 0) add2_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, a, b, n);
}

}  // namespace basalt::models::nemotronh
