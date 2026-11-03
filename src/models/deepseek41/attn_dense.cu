// src/models/deepseek41/attn_dense.cu - small kernels of the batched attention's dense stages.
#include "models/deepseek41/attn_dense.hpp"

#include <cuda_runtime.h>

namespace basalt::models::deepseek41 {
namespace {

// One block per row; the same reduction order as the single-token query path.
__global__ void __launch_bounds__(256) rms_norm_rows_kernel(float* x, const float* w, int n, float eps) {
    float* row = x + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss = fmaf(row[i], row[i], ss);
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float scale = rsqrtf(tot / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) row[i] = row[i] * scale * w[i];
}

}  // namespace

void rms_norm_rows(float* x, const float* w, int n, float eps, int rows, void* stream) {
    if (rows > 0) rms_norm_rows_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, n, eps);
}

}  // namespace basalt::models::deepseek41
