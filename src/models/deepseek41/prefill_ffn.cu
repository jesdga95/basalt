// src/models/deepseek41/prefill_ffn.cu - the clamped SwiGLU between the routed experts' two MMQ products.
#include "models/deepseek41/prefill_ffn.hpp"

#include <cuda_runtime.h>

namespace basalt::models::deepseek41 {
namespace {

// h[r][k] = silu(min(gate, limit)) * clamp(up, -limit, limit); gate = gu[r][k], up = gu[r][n_ff + k]
__global__ void k_swiglu_split(const float* gu, float* h, int64_t n, int n_ff, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / n_ff;
    const int k = (int) (i - r * n_ff);
    const float g = fminf(gu[r * 2 * n_ff + k], limit);
    const float u = fminf(fmaxf(gu[r * 2 * n_ff + n_ff + k], -limit), limit);
    h[i] = g / (1.f + __expf(-g)) * u;
}

__global__ void k_wsum_tokens(const float* rows, const float* w, int n_rows, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = blockIdx.y;
    const float* r = rows + t * n_rows * n;
    const float* wt = w + t * n_rows;
    float s = 0.f;
    for (int k = 0; k < n_rows; ++k) s += wt[k] * r[(size_t) k * n + i];
    out[t * n + i] = s;
}

}  // namespace

void weighted_sum_tokens(const float* rows, const float* w, int n_tokens, int n_rows, int n, float* out, void* stream) {
    if (n_tokens <= 0) return;
    k_wsum_tokens<<<dim3((n + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(rows, w, n_rows, n, out);
}

void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream) {
    const int64_t n = rows * n_ff;
    if (n <= 0) return;
    k_swiglu_split<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gu, h, n, n_ff, limit);
}

}  // namespace basalt::models::deepseek41
