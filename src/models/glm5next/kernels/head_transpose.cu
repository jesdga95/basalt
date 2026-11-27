// src/models/glm5next/kernels/head_transpose.cu - see head_transpose.hpp.
//
// The MLA's per-head weights ([kv_lora][v_dim][n_head] and [qk_dim][kv_lora][n_head]) are one matrix a head, so the
// products over a window want one head's tokens as a contiguous [token][width] block: that is the layout every
// shared projection kernel in the engine reads. The attention and the query projection write the other one, so the
// window pays one permutation. Four floats a thread where the width allows it; the copy is bandwidth, not work.
#include "models/glm5next/kernels/head_transpose.hpp"

namespace basalt::models::glm5next {
namespace {

__global__ void head_major_kernel(const float4* __restrict__ x, float4* __restrict__ xt, int n_tok, int n_head,
                                  int width4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tok * n_head * width4) return;
    const int t = i / (n_head * width4);
    const int r = i - t * n_head * width4;
    const int h = r / width4;
    const int d = r - h * width4;
    xt[((size_t) h * n_tok + t) * width4 + d] = x[((size_t) t * n_head + h) * width4 + d];
}

__global__ void head_store_kernel(const float4* __restrict__ src, float4* __restrict__ y, int n_tok, int n_head,
                                  int head, int width4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tok * width4) return;
    const int t = i / width4;
    const int d = i - t * width4;
    y[((size_t) t * n_head + head) * width4 + d] = src[(size_t) t * width4 + d];
}

}  // namespace

void head_major(const float* x, float* xt, int n_tok, int n_head, int width, cudaStream_t stream) {
    if (width % 4 || n_tok <= 0 || n_head <= 0) return;
    const int total = n_tok * n_head * (width / 4);
    head_major_kernel<<<(total + 255) / 256, 256, 0, stream>>>((const float4*) x, (float4*) xt, n_tok, n_head,
                                                                width / 4);
}

void head_store(const float* src, float* y, int n_tok, int n_head, int head, int width, cudaStream_t stream) {
    if (width % 4 || n_tok <= 0 || n_head <= 0 || head < 0 || head >= n_head) return;
    const int total = n_tok * (width / 4);
    head_store_kernel<<<(total + 255) / 256, 256, 0, stream>>>((const float4*) src, (float4*) y, n_tok, n_head, head,
                                                                width / 4);
}

}  // namespace basalt::models::glm5next
