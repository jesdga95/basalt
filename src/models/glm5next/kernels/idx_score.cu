// src/models/glm5next/kernels/idx_score.cu - see idx_score.hpp: the DSA indexer's pool scores.
//
// One block per token; the token's iq and w go to shared memory once. One thread per pool walks the pools, computing
// each head's dot over head_dim (d ascending) then the relu-weighted sum over heads (h ascending) in the exact order
// the reference does, so the CPU oracle reproduces it bit for bit. fp32 pooled keys, no rotary (the architecture is
// nope), and no candidate levels - every visible pool is scored.
#include "models/glm5next/kernels/idx_score.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

__global__ void idx_score_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                 const float* __restrict__ pooled, const int32_t* __restrict__ pos,
                                 float* __restrict__ scores, int n_pool, int n_head, int head_dim, int pool,
                                 float w_scale) {
    extern __shared__ float smem[];
    float* q = smem;                          // [n_head][head_dim]
    float* ww = smem + n_head * head_dim;     // [n_head]
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n_head * head_dim; i += blockDim.x) q[i] = iq[(size_t) t * n_head * head_dim + i];
    if (threadIdx.x < n_head) ww[threadIdx.x] = w[(size_t) t * n_head + threadIdx.x];
    __syncthreads();

    const int nv = (int) ((pos[t] + 1) / pool);  // pools a token at pos[t] can see
    float* out = scores + (size_t) t * n_pool;
    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < n_pool; p += gridDim.y * blockDim.x) {
        if (p >= nv) {
            out[p] = -INFINITY;
            continue;
        }
        const float* prow = pooled + (size_t) p * head_dim;
        float s = 0.0f;
        for (int h = 0; h < n_head; ++h) {
            const float* qrow = q + h * head_dim;
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) acc += qrow[d] * prow[d];
            s += fmaxf(acc, 0.0f) * (ww[h] * w_scale);
        }
        out[p] = s;
    }
}

}  // namespace

void idx_score(const float* iq, const float* w, const float* pooled, const int32_t* pos, float* scores, int n_tokens,
               int n_pool, int n_head, int head_dim, int pool, float w_scale, cudaStream_t stream) {
    if (n_tokens <= 0 || n_pool <= 0 || n_head <= 0 || head_dim <= 0 || pool <= 0) return;
    const int threads = 128;
    const int row_groups = (n_pool + threads - 1) / threads;
    const int gy = row_groups < 65535 ? row_groups : 65535;
    const size_t smem = (size_t) (n_head * head_dim + n_head) * sizeof(float);
    const dim3 grid(n_tokens, gy);
    idx_score_kernel<<<grid, threads, smem, stream>>>(iq, w, pooled, pos, scores, n_pool, n_head, head_dim, pool,
                                                      w_scale);
}

}  // namespace basalt::models::glm5next
