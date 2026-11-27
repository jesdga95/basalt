// src/models/glm5next/kernels/expert_sum.cu - see expert_sum.hpp.
//
// One block of the model's width per token of the window: a thread owns one channel of one token's row and walks the
// token's K expert rows in order, so the sum comes out in the same order as the host reference the port was checked
// against. The rows arrive as pointers because they are made in three different places; every load is issued before
// any of them is used, which is what keeps the mapped host rows off the critical path.
#include "models/glm5next/kernels/expert_sum.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

__global__ void k_weighted_sum_block(RowPtrs r, const float* __restrict__ w, int n_tokens, int K, int n,
                                     float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = blockIdx.y;
    float v[kMaxExpertsPerToken];
#pragma unroll
    for (int k = 0; k < kMaxExpertsPerToken; ++k) v[k] = k < K ? r.p[t * K + k][i] : 0.f;   // all loads in flight first
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < kMaxExpertsPerToken; ++k)
        if (k < K) s += w[(size_t) t * K + k] * v[k];
    out[(size_t) t * n + i] = s;
}

constexpr int kPutThreads = 256;
struct PutWords {
    int w[kPlanWords];
};

__global__ void k_put_words(const PutWords s, int* __restrict__ dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s.w[i];
}

}  // namespace

void weighted_sum_block(const RowPtrs& rows, const float* w, int n_tokens, int K, int n, float* out,
                        cudaStream_t stream) {
    k_weighted_sum_block<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tokens), 256, 0, stream>>>(
        rows, w, n_tokens, K, n, out);
}

void put_words(const void* src, void* dst, int n_words, cudaStream_t stream) {
    PutWords s;
    __builtin_memcpy(s.w, src, (size_t) n_words * 4);
    k_put_words<<<1, kPutThreads, 0, stream>>>(s, (int*) dst, n_words);
}

}  // namespace basalt::models::glm5next
