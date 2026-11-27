// src/models/glm5next/kernels/kpool.cu - see kpool.hpp: the DSA indexer's key pooling.
//
// One thread per (pool, channel). The softmax is over a pool's own <= kPoolMax slots, so each thread holds the slot
// scores and keys in registers and reduces them itself - no cross-thread reduction. The stable softmax subtracts the
// slot max (llama.cpp soft_max), then weights the keys by exp(score - max) / sum(exp). The pool's result depends only
// on its own slots, so a window of pools is fully parallel.
#include "models/glm5next/kernels/kpool.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

__global__ void kpool_compress_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                      const float* __restrict__ ape, float* __restrict__ out, int n_pools, int kpool,
                                      int head_dim) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_pools * head_dim;
    if (gid >= total) return;
    const int pool = gid / head_dim;
    const int d = gid - pool * head_dim;

    const size_t base = (size_t) pool * kpool * head_dim + d;
    const float* krow = key + base;
    const float* grow = gate + base;

    float sc[kPoolMax];
    float m = -3.402823466e+38f;  // -FLT_MAX
    for (int s = 0; s < kpool; ++s) {
        sc[s] = grow[(size_t) s * head_dim] + ape[(size_t) s * head_dim + d];
        m = fmaxf(m, sc[s]);
    }
    float den = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < kpool; ++s) {
        const float e = expf(sc[s] - m);
        den += e;
        acc = fmaf(e, krow[(size_t) s * head_dim], acc);
    }
    out[(size_t) pool * head_dim + d] = acc / den;
}

}  // namespace

void kpool_compress(const float* key, const float* gate, const float* ape, float* out, int n_pools, int kpool,
                    int head_dim, cudaStream_t stream) {
    if (n_pools <= 0 || kpool <= 0 || kpool > kPoolMax || head_dim <= 0) return;
    const int total = n_pools * head_dim;
    const int block = 256;
    kpool_compress_kernel<<<(total + block - 1) / block, block, 0, stream>>>(key, gate, ape, out, n_pools, kpool,
                                                                              head_dim);
}

}  // namespace basalt::models::glm5next
