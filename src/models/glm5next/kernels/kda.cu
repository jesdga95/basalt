// src/models/glm5next/kernels/kda.cu - see kda.hpp: the recurrent KDA step (gated delta rule, per-channel decay).
//
// One block per head, one thread per value channel (one state row). The head's q, k and the decay factor live in
// shared memory; each thread streams its own row from the state twice - once to accumulate the two dot products
// (against q and k, both after the decay), once to write back the decayed row plus its rank-1 piece - and keeps
// nothing but scalars in registers. The read o = (h_new) q = dot(row .* g, q) + u * dot(q, k), so it needs no
// third pass. dk <= 256 and a multiple of 4 (the model's is 128); the block is dv rounded to a warp multiple (the
// model's is 128 too), and the head's row-owning threads are the first dv of it.
#include "models/glm5next/kernels/kda.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

constexpr int kMaxDk = 256;

__global__ void kda_step_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta,
                                float* __restrict__ out, int dk, int dv) {
    __shared__ float sqg[kMaxDk], skg[kMaxDk], sk[kMaxDk], sdotqk;
    const int head = blockIdx.x;
    const int tid = threadIdx.x;

    // The head's vectors, and the two products every row needs. sdotqk = dot(q, k) folded in a fixed order (thread
    // 0, ascending j) so the fold is bit-stable across launches.
    if (tid < dk) {
        const float qj = q[(size_t) head * dk + tid];
        const float kj = k[(size_t) head * dk + tid];
        const float gj = g[(size_t) head * dk + tid];
        sqg[tid] = qj * gj;
        skg[tid] = kj * gj;
        sk[tid] = kj;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int j = 0; j < dk; ++j) acc = fmaf(q[(size_t) head * dk + j], k[(size_t) head * dk + j], acc);
        sdotqk = acc;
    }
    __syncthreads();

    if (tid >= dv) return;
    float* row = state + ((size_t) head * dv + tid) * dk;

    float acc_o = 0.0f, acc_v = 0.0f;  // dot(row, q .* g) and dot(row, k .* g): v' is against the DECAYED state
    for (int j = 0; j < dk; ++j) {
        const float x = row[j];
        acc_o = fmaf(x, sqg[j], acc_o);
        acc_v = fmaf(x, skg[j], acc_v);
    }
    const float u = beta[head] * (v[(size_t) head * dv + tid] - acc_v);
    for (int j = 0; j < dk; ++j) row[j] = fmaf(sk[j], u, row[j] * g[(size_t) head * dk + j]);
    out[(size_t) head * dv + tid] = acc_o + u * sdotqk;
}

}  // namespace

void kda_step(const float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int n_heads, int dk, int dv, cudaStream_t stream) {
    if (n_heads <= 0 || dk <= 0 || dk > kMaxDk || dv <= 0) return;
    const int threads = ((dv + 31) / 32) * 32;
    kda_step_kernel<<<n_heads, threads, 0, stream>>>(const_cast<float*>(state), q, k, v, g, beta, out, dk, dv);
}

}  // namespace basalt::models::glm5next
