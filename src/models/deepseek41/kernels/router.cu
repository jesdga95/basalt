// Router: adapted from the sqrt-softplus gating and top-k selection of build_moe_ffn in the llama.cpp DeepSeek-V4.1
// support (src/llama-graph.cpp).
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/models/deepseek41/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kMaxTok = 8;          // tokens per logits launch
constexpr int kLogitThreads = 256;  // one block per gate row
constexpr int kMaxVec = 8;          // float4 per thread: n_embd <= 4 * kMaxVec * kLogitThreads
constexpr int kMaxExpert = 512;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Stage: one gate row against up to kMaxTok token vectors. The row is loaded once into registers, each thread owns
// fixed columns, and the warp and block sums run in a fixed order for every token.
__device__ void router_logits_row(const float* __restrict__ gate_row, const float* __restrict__ x, int n_embd,
                                  int n_tokens, float* __restrict__ logits, int n_expert, int row, float* red) {
    const int n4 = n_embd >> 2;
    float4 w[kMaxVec];
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = threadIdx.x + c * kLogitThreads;
        if (i < n4) w[c] = reinterpret_cast<const float4*>(gate_row)[i];
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int t = 0; t < kMaxTok; ++t) {
        if (t >= n_tokens) break;
        const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * n_embd);
        float acc = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxVec; ++c) {
            const int i = threadIdx.x + c * kLogitThreads;
            if (i < n4) {
                const float4 v = xv[i];
                acc = fmaf(w[c].x, v.x, acc);
                acc = fmaf(w[c].y, v.y, acc);
                acc = fmaf(w[c].z, v.z, acc);
                acc = fmaf(w[c].w, v.w, acc);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) red[t * (kLogitThreads / 32) + warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x < n_tokens) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kLogitThreads / 32; ++i) s += red[threadIdx.x * (kLogitThreads / 32) + i];
        logits[(size_t) threadIdx.x * n_expert + row] = s;
    }
}

__global__ void __launch_bounds__(kLogitThreads)
router_logits_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_embd, int n_expert,
                     int n_tokens, float* __restrict__ logits) {
    __shared__ float red[kMaxTok * (kLogitThreads / 32)];
    const int row = blockIdx.x;
    router_logits_row(gate + (size_t) row * n_embd, x, n_embd, n_tokens, logits, n_expert, row, red);
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : logf(1.f + expf(x));
    return sqrtf(sp);
}

// Stage: one token, one thread per expert. A thread ranks its biased score against all others (higher score first,
// the lower index wins a tie); ranks below k are the selection, in rank order.
__device__ void router_select_token(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                    int k, float scale, int32_t* __restrict__ ids, float* __restrict__ weights,
                                    float* sb, float* sel) {
    const int e = threadIdx.x;
    const float s = sqrt_softplus(logits[e]);
    const float b = s + bias[e];
    sb[e] = b;
    __syncthreads();
    int rank = 0;
    const int n4 = n_expert >> 2;
    for (int j4 = 0; j4 < n4; ++j4) {
        const float4 o = reinterpret_cast<const float4*>(sb)[j4];
        const int j = j4 * 4;
        rank += (o.x > b) | ((o.x == b) & (j < e));
        rank += (o.y > b) | ((o.y == b) & (j + 1 < e));
        rank += (o.z > b) | ((o.z == b) & (j + 2 < e));
        rank += (o.w > b) | ((o.w == b) & (j + 3 < e));
    }
    if (rank < k) {
        sel[rank] = s;
        ids[rank] = e;
    }
    __syncthreads();
    if (e == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += sel[r];
        const float denom = sum + 1e-20f;
        for (int r = 0; r < k; ++r) weights[r] = sel[r] / denom * scale;
    }
}

__global__ void router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                     int k, float scale, int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ __align__(16) float sb[kMaxExpert];
    __shared__ float sel[8];
    const int t = blockIdx.x;
    router_select_token(logits + (size_t) t * n_expert, bias, n_expert, k, scale, ids + (size_t) t * k,
                        weights + (size_t) t * k, sb, sel);
}

}  // namespace

void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream) {
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.n_embd % 128 || a.n_embd > 4 * kMaxVec * kLogitThreads || a.k < 1 || a.k > 8 || a.k > a.n_expert)
        throw std::invalid_argument("deepseek41 router: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    for (int t0 = 0; t0 < n_tokens; t0 += kMaxTok) {
        const int nt = n_tokens - t0 < kMaxTok ? n_tokens - t0 : kMaxTok;
        router_logits_kernel<<<a.n_expert, kLogitThreads, 0, st>>>(
            x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
    }
    router_select_kernel<<<n_tokens, a.n_expert, 0, st>>>(logits, bias, a.n_expert, a.k, a.weight_scale, ids, weights);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::deepseek41
