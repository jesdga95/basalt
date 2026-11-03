// Engram: adapted from build_engram and engram_hash in the llama.cpp DeepSeek-V4.1 support (src/models/deepseek41.cpp).
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

#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

// Stage: one (token, copy). Reads its stream, key copy and gate row; adds gate * value to its stream.
__device__ void engram_copy(const float* __restrict__ key, const float* __restrict__ value,
                            const float* __restrict__ gate_w, float keep, int n_embd, float rms_eps,
                            float* __restrict__ x, float* red) {
    const int n4 = n_embd >> 2;
    float xx = 0.f, kk = 0.f, xkg = 0.f;
    for (int i = threadIdx.x; i < n4; i += kThreads) {
        const float4 a = reinterpret_cast<const float4*>(x)[i];
        const float4 b = reinterpret_cast<const float4*>(key)[i];
        const float4 g = reinterpret_cast<const float4*>(gate_w)[i];
        xx = fmaf(a.x, a.x, fmaf(a.y, a.y, fmaf(a.z, a.z, fmaf(a.w, a.w, xx))));
        kk = fmaf(b.x, b.x, fmaf(b.y, b.y, fmaf(b.z, b.z, fmaf(b.w, b.w, kk))));
        xkg = fmaf(a.x * b.x, g.x, fmaf(a.y * b.y, g.y, fmaf(a.z * b.z, g.z, fmaf(a.w * b.w, g.w, xkg))));
    }
    xx = block_sum(xx, red);
    kk = block_sum(kk, red);
    xkg = block_sum(xkg, red);
    const float rx = rsqrtf(xx / (float) n_embd + rms_eps);
    const float rk = rsqrtf(kk / (float) n_embd + rms_eps);
    const float dot = xkg * rx * rk * rsqrtf((float) n_embd);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gate = keep / (1.f + expf(-copysignf(mag, dot)));
    for (int i = threadIdx.x; i < n4; i += kThreads) {
        float4 a = reinterpret_cast<const float4*>(x)[i];
        const float4 v = reinterpret_cast<const float4*>(value)[i];
        a.x = fmaf(v.x, gate, a.x); a.y = fmaf(v.y, gate, a.y); a.z = fmaf(v.z, gate, a.z); a.w = fmaf(v.w, gate, a.w);
        reinterpret_cast<float4*>(x)[i] = a;
    }
}

__global__ void engram_kernel(const float* __restrict__ kv, const float* __restrict__ gate_w,
                              const float* __restrict__ keep, int n_embd, float rms_eps, float* __restrict__ x) {
    __shared__ float red[kThreads / 32];
    const int t = blockIdx.x / kHcMult, c = blockIdx.x % kHcMult;
    const float* kvt = kv + (size_t) t * (kHcMult + 1) * n_embd;
    engram_copy(kvt + (size_t) c * n_embd, kvt + (size_t) kHcMult * n_embd, gate_w + (size_t) c * n_embd,
                keep ? keep[t] : 1.f, n_embd, rms_eps, x + ((size_t) t * kHcMult + c) * n_embd, red);
}

}  // namespace

void engram_apply(const float* kv, const float* gate_w, const float* keep, int n_tokens, const EngramArgs& a,
                  float* x, void* stream) {
    if (a.n_embd % 4) throw std::invalid_argument("deepseek41 engram_apply: n_embd");
    engram_kernel<<<n_tokens * kHcMult, kThreads, 0, (cudaStream_t) stream>>>(kv, gate_w, keep, a.n_embd,
                                                                               a.rms_eps, x);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::deepseek41
