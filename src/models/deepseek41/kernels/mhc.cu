// Hyper-connections: adapted from build_hc_mixes, build_hc_sinkhorn, build_hc_pre and build_hc_post in the llama.cpp
// DeepSeek-V4 support (src/models/deepseek4.cpp).
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
constexpr int kSlots = kHcCoef + 1;    // per partial: 24 dot products and the sum of squares

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Block sum in a fixed order; every thread gets the result.
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

// Stage: partial dot of one hc_fn row slice with one token's flattened streams (and, for mix 0, the sum of squares).
__device__ void hc_partial(const float* __restrict__ x, const float* __restrict__ fn, int flat, int split, int mix,
                           float* __restrict__ part, float* red) {
    const int len = flat / kHcSplits;
    const int base = split * len;
    const float4* xv = reinterpret_cast<const float4*>(x + base);
    const float4* wv = reinterpret_cast<const float4*>(fn + (size_t) mix * flat + base);
    float dot = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < len / 4; i += kThreads) {
        const float4 a = xv[i], w = wv[i];
        dot = fmaf(a.x, w.x, dot); dot = fmaf(a.y, w.y, dot); dot = fmaf(a.z, w.z, dot); dot = fmaf(a.w, w.w, dot);
        if (mix == 0) { ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss); ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss); }
    }
    dot = block_sum(dot, red);
    if (mix == 0) ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        part[mix] = dot;
        if (mix == 0) part[kHcCoef] = ss;
    }
}

__global__ void hc_partial_kernel(const float* __restrict__ x, const float* __restrict__ fn, int flat,
                                  float* __restrict__ ws) {
    __shared__ float red[kThreads / 32];
    const int split = blockIdx.x, mix = blockIdx.y, t = blockIdx.z;
    hc_partial(x + (size_t) t * flat, fn, flat, split, mix, ws + ((size_t) t * kHcSplits + split) * kSlots, red);
}

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

// Stage: one token, one warp. Lanes 0..23 own a mix each; lanes 0..15 (mirrored on 16..31) run the Sinkhorn on comb.
__device__ void hc_coef_token(const float* __restrict__ ws, const float* __restrict__ scale,
                              const float* __restrict__ base, int flat, float rms_eps, float hc_eps, int iters,
                              float* __restrict__ coef) {
    __shared__ float mix_s[32];
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += ws[(size_t) s * kSlots + kHcCoef];
    const float rs = rsqrtf(ss / (float) flat + rms_eps);
    float m = 0.f;
    if (lane < kHcCoef) {
        for (int s = 0; s < kHcSplits; ++s) m += ws[(size_t) s * kSlots + lane];
        m *= rs;
    }
    mix_s[lane] = m;
    __syncwarp();
    const float sc0 = scale[0], sc1 = scale[1], sc2 = scale[2];
    if (lane < 4) coef[lane] = sigmoidf(mix_s[lane] * sc0 + base[lane]) + hc_eps;
    else if (lane < 8) coef[lane] = 2.f * sigmoidf(mix_s[lane] * sc1 + base[lane]);
    // comb element e = src*4 + dst
    const int e = lane & 15;
    float v = mix_s[8 + e] * sc2 + base[8 + e];
    float mx = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    v = expf(v - mx);
    float sm = v + __shfl_xor_sync(0xffffffffu, v, 1);
    sm += __shfl_xor_sync(0xffffffffu, sm, 2);
    v = v / sm + hc_eps;
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            float r = v + __shfl_xor_sync(0xffffffffu, v, 1);
            r += __shfl_xor_sync(0xffffffffu, r, 2);
            v = v / (r + hc_eps);
        }
        float c = v + __shfl_xor_sync(0xffffffffu, v, 4);
        c += __shfl_xor_sync(0xffffffffu, c, 8);
        v = v / (c + hc_eps);
    }
    if (lane < 16) coef[8 + e] = v;
}

__global__ void hc_coef_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                               const float* __restrict__ base, int flat, float rms_eps, float hc_eps, int iters,
                               float* __restrict__ coef) {
    const int t = blockIdx.x;
    hc_coef_token(ws + (size_t) t * kHcSplits * kSlots, scale, base, flat, rms_eps, hc_eps, iters,
                  coef + (size_t) t * kHcCoef);
}

// Stage: one float4 column of one token.
__device__ __forceinline__ void hc_pre_col(const float* __restrict__ x, const float* __restrict__ pre, int n_embd,
                                           int i, float* __restrict__ out) {
    const float4* xv = reinterpret_cast<const float4*>(x);
    const int n4 = n_embd >> 2;
    float4 acc = xv[i];
    acc.x *= pre[0]; acc.y *= pre[0]; acc.z *= pre[0]; acc.w *= pre[0];
#pragma unroll
    for (int s = 1; s < kHcMult; ++s) {
        const float4 v = xv[(size_t) s * n4 + i];
        acc.x = fmaf(v.x, pre[s], acc.x); acc.y = fmaf(v.y, pre[s], acc.y);
        acc.z = fmaf(v.z, pre[s], acc.z); acc.w = fmaf(v.w, pre[s], acc.w);
    }
    reinterpret_cast<float4*>(out)[i] = acc;
}

__global__ void hc_pre_kernel(const float* __restrict__ x, const float* __restrict__ pre, int pre_stride, int n_embd,
                              float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n_embd >> 2)) return;
    float p[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) p[s] = pre[(size_t) t * pre_stride + s];
    hc_pre_col(x + (size_t) t * kHcMult * n_embd, p, n_embd, i, out + (size_t) t * n_embd);
}

// Stage: one float4 column of one token, all four destination streams.
__device__ __forceinline__ void hc_post_col(const float* __restrict__ f, const float* __restrict__ x,
                                            const float* __restrict__ coef, int n_embd, int i,
                                            float* __restrict__ out) {
    const int n4 = n_embd >> 2;
    const float4 fv = reinterpret_cast<const float4*>(f)[i];
    float4 xs[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) xs[s] = reinterpret_cast<const float4*>(x)[(size_t) s * n4 + i];
#pragma unroll
    for (int d = 0; d < kHcMult; ++d) {
        const float post = coef[4 + d];
        float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
        for (int s = 0; s < kHcMult; ++s) {
            const float c = coef[8 + s * 4 + d];
            o.x = fmaf(xs[s].x, c, o.x); o.y = fmaf(xs[s].y, c, o.y);
            o.z = fmaf(xs[s].z, c, o.z); o.w = fmaf(xs[s].w, c, o.w);
        }
        reinterpret_cast<float4*>(out)[(size_t) d * n4 + i] = o;
    }
}

__global__ void hc_post_kernel(const float* __restrict__ f, const float* __restrict__ x,
                               const float* __restrict__ coef, int coef_stride, int n_embd, float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n_embd >> 2)) return;
    hc_post_col(f + (size_t) t * n_embd, x + (size_t) t * kHcMult * n_embd, coef + (size_t) t * coef_stride, n_embd,
                i, out + (size_t) t * kHcMult * n_embd);
}

}  // namespace

void hc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
              const HcArgs& a, float* workspace, float* coef, void* stream) {
    const int flat = kHcMult * a.n_embd;
    if (flat % (kHcSplits * 4) || a.sinkhorn_iters < 1) throw std::invalid_argument("deepseek41 hc_mixes: geometry");
    cudaStream_t st = (cudaStream_t) stream;
    hc_partial_kernel<<<dim3(kHcSplits, kHcCoef, n_tokens), kThreads, 0, st>>>(x, hc_fn, flat, workspace);
    hc_coef_kernel<<<n_tokens, 32, 0, st>>>(workspace, hc_scale, hc_base, flat, a.rms_eps, a.hc_eps,
                                            a.sinkhorn_iters, coef);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void hc_pre(const float* x, const float* pre, int pre_stride, int n_tokens, int n_embd, float* out, void* stream) {
    const int n4 = n_embd / 4;
    hc_pre_kernel<<<dim3((n4 + 127) / 128, n_tokens), 128, 0, (cudaStream_t) stream>>>(x, pre, pre_stride, n_embd,
                                                                                         out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void hc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
             float* out, void* stream) {
    const int n4 = n_embd / 4;
    hc_post_kernel<<<dim3((n4 + 127) / 128, n_tokens), 128, 0, (cudaStream_t) stream>>>(f, x, coef, coef_stride,
                                                                                         n_embd, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::deepseek41
