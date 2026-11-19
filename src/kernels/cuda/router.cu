// Router: adapted from the gating and top-k selection of build_moe_ffn in llama.cpp (src/llama-graph.cpp): sqrt-softplus or
// sigmoid scores, a bias for the choice, the unbiased scores of the chosen normalised.
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
#include "basalt/kernels/router.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kMaxTok = 8;          // tokens per logits launch
constexpr int kLogitThreads = 256;  // one block per gate row
constexpr int kMaxVec = 8;          // float4 per thread: n_embd <= 4 * kMaxVec * kLogitThreads
constexpr int kMaxExpert = 512;
constexpr int kRowsMinTok = 64;     // tokens from which a launch takes several gate rows a block

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
    // blockIdx.y: a group of kMaxTok tokens of the launch
    const int t0 = blockIdx.y * kMaxTok;
    router_logits_row(gate + (size_t) row * n_embd, x + (size_t) t0 * n_embd, n_embd, min(kMaxTok, n_tokens - t0), logits + (size_t) t0 * n_expert,
                      n_expert, row, red);
}

// The same logits, G gate rows a block: a token's float4 is read once for the G rows. Each (row, token) sum runs in exactly
// router_logits_row's order (thread columns, then the warp tree, then the warps in order), so the bits are the same; for
// n_embd = 4 * NV * kLogitThreads.
template <int NV, int G>
__global__ void __launch_bounds__(kLogitThreads)
router_logits_rows_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_embd, int n_expert, int n_tokens,
                          float* __restrict__ logits) {
    __shared__ float red[G][kMaxTok * (kLogitThreads / 32)];
    const int r0 = blockIdx.x * G, t0 = blockIdx.y * kMaxTok, nt = min(kMaxTok, n_tokens - t0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float4 w[G][NV];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int c = 0; c < NV; ++c) w[g][c] = reinterpret_cast<const float4*>(gate + (size_t) (r0 + g) * n_embd)[threadIdx.x + c * kLogitThreads];
    for (int t = 0; t < nt; ++t) {
        const float4* xv = reinterpret_cast<const float4*>(x + (size_t) (t0 + t) * n_embd);
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = 0.f;
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            const float4 v = xv[threadIdx.x + c * kLogitThreads];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                acc[g] = fmaf(w[g][c].x, v.x, acc[g]);
                acc[g] = fmaf(w[g][c].y, v.y, acc[g]);
                acc[g] = fmaf(w[g][c].z, v.z, acc[g]);
                acc[g] = fmaf(w[g][c].w, v.w, acc[g]);
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float sum = warp_sum(acc[g]);
            if (lane == 0) red[g][t * (kLogitThreads / 32) + warp] = sum;
        }
    }
    __syncthreads();
    const int g = threadIdx.x / kMaxTok, t = threadIdx.x % kMaxTok;
    if (g < G && t < nt) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kLogitThreads / 32; ++i) s += red[g][t * (kLogitThreads / 32) + i];
        logits[(size_t) (t0 + t) * n_expert + r0 + g] = s;
    }
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : logf(1.f + expf(x));
    return sqrtf(sp);
}

__device__ __forceinline__ float score(float x, int gate) { return gate == 0 ? sqrt_softplus(x) : 1.f / (1.f + expf(-x)); }

// Shared memory of one token's selection.
struct SelectScratch {
    float wmax[kMaxExpert / 32];   // each warp's largest biased score
    float lb[kMaxExpert];          // the candidates' biased scores and experts, in no particular order
    int le[kMaxExpert];
    float thr;
    int n;
    float sel[16];
};

// Stage: one token, one thread per expert. An expert's rank counts the experts ahead of it (higher biased score, or
// equal and a lower index); ranks below k are the selection, in rank order. The k-th largest warp maximum bounds the
// k-th score from below (k distinct experts reach it), so only experts at or above it can rank below k, and they are
// ranked among themselves.
// kPrior (--route-prior): the experts `held` holds gain `prior` on the score the rank is taken from; their weights stay s.
template <bool kPrior>
__device__ void router_select_token(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                    int k, float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights,
                                    SelectScratch& sh, const ResidentTabs& held, float prior) {
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5, nw = n_expert >> 5;
    const float s = score(logits[e], gate);
    float b = s + bias[e];
    if (kPrior && held.held(e)) b += prior;
    float m = b;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) sh.wmax[warp] = m;
    if (e == 0) sh.n = 0;
    __syncthreads();
    if (warp == 0) {
        if (nw < k) {
            if (lane == 0) sh.thr = -INFINITY;
        } else if (lane < nw) {
            const float v = sh.wmax[lane];
            int r = 0;
            for (int j = 0; j < nw; ++j) r += (sh.wmax[j] > v) | ((sh.wmax[j] == v) & (j < lane));
            if (r == k - 1) sh.thr = v;
        }
    }
    __syncthreads();
    const bool cand = b >= sh.thr;
    if (cand) {
        const int i = atomicAdd(&sh.n, 1);
        sh.lb[i] = b;
        sh.le[i] = e;
    }
    __syncthreads();
    if (cand) {
        int rank = 0;
        for (int j = 0; j < sh.n; ++j) rank += (sh.lb[j] > b) | ((sh.lb[j] == b) & (sh.le[j] < e));
        if (rank < k) {
            sh.sel[rank] = s;
            ids[rank] = e;
        }
    }
    __syncthreads();
    if (e == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += sh.sel[r];
        const float denom = sum + 1e-20f;
        for (int r = 0; r < k; ++r) weights[r] = sh.sel[r] / denom * scale;
    }
}

template <bool kPrior>
__global__ void router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                     int k, float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights,
                                     ResidentTabs held, float prior) {
    __shared__ SelectScratch sh;
    const int t = blockIdx.x;
    router_select_token<kPrior>(logits + (size_t) t * n_expert, bias, n_expert, k, scale, gate, ids + (size_t) t * k,
                                weights + (size_t) t * k, sh, held, prior);
}

}  // namespace

void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream) {
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.n_embd % 128 || a.n_embd > 4 * kMaxVec * kLogitThreads || a.k < 1 || a.k > 16 || a.k > a.n_expert)
        throw std::invalid_argument("router: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    constexpr int kLaunchTok = kMaxTok * 65535;   // grid.y limit
    // a prompt's many tokens: gate rows four to a block where a thread's columns fit (the same bits, a quarter of the reads of x)
    constexpr int kRowsG = 4;
    const bool rows4 = n_tokens >= kRowsMinTok && a.n_embd == 4 * 3 * kLogitThreads && a.n_expert % kRowsG == 0;
    for (int t0 = 0; t0 < n_tokens; t0 += kLaunchTok) {
        const int nt = n_tokens - t0 < kLaunchTok ? n_tokens - t0 : kLaunchTok;
        if (rows4) {
            router_logits_rows_kernel<3, kRowsG><<<dim3(a.n_expert / kRowsG, (nt + kMaxTok - 1) / kMaxTok), kLogitThreads, 0, st>>>(
                x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
            continue;
        }
        router_logits_kernel<<<dim3(a.n_expert, (nt + kMaxTok - 1) / kMaxTok), kLogitThreads, 0, st>>>(
            x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
    }
    if (a.prior > 0.0f && a.held.n > 0)
        router_select_kernel<true><<<n_tokens, a.n_expert, 0, st>>>(logits, bias, a.n_expert, a.k, a.weight_scale, (int) a.gate, ids,
                                                                    weights, a.held, a.prior);
    else
        router_select_kernel<false><<<n_tokens, a.n_expert, 0, st>>>(logits, bias, a.n_expert, a.k, a.weight_scale, (int) a.gate, ids,
                                                                     weights, a.held, 0.0f);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::kernels
