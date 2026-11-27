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

#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace basalt::kernels {
namespace {

constexpr int kMaxTok = 8;          // tokens per logits launch
constexpr int kLogitThreads = 256;  // one block per gate row
constexpr int kMaxVec = 8;          // float4 per thread: n_embd <= 4 * kMaxVec * kLogitThreads
constexpr int kMaxExpert = 512;
constexpr int kRowsMinTok = 64;     // tokens from which a launch takes several gate rows a block
constexpr int kWideTok = 32;        // tokens from which a 2304-wide launch takes 8 gate rows a block (a batch window)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float4 load_w4(const float* __restrict__ p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
__device__ __forceinline__ float4 load_w4(const uint16_t* __restrict__ p, int i) {
    const uint2 u = reinterpret_cast<const uint2*>(p)[i];
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}

// Stage: one gate row against up to kMaxTok token vectors. The row is loaded once into registers, each thread owns
// fixed columns, and the warp and block sums run in a fixed order for every token.
template <typename W>
__device__ void router_logits_row(const W* __restrict__ gate_row, const float* __restrict__ x, int n_embd,
                                  int n_tokens, float* __restrict__ logits, int n_expert, int row, float* red) {
    const int n4 = n_embd >> 2;
    float4 w[kMaxVec];
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = threadIdx.x + c * kLogitThreads;
        if (i < n4) w[c] = load_w4(gate_row, i);
    }
    // the gate row is constant: it is read while the kernel before finishes (inside a PdlScope)
    pdl_wait();
    pdl_launch_dependents();
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

template <typename W>
__global__ void __launch_bounds__(kLogitThreads)
router_logits_kernel(const float* __restrict__ x, const W* __restrict__ gate, int n_embd, int n_expert,
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
template <typename W, int NV, int G>
__global__ void __launch_bounds__(kLogitThreads)
router_logits_rows_kernel(const float* __restrict__ x, const W* __restrict__ gate, int n_embd, int n_expert, int n_tokens,
                          float* __restrict__ logits) {
    __shared__ float red[G][kMaxTok * (kLogitThreads / 32)];
    const int r0 = blockIdx.x * G, t0 = blockIdx.y * kMaxTok, nt = min(kMaxTok, n_tokens - t0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float4 w[G][NV];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int c = 0; c < NV; ++c) w[g][c] = load_w4(gate + (size_t) (r0 + g) * n_embd, threadIdx.x + c * kLogitThreads);
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

__device__ __forceinline__ float score(float x, int gate) {
    return gate == 0 ? sqrt_softplus(x) : gate == 1 ? 1.f / (1.f + expf(-x)) : x;
}

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
    float s = score(logits[e], gate);
    float b = bias ? s + bias[e] : s;
    if (kPrior && held.held(e)) b += prior;
    if (isnan(b)) { b = -INFINITY; s = 0.f; }
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
    const bool cand = nw < k || b >= sh.thr;
    if (nw < k) {
        sh.lb[e] = b;
        sh.le[e] = e;
        if (e == 0) sh.n = n_expert;
    } else if (cand) {
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
    if (e == 0 && gate == 2) {   // softmax over the chosen logits, the largest first
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += expf(sh.sel[r] - sh.sel[0]);
        for (int r = 0; r < k; ++r) weights[r] = expf(sh.sel[r] - sh.sel[0]) / sum * scale;
    } else if (e == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += sh.sel[r];
        if (isfinite(sum)) {
            const float denom = sum + 1e-20f;
            for (int r = 0; r < k; ++r) weights[r] = sh.sel[r] / denom * scale;
        } else {
            int n_inf = 0;
            for (int r = 0; r < k; ++r) n_inf += isinf(sh.sel[r]);
            if (n_inf > 0) {
                const float w_inf = scale / (float) n_inf;
                for (int r = 0; r < k; ++r) weights[r] = isinf(sh.sel[r]) ? w_inf : 0.f;
            } else {
                double dsum = 0.0;
                for (int r = 0; r < k; ++r) dsum += (double) sh.sel[r];
                const double denom = dsum + 1e-20;
                for (int r = 0; r < k; ++r) weights[r] = (float) ((double) sh.sel[r] / denom) * scale;
            }
        }
    }
}

template <bool kPrior>
__global__ void router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                     int k, float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights,
                                     ResidentTabs held, float prior) {
    pdl_wait();
    pdl_launch_dependents();
    __shared__ SelectScratch sh;
    const int t = blockIdx.x;
    router_select_token<kPrior>(logits + (size_t) t * n_expert, bias, n_expert, k, scale, gate, ids + (size_t) t * k,
                                weights + (size_t) t * k, sh, held, prior);
}

// G gate rows per block for n_embd == 2304 (576 float4s = 2 * 256 + 64) with PDL overlap: a block reads its tokens'
// activations once for G rows (2 for a decode window, 8 for a batch window, where those reads bound the launch) while
// preserving router_logits_row's exact FP32 summation order bit-for-bit, whatever G.
template <int G>
__global__ void __launch_bounds__(kLogitThreads)
router_logits_2304_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_expert, int n_tokens,
                          float* __restrict__ logits) {
    constexpr int n_embd = 2304, n4 = n_embd >> 2;
    __shared__ float red[G][kMaxTok * (kLogitThreads / 32)];
    const int r0 = blockIdx.x * G, t0 = blockIdx.y * kMaxTok, nt = min(kMaxTok, n_tokens - t0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float4 w[G][3];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4* gr = reinterpret_cast<const float4*>(gate + (size_t) (r0 + g) * n_embd);
        w[g][0] = gr[tid];
        w[g][1] = gr[tid + kLogitThreads];
        if (tid + 2 * kLogitThreads < n4) w[g][2] = gr[tid + 2 * kLogitThreads];
    }
    pdl_wait();
    pdl_launch_dependents();
    for (int t = 0; t < nt; ++t) {
        const float4* xv = reinterpret_cast<const float4*>(x + (size_t) (t0 + t) * n_embd);
        float acc[G] = {};
        const float4 v0 = xv[tid];
        const float4 v1 = xv[tid + kLogitThreads];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] = fmaf(w[g][0].x, v0.x, acc[g]);
            acc[g] = fmaf(w[g][0].y, v0.y, acc[g]);
            acc[g] = fmaf(w[g][0].z, v0.z, acc[g]);
            acc[g] = fmaf(w[g][0].w, v0.w, acc[g]);
            acc[g] = fmaf(w[g][1].x, v1.x, acc[g]);
            acc[g] = fmaf(w[g][1].y, v1.y, acc[g]);
            acc[g] = fmaf(w[g][1].z, v1.z, acc[g]);
            acc[g] = fmaf(w[g][1].w, v1.w, acc[g]);
        }
        if (tid + 2 * kLogitThreads < n4) {
            const float4 v2 = xv[tid + 2 * kLogitThreads];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                acc[g] = fmaf(w[g][2].x, v2.x, acc[g]);
                acc[g] = fmaf(w[g][2].y, v2.y, acc[g]);
                acc[g] = fmaf(w[g][2].z, v2.z, acc[g]);
                acc[g] = fmaf(w[g][2].w, v2.w, acc[g]);
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float sum = warp_sum(acc[g]);
            if (lane == 0) red[g][t * (kLogitThreads / 32) + warp] = sum;
        }
    }
    __syncthreads();
    static_assert(G * kMaxTok <= kLogitThreads, "a thread a row and token");
    const int g = tid / kMaxTok, t = tid % kMaxTok;
    if (g < G && t < nt) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kLogitThreads / 32; ++i) s += red[g][t * (kLogitThreads / 32) + i];
        logits[(size_t) (t0 + t) * n_expert + r0 + g] = s;
    }
}

// Warp-register Top-k router selection for NE in {128, 256} (4 warps = 4 tokens per block, zero barriers): bitwise identical to
// router_select_kernel, with or without a bias, the softmax gate included.
template <int NE>
__global__ void __launch_bounds__(128)
router_select_warp_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_tokens, int k,
                          float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights) {
    constexpr int EP = NE / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = blockIdx.x * 4 + warp;
    float bias_reg[EP];
#pragma unroll
    for (int j = 0; j < EP; ++j) bias_reg[j] = bias ? bias[lane + 32 * j] : 0.f;
    pdl_wait();
    pdl_launch_dependents();
    if (t >= n_tokens) return;
    const float* tl = logits + (size_t) t * NE;
    float s[EP], b[EP];
#pragma unroll
    for (int j = 0; j < EP; ++j) {
        s[j] = score(tl[lane + 32 * j], gate);
        b[j] = bias ? s[j] + bias_reg[j] : s[j];
    }
    int sel_id = 0;
    float sel_s = 0.f;
    for (int r = 0; r < k; ++r) {
        float bv = b[0], bs = s[0];
        int bi = lane;
#pragma unroll
        for (int j = 1; j < EP; ++j) {
            const int idx = lane + 32 * j;
            if (b[j] > bv || (b[j] == bv && idx < bi)) {
                bv = b[j];
                bi = idx;
                bs = s[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
                bs = os;
            }
        }
        if (lane == r) {
            sel_id = bi;
            sel_s = bs;
        }
#pragma unroll
        for (int j = 0; j < EP; ++j)
            if (bi == lane + 32 * j) b[j] = -INFINITY;
    }
    if (gate == 2) {   // softmax over the chosen logits, the largest first
        const float x = expf(sel_s - __shfl_sync(0xffffffffu, sel_s, 0));
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += __shfl_sync(0xffffffffu, x, r);
        if (lane < k) {
            ids[(size_t) t * k + lane] = sel_id;
            weights[(size_t) t * k + lane] = x / sum * scale;
        }
        return;
    }
    float sum = 0.f;
    for (int r = 0; r < k; ++r) sum += __shfl_sync(0xffffffffu, sel_s, r);
    if (lane < k) {
        ids[(size_t) t * k + lane] = sel_id;
        weights[(size_t) t * k + lane] = sel_s / (sum + 1e-20f) * scale;
    }
}

template <typename W>
void router_logits_impl(const float* x, const W* gate, int n_tokens, const RouterArgs& a, float* logits, void* stream) {
    if (n_tokens <= 0) return;
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.n_embd % 128 || a.n_embd > 4 * kMaxVec * kLogitThreads || a.k < 1 || a.k > 16 || a.k > a.n_expert)
        throw std::invalid_argument("router: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    constexpr int kLaunchTok = kMaxTok * 65535;   // grid.y limit
    // a prompt's many tokens: gate rows four to a block where a thread's columns fit (the same bits, a quarter of the reads of x)
    constexpr int kRowsG = 4;
    const bool rows4 = n_tokens >= kRowsMinTok && a.n_embd == 4 * 3 * kLogitThreads && a.n_expert % kRowsG == 0;
    const bool rows4w = n_tokens >= kRowsMinTok && a.n_embd == 4 * 5 * kLogitThreads && a.n_expert % kRowsG == 0;
    const bool rows2_2304 = n_tokens >= 2 && a.n_embd == 2304 && a.n_expert % 2 == 0;
    for (int t0 = 0; t0 < n_tokens; t0 += kLaunchTok) {
        const int nt = n_tokens - t0 < kLaunchTok ? n_tokens - t0 : kLaunchTok;
        if (rows4) {
            router_logits_rows_kernel<W, 3, kRowsG><<<dim3(a.n_expert / kRowsG, (nt + kMaxTok - 1) / kMaxTok), kLogitThreads, 0, st>>>(
                x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
            continue;
        }
        if (rows4w) {
            router_logits_rows_kernel<W, 5, kRowsG><<<dim3(a.n_expert / kRowsG, (nt + kMaxTok - 1) / kMaxTok), kLogitThreads, 0, st>>>(
                x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
            continue;
        }
        if constexpr (std::is_same_v<W, float>) {
            if (rows2_2304 && nt >= kWideTok && a.n_expert % 8 == 0) {
                launch_pdl(router_logits_2304_kernel<8>, dim3(a.n_expert / 8, (nt + kMaxTok - 1) / kMaxTok), dim3(kLogitThreads), 0, st,
                           x + (size_t) t0 * a.n_embd, gate, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
                continue;
            }
            if (rows2_2304) {
                launch_pdl(router_logits_2304_kernel<2>, dim3(a.n_expert / 2, (nt + kMaxTok - 1) / kMaxTok), dim3(kLogitThreads), 0, st,
                           x + (size_t) t0 * a.n_embd, gate, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
                continue;
            }
        }
        launch_pdl(router_logits_kernel<W>, dim3(a.n_expert, (nt + kMaxTok - 1) / kMaxTok), dim3(kLogitThreads), 0, st,
                   x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, logits + (size_t) t0 * a.n_expert);
    }
}

}  // namespace

void router_logits(const float* x, const float* gate, int n_tokens, const RouterArgs& a, float* logits, void* stream) {
    router_logits_impl(x, gate, n_tokens, a, logits, stream);
}

void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream) {
    router_logits(x, gate, n_tokens, a, logits, stream);
    router_select(logits, bias, n_tokens, a, ids, weights, stream);
}

void router_logits_bf16(const float* x, const uint16_t* gate, int n_tokens, const RouterArgs& a, float* logits, void* stream) {
    router_logits_impl(x, gate, n_tokens, a, logits, stream);
}

void router_bf16(const float* x, const uint16_t* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
                 int32_t* ids, float* weights, void* stream) {
    router_logits_bf16(x, gate, n_tokens, a, logits, stream);
    router_select(logits, bias, n_tokens, a, ids, weights, stream);
}

void router_select(const float* logits, const float* bias, int n_tokens, const RouterArgs& a, int32_t* ids, float* weights,
                   void* stream) {
    if (n_tokens <= 0) return;
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.k < 1 || a.k > 16 || a.k > a.n_expert)
        throw std::invalid_argument("router_select: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    if (a.prior > 0.0f && a.held.n > 0)
        launch_pdl(router_select_kernel<true>, dim3(n_tokens), dim3(a.n_expert), 0, st, logits, bias, a.n_expert, a.k, a.weight_scale,
                   (int) a.gate, ids, weights, a.held, a.prior);
    else if (a.n_expert == 128)
        launch_pdl(router_select_warp_kernel<128>, dim3((n_tokens + 3) / 4), dim3(128), 0, st, logits, bias, n_tokens, a.k, a.weight_scale,
                   (int) a.gate, ids, weights);
    else if (a.n_expert == 256)
        launch_pdl(router_select_warp_kernel<256>, dim3((n_tokens + 3) / 4), dim3(128), 0, st, logits, bias, n_tokens, a.k, a.weight_scale,
                   (int) a.gate, ids, weights);
    else
        launch_pdl(router_select_kernel<false>, dim3(n_tokens), dim3(a.n_expert), 0, st, logits, bias, a.n_expert, a.k, a.weight_scale,
                   (int) a.gate, ids, weights, a.held, 0.0f);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::kernels
