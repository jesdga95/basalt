// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h.
//
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
// DeepSeek-V4.1 wo_a for decode: all 8 groups (and every token) of the block-diagonal projection in one launch.
// Each output row runs the arithmetic of native_q5_k_mmvq / native_q6_k_mmvq with one column (same q8_1
// activation, same per-block order, same cross-warp and shuffle reduction), so the results are bitwise equal.
#include "basalt/kernels/dp4a.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "attn_out_grouped.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

constexpr int QK = 256;
constexpr int WARP = 32;
constexpr int WARPS = 4;
constexpr int Q5_VDR = 2;
constexpr int Q5_QI = 32;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

static_assert(sizeof(Q5KBlock) == 176 && sizeof(Q6KBlock) == 210 && sizeof(Q81Block) == 36);

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = DP4A(v0i, u[2 * i], DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = DP4A(0x01010101, u[2 * i], DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}


struct Q5Fmt {
    using Block = Q5KBlock;
    static constexpr int kBlocksPerIter = Q5_VDR * WARPS * WARP / Q5_QI;
    static __device__ __forceinline__ int kqs(int tid) { return Q5_VDR * (tid % (Q5_QI / Q5_VDR)); }
    static __device__ __forceinline__ int kbx0(int tid) { return tid / (Q5_QI / Q5_VDR); }
    static __device__ __forceinline__ float dot(const Block* w, const Q81Block* x, int iqs) { return q5_q8_dot(w, x, iqs); }
};
struct Q6Fmt {
    using Block = Q6KBlock;
    static constexpr int kBlocksPerIter = WARPS * WARP / 32;
    static __device__ __forceinline__ int kqs(int tid) { return tid % 32; }
    static __device__ __forceinline__ int kbx0(int tid) { return tid / 32; }
    static __device__ __forceinline__ float dot(const Block* w, const Q81Block* x, int iqs) { return q6_q8_dot(w, x, iqs); }
};

// grid (rows per group, tokens * groups); block (32, 4): one output row per block.
template <typename F>
__global__ void __launch_bounds__(WARPS * WARP, 1)
wo_a_grouped_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y) {
    constexpr int kGroupIn = kAttnGroupIn;
    constexpr int blocks_per_row = kGroupIn / QK;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row = int(blockIdx.x);
    const int tg = int(blockIdx.y);          // token * 8 + group
    const int g = tg % kAttnGroups;
    const typename F::Block* wg = w + (size_t) g * kAttnOutRank * blocks_per_row;
    const Q81Block* xc = x + (size_t) tg * (kGroupIn / 32);
    float tmp = 0.0f;
    for (int kbx = F::kbx0(tid); kbx < blocks_per_row; kbx += F::kBlocksPerIter)
        tmp += F::dot(wg + (size_t) row * blocks_per_row + kbx, xc + kbx * (QK / 32), F::kqs(tid));
    __shared__ float partial[WARPS - 1][WARP];
    if (threadIdx.y > 0) partial[threadIdx.y - 1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int l = 0; l < WARPS - 1; ++l) tmp += partial[l][threadIdx.x];
    tmp = warp_sum(tmp);
    if (threadIdx.x == 0) y[(size_t) (tg / kAttnGroups) * (kAttnGroups * kAttnOutRank) + (size_t) g * kAttnOutRank + row] = tmp;
}

// grid (rows per group, groups); block (32, 4): one output row of one group for all NT tokens, each weight block read
// once. A token's sums run in the one-token kernel's order, so its row is bitwise that kernel's.
template <typename F, int NT>
__global__ void __launch_bounds__(WARPS * WARP, 1)
wo_a_grouped_tokens_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y) {
    constexpr int kGroupIn = kAttnGroupIn;
    constexpr int blocks_per_row = kGroupIn / QK;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row = int(blockIdx.x);
    const int g = int(blockIdx.y);
    const typename F::Block* wg = w + (size_t) g * kAttnOutRank * blocks_per_row;
    float tmp[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) tmp[t] = 0.0f;
    for (int kbx = F::kbx0(tid); kbx < blocks_per_row; kbx += F::kBlocksPerIter) {
        const typename F::Block* wb = wg + (size_t) row * blocks_per_row + kbx;
#pragma unroll
        for (int t = 0; t < NT; ++t)
            tmp[t] += F::dot(wb, x + (size_t) (t * kAttnGroups + g) * (kGroupIn / 32) + kbx * (QK / 32), F::kqs(tid));
    }
    __shared__ float partial[WARPS - 1][NT][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) partial[threadIdx.y - 1][t][threadIdx.x] = tmp[t];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[t] += partial[l][t][threadIdx.x];
        tmp[t] = warp_sum(tmp[t]);
        if (threadIdx.x == 0) y[(size_t) t * (kAttnGroups * kAttnOutRank) + (size_t) g * kAttnOutRank + row] = tmp[t];
    }
}

template <typename F>
void launch_tokens(const void* wo_a, const void* xq, float* oa, int n_tokens, cudaStream_t s) {
    const dim3 grid(kAttnOutRank, kAttnGroups), block(WARP, WARPS);
    const auto* w = static_cast<const typename F::Block*>(wo_a);
    const auto* x = static_cast<const Q81Block*>(xq);
    switch (n_tokens) {
        case 2: wo_a_grouped_tokens_kernel<F, 2><<<grid, block, 0, s>>>(w, x, oa); break;
        case 3: wo_a_grouped_tokens_kernel<F, 3><<<grid, block, 0, s>>>(w, x, oa); break;
        case 4: wo_a_grouped_tokens_kernel<F, 4><<<grid, block, 0, s>>>(w, x, oa); break;
        case 5: wo_a_grouped_tokens_kernel<F, 5><<<grid, block, 0, s>>>(w, x, oa); break;
        case 6: wo_a_grouped_tokens_kernel<F, 6><<<grid, block, 0, s>>>(w, x, oa); break;
        case 7: wo_a_grouped_tokens_kernel<F, 7><<<grid, block, 0, s>>>(w, x, oa); break;
        default: wo_a_grouped_tokens_kernel<F, 8><<<grid, block, 0, s>>>(w, x, oa); break;
    }
}

}  // namespace

bool wo_a_grouped(int type, const void* wo_a, const void* xq, float* oa, int n_tokens, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    if (type != 14 && type != 13) return false;
    if (n_tokens < 1 || n_tokens > 8) throw std::invalid_argument("wo_a_grouped: tokens");
    if (n_tokens > 1) {
        if (type == 14) launch_tokens<Q6Fmt>(wo_a, xq, oa, n_tokens, s);
        else launch_tokens<Q5Fmt>(wo_a, xq, oa, n_tokens, s);
    } else {
        const dim3 grid(kAttnOutRank, kAttnGroups), block(WARP, WARPS);
        if (type == 14) {
            wo_a_grouped_kernel<Q6Fmt><<<grid, block, 0, s>>>(static_cast<const Q6KBlock*>(wo_a), static_cast<const Q81Block*>(xq), oa);
        } else {
            wo_a_grouped_kernel<Q5Fmt><<<grid, block, 0, s>>>(static_cast<const Q5KBlock*>(wo_a), static_cast<const Q81Block*>(xq), oa);
        }
    }
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("wo_a_grouped: launch failed");
    return true;
}

}  // namespace basalt::models::deepseek41
