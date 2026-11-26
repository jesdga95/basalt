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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/packed_byte.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace basalt::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

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
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* __restrict__ x,
                                           Q81Block* __restrict__ y, int n_in) {
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                                   const float* __restrict__ up,
                                                   Q81Block* __restrict__ y, int n_in) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint32_t scm, const half2& dm5,
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
        sumf_d += d8[i] * (dot1 * (int) packed_byte(scm, i));
        sumf_m += d8[i] * (dot2 * (int) packed_byte(scm, 2 + i));
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
    const uint32_t scm = pack_scale_min(aux);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, scm, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    pdl_wait();
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = DP4A(u, qx, sumi);
        sumi = DP4A(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = DP4A(v.x, u0, sumi);
        sumi = DP4A(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    pdl_wait();
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint32_t scm, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = DP4A(v1i, u[2 * i + 1], DP4A(v0i, u[2 * i], 0));
        const int dot2 = DP4A(0x01010101, u[2 * i + 1], DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * (int) packed_byte(scm, i));
        sumf_m += d8[i] * (dot2 * (int) packed_byte(scm, 2 + i));
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
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
    const uint32_t scm = pack_scale_min(aux);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, scm, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    pdl_wait();
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
// sc[i] is the block's scales[4 * i] at the lane's scale offset.
__device__ __forceinline__ float q6_q8_dot_sc(int vl, int vh, const int* __restrict__ u, const int (&sc)[2], float d,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (DP4A(vi, u[i], 0) * sc[i]);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    const int sc[2] = {scales[0], scales[4]};
    return q6_q8_dot_sc(vl, vh, u, sc, d, d8);
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

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.  Mapped: output row r is weight row rows[r].
template<bool SmallK, bool Mapped>
__device__ __forceinline__ void q6_k_mmvq_body(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                               float* __restrict__ y, int n_in, int n_out,
                                               const int32_t* __restrict__ rows) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    pdl_wait();
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const int wr = Mapped ? __ldg(rows + row0 + i) : row0 + i;
                const std::size_t block = std::size_t(wr) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    q6_k_mmvq_body<SmallK, false>(w, x, y, n_in, n_out, nullptr);
}
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_rows_kernel(const Q6KBlock* __restrict__ w, const int32_t* __restrict__ rows,
                                             const Q81Block* __restrict__ x, float* __restrict__ y, int n_in,
                                             int n_out) {
    q6_k_mmvq_body<SmallK, true>(w, x, y, n_in, n_out, rows);
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

// Q8_0 weight words and scales, plain or (Stream) cache-streaming; the same values either way.
template<bool Stream> __device__ __forceinline__ int q80_word(const int8_t* qs, int i32) {
    if constexpr (Stream) {
        const auto* x = reinterpret_cast<const unsigned short*>(qs);
        int value = __ldcs(x + 2 * i32);
        value |= __ldcs(x + 2 * i32 + 1) << 16;
        return value;
    } else {
        return load_int_b2(qs, i32);
    }
}
template<bool Stream> __device__ __forceinline__ float q80_scale(const Q80Block* b) {
    if constexpr (Stream) return __half2float(__ldcs(&b->d));
    else return b->d;
}
template<bool Stream> __device__ __forceinline__ float small_q8_dot_s(const Q80Block* __restrict__ w,
                                                                    const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = q80_word<Stream>(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = DP4A(v, u, sumi);
    }
    const float d0 = q80_scale<Stream>(w);
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = DP4A(v.x, q8[i], sumi);
        sumi = DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}


// The Q8_0 one-column kernel: the same per-row arithmetic as native_small_mmvq_kernel, with the first weight blocks
// loaded before pdl_wait (weights are constant) and the dependents let in at the start, so the launch overlaps the
// previous kernel's tail. Plain launches run it unchanged. Stream: the weights are read cache-streaming.
// NCOLS > 1 (not SmallK): columns j read x + j * (n_in / 32) and write y + j * n_out, each weight block loaded once
// for all of them; each column's sums are the one-column kernel's in the same order, so it is BITWISE a one-column
// call on that column, like native_mmvq_multi_kernel's exact layout (which this replaces for Q8_0).
// The body takes the block's index `bx` (blockIdx.x in its own launch; a segment's own index in a segmented launch).
template<bool SmallK, bool Stream, int NCOLS>
__device__ __forceinline__ void small_q80_body(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                               float* __restrict__ y, int n_in, int n_out, int bx) {
    static_assert(NCOLS == 1 || !SmallK, "columns take the one-row layout");
    constexpr int Qi = 8;
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    constexpr int NPRE = 3;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * bx;
    const int blocks_per_row = n_in / 32;
    const int x_stride = n_in / 32;
    const int kbx0 = tid / (Qi / 2);
    const int kqs = 2 * (tid % (Qi / 2));
    float d0[ROWS][NPRE];
    int v0[ROWS][NPRE], v1[ROWS][NPRE];
#pragma unroll
    for (int p = 0; p < NPRE; ++p) {
        const int kbx = kbx0 + p * BLOCKS_PER_ITER;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (kbx < blocks_per_row && row0 + i < n_out) {
                const Q80Block* b = w + std::size_t(row0 + i) * blocks_per_row + kbx;
                v0[i][p] = q80_word<Stream>(b->qs, kqs);
                v1[i][p] = q80_word<Stream>(b->qs, kqs + 1);
                d0[i][p] = q80_scale<Stream>(b);
            }
        }
    }
    pdl_launch_dependents();
    pdl_wait();
    float tmp[NCOLS][ROWS] = {};
#pragma unroll
    for (int p = 0; p < NPRE; ++p) {
        const int kbx = kbx0 + p * BLOCKS_PER_ITER;
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xj = x + std::size_t(j) * x_stride + kbx;
                const int* u = reinterpret_cast<const int*>(xj->qs);
                const float d1 = __low2float(xj->ds);
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    if (row0 + i < n_out) {
                        int sumi = 0;
                        sumi = DP4A(v0[i][p], u[kqs], sumi);
                        sumi = DP4A(v1[i][p], u[kqs + 1], sumi);
                        tmp[j][i] += d0[i][p] * d1 * float(sumi);
                    }
                }
            }
        }
    }
    for (int kbx = kbx0 + NPRE * BLOCKS_PER_ITER; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                if constexpr (NCOLS == 1) {
                    tmp[0][i] += small_q8_dot_s<Stream>(w + block, x + kbx, kqs);
                } else {
                    const int a0 = q80_word<Stream>(w[block].qs, kqs), a1 = q80_word<Stream>(w[block].qs, kqs + 1);
                    const float d = q80_scale<Stream>(w + block);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        const Q81Block* xj = x + std::size_t(j) * x_stride + kbx;
                        const int* u = reinterpret_cast<const int*>(xj->qs);
                        int sumi = 0;
                        sumi = DP4A(a0, u[kqs], sumi);
                        sumi = DP4A(a1, u[kqs + 1], sumi);
                        tmp[j][i] += d * __low2float(xj->ds) * float(sumi);
                    }
                }
            }
        }
    }
    __shared__ float partial[WARPS - 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < WARPS - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<bool SmallK, bool Stream = false, int NCOLS = 1>
__launch_bounds__(WARPS * WARP, NCOLS > 1 ? 8 : 1)
__global__ void native_small_q80_kernel(const Q80Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    small_q80_body<SmallK, Stream, NCOLS>(w, x, y, n_in, n_out, int(blockIdx.x));
}

// A segmented launch: up to three row ranges of one matrix, each with its own input and output, the blocks of segment
// i after those of the segments before it. Each block runs as block (its index - the segment's first) of the segment's
// own launch, so every output is that launch's bits.
struct SegArgs {
    int end[3];               // a segment's last block + 1, cumulative
    int row0[3], n_out[3];    // its first row in the matrix and its rows
    const Q81Block* x[3];
    float* y[3];
};
struct SegPick {
    int bx, row0, n_out;
    const Q81Block* x;
    float* y;
};
__device__ __forceinline__ SegPick seg_pick(const SegArgs& a) {
    const int b = int(blockIdx.x);
    const int i = b < a.end[0] ? 0 : b < a.end[1] ? 1 : 2;
    return {b - (i == 0 ? 0 : i == 1 ? a.end[0] : a.end[1]), i == 0 ? a.row0[0] : i == 1 ? a.row0[1] : a.row0[2],
            i == 0 ? a.n_out[0] : i == 1 ? a.n_out[1] : a.n_out[2], i == 0 ? a.x[0] : i == 1 ? a.x[1] : a.x[2],
            i == 0 ? a.y[0] : i == 1 ? a.y[1] : a.y[2]};
}

template<int NCOLS>
__launch_bounds__(WARPS * WARP, NCOLS > 1 ? 8 : 1)
__global__ void native_small_q80_seg_kernel(const Q80Block* __restrict__ w, SegArgs a, int n_in) {
    const SegPick p = seg_pick(a);
    small_q80_body<false, false, NCOLS>(w + std::size_t(p.row0) * (n_in / 32), p.x, p.y, n_in, p.n_out, p.bx);
}

// ============================ ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by mmvq_multi_parity). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        return q5_q8_dot_impl(r.vl, r.vh, u, pack_scale_min(r.aux), r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        return q4_q8_dot_impl(r.v, u, pack_scale_min(r.aux), r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = DP4A(r.v[j].x, u0, sumi);
            sumi = DP4A(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    __device__ static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    __device__ static W load(const Block* __restrict__ w, int) { return W{w}; }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int k) { return small_q8_dot(r.w, x, k); }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // the upstream layout is not the default

template<typename F, int NCOLS, int NW, int ROWS>
__device__ __forceinline__ void mmvq_multi_body(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                                float* __restrict__ y, int n_in, int n_out, int bx) {
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * bx;
    const int blocks_per_row = n_in / F::DIV;
    pdl_wait();
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<typename F, int NCOLS, int NW, int ROWS>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    mmvq_multi_body<F, NCOLS, NW, ROWS>(w, x, y, n_in, n_out, int(blockIdx.x));
}

template<typename F, int NCOLS>
__launch_bounds__(WARPS * WARP, 4)   // native_mmvq_multi_kernel's bound at one row a block
__global__ void native_mmvq_multi_seg_kernel(const typename F::Block* __restrict__ w, SegArgs a, int n_in) {
    const SegPick p = seg_pick(a);
    mmvq_multi_body<F, NCOLS, WARPS, 1>(w + std::size_t(p.row0) * (n_in / F::DIV), p.x, p.y, n_in, p.n_out, p.bx);
}

// ---------------------------------------------------------------- Q6_K columns on short rows (exact layout)
// With few blocks per row the exact layout's one-row block does little work per launch slot and its threads reload the
// columns' q8_1 words for every row.  Here a block stages the columns once, per lane as {u0, u1, d8_0, d8_1} (the words
// and scales Q6KTraits::apply reads), and each warp walks whole rows.  Lane L adds the blocks kbx = w + 4 i of each
// reference warp w at iqs L in ascending kbx (that warp's sum), then the four sums in warp order (the partial[] fold),
// then warp_sum: the exact layout's order, so every column is bitwise a single-column call's.
constexpr int kStagedWarps = 8;
constexpr int kStagedMaxBlocks = 8;   // blocks of 256 per row that take it
constexpr int kStagedRowsPerSm = 16;  // and the fewest rows per SM
constexpr int kStagedRowsPerWarp = 2;

// A lane's words of one Q6_K block (Q6KTraits::load's, scales read too), so a warp's rows can be read before their dots.
struct Q6Words {
    int vl, vh, sc[2];
    float d;
};
__device__ __forceinline__ Q6Words q6_words(const Q6KBlock* __restrict__ b, int lane) {
    const Q6KTraits::W r = Q6KTraits::load(b, lane);
    return {r.vl, r.vh, {r.scales[0], r.scales[4]}, r.d};
}

// Stage: block `block` of `n_blocks` stages the columns in `xs` and its warps take rows block * kStagedWarps + warp,
// then n_blocks * kStagedWarps further on, kStagedRowsPerWarp of them at a time: all their NB blocks' words are read
// before the first dot, and each row's sums are the ones above in the same order.
template<int NCOLS, int NB>
__device__ __forceinline__ void q6_k_staged_rows(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                                 float* __restrict__ y, int n_out, float4* xs, int block,
                                                 int n_blocks) {
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int x_stride = NB * 256 / Q8K;
    for (int i = warp * WARP + lane; i < NB * NCOLS * WARP; i += kStagedWarps * WARP) {
        const int L = i % WARP, j = (i / WARP) % NCOLS, kbx = i / (WARP * NCOLS);
        const Q81Block* xb = x + std::size_t(j) * x_stride + kbx * 8 + 4 * (L / 16) + (L % 16) / 8;
        xs[i] = make_float4(__int_as_float(reinterpret_cast<const int*>(xb[0].qs)[L % 8]),
                            __int_as_float(reinterpret_cast<const int*>(xb[2].qs)[L % 8]), __low2float(xb[0].ds),
                            __low2float(xb[2].ds));
    }
    __syncthreads();
    const int step = n_blocks * kStagedWarps;
    for (int row0 = block * kStagedWarps + warp; row0 < n_out; row0 += kStagedRowsPerWarp * step) {
        Q6Words wv[kStagedRowsPerWarp][NB];
#pragma unroll
        for (int r = 0; r < kStagedRowsPerWarp; ++r) {
            const Q6KBlock* wr = w + std::size_t(min(row0 + r * step, n_out - 1)) * NB;
#pragma unroll
            for (int kbx = 0; kbx < NB; ++kbx) wv[r][kbx] = q6_words(wr + kbx, lane);
        }
#pragma unroll
        for (int r = 0; r < kStagedRowsPerWarp; ++r) {
            const int row = row0 + r * step;
            if (row >= n_out) break;
            float acc[NCOLS][4];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[j][v] = 0.0f;
#pragma unroll
            for (int kbx = 0; kbx < NB; ++kbx) {
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const float4 xv = xs[(kbx * NCOLS + j) * WARP + lane];
                    const int u[2] = {__float_as_int(xv.x), __float_as_int(xv.y)};
                    const float d8[2] = {xv.z, xv.w};
                    acc[j][kbx % 4] += q6_q8_dot_sc(wv[r][kbx].vl, wv[r][kbx].vh, u, wv[r][kbx].sc, wv[r][kbx].d, d8);
                }
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                float t = acc[j][0];
#pragma unroll
                for (int v = 1; v < 4; ++v) t += acc[j][v];
                t = warp_sum(t);
                if (lane == 0) y[std::size_t(j) * n_out + row] = t;
            }
        }
    }
}

template<int NCOLS, int NB>
__launch_bounds__(kStagedWarps * WARP)
__global__ void native_q6_k_staged_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                          float* __restrict__ y, int n_out) {
    __shared__ float4 xs[NB * NCOLS * WARP];
    pdl_wait();
    q6_k_staged_rows<NCOLS, NB>(w, x, y, n_out, xs, int(blockIdx.x), int(gridDim.x));
}

// The device's SM count, per device.
int device_sms() {
    static int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return 0;
    }
    if (known[dev] == 0 && cudaDeviceGetAttribute(&known[dev], cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        known[dev] = -1;
    }
    return known[dev] > 0 ? known[dev] : 0;
}

// Rows shorter than kStagedMaxBlocks blocks, at least kStagedRowsPerSm per SM. Each warp takes kStagedRowsPerWarp rows:
// the staging is shared by enough rows, and the blocks still end often enough for the kernels of the head's other
// branches (a cluster launch needs several free slots at once) to start beside it.
template<int NCOLS>
bool launch_q6_k_staged(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const int sms = device_sms();
    if (n_in / 256 >= kStagedMaxBlocks || sms == 0 || n_out < kStagedRowsPerSm * sms) return false;
    constexpr int kRowsPerBlock = kStagedWarps * kStagedRowsPerWarp;
    const dim3 grid(unsigned((n_out + kRowsPerBlock - 1) / kRowsPerBlock)), threads(WARP, kStagedWarps);
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    switch (n_in / 256) {
    case 1: launch_pdl(native_q6_k_staged_kernel<NCOLS, 1>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 2: launch_pdl(native_q6_k_staged_kernel<NCOLS, 2>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 3: launch_pdl(native_q6_k_staged_kernel<NCOLS, 3>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 4: launch_pdl(native_q6_k_staged_kernel<NCOLS, 4>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 5: launch_pdl(native_q6_k_staged_kernel<NCOLS, 5>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 6: launch_pdl(native_q6_k_staged_kernel<NCOLS, 6>, grid, threads, 0, s, w, x, y, n_out); return true;
    case 7: launch_pdl(native_q6_k_staged_kernel<NCOLS, 7>, grid, threads, 0, s, w, x, y, n_out); return true;
    default: return false;
    }
}

bool launch_q80_mma(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, NW, 2>, dim3(blocks), dim3(WARP, NW), 0, s, w, x, y, n_in, n_out);
        return;
    }
    if constexpr (std::is_same_v<F, Q6KTraits>) {
        if (launch_q6_k_staged<NCOLS>(weights, x_q8_1, y, n_in, n_out, s)) return;
    }
    const dim3 threads(WARP, WARPS);
    // Q8_0 at 2-3 columns in the one-row layout: the weights loaded before the wait, as at one column (4+ columns
    // measured slower: the preloaded words cost occupancy where the activation loads bind)
    if constexpr (std::is_same_v<F, SmallTraits<Q80Block, 8>> && NCOLS <= 3) {
        if (n_in / F::DIV >= F::BPI) {
            launch_pdl(native_small_q80_kernel<false, false, NCOLS>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in,
                       n_out);
            return;
        }
    }
    if constexpr (std::is_same_v<F, SmallTraits<Q80Block, 8>> && NCOLS >= 4) {
        if (n_in / F::DIV >= F::BPI && launch_q80_mma(weights, x_q8_1, y, n_in, n_out, NCOLS, s)) return;
    }
    if (n_in / F::DIV < F::BPI) {
        constexpr int ROWS = 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out);
    } else {
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in,
                   n_out);
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

// ---------------------------------------------------------------- IQ4_XS columns on int8 tensor cores (MMVQ_MMA)
// For a 32-value sub-block the integer dot of a weight row and a column's q8_1 block is exact in int32 whichever
// instruction computes it, so mma.sync m16n8k32 s8 takes it for 16 rows x 8 columns at once (A: the codebook values,
// B: the q8_1 codes, C = 0).  The rest is IQ4XSTraits::apply's arithmetic on that integer, and the float terms are
// summed in the exact layout's order (native_mmvq_multi_kernel, NW = 4): thread tid = 32 y + L adds the sub-blocks
// k = 8 kbx + s with k % 128 == tid in ascending k, warp 0 adds the other warps' sums in warp order, and warp_sum's
// butterfly (16, 8, 4, 2, 1) adds the lanes.  Here warp w takes sub-block s = w of every block, i.e. the lanes
// L = 8 m + w (m = kbx % 4) of the four reference warps y = kbx % 16 / 4.  It walks the blocks chain by chain
// (c = kbx % 16 ascending, kbx ascending within a chain), so each chain is one reference thread's sum; it folds chain
// c = 4 y + m into v[m] in reference-warp order and does the butterfly's first two steps (m ^ 2, then m ^ 1); the
// block does the last three (w ^ 4, w ^ 2, w ^ 1).  So every column is BITWISE the exact layout's, and with it a
// single-column call's (mmvq_multi_parity checks it).
//
// A block owns 16 MT consecutive rows.  Block kbx of those rows (136 bytes each) and the columns' eight q8_1 blocks
// at kbx are one ring slot, cp.async'd STAGES - 1 slots ahead in the same chain order.
constexpr int kMvRowBytes = 144;   // a staged row: the 136-byte block, padded so the A loads hit 32 banks
constexpr int kMvColWords = 76;    // a staged column: eight 9-word q8_1 blocks, padded so the B loads do too
template<int MT> struct MvCfg {
    static constexpr int R = 16 * MT;                                    // rows per block
    static constexpr int STAGES = MT == 1 ? 8 : 6;
    static constexpr int WBYTES = R * kMvRowBytes;                       // weight bytes of a slot
    static constexpr int SLOT = WBYTES + MAX_NCOLS * kMvColWords * 4;
    static constexpr int SMEM = STAGES * SLOT;
};
static_assert(sizeof(IQ4XSBlock) == 136 && sizeof(Q81Block) == 36, "the blocks this stages");
static_assert(MvCfg<1>::SMEM <= 48 * 1024 && MvCfg<2>::SMEM <= 48 * 1024, "static shared memory");
static_assert(MAX_NCOLS * MvCfg<2>::R * 9 * 4 <= MvCfg<2>::SMEM && MAX_NCOLS * MvCfg<1>::R * 9 * 4 <= MvCfg<1>::SMEM,
              "the partial sums reuse the ring");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<int N> __device__ __forceinline__ void mv_cp(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src), "n"(N) : "memory");
}
__device__ __forceinline__ void mv_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void mv_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
// d = A (16 x 32 s8, row) * B (32 x 8 s8, col), C = 0: the exact int32 dots
__device__ __forceinline__ void mv_mma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
// iq4_table_lookup's CUDA lookup with the codebook words in registers: the same int8 values
__device__ __forceinline__ int2 mv_iq4_lookup(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
    const uint32_t r0 = __byte_perm(__byte_perm(t0, t1, q4), __byte_perm(t2, t3, q4), sel);
    const uint32_t r1 = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), sel >> 16);
    return make_int2(__byte_perm(r0, r1, 0x6420), __byte_perm(r0, r1, 0x7531));
}
// The next block in chain order: kbx + 16 in the same chain, else the next chain's first.
__device__ __forceinline__ int mv_next(int kbx, int nb) { return kbx + 16 < nb ? kbx + 16 : (kbx & 15) + 1; }

// One slot for warp w (sub-block w): MT mma tiles, each term added to acc[mt][q] as IQ4XSTraits::apply computes it.
// C fragment q: row 16 mt + g (q < 2) or + 8, column 2 t + (q & 1).
template<int MT>
__device__ __forceinline__ void mv_slot(float (&acc)[MT][4], const uint8_t* slot, int ncols, int w, int g, int t,
                                        uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t* sx = reinterpret_cast<const uint32_t*>(slot + MvCfg<MT>::WBYTES);
    uint32_t b0 = 0, b1 = 0;
    if (g < ncols) {
        const uint32_t* xp = sx + g * kMvColWords + 9 * w;
        b0 = xp[1 + t];
        b1 = xp[5 + t];
    }
    const float d8a = 2 * t < ncols ? __low2float(*reinterpret_cast<const half2*>(sx + (2 * t) * kMvColWords + 9 * w))
                                    : 0.0f;
    const float d8b = 2 * t + 1 < ncols
                          ? __low2float(*reinterpret_cast<const half2*>(sx + (2 * t + 1) * kMvColWords + 9 * w))
                          : 0.0f;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        int ls[2];
        float dw[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // the row's d | scales_h << 16, its scales_l, and qs word 4 w + t (values 4t.. low, 16+4t.. high)
            const uint32_t* wr = reinterpret_cast<const uint32_t*>(slot + (16 * mt + 8 * h + g) * kMvRowBytes);
            const uint32_t h0 = wr[0], h1 = wr[1];
            const int2 v = mv_iq4_lookup(wr[2 + 4 * w + t], t0, t1, t2, t3);
            a[h] = (uint32_t) v.x;
            a[2 + h] = (uint32_t) v.y;
            ls[h] = (int) (((h1 >> (4 * w)) & 0x0f) | (((h0 >> (16 + 2 * w)) & 0x03) << 4));
            dw[h] = __half2float(__ushort_as_half((unsigned short) (h0 & 0xffffu)));
        }
        int si[4];
        mv_mma(si, a, b0, b1);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int sumi = si[q];
            sumi *= ls[q >> 1] - 32;
            const float d = dw[q >> 1] * ((q & 1) ? d8b : d8a);
            acc[mt][q] += d * sumi;
        }
    }
}
#endif

template<int MT>
__global__ void __launch_bounds__(256, 2) native_iq4_xs_mma_kernel(const IQ4XSBlock* __restrict__ w,
                                                                const Q81Block* __restrict__ x, float* __restrict__ y,
                                                                int n_in, int n_out, int ncols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = MvCfg<MT>;
    __shared__ __align__(16) uint8_t s_mv[C::SMEM];
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * C::R;
    const int nb = n_in / 256;
    const size_t row_bytes = (size_t) nb * 136;
    const size_t col_bytes = (size_t) nb * 8 * 36;   // q8_1 bytes of an activation column
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(w);
    const uint8_t* xb = reinterpret_cast<const uint8_t*>(x);
    const bool w8 = (reinterpret_cast<std::uintptr_t>(wb) & 7) == 0;    // then so is every weight block
    const bool x16 = (reinterpret_cast<std::uintptr_t>(xb) & 15) == 0;  // then so is every run of 8 q8_1 blocks
    const int last = n_out - 1;   // rows past n_out stage a valid row, never stored
    auto issue_w = [&](int kbx, int sl) {
        uint8_t* d = s_mv + sl * C::SLOT;
        const uint8_t* src = wb + (size_t) kbx * 136;
        if (w8) {
            for (int q = tid; q < C::R * 17; q += 256) {
                const int rr = q / 17, wq = q - rr * 17;
                mv_cp<8>(d + rr * kMvRowBytes + 8 * wq, src + (size_t) min(row0 + rr, last) * row_bytes + 8 * wq);
            }
        } else {
            for (int q = tid; q < C::R * 34; q += 256) {
                const int rr = q / 34, wq = q - rr * 34;
                mv_cp<4>(d + rr * kMvRowBytes + 4 * wq, src + (size_t) min(row0 + rr, last) * row_bytes + 4 * wq);
            }
        }
    };
    auto issue_x = [&](int kbx, int sl) {
        uint8_t* d = s_mv + sl * C::SLOT + C::WBYTES;
        const uint8_t* src = xb + (size_t) kbx * 288;
        if (x16) {
            for (int q = tid; q < ncols * 18; q += 256) {
                const int cc = q / 18, wq = q - cc * 18;
                mv_cp<16>(d + cc * (kMvColWords * 4) + 16 * wq, src + cc * col_bytes + 16 * wq);
            }
        } else {
            for (int q = tid; q < ncols * 72; q += 256) {
                const int cc = q / 72, wq = q - cc * 72;
                mv_cp<4>(d + cc * (kMvColWords * 4) + 4 * wq, src + cc * col_bytes + 4 * wq);
            }
        }
    };
    // weights (constant) before pdl_wait, activations after it; group 0 holds all of the first weights
    int kb = 0;
#pragma unroll
    for (int p = 0; p < C::STAGES - 1; ++p) {
        if (p < nb) issue_w(kb, p);
        kb = mv_next(kb, nb);
    }
    pdl_wait();
    kb = 0;
#pragma unroll
    for (int p = 0; p < C::STAGES - 1; ++p) {
        if (p < nb) issue_x(kb, p);
        kb = mv_next(kb, nb);
        mv_commit();
    }
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t t0 = tbl[0], t1 = tbl[1], t2 = tbl[2], t3 = tbl[3];
    int i = 0;   // slots consumed; kb is the block of slot i + STAGES - 1
    // chain c (one reference thread's sum): its blocks c, c + 16, ... in turn, from 0
    auto chain = [&](int c, float (&acc)[MT][4]) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][q] = 0.0f;
        for (int kbx = c; kbx < nb; kbx += 16) {
            mv_wait<C::STAGES - 2>();   // slot i is here (this thread's copies) ...
            __syncthreads();            // ... and everyone's; slot (i - 1) % STAGES is free again
            if (i + C::STAGES - 1 < nb) {
                const int sl = (i + C::STAGES - 1) % C::STAGES;
                issue_w(kb, sl);
                issue_x(kb, sl);
                kb = mv_next(kb, nb);
            }
            mv_commit();
            mv_slot<MT>(acc, s_mv + (i % C::STAGES) * C::SLOT, ncols, wp, g, t, t0, t1, t2, t3);
            ++i;
        }
    };
    float v[MT][4][4];   // [mt][m][q]: lane L = 8 m + wp's sum over the reference warps
#pragma unroll
    for (int m = 0; m < 4; ++m) {   // reference warp 0: the sum starts as its chain
        float acc[MT][4];
        chain(m, acc);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int q = 0; q < 4; ++q) v[mt][m][q] = acc[mt][q];
    }
    for (int y4 = 1; y4 < 4; ++y4) {   // reference warps 1..3 in order
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            float acc[MT][4];
            chain(4 * y4 + m, acc);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int q = 0; q < 4; ++q) v[mt][m][q] = v[mt][m][q] + acc[mt][q];
        }
    }
    mv_wait<0>();
    __syncthreads();
    float* s_part = reinterpret_cast<float*>(s_mv);   // [column][row][9]: warp w's sum at w
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {   // the butterfly's steps 16 (m ^ 2) and 8 (m ^ 1)
            const int row = 16 * mt + 8 * (q >> 1) + g, col = 2 * t + (q & 1);
            const float a0 = v[mt][0][q] + v[mt][2][q];
            const float a1 = v[mt][1][q] + v[mt][3][q];
            s_part[(col * C::R + row) * 9 + wp] = a0 + a1;
        }
    __syncthreads();
    {   // steps 4, 2 and 1: lanes w and w + 4, then 0..1 and 2..3, then 0 and 1
        const int row = tid % C::R, col = tid / C::R;
        if (col < ncols && row0 + row < n_out) {
            const float* p = s_part + (col * C::R + row) * 9;
            const float c0 = p[0] + p[4], c1 = p[1] + p[5], c2 = p[2] + p[6], c3 = p[3] + p[7];
            const float d0 = c0 + c2, d1 = c1 + c3;
            y[(size_t) col * n_out + row0 + row] = d0 + d1;
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();   // the host launches it only where device code for int8 MMA was built
#endif
}

// ncols > 1 IQ4_XS in exact mode on native_iq4_xs_mma_kernel where the device runs it (the same bits);
// native_mmvq_set_mma turns it on for tests.
bool g_mmvq_mma = false;
// the fewest columns that take it (one more where the projection narrows its input)
int g_mmvq_mma_min_t = 4;
std::atomic<long> g_mmvq_mma_launches{0};
// The current device's SM count if it runs native_iq4_xs_mma_kernel (int8 MMA, with
// device code for it in this build), else 0; per device
int mmvq_mma_sms() {
    static int known[64] = {};   // 0 unknown, -1 no, else the SM count
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return 0;
    }
    if (known[dev] == 0) {
        int major = 0, sms = 0;
        cudaFuncAttributes f1{}, f2{};
        const bool q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                       cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
                       cudaFuncGetAttributes(&f1, native_iq4_xs_mma_kernel<1>) == cudaSuccess &&
                       cudaFuncGetAttributes(&f2, native_iq4_xs_mma_kernel<2>) == cudaSuccess;
        if (!q) (void) cudaGetLastError();
        known[dev] = q && major >= 8 && f1.ptxVersion >= 80 && f2.ptxVersion >= 80 && sms > 0
                         ? sms : -1;
    }
    return known[dev] > 0 ? known[dev] : 0;
}
// 32 rows per block while that still gives every SM a block, else 16
void launch_iq4_xs_mma(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, int sms,
                       cudaStream_t s) {
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if ((n_out + 31) / 32 >= sms)
        launch_pdl(native_iq4_xs_mma_kernel<2>, dim3(unsigned((n_out + 31) / 32)), dim3(256), 0, s, w, x, y, n_in,
                   n_out, ncols);
    else
        launch_pdl(native_iq4_xs_mma_kernel<1>, dim3(unsigned((n_out + 15) / 16)), dim3(256), 0, s, w, x, y, n_in,
                   n_out, ncols);
    g_mmvq_mma_launches.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------- Q8_0 columns on int8 tensor cores
// The one-row exact layout (native_small_q80_kernel, 4 warps, 32 blocks an iteration) has thread tid = 32 y + L add, in
// ascending kbx, the terms d_w * d_x * float(sumi) of the blocks kbx = tid / 4 (mod 32) at the 8 values 8 (tid % 4)..,
// then warp 0 adds warps 1, 2, 3 in order, then warp_sum's butterfly.  Each 8-value integer dot is exact whichever
// instruction computes it, so mma.sync m16n8k32 s8 computes it for 16 rows x 8 columns with A masked to the 8 values
// (four mmas a block).  Here warp w takes the chains c = kbx % 32 with c % 8 == w, i.e. reference lanes L = 4 w + grp of
// the four reference warps y = c / 8; it adds a chain's terms in ascending kbx exactly as the reference thread does,
// folds the four y in order, and the block replays the butterfly over L for every (row, column).  So every column is
// BITWISE a one-column call's (mmvq_multi_parity checks it).  A block owns 16 rows; the first chain's weights are read
// before pdl_wait.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void qm_mma(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
#endif
struct QmA { int a0, a1, a2, a3; float da, db; };   // rows g and g + 8: ints t and 4 + t of a block, and the scales

__device__ __forceinline__ void q80_mma_body(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                             float* __restrict__ y, int n_in, int n_out, int ncols, int bx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ float s_part[8 * 16 * 33];
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = bx * 16, nb = n_in / 32;
    const Q80Block* wa = w + (size_t) min(row0 + g, n_out - 1) * nb;
    const Q80Block* wb = w + (size_t) min(row0 + g + 8, n_out - 1) * nb;
    auto load_a = [&](int kbx) {
        QmA r;
        r.a0 = load_int_b2(wa[kbx].qs, t);
        r.a2 = load_int_b2(wa[kbx].qs, 4 + t);
        r.a1 = load_int_b2(wb[kbx].qs, t);
        r.a3 = load_int_b2(wb[kbx].qs, 4 + t);
        r.da = wa[kbx].d;
        r.db = wb[kbx].d;
        return r;
    };
    QmA pre[3];   // chain y = 0: kbx = wp, wp + 32, wp + 64
#pragma unroll
    for (int j = 0; j < 3; ++j)
        if (wp + 32 * j < nb) pre[j] = load_a(wp + 32 * j);
    pdl_launch_dependents();
    pdl_wait();
    const Q81Block* xg = x + (size_t) min(g, ncols - 1) * nb;
    const Q81Block* xc0 = x + (size_t) min(2 * t, ncols - 1) * nb;
    const Q81Block* xc1 = x + (size_t) min(2 * t + 1, ncols - 1) * nb;
    float v[4][4];
    // grp 0: k 0-7 (ints 0, 1: lanes t < 2, a0 a1); 1: k 8-15 (t >= 2, a0 a1); 2: k 16-23 (t < 2, a2 a3); 3: t >= 2, a2 a3
    const bool lo = t < 2;
    auto block = [&](const QmA& A, int kbx, float (&acc)[4][4]) {
        const uint32_t b0 = g < ncols ? (uint32_t) reinterpret_cast<const int*>(xg[kbx].qs)[t] : 0u;
        const uint32_t b1 = g < ncols ? (uint32_t) reinterpret_cast<const int*>(xg[kbx].qs)[4 + t] : 0u;
        const float dx0 = __low2float(xc0[kbx].ds), dx1 = __low2float(xc1[kbx].ds);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const bool mine = (grp & 1) ? !lo : lo, hi = grp >= 2;
            const uint32_t m0 = mine && !hi ? (uint32_t) A.a0 : 0u, m1 = mine && !hi ? (uint32_t) A.a1 : 0u;
            const uint32_t m2 = mine && hi ? (uint32_t) A.a2 : 0u, m3 = mine && hi ? (uint32_t) A.a3 : 0u;
            int si[4];
            qm_mma(si, m0, m1, m2, m3, b0, b1);
            acc[grp][0] += A.da * dx0 * float(si[0]);
            acc[grp][1] += A.da * dx1 * float(si[1]);
            acc[grp][2] += A.db * dx0 * float(si[2]);
            acc[grp][3] += A.db * dx1 * float(si[3]);
        }
    };
#pragma unroll
    for (int yy = 0; yy < 4; ++yy) {   // chain c = 8 yy + wp
        float acc[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[a][q] = 0.0f;
        const int c = 8 * yy + wp;
        if (yy == 0) {
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (c + 32 * j < nb) block(pre[j], c + 32 * j, acc);
            for (int kbx = c + 96; kbx < nb; kbx += 32) block(load_a(kbx), kbx, acc);
        } else {
            for (int kbx = c; kbx < nb; kbx += 32) block(load_a(kbx), kbx, acc);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int q = 0; q < 4; ++q) v[a][q] = yy == 0 ? acc[a][q] : v[a][q] + acc[a][q];
    }
    // lane L = 4 wp + grp's sums, [column][row][L]
#pragma unroll
    for (int grp = 0; grp < 4; ++grp)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = g + 8 * (q >> 1), col = 2 * t + (q & 1);
            s_part[(col * 16 + row) * 33 + 4 * wp + grp] = v[grp][q];
        }
    __syncthreads();
    if ((int) threadIdx.x < 16 * ncols) {   // warp_sum's butterfly (16, 8, 4, 2, 1) as lane 0 sees it
        const int row = threadIdx.x % 16, col = threadIdx.x / 16;
        const float* p = s_part + (col * 16 + row) * 33;
        float r16[16], r8[8], r4[4];
#pragma unroll
        for (int L = 0; L < 16; ++L) r16[L] = p[L] + p[L + 16];
#pragma unroll
        for (int L = 0; L < 8; ++L) r8[L] = r16[L] + r16[L + 8];
#pragma unroll
        for (int L = 0; L < 4; ++L) r4[L] = r8[L] + r8[L + 4];
        const float r2a = r4[0] + r4[2], r2b = r4[1] + r4[3];
        if (row0 + row < n_out) y[(size_t) col * n_out + row0 + row] = r2a + r2b;
    }
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

__global__ void __launch_bounds__(256) native_q80_mma_kernel(const Q80Block* __restrict__ w,
                                                             const Q81Block* __restrict__ x, float* __restrict__ y,
                                                             int n_in, int n_out, int ncols) {
    q80_mma_body(w, x, y, n_in, n_out, ncols, int(blockIdx.x));
}

__global__ void __launch_bounds__(256) native_q80_mma_seg_kernel(const Q80Block* __restrict__ w, SegArgs a, int n_in,
                                                                 int ncols) {
    const SegPick p = seg_pick(a);
    q80_mma_body(w + std::size_t(p.row0) * (n_in / 32), p.x, p.y, n_in, p.n_out, ncols, p.bx);
}

bool launch_q80_mma(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    // a head-sized product (many blocks of 16 rows) from 4 columns, others from 7 (narrow dense products lose at 4-6
    // columns, where 16-row blocks at two a SM hide less latency)
    if (mmvq_mma_sms() == 0 || ncols < (n_out >= 65536 ? 4 : 7)) return false;
    launch_pdl(native_q80_mma_kernel, dim3(unsigned((n_out + 15) / 16)), dim3(256), 0, s,
               static_cast<const Q80Block*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out, ncols);
    return true;
}

constexpr int MAX_PLANES_NCOLS = 128;
constexpr int MAX_QUANT_NCOLS = 128;

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8 (ncols " + std::to_string(ncols) + ", n_in " + std::to_string(n_in) + ")");
}
void validate_planes_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_PLANES_NCOLS) {
        throw std::invalid_argument("native planes MMVQ requires 1 <= ncols <= 128 (ncols " + std::to_string(ncols) + ", n_in " + std::to_string(n_in) + ")");
    }
}
void validate_quant_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_QUANT_NCOLS) {
        throw std::invalid_argument("native q8_1 quantize requires 1 <= ncols <= 128 (ncols " + std::to_string(ncols) + ", n_in " + std::to_string(n_in) + ")");
    }
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}


template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        if constexpr (std::is_same_v<Weight, Q80Block>)
            launch_pdl(native_small_q80_kernel<true>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out);
        else
            native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        if constexpr (std::is_same_v<Weight, Q80Block>)
            launch_pdl(native_small_q80_kernel<false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out);
        else
            native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

} // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

void native_mmvq_set_mma(bool on) { g_mmvq_mma = on; g_mmvq_mma_min_t = 1; }   // tests: every column count
bool native_mmvq_mma() { return g_mmvq_mma; }
bool native_mmvq_mma_available() { return mmvq_mma_sms() > 0; }
long native_mmvq_mma_launches() { return g_mmvq_mma_launches.load(std::memory_order_relaxed); }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_quant_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_quant_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_quantize_q8_1_kernel, dim3(blocks), dim3(QUANT_THREADS), 0, static_cast<cudaStream_t>(stream), x,
               static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_quant_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_swiglu_quantize_q8_1_kernel, dim3(blocks), dim3(QUANT_THREADS), 0, static_cast<cudaStream_t>(stream),
               gate, up, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        if (g_multi_exact && g_mmvq_mma && ncols >= g_mmvq_mma_min_t + (n_in > n_out ? 1 : 0)) {
            if (const int sms = mmvq_mma_sms()) {
                launch_iq4_xs_mma(weights, x_q8_1, y, n_in, n_out, ncols, sms, static_cast<cudaStream_t>(stream));
                launch_check();
                return;
            }
        }
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_mmvq_rows(const void* weights, const int32_t* rows, const void* x_q8_1, float* y, int n_in, int n_out,
                           void* stream) {
    validate_shape(n_in, 1, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(rows);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_rows_kernel<true><<<blocks, threads, 0, s>>>(w, rows, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_rows_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, rows, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq_segments(const void* weights, const Q80Segments& seg, int n_in, int ncols, void* stream) {
    if (seg.n < 1 || seg.n > 3) throw std::invalid_argument("native_q8_0_mmvq_segments takes 1 to 3 segments");
    validate_shape(n_in, ncols, 32);
    validate_pointer(weights);
    validate_stream(stream);
    const auto* w = static_cast<const Q80Block*>(weights);
    const auto s = static_cast<cudaStream_t>(stream);
    const std::size_t row_bytes = std::size_t(n_in / 32) * sizeof(Q80Block);
    const auto separate = [&] {
        for (int i = 0; i < seg.n; ++i)
            native_q8_0_mmvq(static_cast<const uint8_t*>(weights) + seg.row0[i] * row_bytes, seg.x[i], seg.y[i], n_in,
                             seg.n_out[i], ncols, stream);
    };
    // the kernel each segment's own call takes (small_mmvq / launch_multi_n): one launch only where they all take the
    // one-row layout of the same kernel
    constexpr int BPI = 2 * WARPS * WARP / 8;
    bool mma = false;
    for (int i = 0; i < seg.n; ++i) {
        validate_pointer(seg.x[i]);
        validate_pointer(seg.y[i]);
        if (seg.n_out[i] <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
        const bool m = ncols >= 4 && mmvq_mma_sms() != 0 && ncols >= (seg.n_out[i] >= 65536 ? 4 : 7);
        if (i == 0) mma = m;
        else if (m != mma) return separate();
    }
    if (n_in / 32 < BPI || (ncols > 1 && !g_multi_exact)) return separate();
    const int rows_a_block = mma ? 16 : 1;
    SegArgs a{};
    int end = 0;
    for (int i = 0; i < 3; ++i) {
        const int k = i < seg.n ? i : seg.n - 1;   // past the last segment: an empty copy of it
        if (i < seg.n) end += (seg.n_out[i] + rows_a_block - 1) / rows_a_block;
        a.end[i] = end;
        a.row0[i] = seg.row0[k];
        a.n_out[i] = seg.n_out[k];
        a.x[i] = static_cast<const Q81Block*>(seg.x[k]);
        a.y[i] = seg.y[k];
    }
    const dim3 grid{unsigned(end)}, threads(WARP, WARPS);
    using F = SmallTraits<Q80Block, 8>;
    if (mma) {
        launch_pdl(native_q80_mma_seg_kernel, grid, dim3(256), 0, s, w, a, n_in, ncols);
    } else {
        switch (ncols) {
        case 1: launch_pdl(native_small_q80_seg_kernel<1>, grid, threads, 0, s, w, a, n_in); break;
        case 2: launch_pdl(native_small_q80_seg_kernel<2>, grid, threads, 0, s, w, a, n_in); break;
        case 3: launch_pdl(native_small_q80_seg_kernel<3>, grid, threads, 0, s, w, a, n_in); break;
        case 4: launch_pdl(native_mmvq_multi_seg_kernel<F, 4>, grid, threads, 0, s, w, a, n_in); break;
        case 5: launch_pdl(native_mmvq_multi_seg_kernel<F, 5>, grid, threads, 0, s, w, a, n_in); break;
        case 6: launch_pdl(native_mmvq_multi_seg_kernel<F, 6>, grid, threads, 0, s, w, a, n_in); break;
        case 7: launch_pdl(native_mmvq_multi_seg_kernel<F, 7>, grid, threads, 0, s, w, a, n_in); break;
        default: launch_pdl(native_mmvq_multi_seg_kernel<F, 8>, grid, threads, 0, s, w, a, n_in); break;
        }
    }
    launch_check();
}

// ============================ Q8_0 in planes (aligned codes [rows][n_in], fp16 scales [rows][n_in/32]) ============================

namespace {

struct Q8Planes {
    const int8_t* codes;
    const half* scales;
};

__host__ __device__ __forceinline__ Q8Planes q8p_offset(Q8Planes p, int row0, int n_in) {
    return {p.codes + std::size_t(row0) * n_in, p.scales + std::size_t(row0) * (n_in / 32)};
}

// 1 warp per row, 4 rows per CTA, 16-byte coalesced int4 (LDG.128) weight loads, zero shared memory & zero __syncthreads.
// Two lanes cover each 32-value block (lane&1 picks bytes 0..15 or 16..31), so one warp sweeps 16 blocks (512 bytes) per step.
template<int NCOLS, int NPRE>
__device__ __forceinline__ void q8p_w1_body(Q8Planes w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                            int n_in, int n_out, int bx, float softcap = 0.0f) {
    const int row = bx * WARPS + int(threadIdx.y);
    if (row >= n_out) {
        pdl_launch_dependents();
        pdl_wait();
        return;
    }
    const int lane = int(threadIdx.x);
    const int nb = n_in / 32;
    const int b0 = lane >> 1;
    const int h = lane & 1;   // 0: bytes 0..15, 1: bytes 16..31
    const int8_t* wc = w.codes + std::size_t(row) * n_in;
    const half* ws = w.scales + std::size_t(row) * nb;
    float tmp[NCOLS] = {};
    if constexpr (NPRE > 0) {
        int4 q[NPRE];
        float d[NPRE];
#pragma unroll
        for (int p = 0; p < NPRE; ++p) {
            const int kbx = b0 + p * 16;
            if (kbx < nb) {
                q[p] = __ldg(reinterpret_cast<const int4*>(wc + kbx * 32 + h * 16));
                d[p] = __half2float(__ldg(ws + kbx));
            }
        }
        pdl_launch_dependents();
        pdl_wait();
#pragma unroll
        for (int p = 0; p < NPRE; ++p) {
            const int kbx = b0 + p * 16;
            if (kbx < nb) {
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const Q81Block* xj = x + std::size_t(j) * nb + kbx;
                    const int* u = reinterpret_cast<const int*>(xj->qs) + h * 4;
                    int sumi = DP4A(q[p].x, u[0], 0);
                    sumi = DP4A(q[p].y, u[1], sumi);
                    sumi = DP4A(q[p].z, u[2], sumi);
                    sumi = DP4A(q[p].w, u[3], sumi);
                    tmp[j] = fmaf(__fmul_rn(d[p], __low2float(xj->ds)), float(sumi), tmp[j]);
                }
            }
        }
    } else {
        pdl_launch_dependents();
        pdl_wait();
    }
    for (int kbx = b0 + NPRE * 16; kbx < nb; kbx += 16) {
        const int4 q = __ldg(reinterpret_cast<const int4*>(wc + kbx * 32 + h * 16));
        const float d = __half2float(__ldg(ws + kbx));
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xj = x + std::size_t(j) * nb + kbx;
            const int* u = reinterpret_cast<const int*>(xj->qs) + h * 4;
            int sumi = DP4A(q.x, u[0], 0);
            sumi = DP4A(q.y, u[1], sumi);
            sumi = DP4A(q.z, u[2], sumi);
            sumi = DP4A(q.w, u[3], sumi);
            tmp[j] = fmaf(__fmul_rn(d, __low2float(xj->ds)), float(sumi), tmp[j]);
        }
    }
    const float inv_cap = softcap > 0.0f ? 1.0f / softcap : 0.0f;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float val = warp_sum(tmp[j]);
        if (lane == 0) y[std::size_t(j) * n_out + row] = softcap > 0.0f ? softcap * tanhf(val * inv_cap) : val;
    }
}

template<int NCOLS, int NPRE>
__launch_bounds__(WARPS * WARP)
__global__ void native_q8p_w1_kernel(Q8Planes w, const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out, float softcap) {
    q8p_w1_body<NCOLS, NPRE>(w, x, y, n_in, n_out, int(blockIdx.x), softcap);
}

template<int NCOLS, int NPRE>
__launch_bounds__(WARPS * WARP)
__global__ void native_q8p_w1_seg_kernel(Q8Planes w, SegArgs a, int n_in) {
    const SegPick p = seg_pick(a);
    q8p_w1_body<NCOLS, NPRE>(q8p_offset(w, p.row0, n_in), p.x, p.y, n_in, p.n_out, p.bx);
}

// Matches q8p_w1_body's reduction order (lane = 2 * b0 + h, 16 blocks per step, two 16-byte halves h = 0, 1 per block, then
// warp_sum's 16 -> 8 -> 4 -> 2 -> 1 butterfly), using two mma.m16n8k32 s8 instructions per 32-element block and CTILES 8-column
// tiles sharing one weight stream so every column at 1..32 columns is bitwise identical to a single-column call.
template<int CTILES>
__device__ __forceinline__ void q8p_mma_body(Q8Planes w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                             int n_in, int n_out, int ncols, int bx, float softcap = 0.0f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NPRE = CTILES <= 2 ? 3 : 1;
    __shared__ float s_part[8 * 16 * 17];
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = bx * 16, nb = n_in / 32;
    const int ra = min(row0 + g, n_out - 1), rb = min(row0 + g + 8, n_out - 1);
    const int8_t* ca = w.codes + (size_t) ra * n_in;
    const int8_t* cb = w.codes + (size_t) rb * n_in;
    const half* sa = w.scales + (size_t) ra * nb;
    const half* sb = w.scales + (size_t) rb * nb;
    auto load_a = [&](int kbx) {
        QmA r;
        const int* qa = reinterpret_cast<const int*>(ca + (size_t) kbx * 32);
        const int* qb = reinterpret_cast<const int*>(cb + (size_t) kbx * 32);
        r.a0 = __ldg(qa + t);
        r.a2 = __ldg(qa + 4 + t);
        r.a1 = __ldg(qb + t);
        r.a3 = __ldg(qb + 4 + t);
        r.da = __half2float(__ldg(sa + kbx));
        r.db = __half2float(__ldg(sb + kbx));
        return r;
    };
    QmA pre[NPRE];
#pragma unroll
    for (int j = 0; j < NPRE; ++j)
        if (wp + 16 * j < nb) pre[j] = load_a(wp + 16 * j);
    pdl_launch_dependents();
    pdl_wait();
    auto block = [&](const QmA& A, int kbx, float (&acc)[CTILES][2][4]) {
#pragma unroll
        for (int ct = 0; ct < CTILES; ++ct) {
            const int c0 = 8 * ct;
            const int cg = c0 + g, cc0 = c0 + 2 * t, cc1 = cc0 + 1;
            const Q81Block* xg = x + (size_t) min(cg, ncols - 1) * nb;
            const Q81Block* xc0 = x + (size_t) min(cc0, ncols - 1) * nb;
            const Q81Block* xc1 = x + (size_t) min(cc1, ncols - 1) * nb;
            const uint32_t b0 = cg < ncols ? (uint32_t) reinterpret_cast<const int*>(xg[kbx].qs)[t] : 0u;
            const uint32_t b1 = cg < ncols ? (uint32_t) reinterpret_cast<const int*>(xg[kbx].qs)[4 + t] : 0u;
            const float dx0 = __low2float(xc0[kbx].ds), dx1 = __low2float(xc1[kbx].ds);
            const float sax0 = __fmul_rn(A.da, dx0), sax1 = __fmul_rn(A.da, dx1);
            const float sbx0 = __fmul_rn(A.db, dx0), sbx1 = __fmul_rn(A.db, dx1);
            int si0[4], si1[4];
            qm_mma(si0, (uint32_t) A.a0, (uint32_t) A.a1, 0u, 0u, b0, 0u);
            qm_mma(si1, (uint32_t) A.a2, (uint32_t) A.a3, 0u, 0u, b1, 0u);
            acc[ct][0][0] = fmaf(sax0, float(si0[0]), acc[ct][0][0]);
            acc[ct][0][1] = fmaf(sax1, float(si0[1]), acc[ct][0][1]);
            acc[ct][0][2] = fmaf(sbx0, float(si0[2]), acc[ct][0][2]);
            acc[ct][0][3] = fmaf(sbx1, float(si0[3]), acc[ct][0][3]);
            acc[ct][1][0] = fmaf(sax0, float(si1[0]), acc[ct][1][0]);
            acc[ct][1][1] = fmaf(sax1, float(si1[1]), acc[ct][1][1]);
            acc[ct][1][2] = fmaf(sbx0, float(si1[2]), acc[ct][1][2]);
            acc[ct][1][3] = fmaf(sbx1, float(si1[3]), acc[ct][1][3]);
        }
    };
    float acc0[CTILES][2][4] = {}, acc1[CTILES][2][4] = {};
#pragma unroll
    for (int j = 0; j < NPRE; ++j)
        if (wp + 16 * j < nb) block(pre[j], wp + 16 * j, acc0);
    for (int kbx = wp + 16 * NPRE; kbx < nb; kbx += 16) block(load_a(kbx), kbx, acc0);
    for (int kbx = wp + 8; kbx < nb; kbx += 16) block(load_a(kbx), kbx, acc1);
    const int tid = wp * 32 + lane;
#pragma unroll
    for (int ct = 0; ct < CTILES; ++ct) {
        if constexpr (CTILES > 1) {
            if (ct > 0) __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = 8 * (q >> 1) + g, col = 2 * t + (q & 1);
            s_part[(col * 16 + row) * 17 + 2 * wp] = __fadd_rn(acc0[ct][0][q], acc1[ct][0][q]);
            s_part[(col * 16 + row) * 17 + 2 * wp + 1] = __fadd_rn(acc0[ct][1][q], acc1[ct][1][q]);
        }
        __syncthreads();
        const int ntc = min(8, ncols - 8 * ct);
        if (tid < ntc * 16) {
            const int row = tid & 15, col = tid >> 4;
            const float* p = s_part + (col * 16 + row) * 17;
            float r8[8];
#pragma unroll
            for (int L = 0; L < 8; ++L) r8[L] = __fadd_rn(p[L], p[L + 8]);
            float r4[4];
#pragma unroll
            for (int L = 0; L < 4; ++L) r4[L] = __fadd_rn(r8[L], r8[L + 4]);
            const float r2a = __fadd_rn(r4[0], r4[2]), r2b = __fadd_rn(r4[1], r4[3]);
            if (row0 + row < n_out) {
                const float val = __fadd_rn(r2a, r2b);
                y[(size_t) (8 * ct + col) * n_out + row0 + row] = softcap > 0.0f ? softcap * tanhf(val * (1.0f / softcap)) : val;
            }
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

template<int CTILES>
__global__ void __launch_bounds__(256, CTILES <= 4 ? 2 : 1) native_q8p_mma_kernel(Q8Planes w, const Q81Block* __restrict__ x,
                                                                                   float* __restrict__ y, int n_in, int n_out, int ncols, float softcap) {
    const int c0 = int(blockIdx.y) * (8 * CTILES);
    const int sub = min(8 * CTILES, ncols - c0);
    if (sub > 0)
        q8p_mma_body<CTILES>(w, x + std::size_t(c0) * (n_in / 32), y + std::size_t(c0) * n_out, n_in, n_out, sub, int(blockIdx.x), softcap);
}

template<int CTILES>
__global__ void __launch_bounds__(256, CTILES <= 4 ? 2 : 1) native_q8p_mma_seg_kernel(Q8Planes w, SegArgs a, int n_in, int ncols) {
    const SegPick p = seg_pick(a);
    const int c0 = int(blockIdx.y) * (8 * CTILES);
    const int sub = min(8 * CTILES, ncols - c0);
    if (sub > 0)
        q8p_mma_body<CTILES>(q8p_offset(w, p.row0, n_in), p.x + std::size_t(c0) * (n_in / 32), p.y + std::size_t(c0) * p.n_out, n_in, p.n_out, sub, p.bx);
}

void launch_q8p(Q8Planes w, const Q81Block* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s, float softcap = 0.0f) {
    const dim3 threads(WARP, WARPS);
    if (n_in >= 1024 && mmvq_mma_sms() != 0 && (ncols > 8 || (n_out >= 65536 && ncols >= 4) || (n_out >= 4096 && ncols >= 7))) {
        const dim3 grid{unsigned((n_out + 15) / 16)};
        if (ncols > 32 && std::size_t(n_out) * std::size_t(n_in) <= (32u << 20)) {
            launch_pdl(native_q8p_mma_kernel<4>, dim3(grid.x, unsigned((ncols + 31) / 32)), dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap);
            return;
        }
        switch ((ncols + 7) / 8) {
        case 1: launch_pdl(native_q8p_mma_kernel<1>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 2: launch_pdl(native_q8p_mma_kernel<2>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 3: launch_pdl(native_q8p_mma_kernel<3>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 4: launch_pdl(native_q8p_mma_kernel<4>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 5: launch_pdl(native_q8p_mma_kernel<5>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 6: launch_pdl(native_q8p_mma_kernel<6>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        case 7: launch_pdl(native_q8p_mma_kernel<7>, grid, dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        default: launch_pdl(native_q8p_mma_kernel<8>, dim3(grid.x, unsigned((ncols + 63) / 64)), dim3(256), 0, s, w, x, y, n_in, n_out, ncols, softcap); break;
        }
        return;
    }
    if (ncols > 8) {
        for (int c0 = 0; c0 < ncols; c0 += 8)
            launch_q8p(w, x + std::size_t(c0) * (n_in / 32), y + std::size_t(c0) * n_out, n_in, n_out, std::min(8, ncols - c0), s, softcap);
        return;
    }
    const dim3 grid{unsigned((n_out + WARPS - 1) / WARPS)};
    switch (ncols) {
    case 1: launch_pdl(native_q8p_w1_kernel<1, 2>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 2: launch_pdl(native_q8p_w1_kernel<2, 2>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 3: launch_pdl(native_q8p_w1_kernel<3, 2>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 4: launch_pdl(native_q8p_w1_kernel<4, 1>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 5: launch_pdl(native_q8p_w1_kernel<5, 1>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 6: launch_pdl(native_q8p_w1_kernel<6, 1>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    case 7: launch_pdl(native_q8p_w1_kernel<7, 1>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    default: launch_pdl(native_q8p_w1_kernel<8, 0>, grid, threads, 0, s, w, x, y, n_in, n_out, softcap); break;
    }
}

}  // namespace

void native_q8_planes_mmvq(const void* planes, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                           void* stream, float softcap) {
    validate_planes_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(planes);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* codes = static_cast<const int8_t*>(planes);
    const auto* scales = reinterpret_cast<const half*>(codes + std::size_t(n_out) * n_in);
    launch_q8p({codes, scales}, static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out, ncols,
               static_cast<cudaStream_t>(stream), softcap);
    launch_check();
}

void native_q8_planes_mmvq_segments(const void* planes, int total_rows, const Q80Segments& seg, int n_in, int ncols,
                                    void* stream) {
    if (seg.n < 1 || seg.n > 3) throw std::invalid_argument("native_q8_planes_mmvq_segments takes 1 to 3 segments");
    validate_planes_shape(n_in, ncols, 32);
    validate_pointer(planes);
    validate_stream(stream);
    const auto* codes = static_cast<const int8_t*>(planes);
    const auto* scales = reinterpret_cast<const half*>(codes + std::size_t(total_rows) * n_in);
    const Q8Planes w{codes, scales};
    const auto s = static_cast<cudaStream_t>(stream);
    constexpr int BPI = 2 * WARPS * WARP / 8;
    for (int i = 0; i < seg.n; ++i) {
        validate_pointer(seg.x[i]);
        validate_pointer(seg.y[i]);
        if (seg.n_out[i] <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    }
    int sum_rows = 0;
    for (int i = 0; i < seg.n; ++i) sum_rows += seg.n_out[i];
    const bool mma = n_in >= 1024 && mmvq_mma_sms() != 0 &&
                     (ncols > 8 || (sum_rows >= 65536 && ncols >= 4) || (sum_rows >= 4096 && ncols >= 7));
    if (n_in / 32 < BPI || (ncols > 8 && !mma)) {
        for (int j = 0; j < seg.n; ++j)
            launch_q8p(q8p_offset(w, seg.row0[j], n_in), static_cast<const Q81Block*>(seg.x[j]), seg.y[j], n_in,
                       seg.n_out[j], ncols, s);
        launch_check();
        return;
    }
    const int rows_a_block = mma ? 16 : WARPS;
    SegArgs a{};
    int end = 0;
    for (int i = 0; i < 3; ++i) {
        const int k = i < seg.n ? i : seg.n - 1;
        if (i < seg.n) end += (seg.n_out[i] + rows_a_block - 1) / rows_a_block;
        a.end[i] = end;
        a.row0[i] = seg.row0[k];
        a.n_out[i] = seg.n_out[k];
        a.x[i] = static_cast<const Q81Block*>(seg.x[k]);
        a.y[i] = seg.y[k];
    }
    const dim3 grid{unsigned(end)}, threads(WARP, WARPS);
    if (mma) {
        if (ncols > 32 && std::size_t(sum_rows) * std::size_t(n_in) <= (32u << 20)) {
            launch_pdl(native_q8p_mma_seg_kernel<4>, dim3(grid.x, unsigned((ncols + 31) / 32)), dim3(256), 0, s, w, a, n_in, ncols);
            launch_check();
            return;
        }
        switch ((ncols + 7) / 8) {
        case 1: launch_pdl(native_q8p_mma_seg_kernel<1>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 2: launch_pdl(native_q8p_mma_seg_kernel<2>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 3: launch_pdl(native_q8p_mma_seg_kernel<3>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 4: launch_pdl(native_q8p_mma_seg_kernel<4>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 5: launch_pdl(native_q8p_mma_seg_kernel<5>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 6: launch_pdl(native_q8p_mma_seg_kernel<6>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        case 7: launch_pdl(native_q8p_mma_seg_kernel<7>, grid, dim3(256), 0, s, w, a, n_in, ncols); break;
        default: launch_pdl(native_q8p_mma_seg_kernel<8>, dim3(grid.x, unsigned((ncols + 63) / 64)), dim3(256), 0, s, w, a, n_in, ncols); break;
        }
        launch_check();
        return;
    }
    switch (ncols) {
    case 1: launch_pdl(native_q8p_w1_seg_kernel<1, 2>, grid, threads, 0, s, w, a, n_in); break;
    case 2: launch_pdl(native_q8p_w1_seg_kernel<2, 2>, grid, threads, 0, s, w, a, n_in); break;
    case 3: launch_pdl(native_q8p_w1_seg_kernel<3, 2>, grid, threads, 0, s, w, a, n_in); break;
    case 4: launch_pdl(native_q8p_w1_seg_kernel<4, 1>, grid, threads, 0, s, w, a, n_in); break;
    case 5: launch_pdl(native_q8p_w1_seg_kernel<5, 1>, grid, threads, 0, s, w, a, n_in); break;
    case 6: launch_pdl(native_q8p_w1_seg_kernel<6, 1>, grid, threads, 0, s, w, a, n_in); break;
    case 7: launch_pdl(native_q8p_w1_seg_kernel<7, 1>, grid, threads, 0, s, w, a, n_in); break;
    default: launch_pdl(native_q8p_w1_seg_kernel<8, 0>, grid, threads, 0, s, w, a, n_in); break;
    }
    launch_check();
}

void native_q8_0_mmvq_stream(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, void* stream) {
    validate_shape(n_in, 1, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* w = static_cast<const Q80Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / 8)   // small_mmvq<Q80Block, 8>'s choice, so the same layout
        launch_pdl(native_small_q80_kernel<true, true>, dim3(unsigned((std::size_t(n_out) + WARPS - 1) / WARPS)), threads,
                   0, s, w, x, y, n_in, n_out);
    else
        launch_pdl(native_small_q80_kernel<false, true>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out);
    launch_check();
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}


} // namespace basalt::kernels
