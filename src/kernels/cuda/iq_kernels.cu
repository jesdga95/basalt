// src/kernels/cuda/iq_kernels.cu - see include/basalt/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at llama.cpp commit
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; MIT, Copyright (c) 2023-2026 The ggml authors, see
// THIRD_PARTY_LICENSES).  The block structs and codebook grids are basalt/ggml/ggml_common.h.
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/diag.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
#define ggml_cuda_dp4a(a, b, c) DP4A((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// ---------------------------------------------------------------- UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q8_0 = 2;

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(const int* __restrict__ v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                                             const half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(const int* __restrict__ vl, const int* __restrict__ vh,
                                                             const int* __restrict__ u, const uint8_t* __restrict__ sc,
                                                             const uint8_t* __restrict__ m, const half2& dm5,
                                                             const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
__device__ __forceinline__ void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation, which has the lower relative error per
// expert.  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
// Q5_0: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q5_0_q8_1_impl) -- symmetric (one delta, no
// learned min), so unlike Q5_1 there is no min-term rounding choice to make: the constant -16 per-weight offset
// is accounted for via the q8_1 block's `ds.y` (sum of the ORIGINAL activations), exactly as upstream.
__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    const float s8 = __high2float(bq8_1->ds);
    return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
}
__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}
__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}

// NVFP4's UE4M3 sub-block scale as ggml decodes it (half the value, kvalues_fp4 being doubled; 0 and 0x7F as 0, bit 7
// ignored), built from its bits: exact, and the same float as the CPU tier's nvfp4_scale.
__device__ __forceinline__ float nvfp4_scale_dev(uint32_t x) {
    const uint32_t v = x & 0x7F;
    if (v == 0 || v == 0x7F) return 0.0f;
    const uint32_t e = v >> 3, m = v & 7;
    if (e == 0) return (float) m * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119) << 23) | (m << 20));
}
// NVFP4: one call per 16-value sub-block (iqs 0..3).  Sub-block s meets q8_1 block s/2, half s&1; its byte j holds
// value j (low nibble) and value j + 8 (high nibble), so the low nibbles of a word meet activation word w and the
// high ones word w + 2.  Integer sum per sub-block, scaled by d8 * d4 in float (the per-expert FP32 scale is applied
// by the expert kernels).
__device__ __forceinline__ float vec_dot_nvfp4_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_nvfp4* b = (const block_nvfp4*) vbq + kbx;
    const block_q8_1* y = bq8_1 + (iqs >> 1);
    const int a0 = 4 * (iqs & 1);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = get_int_from_table_16(get_int_b4(b->qs, 2 * iqs + i), kvalues_fp4);
        sumi = ggml_cuda_dp4a(v.x, get_int_b4(y->qs, a0 + i), sumi);
        sumi = ggml_cuda_dp4a(v.y, get_int_b4(y->qs, a0 + i + 2), sumi);
    }
    return __low2float(y->ds) * nvfp4_scale_dev(b->d[iqs]) * (float) sumi;
}

// Q6_K (vecdotq.cuh's vec_dot_q6_K_q8_1, VDR 1)
__device__ __forceinline__ float vec_dot_q6_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q6_K* bq6 = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6->ql, iqs);
    const int vh = get_int_b2(bq6->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8 * (ggml_cuda_dp4a(vi, u, 0) * sc);
    }
    return __half2float(bq6->d) * sumf;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<40> { static constexpr int qk = 64, ipb = 4, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_nvfp4_q8_1(v, y, kbx, iqs); } };

// NVFP4 experts end in a 16-byte tail {s_gate, s_up, s_down, 0}: the FP32 scale of each matrix, on its output.
constexpr size_t kNvfp4Tail = 16;
__device__ __forceinline__ const float* nvfp4_tail(const void* blob, const NativeExpertLayout& L) {
    return (const float*) ((const uint8_t*) blob + L.bytes - kNvfp4Tail);
}
// The output scale of an expert's matrix m (0 gate, 1 up, 2 down): NVFP4's tail, 1 for the other formats.  The
// decode-once kernels apply it as the per-entry ones do, one multiply of each finished dot, so the bits match.
template<int T>
__device__ __forceinline__ float expert_out_scale(unsigned long long blob, const NativeExpertLayout& L, int m) {
    if constexpr (T == 40) return nvfp4_tail((const void*) blob, L)[m];
    else return 1.0f;
}
template<int T, int NC>
__device__ __forceinline__ void tail_scale(float (&s)[NC], float sc) {
    if constexpr (T == 40) {
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] *= sc;
    }
}

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#define GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(14) X(6) X(8) X(40)
#define D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(8) X(40)
#define MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(8) X(40)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    pdl_wait();
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it).
template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q5_0) the
// per-entry ones, which call Fmt<TY>::dot per column (the launchers test kSplit at compile time, so the multi kernels
// are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 18 || TY == 22 || TY == 29);

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xxs_grid;
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[k0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[k0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = __half2float(bq2->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xs_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint64_t*) s_grid;
        else grid_lut = iq2s_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq3xxs_grid;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(grid_lut[q3[l0 + 0]], grid_lut[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 28;
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = __half2float(scale.f16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * __low2float(bq8_1[iqs].ds);
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { int2 v[2]; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        const float d = r.dw * __low2float(bq8_1->ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { int2 v[4]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x, u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(bq8_1[iqs / 4].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = __low2float(bq8_1_chunk->ds);
        return r.d2 * d8 * sumi;
    }
};

// The UD-Q4_K_XL expert formats.  `load` reads the weight side of vec_dot_*_q8_1 (the quantized words, the unpacked
// scales and mins, the half2 block scales); `apply` takes the activation words and runs the same *_impl_vmmq /
// float expression on them, so each column is bitwise the per-entry dot's.
template<> inline constexpr bool kSplit<12> = true;
template<> struct Split<12> {   // Q4_K
    struct W { int v[2]; uint16_t aux[2]; half2 dm; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(bq4_K->scales, bq8_offset, r.aux);
        r.dm = bq4_K->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int u[2 * QR4_K];
        float d8[QR4_K];
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
template<> inline constexpr bool kSplit<13> = true;
template<> struct Split<13> {   // Q5_K
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; half2 dm; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        k_scale_min(bq5_K->scales, bq8_offset, r.aux);
        r.dm = bq5_K->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int u[2 * QR5_K];
        float d8[QR5_K];
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
template<> inline constexpr bool kSplit<14> = true;
template<> struct Split<14> {   // Q6_K: vec_dot_q6_K_q8_1 with the weight side loaded apart from the activation side
    struct W { int vl, vh; int sc[QR6_K]; float d; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q6_K* bq6 = (const block_q6_K*) vbq + kbx;
        W r;
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        r.vl = get_int_b2(bq6->ql, iqs);
        r.vh = get_int_b2(bq6->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
        const int8_t* scales = bq6->scales + scale_offset;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) r.sc[i] = scales[4 * i];
        r.d = __half2float(bq6->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const int u = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
            const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
            const int vil = (r.vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((r.vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4((vil | vih), 0x20202020);
            sumf += d8 * (ggml_cuda_dp4a(vi, u, 0) * r.sc[i]);
        }
        return r.d * sumf;
    }
};
template<> inline constexpr bool kSplit<7> = true;
template<> struct Split<7> {   // Q5_1
    struct W { int lo[VDR_Q5_1], hi[VDR_Q5_1]; float2 dm; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = get_int_b4(bq5_1->qs, iqs + i);
            const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.lo[i] = vi0;
            r.hi[i] = vi1;
        }
        r.dm = __half22float2(bq5_1->dm);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            sumi = ggml_cuda_dp4a(r.lo[i], u0, sumi);
            sumi = ggml_cuda_dp4a(r.hi[i], u1, sumi);
            sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
        }
        const float d8 = __low2float(bq8_1->ds);
        return sumi * (r.dm.x * d8) + sumu * (r.dm.y * d8);
    }
};
template<> inline constexpr bool kSplit<8> = true;
template<> struct Split<8> {   // Q8_0
    struct W { int q[VDR_Q8_0]; float d; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) r.q[i] = get_int_b2(bq8_0->qs, iqs + i);
        r.d = __half2float(bq8_0->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = ggml_cuda_dp4a(r.q[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        const float d8_1 = __low2float(bq8_1->ds);
        return r.d * d8_1 * ((float) sumi);
    }
};

template<> inline constexpr bool kSplit<40> = true;
template<> struct Split<40> {   // NVFP4: vec_dot_nvfp4_q8_1 split; the per-expert FP32 scale is the kernels' (tail_scale)
    struct W { int2 v[2]; float d4; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_nvfp4* b = (const block_nvfp4*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = get_int_from_table_16(get_int_b4(b->qs, 2 * iqs + i), kvalues_fp4);
        r.d4 = nvfp4_scale_dev(b->d[iqs]);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* y = bq8_1 + (iqs >> 1);
        const int a0 = 4 * (iqs & 1);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = ggml_cuda_dp4a(r.v[i].x, get_int_b4(y->qs, a0 + i), sumi);
            sumi = ggml_cuda_dp4a(r.v[i].y, get_int_b4(y->qs, a0 + i + 2), sumi);
        }
        return __low2float(y->ds) * r.d4 * (float) sumi;
    }
};

// Down formats whose row is K = 40 calls (n_ff 640): IQ4_NL, Q5_1 (32-value blocks, two calls) and NVFP4 (64-value
// blocks, four): eight lanes a row.
template<int TD> inline constexpr bool kDown40 = TD == 20 || TD == 40 || TD == 7;
__host__ __device__ constexpr bool kDown40Type(int t) { return t == 20 || t == 40 || t == 7; }

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
// spare registers), the whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way,
// summed in the same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_multi(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                              int nb, int lane, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}

// 8-lane sub-warp row dot for K = nb * ipb == 40 (IQ4_NL or NVFP4 down with n_ff == 640):
// 4 rows per warp (32 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false, bool PRE = false>
__device__ __forceinline__ void row_dot_40_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                int t, float (&s)[NC], const typename Split<TY>::W* pw = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int k = t + 32 * i, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = PRE ? pw[i] : S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = PRE ? pw[2] : S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s16 = 0.0f;
                s16 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], s16);
            }
        }
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = PRE ? pw[3] : S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 24, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = PRE ? pw[4] : S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s24 = 0.0f;
                s24 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], __fadd_rn(s8[c], s24));
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 20 (TD == 42, Q2_0 with n_ff == 640):
// 2 rows per warp (16 rows per 256-thread block), bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_20_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
__device__ __forceinline__ const uint32_t* stage_iq_grid(uint32_t* s_buf, int tid, int nthreads) {
    if constexpr (!STAGE_GRID) {
        return nullptr;
    } else if constexpr (TY == 16) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xxs_grid;
        for (int i = tid; i < 256; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 17) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xs_grid;
        for (int i = tid; i < 512; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 22) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2s_grid;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 18) {
        for (int i = tid; i < 256; i += nthreads) s_buf[i] = iq3xxs_grid[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 29) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq1s_grid_gpu;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else {
        return nullptr;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
struct IqGridWords {
    static constexpr int value = !STAGE_GRID ? 4 : (TY == 22 || TY == 29) ? 2048 : (TY == 17) ? 1024 : (TY == 16) ? 512 :
                                 (TY == 18) ? 256 : 4;
};

// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
template<int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>>
__global__ void __launch_bounds__(128) mmvq_multi_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                         const block_q8_1* __restrict__ x, float* __restrict__ y,
                                                         int n_in, int n_out, int ncols) {
    __shared__ alignas(16) uint32_t s_grid_buf[IqGridWords<TY, STAGE_GRID>::value];
    const uint32_t* s_grid = stage_iq_grid<TY, STAGE_GRID>(s_grid_buf, threadIdx.y * 32 + threadIdx.x, 128);
    pdl_wait();   // the grid table above is constant
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int n = min(NC, ncols - c0);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (c0 + min(c, n - 1)) * xb;
        float s[NC];
        row_dot_multi<TY, NC, false, STAGE_GRID>(wr, x, off, n, nb, lane, s, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

// Group g is computed by block row g % gridDim.y: rows y, y + gridDim.y, ... < *n_groups.  The callers size the
// launch for the most groups a call can have (cap: every entry its own expert), but the count is only known on the
// device, and a verify window's PCIe call usually has no group at all: with one block row per possible group every
// unused one was cap x (2 n_ff / 8 + n_embd / 8) blocks that started only to return.  A smaller gridDim.y strides
// instead.  Each group's rows are computed by the same warp code in the same order either way, so the results do
// not depend on gridDim.y (gridDim.y = cap_groups is the one-row-per-group launch).
template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        [[maybe_unused]] float sc = 1.0f;
        if constexpr (TG == 40) sc = nvfp4_tail((const void*) grp_ptr[g], L)[is_up ? 1 : 0];
        for (int e = e0; e < e1; ++e) {
            float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if constexpr (TG == 40) s *= sc;
            if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
        }
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
// A group has at most one entry per token of the window (kVerifyMaxT = 8; a --spec 5 window carries up to 6 tokens),
// so 6 takes such windows in one pass; 8 would take ~64-80 registers.
constexpr int GRP_NC = 6;

template<int TG, bool STAGE_GRID = kStageIqGrid<TG>>
__global__ void __launch_bounds__(256) native_gu_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ alignas(16) uint32_t s_grid_buf[IqGridWords<TG, STAGE_GRID>::value];
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        [[maybe_unused]] const float sc = expert_out_scale<TG>(grp_ptr[g], L, is_up ? 1 : 0);
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };
                float s1[1];
                row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, nb, lane, s1, s_grid);
                tail_scale<TG>(s1, sc);
                if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
                float s2[2];
                row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, nb, lane, s2, s_grid);
                tail_scale<TG>(s2, sc);
                if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
                float s[GRP_NC];
                row_dot_multi<TG, GRP_NC, false, STAGE_GRID>(wr, xq, off, n, nb, lane, s, s_grid);
                tail_scale<TG>(s, sc);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
            }
        }
    }
}

// native_gu_multi_kernel with the group's activations in shared memory, kGuStageT entries at a time: a q8_1 block
// is 9 words, so the lanes' block-strided reads fall in distinct banks instead of spanning ~9 cache lines per load.
// The same loads in the same order from shared memory, so the same bits.
constexpr int kGuStageT = 8;

template<int TG, bool STAGE_GRID = kStageIqGrid<TG>>
__global__ void __launch_bounds__(256) native_gu_multi_sx_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                 const int32_t* __restrict__ grp_start,
                                                                 const int32_t* __restrict__ n_groups,
                                                                 const int32_t* __restrict__ ent_tok,
                                                                 const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                                 float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ alignas(16) uint32_t s_grid_buf[IqGridWords<TG, STAGE_GRID>::value];
    extern __shared__ __align__(16) uint32_t s_x[];
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    const bool valid = row < 2 * L.n_ff;                     // every thread stages, so none returns early
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = valid ? (is_up ? L.up_off : 0) + (size_t) r * L.gu_row : 0;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int xw = xb * (int) (sizeof(block_q8_1) / sizeof(uint32_t));   // words per entry
    const block_q8_1* sx = reinterpret_cast<const block_q8_1*>(s_x);
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        [[maybe_unused]] const float sc = expert_out_scale<TG>(grp_ptr[g], L, is_up ? 1 : 0);
        for (int c0 = e0; c0 < e1; c0 += kGuStageT) {
            const int nc = min(kGuStageT, e1 - c0);
            __syncthreads();                                 // the previous entries' reads are done
            for (int i = threadIdx.x; i < nc * xw; i += 256) {
                const int c = i / xw, w = i - c * xw;
                s_x[i] = reinterpret_cast<const uint32_t*>(xq + (size_t) ent_tok[c0 + c] * xb)[w];
            }
            __syncthreads();
            if (!valid) continue;
            for (int e = c0; e < c0 + nc; e += GRP_NC) {     // the GRP_NC passes of native_gu_multi_kernel
                const int n = min(GRP_NC, c0 + nc - e), b = e - c0;
                if (n == 1) {
                    const int off1[1] = { b * xb };
                    float s1[1];
                    row_dot_multi<TG, 1, true, STAGE_GRID>(wr, sx, off1, 1, nb, lane, s1, s_grid);
                    tail_scale<TG>(s1, sc);
                    if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
                } else if (n == 2) {
                    const int off2[2] = { b * xb, (b + 1) * xb };
                    float s2[2];
                    row_dot_multi<TG, 2, true, STAGE_GRID>(wr, sx, off2, 2, nb, lane, s2, s_grid);
                    tail_scale<TG>(s2, sc);
                    if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
                } else {
                    int off[GRP_NC];
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c) off[c] = (b + min(c, n - 1)) * xb;
                    float sm[GRP_NC];
                    row_dot_multi<TG, GRP_NC, false, STAGE_GRID>(wr, sx, off, n, nb, lane, sm, s_grid);
                    tail_scale<TG>(sm, sc);
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n && lane == c) dst[(size_t) (e + c) * L.n_ff + r] = sm[c];
                }
            }
        }
    }
}

// ---------------------------------------------------------------- IQ3_XXS gate/up on int8 tensor cores
// Per 32-value sub-block k the integer dot of a weight row and an entry's q8_1 block is exact in int32 whichever
// instruction computes it, so mma.sync m16n8k32 s8 takes it for 16 rows x 8 entries at once (A: the signed grid
// values, decoded once per row and sub-block straight into fragments; B: the entries' q8_1 codes; C starts at 0).
// The rest is Split<18>::apply's arithmetic on that integer, and the float terms are summed in row_dot_multi's
// order: the reference's lane L adds the terms k = L, L + 32, L + 64 in turn and the warp butterfly then adds the
// lanes' sums.  Here warp w holds lanes L = w, w + 8, w + 16, w + 24 (the sub-blocks k with k % 8 == w), adds each
// lane's terms in the same order with the same expression, and does the butterfly's first two steps; the block does
// the last three.  So the output is BITWISE native_gu_multi_kernel's (native_grouped_parity checks it).
//
// A block owns 32 consecutive gate or up rows of one group at a time and walks K in chunks of two 256-value blocks
// (196 bytes per row), cp.async'd kGmStages - 1 chunks ahead; the group's entries are staged 8 at a time with each
// q8_1 scale already a float.
constexpr int kGmRows = 32;                           // gate or up rows per block: two m16 tiles
constexpr int kGmCols = 8;                            // entries per pass: the mma's n8
constexpr int kGmStages = 3;                          // weight chunks in the ring
constexpr int kGmSeg = 2 * 98;                        // bytes of a row per chunk: two IQ3_XXS blocks
constexpr int kGmChunk = kGmRows * kGmSeg;            // bytes per ring slot
static_assert(sizeof(block_iq3_xxs) == 98, "the IQ3_XXS block this decodes");
static_assert(kGmCols * kGmRows * 9 * 4 <= kGmStages * kGmChunk, "the partial sums reuse the ring");

// shared memory: the ring, the grid, then kGmCols entries of 9 * n_embd / 32 words each + 4 (no bank conflicts on
// the B fragment loads: the entries' strides fall 4 banks apart)
size_t gu_mma_smem(int64_t n_embd) {
    return (size_t) kGmStages * kGmChunk + 256 * sizeof(uint32_t) +
           (size_t) kGmCols * (size_t) (9 * (n_embd / 32) + 4) * sizeof(uint32_t);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void gm_cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void gm_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void gm_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
// d = A (16 x 32 s8, row) * B (32 x 8 s8, col), C = 0: the exact int32 dots
__device__ __forceinline__ void gm_mma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
// the 4 sign bits of grid word w (0..7) of a sub-block: group w / 2's 7 bits with their parity as bit 7
// (unpack_ksigns), bits 4 (w & 1) .. 4 (w & 1) + 3
__device__ __forceinline__ uint32_t gm_sign_nib(uint32_t aux, int w) {
    const uint32_t v = (aux >> (7 * (w >> 1))) & 127u;
    const uint32_t s = v ^ ((__popc(v) & 1u) << 7);
    return (s >> (4 * (w & 1))) & 15u;
}
// a grid word with byte j negated where bit j of `nib` is set: (g ^ 0xff) + 1 per byte, no carry out of a byte as
// every grid value is >= 1 - the bytes __vsub4(g ^ m, m) gives
__device__ __forceinline__ uint32_t gm_signed(uint32_t g, uint32_t nib) {
    const uint32_t b = (nib * 0x204081u) & 0x01010101u;   // bit j of nib to bit 0 of byte j
    const uint32_t m = (b << 8) - b;                      // 0xff in those bytes
    return (g ^ m) + b;
}
// Row `bp` (an IQ3_XXS block in shared memory, 2-byte aligned), sub-block ib: grid words t and 4 + t (values
// 4t..4t+3 and 16+4t..16+4t+3, the A fragment's), the 4-bit scale and the block scale
__device__ __forceinline__ void gm_decode(const uint8_t* bp, int ib, int t, const uint32_t* grid, uint32_t& lo,
                                          uint32_t& hi, int& ls, float& dw) {
    const uint16_t* a16 = (const uint16_t*) (bp + 66 + 4 * ib);
    const uint32_t aux = (uint32_t) a16[0] | ((uint32_t) a16[1] << 16);
    lo = gm_signed(grid[bp[2 + 8 * ib + t]], gm_sign_nib(aux, t));
    hi = gm_signed(grid[bp[6 + 8 * ib + t]], gm_sign_nib(aux, 4 + t));
    ls = (int) (aux >> 28);
    dw = __half2float(*(const __half*) bp);
}
// Chunk c (ring slot `ring`) for warp w: sub-blocks k = 16c + w and 16c + 8 + w, i.e. the reference lanes
// L = k % 32 = w + 8 ((2c + s) % 4), held in acc[.][IIB + s] (IIB = 2 (c & 1)).
template<int IIB>
__device__ __forceinline__ void gm_chunk(float (&acc)[2][4][4], const uint8_t* ring, const uint32_t* s_grid,
                                         const uint32_t* s_x, int xs, int c, int ne, int w, int g, int t) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int k = 16 * c + 8 * s + w;
        uint32_t b0 = 0, b1 = 0;
        if (g < ne) {
            const uint32_t* xp = s_x + g * xs + 9 * k;
            b0 = xp[1 + t];
            b1 = xp[5 + t];
        }
        const float d8a = 2 * t < ne ? __uint_as_float(s_x[(2 * t) * xs + 9 * k]) : 0.0f;
        const float d8b = 2 * t + 1 < ne ? __uint_as_float(s_x[(2 * t + 1) * xs + 9 * k]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            uint32_t a[4];
            int ls[2];
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
                gm_decode(ring + (16 * mt + 8 * h + g) * kGmSeg + 98 * s, w, t, s_grid, a[h], a[2 + h], ls[h], dw[h]);
            int si[4];
            gm_mma(si, a, b0, b1);
#pragma unroll
            for (int q = 0; q < 4; ++q) {   // C: rows g (q < 2) and g + 8, entries 2t + (q & 1)
                int sumi = si[q];
                sumi = (ls[q >> 1] * sumi + sumi / 2) / 2;
                const float d = dw[q >> 1] * ((q & 1) ? d8b : d8a);
                acc[mt][IIB + s][q] += d * sumi;
            }
        }
    }
}
#endif

__global__ void __launch_bounds__(256, 2) native_gu_mma_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                               const int32_t* __restrict__ grp_start,
                                                               const int32_t* __restrict__ n_groups,
                                                               const int32_t* __restrict__ ent_tok,
                                                               const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                               float* __restrict__ gate, float* __restrict__ up) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    extern __shared__ __align__(16) uint8_t s_gm[];
    uint8_t* s_ring = s_gm;
    uint32_t* s_grid = (uint32_t*) (s_gm + kGmStages * kGmChunk);
    uint32_t* s_x = s_grid + 256;
    float* s_part = (float*) s_gm;   // after a pass's K loop: [entry][row][9] (warp w's sum at w)
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    s_grid[tid] = iq3xxs_grid[tid];  // visible from the first barrier of the K loop
    const int row0 = blockIdx.x * kGmRows;                   // 0 .. 2*n_ff, a tile of gate or of up rows
    const bool is_up = row0 >= (int) L.n_ff;
    const int r0 = is_up ? row0 - (int) L.n_ff : row0;
    float* dst = is_up ? up : gate;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r0 * L.gu_row;
    const int xb = (int) (L.n_embd / 32), xw = 9 * xb, xs = xw + 4;
    const int nch = (int) (L.n_embd / 512);
    for (int gi = blockIdx.y; gi < ng; gi += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wbase = (const uint8_t*) grp_ptr[gi] + w_off;
        const bool al4 = ((size_t) wbase & 3) == 0;
        const int e0 = grp_start[gi], e1 = grp_start[gi + 1];
        for (int e = e0; e < e1; e += kGmCols) {
            const int ne = min(kGmCols, e1 - e);
            // chunk c into its ring slot: 4-byte cp.async (the rows are 4-byte aligned when the blob is), else
            // 2-byte copies
            auto issue = [&](int c) {
                uint8_t* d = s_ring + (c % kGmStages) * kGmChunk;
                const uint8_t* src = wbase + (size_t) c * kGmSeg;
                if (al4) {
                    for (int q = tid; q < kGmRows * (kGmSeg / 4); q += 256) {
                        const int rr = q / (kGmSeg / 4), wq = q - rr * (kGmSeg / 4);
                        gm_cp4(d + rr * kGmSeg + 4 * wq, src + (size_t) rr * L.gu_row + 4 * wq);
                    }
                } else {
                    for (int q = tid; q < kGmRows * (kGmSeg / 2); q += 256) {
                        const int rr = q / (kGmSeg / 2), wq = q - rr * (kGmSeg / 2);
                        *(uint16_t*) (d + rr * kGmSeg + 2 * wq) =
                            *(const uint16_t*) (src + (size_t) rr * L.gu_row + 2 * wq);
                    }
                }
            };
#pragma unroll
            for (int c = 0; c < kGmStages - 1; ++c) {
                if (c < nch) issue(c);
                gm_commit();
            }
            for (int c = 0; c < ne; ++c) {   // the entries' q8_1 rows, the scale word as the float __low2float gives
                const uint32_t* xsrc = (const uint32_t*) (xq + (size_t) ent_tok[e + c] * xb);
                uint32_t* xd = s_x + c * xs;
                for (int i = tid; i < xw; i += 256) {
                    uint32_t v = xsrc[i];
                    if (i % 9 == 0) v = __float_as_uint(__half2float(__ushort_as_half((unsigned short) (v & 0xFFFFu))));
                    xd[i] = v;
                }
            }
            float acc[2][4][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[mt][i][q] = 0.0f;
            for (int c = 0; c < nch; ++c) {
                gm_wait<kGmStages - 2>();   // chunk c is here (this thread's copies) ...
                __syncthreads();            // ... and everyone's; slot (c - 1) % kGmStages is free again
                if (c + kGmStages - 1 < nch) issue(c + kGmStages - 1);
                gm_commit();
                const uint8_t* ring = s_ring + (c % kGmStages) * kGmChunk;
                if (c & 1) gm_chunk<2>(acc, ring, s_grid, s_x, xs, c, ne, w, g, t);
                else gm_chunk<0>(acc, ring, s_grid, s_x, xs, c, ne, w, g, t);
            }
            gm_wait<0>();
            __syncthreads();
            // the butterfly's first two steps in the warp: (L + (L + 16)) for L = w and L = w + 8, then their sum
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int row = 16 * mt + 8 * (q >> 1) + g, col = 2 * t + (q & 1);
                    const float lo = acc[mt][0][q] + acc[mt][2][q];
                    const float hi = acc[mt][1][q] + acc[mt][3][q];
                    s_part[(col * kGmRows + row) * 9 + w] = lo + hi;
                }
            __syncthreads();
            {   // its last three: lanes 0..7 -> 0..3 -> 0..1 -> 0
                const int row = tid & (kGmRows - 1), col = tid / kGmRows;
                if (col < ne) {
                    const float* p = s_part + (col * kGmRows + row) * 9;
                    const float c0 = p[0] + p[4], c1 = p[1] + p[5], c2 = p[2] + p[6], c3 = p[3] + p[7];
                    const float d0 = c0 + c2, d1 = c1 + c3;
                    dst[(size_t) (e + col) * L.n_ff + r0 + row] = d0 + d1;
                }
            }
            __syncthreads();   // the ring and the entries are free for the next pass
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();   // the host launches it only where device code for the tensor-core instructions was built
#endif
}

// ---------------------------------------------------------------- IQ3_S mmvq columns on int8 tensor cores
// mmvq_multi_kernel<21> with the 32-value integer dots on mma.sync m16n8k32 s8, as native_gu_mma_kernel does for
// IQ3_XXS: Split<21>::apply's arithmetic on each integer, the float terms summed in row_dot_multi's order (lane L adds
// k = L, L + 32, ... in turn, then warp_sum).  Warp w takes sub-block w of every block, i.e. the lanes
// L = 8 (kbx % 4) + w; walking the blocks in order it keeps one sum per such lane (acc[.][kbx % 4]), does the
// butterfly's first two steps, and the block does the last three.  So every column is BITWISE mmvq_multi_kernel's
// (iq_multi_parity checks it).
//
// A block owns 16 MT consecutive rows and walks K two blocks (220 bytes per row) at a time; a ring slot holds those
// and the columns' sixteen q8_1 blocks, cp.async'd STAGES - 1 slots ahead.
constexpr int kM3Seg = 2 * 110;    // bytes of a row per slot: two IQ3_S blocks
constexpr int kM3ColWords = 148;   // a staged column: sixteen 9-word q8_1 blocks, padded (conflict-free B loads)
template<int MT> struct M3Cfg {
    static constexpr int R = 16 * MT;                              // rows per block
    static constexpr int STAGES = MT == 1 ? 5 : 3;
    static constexpr int WBYTES = R * kM3Seg;                      // weight bytes of a slot
    static constexpr int SLOT = WBYTES + kGmCols * kM3ColWords * 4;
    static constexpr int RING = STAGES * SLOT;
    static constexpr int SMEM = RING + 512 * 4;                    // and the grid
};
static_assert(sizeof(block_iq3_s) == 110 && sizeof(block_q8_1) == 36, "the blocks this stages");
static_assert(M3Cfg<1>::SMEM <= 48 * 1024 && M3Cfg<2>::SMEM <= 48 * 1024, "static shared memory");
static_assert(M3Cfg<1>::SLOT % 16 == 0 && M3Cfg<2>::SLOT % 16 == 0 && M3Cfg<1>::WBYTES % 16 == 0 &&
              M3Cfg<2>::WBYTES % 16 == 0, "16-byte copies into the column part");
static_assert(kGmCols * M3Cfg<1>::R * 9 * 4 <= M3Cfg<1>::RING && kGmCols * M3Cfg<2>::R * 9 * 4 <= M3Cfg<2>::RING,
              "the partial sums reuse the ring");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void gm_cp16(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src) : "memory");
}
// Row `bp` (an IQ3_S block in shared memory, 2-byte aligned), sub-block ib: the signed grid words t and 4 + t (values
// 4t..4t+3 and 16+4t..16+4t+3, the A fragment's; sign bit v of the sub-block's 32 is value v's), the integer scale
// and the block scale
__device__ __forceinline__ void m3_decode(const uint8_t* bp, int ib, int t, const uint32_t* grid, uint32_t& lo,
                                          uint32_t& hi, int& ls, float& dw) {
    const uint32_t qh = bp[66 + ib];
    const uint16_t* s16 = (const uint16_t*) (bp + 74 + 4 * ib);
    const uint32_t sg = (uint32_t) s16[0] | ((uint32_t) s16[1] << 16);
    lo = gm_signed(grid[bp[2 + 8 * ib + t] | (((qh >> t) & 1u) << 8)], (sg >> (4 * t)) & 15u);
    hi = gm_signed(grid[bp[6 + 8 * ib + t] | (((qh >> (4 + t)) & 1u) << 8)], (sg >> (16 + 4 * t)) & 15u);
    ls = 1 + 2 * ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 0x0F);
    dw = __half2float(*(const __half*) bp);
}
// One slot for warp w: blocks 2c and 2c + 1, sub-block w of each, i.e. the lanes L = w + 8 (kbx % 4) held in
// acc[.][IIB + s] (IIB = 2 (c & 1)).  C fragment q: row 16 mt + g (q < 2) or + 8, column 2 t + (q & 1).
template<int MT, int IIB>
__device__ __forceinline__ void m3_slot(float (&acc)[MT][4][4], const uint8_t* slot, const uint32_t* s_grid,
                                        int ncols, int w, int g, int t) {
    const uint32_t* sx = (const uint32_t*) (slot + M3Cfg<MT>::WBYTES);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int k = 8 * s + w;   // the slot's q8_1 block
        uint32_t b0 = 0, b1 = 0;
        if (g < ncols) {
            const uint32_t* xp = sx + g * kM3ColWords + 9 * k;
            b0 = xp[1 + t];
            b1 = xp[5 + t];
        }
        const float d8a = 2 * t < ncols ? __low2float(*(const half2*) (sx + (2 * t) * kM3ColWords + 9 * k)) : 0.0f;
        const float d8b =
            2 * t + 1 < ncols ? __low2float(*(const half2*) (sx + (2 * t + 1) * kM3ColWords + 9 * k)) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t a[4];
            int ls[2];
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
                m3_decode(slot + (16 * mt + 8 * h + g) * kM3Seg + 110 * s, w, t, s_grid, a[h], a[2 + h], ls[h], dw[h]);
            int si[4];
            gm_mma(si, a, b0, b1);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int sumi = si[q];
                sumi *= ls[q >> 1];
                const float d = dw[q >> 1] * ((q & 1) ? d8b : d8a);
                acc[mt][IIB + s][q] += d * sumi;
            }
        }
    }
}
#endif

template<int MT>
__global__ void __launch_bounds__(256, 2) mmvq_iq3s_mma_kernel(const uint8_t* __restrict__ W, size_t rb,
                                                               const block_q8_1* __restrict__ X, float* __restrict__ y,
                                                               int n_in, int n_out, int ncols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = M3Cfg<MT>;
    __shared__ __align__(16) uint8_t s_m3[C::SMEM];
    uint32_t* s_grid = (uint32_t*) (s_m3 + C::RING);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    for (int i = tid; i < 512; i += 256) s_grid[i] = iq3s_grid[i];   // visible from the first barrier of the K loop
    const int row0 = blockIdx.x * C::R;
    const int nch = n_in / 512;
    const int last = n_out - 1;   // rows past n_out stage a valid row, never stored
    const uint8_t* xb = (const uint8_t*) X;
    const size_t col_bytes = (size_t) (n_in / 32) * 36;
    const bool w4 = (((size_t) W | rb) & 3) == 0;   // every row 4-byte aligned (rb is a multiple of 4 here)
    const bool x16 = ((size_t) xb & 15) == 0;      // then so is every run of 16 q8_1 blocks
    auto issue_w = [&](int c) {
        uint8_t* d = s_m3 + (c % C::STAGES) * C::SLOT;
        const uint8_t* src = W + (size_t) c * kM3Seg;
        if (w4) {
            for (int q = tid; q < C::R * (kM3Seg / 4); q += 256) {
                const int rr = q / (kM3Seg / 4), wq = q - rr * (kM3Seg / 4);
                gm_cp4(d + rr * kM3Seg + 4 * wq, src + (size_t) min(row0 + rr, last) * rb + 4 * wq);
            }
        } else {
            for (int q = tid; q < C::R * (kM3Seg / 2); q += 256) {
                const int rr = q / (kM3Seg / 2), wq = q - rr * (kM3Seg / 2);
                *(uint16_t*) (d + rr * kM3Seg + 2 * wq) =
                    *(const uint16_t*) (src + (size_t) min(row0 + rr, last) * rb + 2 * wq);
            }
        }
    };
    auto issue_x = [&](int c) {
        uint8_t* d = s_m3 + (c % C::STAGES) * C::SLOT + C::WBYTES;
        const uint8_t* src = xb + (size_t) c * 576;
        if (x16) {
            for (int q = tid; q < ncols * 36; q += 256) {
                const int cc = q / 36, wq = q - cc * 36;
                gm_cp16(d + cc * (kM3ColWords * 4) + 16 * wq, src + cc * col_bytes + 16 * wq);
            }
        } else {
            for (int q = tid; q < ncols * 144; q += 256) {
                const int cc = q / 144, wq = q - cc * 144;
                gm_cp4(d + cc * (kM3ColWords * 4) + 4 * wq, src + cc * col_bytes + 4 * wq);
            }
        }
    };
    // weights (constant) before pdl_wait, activations after it; group 0 holds all of the first weights
#pragma unroll
    for (int c = 0; c < C::STAGES - 1; ++c)
        if (c < nch) issue_w(c);
    pdl_wait();
#pragma unroll
    for (int c = 0; c < C::STAGES - 1; ++c) {
        if (c < nch) issue_x(c);
        gm_commit();
    }
    float acc[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][i][q] = 0.0f;
    for (int c = 0; c < nch; ++c) {
        gm_wait<C::STAGES - 2>();   // slot c is here (this thread's copies) ...
        __syncthreads();            // ... and everyone's; slot (c - 1) % STAGES is free again
        if (c + C::STAGES - 1 < nch) {
            issue_w(c + C::STAGES - 1);
            issue_x(c + C::STAGES - 1);
        }
        gm_commit();
        const uint8_t* slot = s_m3 + (c % C::STAGES) * C::SLOT;
        if (c & 1) m3_slot<MT, 2>(acc, slot, s_grid, ncols, w, g, t);
        else m3_slot<MT, 0>(acc, slot, s_grid, ncols, w, g, t);
    }
    gm_wait<0>();
    __syncthreads();
    float* s_part = (float*) s_m3;   // [column][row][9]: warp w's sum at w
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {   // the butterfly's steps 16 (L and L + 16) and 8
            const int row = 16 * mt + 8 * (q >> 1) + g, col = 2 * t + (q & 1);
            const float lo = acc[mt][0][q] + acc[mt][2][q];
            const float hi = acc[mt][1][q] + acc[mt][3][q];
            s_part[(col * C::R + row) * 9 + w] = lo + hi;
        }
    __syncthreads();
    {   // steps 4, 2 and 1: lanes 0..7 -> 0..3 -> 0..1 -> 0
        const int row = tid % C::R, col = tid / C::R;
        if (col < ncols && row0 + row < n_out) {
            const float* p = s_part + (col * C::R + row) * 9;
            const float c0 = p[0] + p[4], c1 = p[1] + p[5], c2 = p[2] + p[6], c3 = p[3] + p[7];
            const float d0 = c0 + c2, d1 = c1 + c3;
            y[(size_t) col * n_out + row0 + row] = d0 + d1;
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();   // the host launches it only where device code for the tensor-core instructions was built
#endif
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        [[maybe_unused]] float sc = 1.0f;
        if constexpr (TD == 40) sc = nvfp4_tail((const void*) grp_ptr[g], L)[2];
        for (int e = e0; e < e1; ++e) {
            float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if constexpr (TD == 40) s *= sc;
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
        }
    }
}

__device__ __forceinline__ unsigned int ld_acquire_gpu(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned int* p, unsigned int v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void red_release_add_gpu(unsigned int* p, unsigned int v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// native_down_kernel with the entries taken GRP_NC at a time.  DEP: launched as a programmatic dependent - the
// block's first group's weight rows are prefetched into L2 before the wait (and, for IQ4_NL, loaded into registers).
template<int TD, bool DEP = false>
__global__ void __launch_bounds__(256) native_down_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                const int32_t* __restrict__ grp_start,
                                                                const int32_t* __restrict__ n_groups,
                                                                const int32_t* __restrict__ ent_dst,
                                                                const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                float* __restrict__ out) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    constexpr bool PRE = DEP && TD == 20;
    // PRE: this thread's weight words for the block's first group are loaded before the wait (the weights do not
    // depend on the previous kernel), so the first entry batch starts with them in registers
    typename Split<TD>::W pre[5];
    bool pre_ok = false;
    if constexpr (DEP && PRE && TD == 20) {
        if (L.n_ff == 640) {
            const int r = blockIdx.x * 32 + (threadIdx.x >> 3), t = threadIdx.x & 7;
            if (r < L.n_embd) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[blockIdx.y] + L.down_off + (size_t) r * L.d_row;
                using F = Fmt<TD>;
                constexpr int ks[5] = { 0, 32, 16, 8, 24 };
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int k = t + ks[i];
                    pre[i] = Split<TD>::load(wr, k / F::ipb, F::step * (k % F::ipb));
                }
                pre_ok = true;
            }
        }
    }
    if constexpr (DEP) {
        const int rows = L.n_ff == 640 ? (kDown40<TD> ? 32 : TD == 42 ? 16 : 8) : 8;
        const int r0 = blockIdx.x * rows, nr = min(rows, (int) L.n_embd - r0);
        const uint8_t* p = (const uint8_t*) grp_ptr[blockIdx.y] + L.down_off + (size_t) r0 * L.d_row;
        for (int o = threadIdx.x * 128; o < nr * (int) L.d_row; o += 256 * 128)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
        pdl_wait();
        pdl_launch_dependents();   // the tail wait may start polling the tiers once every block is past its wait
    }
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    if constexpr (kDown40<TD>) {   // K = 40 calls per row either way: 8 lanes a row
        if (hb == 20) {
            const int r = blockIdx.x * 32 + (threadIdx.x >> 3);
            const int t = threadIdx.x & 7;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                [[maybe_unused]] const float sc = expert_out_scale<TD>(grp_ptr[g], L, 2);
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const bool use_pre = PRE && pre_ok && g == blockIdx.y && e == e0;
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            if constexpr (PRE) {
                                if (use_pre) row_dot_40_sub8<TD, 1, true, true>(wr, s_hq, off1, 1, t, s1, pre);
                                else row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            } else row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            tail_scale<TD>(s1, sc);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            if constexpr (PRE) {
                                if (use_pre) row_dot_40_sub8<TD, 2, true, true>(wr, s_hq, off2, 2, t, s2, pre);
                                else row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            } else row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            tail_scale<TD>(s2, sc);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = t == 0 ? s2[0] : s2[1];
                        } else {
                            int off[GRP_NC];
#pragma unroll
                            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                            float s[GRP_NC];
                            if constexpr (PRE) {
                                if (use_pre) row_dot_40_sub8<TD, GRP_NC, false, true>(wr, s_hq, off, n, t, s, pre);
                                else row_dot_40_sub8<TD, GRP_NC>(wr, s_hq, off, n, t, s);
                            } else row_dot_40_sub8<TD, GRP_NC>(wr, s_hq, off, n, t, s);
                            tail_scale<TD>(s, sc);
#pragma unroll
                            for (int c = 0; c < GRP_NC; ++c)
                                if (c < n && t == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
                        }
                    }
                }
            }
            return;
        }
    } else if constexpr (TD == 42) {
        if (hb == 20) {
            const int r = blockIdx.x * 16 + (threadIdx.x >> 4);
            const int t = threadIdx.x & 15;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = t == 0 ? s2[0] : s2[1];
                        } else {
                            int off[GRP_NC];
#pragma unroll
                            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                            float s[GRP_NC];
                            row_dot_20_sub16<TD, GRP_NC>(wr, s_hq, off, n, t, s);
#pragma unroll
                            for (int c = 0; c < GRP_NC; ++c)
                                if (c < n && t == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
                        }
                    }
                }
            }
            return;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    const bool valid_r = (r < L.n_embd);
    const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        [[maybe_unused]] const float sc = expert_out_scale<TD>(grp_ptr[g], L, 2);
        if (hb == 20) {
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                const int words = n * (20 * 9);
                __syncthreads();
                for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                __syncthreads();
                if (valid_r) {
                    if (n == 1) {
                        const int off1[1] = { 0 };
                        float s1[1];
                        row_dot_multi<TD, 1, true>(wr, s_hq, off1, 1, nb, lane, s1);
                        tail_scale<TD>(s1, sc);
                        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                    } else if (n == 2) {
                        const int off2[2] = { 0, 20 };
                        float s2[2];
                        row_dot_multi<TD, 2, true>(wr, s_hq, off2, 2, nb, lane, s2);
                        tail_scale<TD>(s2, sc);
                        if (lane < 2) out[(size_t) ent_dst[e + lane] * L.n_embd + r] = s2[lane];
                    } else {
                        int off[GRP_NC];
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                        float s[GRP_NC];
                        row_dot_multi<TD, GRP_NC>(wr, s_hq, off, n, nb, lane, s);
                        tail_scale<TD>(s, sc);
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
                    }
                }
            }
        } else if (valid_r) {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC>(wr, hq, off, n, nb, lane, s);
                tail_scale<TD>(s, sc);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
            }
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q8_1_store(x[i], y, i);
}

// swiglu_entries_kernel and quantize_q8_1_kernel in one pass, over the call's own entries only:
// [grp_start[0], grp_start[*n_groups]) (a call's entries are contiguous; the verify window's PCIe call starts after
// its VRAM call) instead of all cap_entries twice - nothing reads the others: the down kernel reads its groups'.
// n_ff is a multiple of 32, so a warp is one q8_1 block and the bounds are warp-uniform.  The values are the two
// kernels': the SwiGLU product is rounded on its own - it must not contract into the first add of q8_1_store's block
// sum, as it could not when it went through memory (__fmul_rn) - then q8_1_store unchanged: the same blocks, bit
// for bit.
__global__ void __launch_bounds__(256) swiglu_q8_1_entries_kernel(const float* __restrict__ gate,
                                                                  const float* __restrict__ up,
                                                                  const int32_t* __restrict__ grp_start,
                                                                  const int32_t* __restrict__ n_groups, int n_ff,
                                                                  block_q8_1* __restrict__ hq) {
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    for (long long i = lo + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += (long long) gridDim.x * blockDim.x) {
        const float g = gate[i];
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
        q8_1_store(h, hq, i);
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = __low2half(x[ibs].dm);
        const float dmin = __high2half(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
// llama.cpp's dequantize_block_q6_K (64 threads folded onto 32): each value is d * scale * (6-bit code - 32).
template<typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q6_K* x = (const block_q6_K*) vx;
    const float d = __half2float(x[ibs].d);
    for (int tt = tid; tt < 64; tt += 32) {
        const int ip = tt / 32, il = tt % 32, is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0] = cvt<dst_t>(d * sc[0] * (float) ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * (float) ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * (float) ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * (float) ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template<typename dst_t>
__device__ void dq_q5_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>(((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d);
    }
}
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it): 8 values per thread.
template<typename dst_t>
__device__ void dq_bf16(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] = cvt<dst_t>(__uint_as_float((uint32_t) x[j] << 16));
}

// NVFP4: superblock ibs is four 64-value blocks; thread tid writes 8 values of block tid/8, sub-block (tid%8)/2, the
// low (even tid) or high nibbles.  `s` is the matrix's FP32 scale (1 leaves ggml's to_float values unchanged).
template<typename dst_t>
__device__ void dq_nvfp4(const void* vx, int64_t ibs, dst_t* yy, int tid, float s) {
    const block_nvfp4* x = (const block_nvfp4*) vx + 4 * ibs + tid / 8;
    const int sb = (tid % 8) / 2, sh = 4 * (tid & 1);
    const float d = nvfp4_scale_dev(x->d[sb]);
    const uint8_t* q = x->qs + 8 * sb;
    dst_t* y = yy + 64 * (tid / 8) + 16 * sb + 2 * sh;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) kvalues_fp4[(q[j] >> sh) & 0xf] * d * s);
}

// Every type below must also be in is_iq() (BF16 and Q6_K: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid, float s = 1.0f) {
    switch (ty) {
        case 40: dq_nvfp4(vx, ibs, y, tid, s); break;
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 14: dq_q6_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y,
                                    const float* __restrict__ scale = nullptr) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x, scale ? *scale : 1.0f);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y, const float* __restrict__ scale) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x,
                        scale ? scale[parity] : 1.0f);
}

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 8 || t == 40;
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define QK_CASE(T) case T: return Fmt<T>::qk;
        GU_FMTS(QK_CASE)
#undef QK_CASE
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define QK_CASE(T) case T: return Fmt<T>::qk;
        D_FMTS(QK_CASE)
#undef QK_CASE
        default: return 0;
    }
}

// The per-column kernels instead of the ones that decode a weight part once for every column (the same bits; tests)
bool g_old_kernels = false;
// 64-bit i-quant codebook tables staged into shared memory
constexpr bool g_stage_grid = true;
// five or six columns in one pass of six rather than of eight (the same bits)
constexpr bool g_nc6 = true;
// the grouped gate/up kernel reads the group's activations from global memory (the same bits as staging them)
constexpr bool g_gu_stage = false;
// the grid formats' gate/up rows two a warp (native_gu2_kernel), and the SwiGLU pass inside the down kernel
// (native_down2_kernel); native_grouped_set_gu2 sets it for tests.  The same bits.
int g_gu2 = 2;
// the most block rows a grouped call launches when its caller leaves grid_groups to it (more groups take the stride;
// the same bits)
constexpr int g_dec_gy = 24;
// IQ3_XXS gate/up on int8 tensor cores (1), or only for the calls whose caller asks (2): off (0)
constexpr int g_gu_mma = 0;
thread_local int t_gu_mma_hint = 0;
// Whether native_gu_mma_kernel runs on the current device at L's shapes: int8 MMA with device code for it in this
// build, n_embd in whole chunks (rows 4-byte aligned), the shared memory without opt-in
bool gu_mma_usable(const NativeExpertLayout& L) {
    if (L.n_embd <= 0 || L.n_embd % 512 != 0 || L.n_ff % kGmRows != 0 || gu_mma_smem(L.n_embd) > 48 * 1024)
        return false;
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        cudaFuncAttributes fa{};
        const bool q1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess;
        const bool q2 = q1 && cudaFuncGetAttributes(&fa, native_gu_mma_kernel) == cudaSuccess;
        if (!q1 || !q2) cudaGetLastError();
        ok[dev] = q2 && major >= 8 && fa.ptxVersion >= 80 ? 1 : 2;
    }
    return ok[dev] == 1;
}
// iq_mmvq's IQ3_S calls of 2..8 columns with n_in % 512 == 0 on mmvq_iq3s_mma_kernel where the device runs it (the
// same bits); iq_set_mmvq_mma turns it on for tests.
bool g_mmvq_mma = false;
// the fewest columns that take it
int g_mmvq_mma_min_t = 4;
std::atomic<long> g_mmvq_mma_launches{0};
// The current device's SM count if it runs mmvq_iq3s_mma_kernel (int8 MMA with device code for it in this build),
// else 0; per device
int mmvq_mma_sms() {
    static int known[64] = {};   // 0 unknown, -1 no, else the SM count
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (known[dev] == 0) {
        int major = 0, sms = 0;
        cudaFuncAttributes f1{}, f2{};
        const bool q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                       cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
                       cudaFuncGetAttributes(&f1, mmvq_iq3s_mma_kernel<1>) == cudaSuccess &&
                       cudaFuncGetAttributes(&f2, mmvq_iq3s_mma_kernel<2>) == cudaSuccess;
        if (!q) cudaGetLastError();
        known[dev] = q && major >= 8 && f1.ptxVersion >= 80 && f2.ptxVersion >= 80 && sms > 0
                         ? sms : -1;
    }
    return known[dev] > 0 ? known[dev] : 0;
}
// The single-matrix mmvq (128-thread blocks, one per 4 rows) does not stage the tables: the 2-8 KB per block cost
// more than it saved there, while the grouped expert kernels gain (bitwise the same either way).
constexpr bool g_stage_grid_mmvq = false;

template<int TY>
void launch_mmvq(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, int ncols,
                 cudaStream_t s) {
    if constexpr (TY == 21) {
        if (g_mmvq_mma && !g_old_kernels && ncols >= std::max(2, g_mmvq_mma_min_t + (n_in > n_out ? 1 : 0)) && ncols <= kGmCols &&
            n_in % 512 == 0) {
            if (const int sms = mmvq_mma_sms()) {   // 32 rows per block while that gives every SM one, else 16
                if ((n_out + 31) / 32 >= sms)
                    launch_pdl(mmvq_iq3s_mma_kernel<2>, dim3((unsigned) ((n_out + 31) / 32)), dim3(256), 0, s, W, rb, X,
                               y, n_in, n_out, ncols);
                else
                    launch_pdl(mmvq_iq3s_mma_kernel<1>, dim3((unsigned) ((n_out + 15) / 16)), dim3(256), 0, s, W, rb, X,
                               y, n_in, n_out, ncols);
                g_mmvq_mma_launches.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    auto go = [&](auto kernel) { launch_pdl(kernel, grid, block, 0, s, W, rb, X, y, n_in, n_out, ncols); };
    if constexpr (!kSplit<TY>) go(mmvq_kernel<TY>);
    else if (g_old_kernels) go(mmvq_kernel<TY>);
    else if constexpr (kStageIqGrid<TY>) {
        if (!g_stage_grid_mmvq) {
            if (ncols <= 1) go(mmvq_multi_kernel<TY, 1, false>);
            else if (ncols == 2) go(mmvq_multi_kernel<TY, 2, false>);
            else if (ncols <= 4) go(mmvq_multi_kernel<TY, 4, false>);
            else if (ncols <= 6 && g_nc6) go(mmvq_multi_kernel<TY, 6, false>);
            else go(mmvq_multi_kernel<TY, 8, false>);
            return;
        }
        if (ncols <= 1) go(mmvq_multi_kernel<TY, 1>);
        else if (ncols == 2) go(mmvq_multi_kernel<TY, 2>);
        else if (ncols <= 4) go(mmvq_multi_kernel<TY, 4>);
        else if (ncols <= 6 && g_nc6) go(mmvq_multi_kernel<TY, 6>);
        else go(mmvq_multi_kernel<TY, 8>);
    } else {
        if (ncols <= 1) go(mmvq_multi_kernel<TY, 1>);
        else if (ncols == 2) go(mmvq_multi_kernel<TY, 2>);
        else if (ncols <= 4) go(mmvq_multi_kernel<TY, 4>);
        else if (ncols <= 6 && g_nc6) go(mmvq_multi_kernel<TY, 6>);
        else go(mmvq_multi_kernel<TY, 8>);   // 8 at a time past 8
    }
}

#include "gu2.cuh"

template<int TG>
void launch_gu(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
               const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
               float* gate, float* up, bool allow_mma) {
    if constexpr (TG == 18) {
        if (allow_mma && (g_gu_mma == 1 || (g_gu_mma == 2 && t_gu_mma_hint)) && !g_old_kernels && gu_mma_usable(L)) {
            const dim3 gm((unsigned) (2 * L.n_ff / kGmRows), grid.y);
            native_gu_mma_kernel<<<gm, 256, gu_mma_smem(L.n_embd), s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L,
                                                                         gate, up);
            return;
        }
    }
    if constexpr (kGu2Q2<TG>) {
        if (allow_mma && g_gu2 && !g_old_kernels && gu2_usable(L)) {
            const dim3 g2((unsigned) ((2 * L.n_ff + GU2_ROWS - 1) / GU2_ROWS), grid.y);
            native_gu2_kernel<TG><<<g2, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
    }
    const size_t sx_bytes = (size_t) kGuStageT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
    if constexpr (!kSplit<TG>) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if (g_old_kernels) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if (g_gu_stage && sx_bytes <= 40 * 1024) {
        if constexpr (kStageIqGrid<TG>) {
            if (!g_stage_grid) {
                native_gu_multi_sx_kernel<TG, false><<<grid, 256, sx_bytes, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                return;
            }
        }
        native_gu_multi_sx_kernel<TG><<<grid, 256, sx_bytes, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
    else if constexpr (kStageIqGrid<TG>) {
        if (!g_stage_grid) {
            native_gu_multi_kernel<TG, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
        native_gu_multi_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else native_gu_multi_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
}

template<int TD>
void launch_down(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L,
                 float* out, bool dep = false) {
    if constexpr (!kSplit<TD>) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if (g_old_kernels) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if (dep) launch_dependent(native_down_multi_kernel<TD, true>, grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else native_down_multi_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
void native_grouped_set_gu2(int mode) { g_gu2 = mode; }
bool iq_old_kernels() { return g_old_kernels; }
void iq_set_mmvq_mma(bool on) { g_mmvq_mma = on; g_mmvq_mma_min_t = 1; }   // tests: every column count
bool iq_mmvq_mma() { return g_mmvq_mma; }
bool iq_mmvq_mma_available() { return mmvq_mma_sms() > 0; }
long iq_mmvq_mma_launches() { return g_mmvq_mma_launches.load(std::memory_order_relaxed); }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 14 || t == 30; }   // Q6_K: the embedding and the expert gate/up

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 40: return (size_t) (n / 64) * sizeof(block_nvfp4);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define MMVQ_CASE(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        MMVQ_FMTS(MMVQ_CASE)
#undef MMVQ_CASE
        default: basalt::log::error("basalt", "iq_mmvq: type %d is not supported", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale) {
    if (n % 256 != 0 || !is_iq(t)) { basalt::log::error("basalt", "iq_dequant_f16: bad arguments"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst, scale);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, const uint8_t* __restrict__ hot,
                                  int64_t n_hot, size_t row_bytes, const int32_t* __restrict__ tokens, int64_t n_embd,
                                  float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const int32_t tok = tokens[t];
    const uint8_t* row = (tok < n_hot ? hot : table) + (size_t) tok * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream, const void* hot, int64_t n_hot) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { basalt::log::error("basalt", "iq_embed_rows: bad arguments"); std::exit(1); }
    if (hot == nullptr) n_hot = 0;
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, (const uint8_t*) hot, n_hot, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { basalt::log::error("basalt", "iq_dequant_f32: bad arguments"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream,
                       const float* scale) {
    // checked like the other entry points: an unknown type would leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { basalt::log::error("basalt", "iq_dequant_gu_f16: type %d / %lld", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst, scale);
    check("iq_dequant_gu_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row + (gu_type == 40 ? kNvfp4Tail : 0);   // NVFP4's scale tail
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}


int g_exp_phase = 0;   // the bench: 0 all, 1 gate/up + swiglu + quantize only, 2 down only

void native_expert_set_mode(int mode, int phase) {
    (void) mode;
    g_exp_phase = phase;
}

namespace {
bool g_grouped_v1 = false;
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }
void native_grouped_set_mma_hint(bool on) {
    t_gu_mma_hint = on ? 1 : 0;
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { basalt::log::error("basalt", "native_expert_grouped: n_ff %lld", (long long) L.n_ff); std::exit(1); }
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const bool v1 = g_grouped_v1;
    const bool chain = g_gu2 >= 2 && !v1 && !g_old_kernels;   // SwiGLU and down as dependents
    int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    if (chain && grid_groups <= 0 && g_dec_gy > 0 && gy > g_dec_gy) gy = g_dec_gy;
    const dim3 ggu((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) gy);
    if (g_exp_phase != 2) {
    switch (L.gu_type) {
#define GU_CASE(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, !v1); break;
        GU_FMTS(GU_CASE)
#undef GU_CASE
        default: basalt::log::error("basalt", "native_expert_grouped: gate/up type %d", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    const bool sw_v1 = v1;
    if (sw_v1) {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
        quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    } else if (chain) {
        launch_dependent(swiglu_q8_1_entries_pdl_kernel, dim3((unsigned) ((nh + 255) / 256)), s, (const float*) gate,
                         (const float*) up, grp_start, n_groups, (int) L.n_ff, hq);
    } else {
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups,
                                                                                (int) L.n_ff, hq);
    }
    check("native_expert_grouped/swiglu");
    }
    if (g_exp_phase == 1) return;
    const int d_rows = (!g_old_kernels && L.n_ff == 640) ? (kDown40Type(L.d_type) ? 32 : (L.d_type == 42 ? 16 : 8)) : 8;
    const dim3 gd((unsigned) ((L.n_embd + d_rows - 1) / d_rows), (unsigned) gy);
    switch (L.d_type) {
#define DOWN_CASE(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, chain); break;
        D_FMTS(DOWN_CASE)
#undef DOWN_CASE
        default: basalt::log::error("basalt", "native_expert_grouped: down type %d", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace basalt::kernels

#define FZ_DEV_ONLY
#include "dec_gdn_rec.cu"   // the peer mixer item's recurrence (device code only)
#undef FZ_DEV_ONLY
#include "dec_moe.cuh"   // DEC_MOE: the decode MoE tail's kernels, on this file's row dots
#include "pk_moe.cuh"   // persistent experts + combine, on the same device code
