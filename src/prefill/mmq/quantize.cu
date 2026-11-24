// SPDX-License-Identifier: MIT
// Quantizes float activation rows to the block_q8_1_mmq layout, gathering rows through an index map.
// llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (https://github.com/ggml-org/llama.cpp),
// ggml/src/ggml-cuda/quantize.cu.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "quantize.cuh"
#include <cstdint>

// One thread quantizes 4 consecutive floats; the 8 threads of a 32-value block share one scale (and, for DS4, the
// sum of its rounded values times the scale); for D2S6 the 16 threads of 64 values share one scale, and each 16 values
// keep the sum of their rounded values times it.  Row i of the output is row ids[i] of the input (row i without `ids`).
// RELU2 (Basalt): each value first becomes relu(x s_in)^2 s_down with its row's expert's scales (row r is expert j's for
// bounds[j] <= r < bounds[j + 1]; s_in = tails[4 j], s_down = tails[4 j + 2]), a squared-ReLU expert's hidden row.
template <mmq_q8_1_ds_layout ds_layout, bool RELU2>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int32_t * __restrict__ bounds, const int n_experts,
        const float * __restrict__ tails) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    const int64_t i2  = blockIdx.z % ne2;
    const int64_t i3  = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
    const int64_t base_idx = i3*s03 + i2*s02 + i01*s01;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (RELU2) {
        int j = 0;
        while (j + 1 < n_experts && bounds[j + 1] <= i01) ++j;
        const float si = tails[4*j], sd = tails[4*j + 2];
        const auto r2 = [&](float v) {
            const float r = fmaxf(__fmul_rn(v, si), 0.0f);
            return __fmul_rn(__fmul_rn(r, r), sd);
        };
        xi = make_float4(r2(xi.x), r2(xi.y), r2(xi.z), r2(xi.w));
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // DS4: the block's sum of the rounded values times d, so the minimum's term meets the same activations as the codes
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        int isum = (int) q.x + (int) q.y + (int) q.z + (int) q.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, offset, WARP_SIZE);
        }
        sum = d * (float) isum;
    }

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + k_block*ne1 + blockIdx.x;

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs/16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs/64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32]  = d;
        }
    }
}

template <bool RELU2>
static void quantize_launch(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_x,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream,
        const int32_t * bounds, const int n_experts, const float * tails) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, RELU2><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, bounds, n_experts, tails);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, RELU2><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, bounds, n_experts, tails);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, RELU2><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, bounds, n_experts, tails);
            break;
    }
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_x,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_launch<false>(x, ids, vy, type_x, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream, nullptr, 0, nullptr);
}

void quantize_mmq_q8_1_relu2_cuda(
        const float * x, void * vy, const ggml_type type_x, const int64_t ne00, const int64_t s01, const int64_t ne0,
        const int64_t ne1, const int32_t * bounds, const int n_experts, const float * tails, cudaStream_t stream) {
    quantize_launch<true>(x, nullptr, vy, type_x, ne00, s01, ne1 * s01, ne1 * s01, ne0, ne1, 1, 1, stream, bounds, n_experts, tails);
}
