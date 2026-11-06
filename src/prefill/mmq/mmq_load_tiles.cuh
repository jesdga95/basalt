// SPDX-License-Identifier: MIT
// Loads of the quantized weight tiles into shared memory, per format.
// llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (https://github.com/ggml-org/llama.cpp),
// ggml/src/ggml-cuda/mmq-load-tiles.cuh.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#pragma once

#include "vecdotq.cuh"

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int blocks_per_iter = MMQ_ITER_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iter * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 4;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qxi[j];

            // unpack even and odd crumbs into byte values
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            // unshuffle values
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);

            x_qs[i*sram_stride           + dst_offset + j*2+0] = qx;
            x_qs[i*sram_stride           + dst_offset + j*2+1] = qy;
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + scale_block;

        x_df[i*sram_stride                           + ksx] = bxi->d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i*sram_stride + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride                           + kbxd] = bxi->d;
    }
}

// IQ1_S: eight grid values per call from the 11-bit index, one 3-bit scale and one grid shift per 32 values; the shift
// term rides in the half2 (scale, scale*delta) the q8_1 x q8_1 dot reads.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_S);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

#pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid1;
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

        x_ds[i*sram_stride + kqsx] = make_half2(d1q, d1q*delta);
    }
}

// IQ1_M: eight grid values per 11-bit index, a grid shift of +-1/8 per eight values and a 3-bit scale per 16 values. The
// int8 weight is 8 * (code - 1) +- 1, exact; the scale per 16 values is d * (2s + 1) / 8.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_m(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b4(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl  = bxi->qh[2*kqsx + l/2] >> (4*(l%2));
            const int grid = iq1s_grid_gpu[qs[l] | ((qhl & 0x07) << 8)];
            const int sub  = (qhl & 0x08) ? 0xF7F7F7F7 : 0xF9F9F9F9;   // -9 or -7 in every byte

            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = __vadd4(((grid >> 0) & 0x0F0F0F0F) << 3, sub);
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = __vadd4(((grid >> 4) & 0x0F0F0F0F) << 3, sub);
        }

        const uint16_t * sc = (const uint16_t *) bxi->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d   = __half2float(scale.f16) / 8.0f;
        const int   tmp = sc[kqsx/2] >> (6*(kqsx%2));

        x_df[i*sram_stride                             + 2*kqsx+0] = d * (2*((tmp >> 0) & 0x07) + 1);
        x_df[i*sram_stride                             + 2*kqsx+1] = d * (2*((tmp >> 3) & 0x07) + 1);
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[packed_byte(q2, l)];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid1;
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (packed_byte(qs_packed, l) | ((qh << (8-2*l)) & 0x300)));

            const uint32_t sgn = packed_byte(signs_packed_32, l);
            const int signs0 = __vcmpne4(((sgn & 0x03) << 7) | ((sgn & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sgn & 0x30) << 3) | ((sgn & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[packed_byte(q3_packed, 2*l+0)], iq3xxs_grid[packed_byte(q3_packed, 2*l+1)]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = (ls*d + d/2)/2;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[packed_byte(qs_packed, 2*l+0) | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[packed_byte(qs_packed, 2*l+1) | ((qh << (7 - 2*l)) & 0x100)]);

            const uint32_t sgn = packed_byte(signs_packed_32, l);
            const int signs0 = __vcmpne4(((sgn & 0x03) << 7) | ((sgn & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sgn & 0x30) << 3) | ((sgn & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = ls*d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

        x_df[i*sram_stride             + threadIdx.x % 8] = d * (ls - 32);
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;

        x_qs[i*sram_stride + k0 + 0]      = v.x;
        x_qs[i*sram_stride + k0 + QI4_NL] = v.y;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride                       + kbxd] = __half2float(bxi->d);
    }
}

// NVFP4 (Basalt): the UE4M3 sub-block scale as ggml decodes it (half the value, kvalues_fp4 being doubled; 0 and 0x7F
// as 0, bit 7 ignored), built from its bits: the same float as the decode kernels' nvfp4_scale_dev.
static __device__ __forceinline__ float ggml_cuda_mmq_nvfp4_scale(const uint32_t x) {
    const uint32_t v = x & 0x7F;
    if (v == 0 || v == 0x7F) return 0.0f;
    const uint32_t e = v >> 3, m = v & 7;
    if (e == 0) return (float) m * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119) << 23) | (m << 20));
}

// NVFP4 (Basalt): one thread per 16-value sub-block.  Its 8 code bytes hold value j in the low nibble of byte j and
// value j + 8 in the high one, so the tile gets 4 ints in value order and the sub-block's scale goes into the
// per-16 scale slot (the Q3_K layout that ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma reads).  The per-expert FP32 scale
// is not applied here (the caller applies it to the products).
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int nsub            = QK_NVFP4 / QK_NVFP4_SUB;
    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4_SUB;
    constexpr int nrows           = warp_size / threads_per_row;
    const int txi = threadIdx.x % threads_per_row;
    const int kbx = txi / nsub;   // block within the tile row
    const int sb  = txi % nsub;   // sub-block within the block

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = (const block_nvfp4 *) x + kbx0 + i*stride + kbx;

        const int2 v0 = get_int_from_table_16(get_int_b4(bxi->qs, 2*sb + 0), kvalues_fp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bxi->qs, 2*sb + 1), kvalues_fp4);
        const int k0 = 4*txi;

        x_qs[i*sram_stride + k0 + 0] = v0.x;
        x_qs[i*sram_stride + k0 + 1] = v1.x;
        x_qs[i*sram_stride + k0 + 2] = v0.y;
        x_qs[i*sram_stride + k0 + 3] = v1.y;
        x_df[i*sram_stride + txi]    = ggml_cuda_mmq_nvfp4_scale(bxi->d[sb]);
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_0;
    const int kqsx = txi % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride                     + kbxd] = bxi->d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_1(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_1;
    const int kqsx = txi % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

        x_dm[i*sram_stride + kbxd] = bxi->dm;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

            x_qs[i*sram_stride + k] = x_qs_k;
        }

        const int sc_m = bxi->scales[kqsx];
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));

        x_dm[i*sram_stride + kqsx] = x_dm_ik;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q3_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

            x_qs[i*sram_stride + k] = x_qs_k;
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*sram_stride + sizeof(int)*ksc + l] = d*sc8[l];
        }
    }
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2((int) packed_byte(sc32, l), (int) packed_byte(m32, l));
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

        x_qs[i*sram_stride + kq0] = ql0 | qh0;
        x_qs[i*sram_stride + kq1] = ql1 | qh1;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2((int) packed_byte(sc32, l), (int) packed_byte(m32, l));
        }
    }
}

// Q6_K: the 6-bit codes (minus 32) as signed int8 in value order, and one float scale (d times the int8 sub-scale)
// per 16 values, in the layout ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma reads.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    static_assert(MMQ_ITER_K == QK_K && warp_size == 32, "one block per tile row, one warp per row");

    // A thread: half h of the block, 4 consecutive positions l of 32, and one of the two ql/qh bit pairs pr.
    const int l4 = threadIdx.x % 8;
    const int h  = (threadIdx.x / 8) % 2;
    const int pr = threadIdx.x / 16;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, h*16 + pr*8 + l4);
        const int qh = get_int_b2(bxi->qh, h*8 + l4);

        // Positions 128h + 32s + l: ql's low nibble (s = pr) or high nibble (s = 2 + pr), qh bits 2s..2s+1.
        const int lo = (ql & 0x0F0F0F0F)        | (((qh >> (2*pr))     << 4) & 0x30303030);
        const int hi = ((ql >> 4) & 0x0F0F0F0F) | (((qh >> (4 + 2*pr)) << 4) & 0x30303030);

        x_qs[i*sram_stride + h*32 + pr*8      + l4] = __vsubss4(lo, 0x20202020);
        x_qs[i*sram_stride + h*32 + pr*8 + 16 + l4] = __vsubss4(hi, 0x20202020);
    }

    constexpr int rows_per_warp = warp_size / 16;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/16;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        x_df[i*sram_stride + threadIdx.x % 16] = __half2float(bxi->d) * bxi->scales[threadIdx.x % 16];
    }
}
