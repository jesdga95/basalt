// src/kernels/cuda/window_gemm_sliced.cuh - the decode window products against FP8 E4M3 matrices, K sliced across a
// block's warps (see include/basalt/kernels/window_gemm.hpp).  No kernel is instantiated here.
//
// A block owns 16 rows of one matrix over the whole K; its KW warps each take a fixed slice of every group of KW * 64 KB
// values (warp w: values w * 64 KB .. + 64 KB - 1 of the group), so there is no split across blocks, no workspace and
// no ticket.  The group's weights and the window's activation rows stream through ST shared-memory stages together
// (cp.async; the first stages' weights issued before the PDL wait, since they are constant).  Thread (g, c) of a warp
// holds weight rows g and g + 8 and activation column g of each 8-column tile; in each 64-value block of its slice it
// takes values 16c .. 16c + 15, step s's k-slots {2c, 2c+1} values 16c + 4s, +1 and {2c+8, 2c+9} values 16c + 4s + 2,
// +3, and multiplies the same values of its activation column, so the sums are plain dot products.  The warps' sums
// meet in shared memory and are added in warp order.  KW and KB depend on the call's shapes only, and the column tiles
// (NT) never meet: a column's bits are the same at any T and any NT.
//
// Adapts the layout of Infernix's fp8_a16_sliced_k_mma (Apache-2.0, by Ian Ranson and the Infernix contributors; see
// THIRD_PARTY_LICENSES): a 16-row block over the whole K, warps on fixed K slices, the activations staged beside the
// weights, the slices reduced in the block.
#pragma once

#include "window_gemm.cuh"

namespace basalt::kernels::wgemm::detail {

template <int KW, int KB, int NT, int ST>
struct SlicedCfg {
    static constexpr int kThreads = 32 * KW;
    static constexpr int kTileK = 64 * KB;           // values of a warp's slice per group
    static constexpr int kGroupK = KW * kTileK;      // values per group
    static constexpr int kCols = 8 * NT;             // activation rows a stage holds
    static constexpr int kWBytes = 16 * kGroupK;     // E4M3 weights per stage: [KW][16][kTileK]
    static constexpr int kABytes = kCols * kGroupK * 2;   // bf16 activations per stage: [KW][kCols][kTileK]
    static constexpr int kStage = kWBytes + kABytes;
    static constexpr int kPartStride = 20;           // floats per column of a warp's sums (16 rows, padded for banks)
    static constexpr int kPart = KW * kCols * kPartStride * 4;   // over the stages once the last group is read
    static constexpr int kSmem = ST * kStage > kPart ? ST * kStage : kPart;
    static_assert(KW == 2 || KW == 4 || KW == 8, "window gemm: warps per block");
};

template <int N> __device__ __forceinline__ void cp_async_ca(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(gmem), "n"(N));
}

// One block's 16 rows: tile `tile` of the call's tiles.
template <int KW, int KB, int NT, int ST>
__device__ __forceinline__ void sliced_stage(const Args& a, int tile, uint8_t* smem) {
    using C = SlicedCfg<KW, KB, NT, ST>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    int mi = 0;
    while (mi + 1 < a.n && tile >= a.tile0[mi + 1]) ++mi;
    const int64_t row0 = (int64_t) (tile - a.tile0[mi]) * 16, rows = a.rows[mi], K = a.K;
    const uint8_t* W = a.w[mi];
    const int groups = (int) (K / C::kGroupK), T = a.T;

    // a weight row's 16-byte piece q of a warp's slice lands on piece q ^ wz(r); an activation row's on q ^ (t & 1)
    auto wz = [](int r) { return KB >= 2 ? (r & 1) << 2 : 0; };
    auto issue_w = [&](int stage, int grp) {
        uint8_t* st = smem + stage * C::kStage;
        constexpr int kPieces = 16 * (C::kGroupK / 16), kPerRow = C::kGroupK / 16, kPerSlice = C::kTileK / 16;
        for (int p = threadIdx.x; p < kPieces; p += C::kThreads) {
            const int r = p / kPerRow, j = p - r * kPerRow, w = j / kPerSlice, q = j - w * kPerSlice;
            const int64_t row = min(row0 + r, rows - 1);   // rows past the last read the last (not written)
            cp_async<16>(st + (w * 16 + r) * C::kTileK + ((q ^ wz(r)) << 4), W + row * K + (int64_t) grp * C::kGroupK + j * 16);
        }
    };
    auto issue_a = [&](int stage, int grp) {   // the live rows only: the other columns' sums are never read
        uint8_t* st = smem + stage * C::kStage + C::kWBytes;
        constexpr int kPerRow = C::kGroupK / 8, kPerSlice = C::kTileK / 8;
        const int n = T * kPerRow;
        for (int p = threadIdx.x; p < n; p += C::kThreads) {
            const int t = p / kPerRow, j = p - t * kPerRow, w = j / kPerSlice, q = j - w * kPerSlice;
            cp_async_ca<16>(st + (w * C::kCols + t) * (2 * C::kTileK) + ((q ^ (t & 1)) << 4),
                            a.act + (int64_t) t * K + (int64_t) grp * C::kGroupK + j * 8);
        }
    };

#pragma unroll
    for (int s = 0; s < ST; ++s) {
        if (s < groups) issue_w(s, s);
        cp_commit();
    }
    pdl_wait();   // the activations come from the kernels before this one
    pdl_launch_dependents();
#pragma unroll
    for (int s = 0; s < ST; ++s) {
        if (s < groups) issue_a(s, s);
        cp_commit();
    }

    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    for (int grp = 0; grp < groups; ++grp) {
        const int stage = grp % ST;
        cp_wait<ST - 1>();
        __syncthreads();
        const uint8_t* ws = smem + stage * C::kStage + warp * 16 * C::kTileK;
        const uint8_t* as = smem + stage * C::kStage + C::kWBytes + warp * C::kCols * (2 * C::kTileK);
#pragma unroll
        for (int b = 0; b < KB; ++b) {
            const int pc = 4 * b + c;
            const uint4 w0 = *(const uint4*) (ws + g * C::kTileK + ((pc ^ wz(g)) << 4));
            const uint4 w1 = *(const uint4*) (ws + (g + 8) * C::kTileK + ((pc ^ wz(g + 8)) << 4));
            const unsigned u0[4] = {w0.x, w0.y, w0.z, w0.w}, u1[4] = {w1.x, w1.y, w1.z, w1.w};
            unsigned a0[4], a1[4], a2[4], a3[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                a0[s] = e4m3x2_f16(u0[s]);
                a1[s] = e4m3x2_f16(u1[s]);
                a2[s] = e4m3x2_f16(u0[s] >> 16);
                a3[s] = e4m3x2_f16(u1[s] >> 16);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t * 8 >= T) continue;   // the whole tile past the window
                const int col = t * 8 + g, xq = 8 * b + 2 * c;
                const uint8_t* xr = as + col * (2 * C::kTileK);
                const uint4 xa = *(const uint4*) (xr + ((xq ^ (col & 1)) << 4));
                const uint4 xb = *(const uint4*) (xr + (((xq + 1) ^ (col & 1)) << 4));
                const unsigned xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int s = 0; s < 4; ++s) mma(acc[t], a0[s], a1[s], a2[s], a3[s], xw[2 * s], xw[2 * s + 1]);
            }
        }
        __syncthreads();
        if (grp + ST < groups) {
            issue_w(stage, grp + ST);
            issue_a(stage, grp + ST);
        }
        cp_commit();
    }
    cp_wait<0>();

    // the warps' sums, added in warp order, then the epilogue (every warp is past its last read of the stages)
    float* part = (float*) smem;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = t * 8 + 2 * c + (j & 1), r = g + 8 * (j >> 1);
            part[(warp * C::kCols + col) * C::kPartStride + r] = acc[t][j];
        }
    __syncthreads();
    const float sc = a.scale[mi] ? *a.scale[mi] : 1.0f;
    const int64_t ld = a.ld_y[mi];
    for (int e = threadIdx.x; e < T * 16; e += C::kThreads) {
        const int col = e >> 4, r = e & 15;
        float sum = part[col * C::kPartStride + r];
#pragma unroll
        for (int w = 1; w < KW; ++w) sum += part[(w * C::kCols + col) * C::kPartStride + r];
        const int64_t row = row0 + r;
        if (row >= rows) continue;
        if (a.epi == (int) Epilogue::AddBf16) {
            uint16_t* d16 = a.y16[mi] + col * ld + row;
            // the product rounded as Store leaves it (no fma into the add), then one add, one rounding
            *d16 = __bfloat16_as_ushort(
                __float2bfloat16_rn(__fadd_rn(__bfloat162float(__ushort_as_bfloat16(*d16)), __fmul_rn(sum, sc))));
            continue;
        }
        float* dst = a.y[mi] + col * ld + row;
        const float v = sum * sc;
        if (a.epi == (int) Epilogue::Add) *dst += v;
        else *dst = v;
    }
}

template <int KW, int KB, int NT, int ST>
__global__ void __launch_bounds__(SlicedCfg<KW, KB, NT, ST>::kThreads) wgemm_sliced_kernel(const Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    sliced_stage<KW, KB, NT, ST>(a, blockIdx.x, smem);
}

}  // namespace basalt::kernels::wgemm::detail
