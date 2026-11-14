// src/kernels/cuda/window_gemm.cuh - the decode window products' kernel for NVFP4, BF16 and Q8 matrices (see
// include/basalt/kernels/window_gemm.hpp; FP8 runs window_gemm_sliced.cuh).  No kernel is instantiated here.
//
// A work item is RS rows of each of NM matrices (NM 2: a SwiGLU pair, gate rows in the first warps, up rows in the
// others) over one slice of K; a warp owns 16 MW rows.  The grid is at most one wave of blocks, each taking items
// blockIdx.x, + gridDim.x, ...: their chunks of CH values stream in order through shared memory as one sequence (cp.async,
// ST stages, the first ones issued before the PDL wait since weights are constant), and each warp runs its rows over them on mma m16n8k16 with the window's rows as the columns, NT tiles of 8.  Thread (g, c) holds weight rows
// g and g + 8 of its warp and activation column g of each tile; in every 64-value block of K it takes values
// 16c .. 16c + 15 (for NVFP4 exactly one UE4M3 group per row), step s's k-slots {2c, 2c+1} values 16c + 4s, +1 and
// {2c+8, 2c+9} values 16c + 4s + 2, +3, and multiplies the same values of its activation column, so the sums are plain
// dot products.  With several slices each block writes its sums to the workspace, and the last block of a row tile to
// finish (a ticket) adds them in slice order and applies the epilogue.  The slice count depends on the call's shapes and
// the device only, never on T: a column's bits are the same at any T and any NT.
//
// Adapts the launch protocol of Infernix's decode products (Apache-2.0, by Ian Ranson and the Infernix contributors;
// see THIRD_PARTY_LICENSES): weights staged before the PDL wait, dependents started once the weights are read.
#pragma once

#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace basalt::kernels::wgemm::detail {

struct Args {
    const uint8_t* w[kMaxMatrices];
    const uint8_t* sp[kMaxMatrices];   // NVFP4 scale planes
    int64_t rows[kMaxMatrices];
    const float* scale[kMaxMatrices];
    float* y[kMaxMatrices];
    uint16_t* y16[kMaxMatrices];
    int64_t ld_y[kMaxMatrices];
    int64_t row_off[kMaxMatrices];   // the matrix's first row in the workspace's rows
    int tile0[kMaxMatrices + 1];     // the matrix's first tile (tile0[n]: all tiles)
    int n;
    int64_t K, part_rows;
    int T, tiles, ks, cps, nchunks, epi;
    const uint16_t* act;   // bf16 rows
    uint16_t* act_out;     // SwiGLU: h as the next product's bf16 rows ([16][rows]), or null
    float* partial;
    int32_t* tickets;
};

template <int F, int RS, int NM, int CH, int ST>
struct Cfg {
    static constexpr int kThreads = 32 * NM * RS / 16;   // with one 16-row tile per warp
    static constexpr int CB = F == (int) Format::Q8 ? CH : F == (int) Format::Bf16 ? 2 * CH : CH / 2;   // weight bytes per row per chunk
    static constexpr int P = CB / 16;                                        // 16-byte pieces per row per chunk
    static constexpr int SB = F == (int) Format::Nvfp4 ? (CH / 64) * RS * 4 : F == (int) Format::Q8 ? RS * (CH / 16) : 0;   // scale bytes per matrix per chunk
    static constexpr int MB = RS * CB + SB;                                  // bytes per matrix per stage
    static constexpr int kStage = NM * MB;
    static constexpr int kSmem = ST * kStage;
    static constexpr int kScalePiece = RS / 8;                               // a scale tile line's bytes for RS rows
    static_assert(RS % 16 == 0 && CH % 64 == 0, "window gemm: tile shape");
    static_assert(F != (int) Format::Nvfp4 || (RS == 32 || RS == 64 || RS == 128), "window gemm: NVFP4 rows per block");
};

// the piece a row's 16-byte piece q lands on in shared memory (bank spread for the reads below)
template <int F, int P> __device__ __forceinline__ int swz(int r) {
    if constexpr (F == (int) Format::Q8) return P >= 8 ? (r & 1) << 2 : 0;
    else if constexpr (F == (int) Format::Bf16) return P >= 16 ? (r & 1) << 3 : 0;
    else if constexpr (P >= 8) return (r & 3) << 1;
    else if constexpr (P == 4) return ((r >> 1) & 1) << 1;
    else return 0;
}

template <int N> __device__ __forceinline__ void cp_async(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    if constexpr (N == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(gmem), "n"(N));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_f16(unsigned x) {
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
}
// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_f16(unsigned w, unsigned sc, unsigned (&h)[4]) {
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 t0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 t1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 t2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 t3, b3;\n"
        "mul.rn.bf16x2 %0, t0, %5;\n"
        "mul.rn.bf16x2 %1, t1, %5;\n"
        "mul.rn.bf16x2 %2, t2, %5;\n"
        "mul.rn.bf16x2 %3, t3, %5;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w), "r"(sc));
}
// two int8 (the low 16 bits of x, the low byte the lower half) as bf16x2, exact
__device__ __forceinline__ unsigned s8x2_bf16(unsigned x) {
    const float f0 = __int_as_float(0x4B000000 | ((x ^ 0x80u) & 0xffu)) - 8388736.0f;
    const float f1 = __int_as_float(0x4B000000 | (((x >> 8) ^ 0x80u) & 0xffu)) - 8388736.0f;
    unsigned h;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(f1), "f"(f0));
    return h;
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    unsigned d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint4 ld_nc16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int F, int RS, int NM, int CH, int ST, int NT, int MW>
__global__ void __launch_bounds__(Cfg<F, RS, NM, CH, ST>::kThreads / MW, 1) wgemm_kernel(const Args a) {
    using C = Cfg<F, RS, NM, CH, ST>;
    constexpr bool kBf16 = F == (int) Format::Bf16, kQ8 = F == (int) Format::Q8;
    constexpr int KB = CH / 64;                       // 64-value blocks per chunk
    constexpr int kWarpsPerM = RS / (16 * MW);        // warps per matrix
    constexpr int kThreads = 32 * NM * kWarpsPerM;
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int last;
    __shared__ float xchg[NM == 2 ? RS * kRows : 1];   // SwiGLU: the up rows for the gate warps
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int ks = a.ks, items = a.tiles * ks, G = gridDim.x;
    const int64_t row_bytes = kQ8 ? a.K : kBf16 ? 2 * a.K : a.K / 2;

    // a work item is (row tile, K slice); a block takes items blockIdx.x, + G, ... and streams their chunks as one
    // sequence through the stages
    auto matrix_of = [&](int tile) {
        int mi = 0;
        if constexpr (NM == 1)
            while (mi + 1 < a.n && tile >= a.tile0[mi + 1]) ++mi;
        return mi;
    };
    auto chunks_of = [&](int item) { return min(a.cps, a.nchunks - (item % ks) * a.cps); };
    auto issue = [&](int item, int i, int stage) {
        const int tile = item / ks, mi = matrix_of(tile);
        const int64_t row0 = (int64_t) (tile - a.tile0[mi]) * RS;
        const int64_t kc = (int64_t) ((item % ks) * a.cps + i);   // chunk index along K
        uint8_t* st = smem + stage * C::kStage;
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            const int mat = NM == 1 ? mi : m;
            const uint8_t* W = a.w[mat];
            const int64_t rows = a.rows[mat];
            uint8_t* sm = st + m * C::MB;
            for (int p = threadIdx.x; p < RS * C::P; p += kThreads) {
                const int r = p / C::P, q = p - r * C::P;
                const int64_t row = min(row0 + r, rows - 1);   // rows past the last read the last (not written)
                cp_async<16>(sm + r * C::CB + ((q ^ swz<F, C::P>(r)) << 4), W + row * row_bytes + kc * C::CB + q * 16);
            }
            if constexpr (kQ8) {   // each row's CH / 32 fp16 scales of the chunk
                const uint8_t* S = a.sp[mat];
                for (int r = threadIdx.x; r < RS; r += kThreads) {
                    const int64_t row = min(row0 + r, rows - 1);
                    cp_async<CH / 16>(sm + RS * C::CB + r * (CH / 16), S + row * (a.K / 16) + kc * (CH / 16));
                }
            }
            if constexpr (F == (int) Format::Nvfp4) {
                const uint8_t* S = a.sp[mat];
                const int64_t tbase = (row0 >> 7) * (a.K / 64);
                const int quart = (int) ((row0 & 127) >> 5) * 4;
                for (int p = threadIdx.x; p < KB * 32; p += kThreads) {
                    const int kb = p >> 5, l = p & 31;
                    cp_async<C::kScalePiece>(sm + RS * C::CB + kb * RS * 4 + l * C::kScalePiece,
                                             S + (tbase + kc * KB + kb) * 512 + l * 16 + quart);
                }
            }
        }
    };
    int iss_item = blockIdx.x, iss_c = 0, iss_q = 0;
    int iss_n = iss_item < items ? chunks_of(iss_item) : 0;
    auto issue_next = [&]() {
        if (iss_item < items) {
            issue(iss_item, iss_c, iss_q % ST);
            ++iss_q;
            if (++iss_c == iss_n) {
                iss_c = 0;
                iss_item += G;
                if (iss_item < items) iss_n = chunks_of(iss_item);
            }
        }
        cp_commit();
    };

#pragma unroll
    for (int i = 0; i < ST - 1; ++i) issue_next();
    pdl_wait();   // the activations and the workspace come from the kernels before this one
    // the next kernel may take SM room as blocks finish and stage its weights meanwhile; only after the wait, since a
    // dependent may read what the kernel before this one wrote before its own wait
    pdl_launch_dependents();

    const int wm = NM == 1 ? 0 : warp / kWarpsPerM;   // this warp's matrix slot in the stage
    const int rw = warp % kWarpsPerM;
    bool live[NT];
    const uint16_t* xcol[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        live[t] = t * 8 + g < a.T;
        xcol[t] = a.act + (int64_t) (t * 8 + g) * a.K + 16 * c;
    }
    int q = 0;   // chunks consumed
    for (int item = blockIdx.x; item < items; item += G) {
        const int tile = item / ks, slice = item % ks, mi = matrix_of(tile);
        const int64_t row0 = (int64_t) (tile - a.tile0[mi]) * RS;
        const int n = chunks_of(item);
        const int64_t k0 = (int64_t) slice * a.cps * CH;
        float acc[MW][NT][4];
#pragma unroll
        for (int mt = 0; mt < MW; ++mt)
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[mt][t][0] = acc[mt][t][1] = acc[mt][t][2] = acc[mt][t][3] = 0.0f;

        // the activations of chunk i + 1 load while chunk i computes (not at NT 4: no registers to spare)
        constexpr bool kAhead = NT <= 2;
        uint4 na[KB][NT], nb[KB][NT];
        auto load_x = [&](int i) {
#pragma unroll
            for (int b = 0; b < KB; ++b)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const uint16_t* xp = xcol[t] + k0 + i * CH + b * 64;
                    na[b][t] = live[t] ? ld_nc16(xp) : make_uint4(0, 0, 0, 0);
                    nb[b][t] = live[t] ? ld_nc16(xp + 8) : make_uint4(0, 0, 0, 0);
                }
        };
        if constexpr (kAhead) load_x(0);
        for (int i = 0; i < n; ++i, ++q) {
            issue_next();
            if constexpr (!kAhead) load_x(i);
            uint4 xa[KB][NT], xb[KB][NT];
#pragma unroll
            for (int b = 0; b < KB; ++b)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    xa[b][t] = na[b][t];
                    xb[b][t] = nb[b][t];
                }
            if (kAhead && i + 1 < n) load_x(i + 1);
            cp_wait<ST - 1>();
            __syncthreads();
            const uint8_t* sm = smem + (q % ST) * C::kStage + wm * C::MB;
#pragma unroll
            for (int mt = 0; mt < MW; ++mt) {
                const int r0l = (rw * MW + mt) * 16 + g, r1l = r0l + 8;
                const uint8_t* q0 = sm + r0l * C::CB;
                const uint8_t* q1 = sm + r1l * C::CB;
#pragma unroll
                for (int b = 0; b < KB; ++b) {
                    unsigned a0[4], a1[4], a2[4], a3[4];   // step s: rows g, g + 8 at k-slots 2c.., then 2c + 8..
                    if constexpr (kBf16) {   // values 16c .. 16c + 15 as eight bf16 pairs, two 16-byte pieces
                        const int pc = 8 * b + 2 * c;
                        const uint4 p0 = *(const uint4*) (q0 + ((pc ^ swz<F, C::P>(r0l)) << 4));
                        const uint4 p1 = *(const uint4*) (q0 + (((pc + 1) ^ swz<F, C::P>(r0l)) << 4));
                        const uint4 p2 = *(const uint4*) (q1 + ((pc ^ swz<F, C::P>(r1l)) << 4));
                        const uint4 p3 = *(const uint4*) (q1 + (((pc + 1) ^ swz<F, C::P>(r1l)) << 4));
                        const unsigned w0[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
                        const unsigned w1[8] = {p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w};
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            a0[s] = w0[2 * s];
                            a1[s] = w1[2 * s];
                            a2[s] = w0[2 * s + 1];
                            a3[s] = w1[2 * s + 1];
                        }
                    } else if constexpr (kQ8) {   // values 16c .. 16c + 15: half of 32-value block 2b + c / 2
                        const int pc = 4 * b + c;
                        const uint4 w0 = *(const uint4*) (q0 + ((pc ^ swz<F, C::P>(r0l)) << 4));
                        const uint4 w1 = *(const uint4*) (q1 + ((pc ^ swz<F, C::P>(r1l)) << 4));
                        const __half* ss = (const __half*) (sm + RS * C::CB);
                        const int jb = 2 * b + (c >> 1);
                        const unsigned short s0 = __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(ss[r0l * (CH / 32) + jb])));
                        const unsigned short s1 = __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(ss[r1l * (CH / 32) + jb])));
                        const unsigned sc0 = s0 | ((unsigned) s0 << 16), sc1 = s1 | ((unsigned) s1 << 16);
                        const unsigned u0[4] = {w0.x, w0.y, w0.z, w0.w}, u1[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            a0[s] = bf16x2_mul(s8x2_bf16(u0[s]), sc0);
                            a1[s] = bf16x2_mul(s8x2_bf16(u1[s]), sc1);
                            a2[s] = bf16x2_mul(s8x2_bf16(u0[s] >> 16), sc0);
                            a3[s] = bf16x2_mul(s8x2_bf16(u1[s] >> 16), sc1);
                        }
                    } else {
                        const int pc = 2 * b + (c >> 1), half = (c & 1) * 8;
                        const uint2 w0 = *(const uint2*) (q0 + ((pc ^ swz<F, C::P>(r0l)) << 4) + half);
                        const uint2 w1 = *(const uint2*) (q1 + ((pc ^ swz<F, C::P>(r1l)) << 4) + half);
                        const uint8_t* ss = sm + RS * C::CB + b * RS * 4;
                        const unsigned s0 = ss[(r0l & 31) * C::kScalePiece + (r0l >> 5) * 4 + c];
                        const unsigned s1 = ss[(r1l & 31) * C::kScalePiece + (r1l >> 5) * 4 + c];
                        const unsigned sc0 = e4m3x2_f16(s0 | (s0 << 8)), sc1 = e4m3x2_f16(s1 | (s1 << 8));
                        unsigned h0[8], h1[8];
                        e2m1x8_f16(w0.x, sc0, *reinterpret_cast<unsigned(*)[4]>(h0));
                        e2m1x8_f16(w0.y, sc0, *reinterpret_cast<unsigned(*)[4]>(h0 + 4));
                        e2m1x8_f16(w1.x, sc1, *reinterpret_cast<unsigned(*)[4]>(h1));
                        e2m1x8_f16(w1.y, sc1, *reinterpret_cast<unsigned(*)[4]>(h1 + 4));
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            a0[s] = h0[2 * s];
                            a1[s] = h1[2 * s];
                            a2[s] = h0[2 * s + 1];
                            a3[s] = h1[2 * s + 1];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const unsigned xw[8] = {xa[b][t].x, xa[b][t].y, xa[b][t].z, xa[b][t].w,
                                                xb[b][t].x, xb[b][t].y, xb[b][t].z, xb[b][t].w};
#pragma unroll
                        for (int s = 0; s < 4; ++s) mma(acc[mt][t], a0[s], a1[s], a2[s], a3[s], xw[2 * s], xw[2 * s + 1]);
                    }
                }
            }
            __syncthreads();
        }
        // (dependents were started at the top: see the trigger before the wait)

        // ---- the item's epilogue: its slice's sums to the workspace, the last slice of the tile adds them in order
        const int mat = NM == 1 ? mi : wm;
        const int64_t rows = a.rows[mat];
        bool final_block = true;
        if (ks > 1) {
            float* part = a.partial + (int64_t) slice * kRows * a.part_rows + a.row_off[mat];
#pragma unroll
            for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int col = t * 8 + 2 * c + (j & 1);
                        const int64_t r = row0 + (rw * MW + mt) * 16 + g + 8 * (j >> 1);
                        if (col < a.T && r < rows) part[col * a.part_rows + r] = acc[mt][t][j];
                    }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(a.tickets + tile, 1) == ks - 1;
            __syncthreads();
            final_block = last;
            if (final_block) {
                __threadfence();
                const float* base = a.partial + a.row_off[mat];
                // slice by slice, every element's load of a slice issued together (each element still sums its
                // slices in order from 0)
#pragma unroll
                for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
#pragma unroll
                        for (int j = 0; j < 4; ++j) acc[mt][t][j] = 0.0f;
                for (int k = 0; k < ks; ++k) {
                    const float* bk = base + (int64_t) k * kRows * a.part_rows;
#pragma unroll
                    for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                        for (int t = 0; t < NT; ++t)
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int col = t * 8 + 2 * c + (j & 1);
                                const int64_t r = row0 + (rw * MW + mt) * 16 + g + 8 * (j >> 1);
                                if (col < a.T && r < rows) acc[mt][t][j] += __ldcg(bk + (int64_t) col * a.part_rows + r);
                            }
                }
                if (threadIdx.x == 0) a.tickets[tile] = 0;
            }
        }
        if (!final_block) continue;
        if constexpr (NM == 2) {
            // SwiGLU: the up warps hand their rows to the gate warps
            const float su = a.scale[1] ? *a.scale[1] : 1.0f, sg = a.scale[0] ? *a.scale[0] : 1.0f;
            if (wm == 1)
#pragma unroll
                for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int col = t * 8 + 2 * c + (j & 1);
                            const int rl = (rw * MW + mt) * 16 + g + 8 * (j >> 1);
                            if (col < a.T) xchg[rl * kRows + col] = acc[mt][t][j] * su;
                        }
            __syncthreads();
            if (wm == 0)
#pragma unroll
                for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int col = t * 8 + 2 * c + (j & 1);
                            const int rl = (rw * MW + mt) * 16 + g + 8 * (j >> 1);
                            if (col >= a.T || row0 + rl >= rows) continue;
                            const float gv = acc[mt][t][j] * sg, uv = xchg[rl * kRows + col];
                            const float hv = gv / (1.0f + expf(-gv)) * uv;
                            if (a.y[0]) a.y[0][col * a.ld_y[0] + row0 + rl] = hv;
                            if (a.act_out) a.act_out[col * rows + row0 + rl] = __bfloat16_as_ushort(__float2bfloat16_rn(hv));
                        }
            __syncthreads();   // xchg is free for the next item
        } else {
            const float sc = a.scale[mat] ? *a.scale[mat] : 1.0f;
            float* y = a.y[mat];
            const int64_t ld = a.ld_y[mat];
#pragma unroll
            for (int mt = 0; mt < MW; ++mt)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int col = t * 8 + 2 * c + (j & 1);
                        const int64_t r = row0 + (rw * MW + mt) * 16 + g + 8 * (j >> 1);
                        if (col >= a.T || r >= rows) continue;
                        const float v = acc[mt][t][j] * sc;
                        if (a.epi == (int) Epilogue::AddBf16) {
                            uint16_t* d16 = a.y16[mat] + col * ld + r;
                            // the product rounded as Store leaves it (no fma into the add), then one add, one rounding
                            *d16 = __bfloat16_as_ushort(__float2bfloat16_rn(
                                __fadd_rn(__bfloat162float(__ushort_as_bfloat16(*d16)), __fmul_rn(acc[mt][t][j], sc))));
                            continue;
                        }
                        float* dst = y + col * ld + r;
                        if (a.epi == (int) Epilogue::Add) *dst += v;
                        else *dst = v;
                    }
        }
    }
    cp_wait<0>();
}

/// A kernel instantiation as the host dispatch sees it.
struct Kernel {
    void (*fn)(Args) = nullptr;
    int threads = 0, smem = 0, rs = 0, ch = 0, nm = 1;
};

}  // namespace basalt::kernels::wgemm::detail
