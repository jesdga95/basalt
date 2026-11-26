#pragma once
// src/kernels/cuda/expert_tile.cuh - grouped expert products (NVFP4 / MXFP8 weights) on tensor cores, the weights as the
// MMA's M side. Device code only: a model's translation unit instantiates the kernels with its own finish (see
// down_kernel) and must target the a-variant of the architecture (the E2M1 codes convert with cvt.rn.f16x2.e2m1x2).
//
// The tile design is adapted from ingine-batch's batch/moe.cu (Apache-2.0): few entries an expert, so a work item is ROWS
// weight rows (warps of 16) x one tile of up to TT entries of one expert over the whole K; each weight is decoded once a
// tile, straight from the staged codes into A fragments in registers; K streams in chunks of 128 values through NS
// shared-memory stages (cp.async, weights read once: evict-first), two or more blocks an SM.
//
// Numerics. A weight decodes exactly into the MMA's A type: NVFP4 as E2M1 times its UE4M3 sub-block scale in f16 (at
// most 5 significant bits, |w| in [2^-10, 2688]); MXFP8 as E4M3 times its UE8M0 block scale in bf16. The activations
// come as rows of f16 (NVFP4) or bf16 (MXFP8) values in the pair layout (xh_pos). An output is the sum of four fp32 MMA
// chains over K, chain m taking the m-th k16 product of every 64 values in K order, added as (c0 + c1) + (c2 + c3), then
// times the expert's FP32 scale (NVFP4). Nothing in it depends on the tile's shape, its other entries or the call's
// size, so an entry gives the same bits alone and batched, at every tile shape.
//
// K order. Thread c (lane & 3) of a quad owns values 16 c .. 16 c + 15 of every 64 (an NVFP4 sub-block); its pair j
// (j = 0..7) is values (16 c + j, 16 c + j + 8): the two nibbles of NVFP4 code byte j, and the activation row's word j
// of the group (xh_pos). k16 product m takes pairs 2 m and 2 m + 1 of every thread.

#include "basalt/kernels/expert_rows.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace basalt::kernels::xt {

constexpr int kNvfp4 = 40, kMxfp8 = 46;
constexpr int kChunk = 128;          ///< k a pipeline stage carries
constexpr int kTicketCh = 64;        ///< channels one finish ticket counts
constexpr int kMaxGroups = 256;      ///< groups (experts) a call has at most

/// The tile shape of a call: work items of ROWS weight rows x TT entries, NS stages.
template<int TY, int K, int TT_, int ROWS_, int NS_>
struct Cfg {
    static constexpr int TT = TT_, ROWS = ROWS_, NS = NS_, NT = 2 * ROWS_;
    static constexpr int ROWB = TY == kNvfp4 ? K / 64 * 36 : K / 128 * 132;   ///< bytes of a weight row
    static constexpr int CB = TY == kNvfp4 ? 72 : 132;                         ///< bytes of a row's chunk
    static constexpr int WSZ = TY == kNvfp4 ? 72 : 136;  ///< bytes staged a row a chunk: 8-byte pieces from the 8-aligned byte at or before it
    static constexpr int WP = WSZ / 8;
    static constexpr int KC = K / kChunk;
    static constexpr int AST = 2 * kChunk + 16;          ///< a staged activation chunk's stride (16 more: no bank conflicts)
    static constexpr int W_SLOT = ROWS * WSZ, A_SLOT = TT * AST, SLOT = W_SLOT + A_SLOT;
    static constexpr int SMEM = NS * SLOT;
    static_assert(K % kChunk == 0 && ROWS % 16 == 0 && TT % 8 == 0 && W_SLOT % 16 == 0, "tile shape");
    static_assert(SMEM <= 100 * 1024, "stages fit an SM's shared memory");
};

// ---- device helpers -------------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }

__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
// (the 8-byte copy takes no L2 cache hint: the hinted form traps on sm_120)
__device__ __forceinline__ void cp_async8(void* dst, const void* src, uint64_t pol) {
    (void) pol;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template<bool BF>
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr (BF)
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// the four bytes of a word as four f16 pairs of E2M1 codes, exactly: byte j (first code in its low nibble) as d[j] (the
// conversions read the bytes in place, no shifts)
__device__ __forceinline__ void e2m1x8_f16x2(uint32_t w, uint32_t (&d)[4]) {
    asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0; cvt.rn.f16x2.e2m1x2 %1, b1; cvt.rn.f16x2.e2m1x2 %2, b2; cvt.rn.f16x2.e2m1x2 %3, b3; }"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(w));
}
// the two halves of a word as two f16 pairs of E4M3 codes, exactly (read in place)
__device__ __forceinline__ void e4m3x4_f16x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
    asm("{ .reg .b16 h0, h1; mov.b32 {h0, h1}, %2; cvt.rn.f16x2.e4m3x2 %0, h0; cvt.rn.f16x2.e4m3x2 %1, h1; }"
        : "=r"(lo), "=r"(hi) : "r"(w));
}
// NVFP4's UE4M3 sub-block scale, whole, in both halves of an f16 pair (bit 7 ignored, 0x7F as 0): the value the other
// NVFP4 kernels use (ggml halves it and doubles the codes)
// Branch free: the byte converts in both halves, its magnitude kept (bit 7 is E4M3's sign), and 0x7F's NaN becomes 0 in
// the max with 0 (max takes the other operand of a NaN).
__device__ __forceinline__ uint32_t ue4m3_f16x2(uint32_t x) {
    uint32_t d;
    asm("{ .reg .b32 r, z; .reg .b16 t; prmt.b32 r, %1, 0, 0x4400; cvt.u16.u32 t, r; cvt.rn.f16x2.e4m3x2 %0, t;"
        " and.b32 %0, %0, 0x7FFF7FFF; mov.b32 z, 0; max.f16x2 %0, %0, z; }" : "=r"(d) : "r"(x));
    return d;
}
// MXFP8's UE8M0 block scale (0xFF as 0, 0 as 2^-127)
__device__ __forceinline__ float ue8m0_f32(uint32_t u) {
    return u == 0xFFu ? 0.0f : u == 0u ? __uint_as_float(0x00400000u) : __uint_as_float(u << 23);
}
__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
    return *reinterpret_cast<const uint32_t*>(&r);
}
// an f16 pair (exact E4M3 values) times a power of two, as a bf16 pair (exact)
__device__ __forceinline__ uint32_t f16x2_scaled_bf16x2(uint32_t h, float s) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * s, f.y * s);
    return *reinterpret_cast<const uint32_t*>(&b);
}

/// A thread's 8 pairs of one staged weight row for 64-value step h of the chunk: pair j as a register (f16 NVFP4, bf16
/// MXFP8). `row` is the staged row (WSZ bytes), `o` the chunk's offset in it (MXFP8: 0 or 4).
template<int TY>
__device__ __forceinline__ void decode_row(const uint8_t* row, int o, int h, int c, uint32_t (&p)[8]) {
    if constexpr (TY == kNvfp4) {   // block h of the chunk: 4 scale bytes, then 4 sub-blocks of 8 code bytes
        const uint8_t* blk = row + h * 36;
        const uint32_t s2 = ue4m3_f16x2(blk[c]);
        const uint32_t u0 = *reinterpret_cast<const uint32_t*>(blk + 4 + 8 * c);
        const uint32_t u1 = *reinterpret_cast<const uint32_t*>(blk + 8 + 8 * c);
        uint32_t d0[4], d1[4];
        e2m1x8_f16x2(u0, d0);
        e2m1x8_f16x2(u1, d1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            p[j] = hmul2_u(d0[j], s2);
            p[4 + j] = hmul2_u(d1[j], s2);
        }
    } else {   // the chunk is one block: 4 scale bytes, then 128 codes; this thread's 16 in 32-value block 2 h + c / 2
        const uint8_t* blk = row + o;
        const float s = ue8m0_f32(blk[2 * h + (c >> 1)]);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 4 + 64 * h + 16 * c);
        const uint32_t w0 = q[0], w1 = q[1], w2 = q[2], w3 = q[3];
        const uint32_t x01 = __byte_perm(w0, w2, 0x5140), x23 = __byte_perm(w0, w2, 0x7362);
        const uint32_t x45 = __byte_perm(w1, w3, 0x5140), x67 = __byte_perm(w1, w3, 0x7362);
        uint32_t h[8];
        e4m3x4_f16x2(x01, h[0], h[1]);
        e4m3x4_f16x2(x23, h[2], h[3]);
        e4m3x4_f16x2(x45, h[4], h[5]);
        e4m3x4_f16x2(x67, h[6], h[7]);
#pragma unroll
        for (int j = 0; j < 8; ++j) p[j] = f16x2_scaled_bf16x2(h[j], s);
    }
}

/// The NVFP4 expert's FP32 scales {gate, up, down} at the blob's end; MXFP8: 1.
template<int TY>
__device__ __forceinline__ float expert_scale(const void* blob, const NativeExpertLayout& L, int m) {
    if constexpr (TY == kNvfp4) return reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(blob) + L.bytes - 16)[m];
    else return 1.0f;
}

/// The layout's SwiGLU (softcapped, limited or plain), the other expert kernels' arithmetic.
__device__ __forceinline__ float swiglu(const NativeExpertLayout& L, float g, float u) {
    if (L.gate_softcap > 0.0f) {
        const float cg = L.gate_softcap, cu = L.up_softcap;
        return __fmul_rn(__fmul_rn(cg * tanhf(g / cg), 1.0f / (1.0f + expf(-g))), cu * tanhf(u / cu));
    }
    if (L.swiglu_limit > 0.0f) {
        const float gl = fminf(g, L.swiglu_limit);
        const float ul = fmaxf(fminf(u, L.swiglu_limit), -L.swiglu_limit);
        return __fmul_rn(gl / (1.0f + expf(-gl)), ul);
    }
    return __fmul_rn(g / (1.0f + expf(-g)), u);
}

/// The call's plan (one group an expert on the card, its entries in row order): ptr[g] the blob, start[g]..start[g + 1]
/// its entries, dst[e] the output row of entry e (row * k + slot), tok[e] its row.
struct Plan {
    const unsigned long long* ptr = nullptr;
    const int32_t* start = nullptr;
    const int32_t* n_groups = nullptr;
    const int32_t* dst = nullptr;
    const int32_t* tok = nullptr;
};

/// Work item `item` of a grid of tiles x nblk: its group, first entry and entries (group -1: past the call's tiles).
template<int TT>
__device__ __forceinline__ void find_tile(const Plan& P, int tile, int* s_t) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) s_t[0] = -1;
    __syncwarp();
    const int ng = *P.n_groups;
    int before = 0;
    for (int g0 = 0; g0 < ng; g0 += 32) {
        const int gi = g0 + lane;
        const int e0 = gi < ng ? P.start[gi] : 0;
        const int cnt = gi < ng ? P.start[gi + 1] - e0 : 0;
        const int nt = (cnt + TT - 1) / TT;
        int inc = nt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += v;
        }
        const int ex = before + inc - nt;
        if (tile >= ex && tile < ex + nt) {
            const int u = tile - ex;
            s_t[0] = gi;
            s_t[1] = e0 + u * TT;
            s_t[2] = min(TT, cnt - u * TT);
        }
        before += __shfl_sync(0xffffffffu, inc, 31);
    }
}

/// The work of one item: the staged chunks' products, kept in acc. GU: smem rows [0, ROWS / 2) are gate rows i0.., the
/// rest the same up rows; the thread's A rows g and g + 8 are gate and up row i0 + 8 w + g. Down: smem row r is output
/// channel n0 + r; A rows g and g + 8 are channels n0 + 16 w + 2 g and the next.
template<int TY, int K, bool GU, class C, int NB>
__device__ __forceinline__ void item_loop(uint8_t* smem, const uint8_t* const* rowp, int n, const int* s_arow,
                                          const uint16_t* __restrict__ X, float (&acc)[NB][4][4], uint64_t pol,
                                          bool pre_issued) {
    constexpr bool BF = TY == kMxfp8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
    const int ra = GU ? 8 * w + g : 16 * w + 2 * g;
    const int rb = GU ? C::ROWS / 2 + 8 * w + g : 16 * w + 2 * g + 1;
    // a thread's copies of every chunk, worked out once: weight piece id = tid + NT k (row id / WP, piece id % WP) at its
    // row's offset from the first row's 8-aligned start, activation piece id = tid + NT k (entry id / 16, piece id % 16)
    constexpr int WS = (C::ROWS * C::WP + C::NT - 1) / C::NT, AS = (C::TT * 16 + C::NT - 1) / C::NT;
    const uint8_t* const wbase =
        reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(rowp[0]) & ~(uintptr_t) 7);
    uint32_t woff[WS], wdst[WS];
#pragma unroll
    for (int k = 0; k < WS; ++k) {
        const int id = tid + C::NT * k, r = id / C::WP, p = id - r * C::WP;
        woff[k] = id < C::ROWS * C::WP ? (uint32_t) (rowp[r] - wbase) : 0u;
        wdst[k] = (uint32_t) (r * C::WSZ + 8 * p) | (uint32_t) (8 * p) << 16;
    }
    uint32_t aoff[AS];
#pragma unroll
    for (int k = 0; k < AS; ++k) {
        const int id = tid + C::NT * k, t = id >> 4, p = id & 15;
        aoff[k] = id < C::TT * 16 && t < n ? (uint32_t) (s_arow[t] * K + 8 * p) : ~0u;
    }
    const auto load_w = [&](int q, int s) {
        uint8_t* ws = smem + s * C::SLOT;
#pragma unroll
        for (int k = 0; k < WS; ++k) {
            if (C::ROWS * C::WP % C::NT == 0 || tid + C::NT * k < C::ROWS * C::WP)
                cp_async8(ws + (wdst[k] & 0xffffu), wbase + ((woff[k] + (uint32_t) (q * C::CB)) & ~7u) + (wdst[k] >> 16), pol);
        }
    };
    const auto load_a = [&](int q, int s) {
        uint8_t* as = smem + s * C::SLOT + C::W_SLOT;
#pragma unroll
        for (int k = 0; k < AS; ++k) {
            const int id = tid + C::NT * k;
            if (aoff[k] != ~0u) cp_async16(as + (id >> 4) * C::AST + 16 * (id & 15), X + aoff[k] + q * kChunk);
        }
    };
    // the first NS - 1 chunks: the weights may have been issued before the caller's wait (one commit group with chunk 0's
    // activations then)
#pragma unroll
    for (int q = 0; q < C::NS - 1; ++q) {
        if (!pre_issued) load_w(q, q);
        load_a(q, q);
        cp_commit();
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int m = 0; m < 4; ++m) acc[nb][m][0] = acc[nb][m][1] = acc[nb][m][2] = acc[nb][m][3] = 0.0f;
    const int oa = TY == kMxfp8 ? (int) (reinterpret_cast<uintptr_t>(rowp[ra]) & 7) : 0;
    const int ob = TY == kMxfp8 ? (int) (reinterpret_cast<uintptr_t>(rowp[rb]) & 7) : 0;
#pragma unroll 1
    for (int q = 0; q < C::KC; ++q) {
        cp_wait<C::NS - 2>();
        __syncthreads();   // chunk q is in, and every thread is done with the slot chunk q + NS - 1 takes
        if (q + C::NS - 1 < C::KC) {
            load_w(q + C::NS - 1, (q + C::NS - 1) % C::NS);
            load_a(q + C::NS - 1, (q + C::NS - 1) % C::NS);
        }
        cp_commit();
        const uint8_t* W = smem + (q % C::NS) * C::SLOT;
        const uint8_t* A = W + C::W_SLOT;
        // MXFP8: the chunk's offset in a staged row (the row's start plus q blocks of 132 bytes, mod 8)
        const int qa = TY == kMxfp8 ? ((oa + 4 * q) & 7) : 0, qb = TY == kMxfp8 ? ((ob + 4 * q) & 7) : 0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t pa[8], pb[8];
            decode_row<TY>(W + ra * C::WSZ, qa, h, c, pa);
            decode_row<TY>(W + rb * C::WSZ, qb, h, c, pb);
            uint4 b[NB][2];
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const uint8_t* ar = A + (nb * 8 + g) * C::AST + 128 * h + 32 * c;
                b[nb][0] = *reinterpret_cast<const uint4*>(ar);
                b[nb][1] = *reinterpret_cast<const uint4*>(ar + 16);
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t a[4] = {pa[2 * m], pb[2 * m], pa[2 * m + 1], pb[2 * m + 1]};
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    const uint4& bb = b[nb][m >> 1];
                    if (m & 1) mma16816<BF>(acc[nb][m], a, bb.z, bb.w);
                    else mma16816<BF>(acc[nb][m], a, bb.x, bb.y);
                }
            }
        }
    }
    cp_wait<0>();
}

__device__ __forceinline__ float chains(const float (&a)[4][4], int i) {
    return __fadd_rn(__fadd_rn(a[0][i], a[1][i]), __fadd_rn(a[2][i], a[3][i]));
}

/// Gate/up with the SwiGLU: work item blockIdx.x = tile * (2 n_ff / ROWS) + row block; writes h [entry][n_ff] in the
/// down's activation layout (f16 for an NVFP4 down, bf16 for MXFP8). X: the rows of the entries' tokens [row][n_embd].
template<int TY, int K, class C>
__global__ void __launch_bounds__(C::NT, 2) gate_up_kernel(Plan P, NativeExpertLayout L, const uint16_t* __restrict__ X,
                                                           uint16_t* __restrict__ H) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int s_t[3];
    __shared__ int s_arow[C::TT];
    __shared__ const uint8_t* s_rowp[C::ROWS];
    const int nblk = 2 * (int) L.n_ff / C::ROWS;
    const int tile = (int) blockIdx.x / nblk, rblk = (int) blockIdx.x - tile * nblk;
    pdl_wait();   // the plan
    pdl_launch_dependents();
    if (threadIdx.x < 32) find_tile<C::TT>(P, tile, s_t);
    __syncthreads();
    const int gi = s_t[0];
    if (gi < 0) return;
    const int e0 = s_t[1], n = s_t[2], i0 = rblk * C::ROWS / 2;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(P.ptr[gi]);
    for (int r = threadIdx.x; r < C::ROWS; r += C::NT) {
        const bool up = r >= C::ROWS / 2;
        s_rowp[r] = blob + (up ? L.up_off : 0) + (size_t) (i0 + (up ? r - C::ROWS / 2 : r)) * C::ROWB;
    }
    if (threadIdx.x < C::TT) s_arow[threadIdx.x] = (int) threadIdx.x < n ? P.tok[e0 + threadIdx.x] : 0;
    __syncthreads();
    const uint64_t pol = evict_first_policy();
    const float sg = expert_scale<TY>(blob, L, 0), su = expert_scale<TY>(blob, L, 1);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, c = lane & 3;
    const int i = i0 + 8 * w + g, pos = xh_pos(i);
    const bool hbf = L.d_type == kMxfp8;
    const auto run = [&](auto nbc) {
        constexpr int NB = decltype(nbc)::value;
        float acc[NB][4][4];
        item_loop<TY, K, true, C, NB>(smem, s_rowp, n, s_arow, X, acc, pol, false);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int t = nb * 8 + 2 * c + p;
                if (t >= n) continue;
                const float hv = swiglu(L, __fmul_rn(chains(acc[nb], p), sg), __fmul_rn(chains(acc[nb], 2 + p), su));
                const size_t at = (size_t) (e0 + t) * L.n_ff + pos;
                if (hbf) H[at] = __bfloat16_as_ushort(__float2bfloat16_rn(hv));
                else H[at] = __half_as_ushort(__float2half_rn(hv));
            }
    };
    // the n8 blocks as a template constant (a runtime bound leaves predicated MMAs in the pipe)
    switch ((n + 7) >> 3) {
        case 1: run(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (C::TT >= 16) run(std::integral_constant<int, 2>{}); break;
        case 3: if constexpr (C::TT >= 24) run(std::integral_constant<int, 3>{}); break;
        default: if constexpr (C::TT >= 32) run(std::integral_constant<int, 4>{}); break;
    }
#else
    (void) P; (void) L; (void) X; (void) H;
    __trap();
#endif
}

/// What down_kernel's finish reads beside the plan. F provides, on the device:
///   int seg(int row) const;              the segment (rows finished together) of a row
///   int target(int seg) const;           the pairs a segment's ticket counts: its rows x k
///   void finish(int seg, int c0, int tid, int nt) const;   its channels c0 .. c0 + kTicketCh - 1, every pair's row in
///   int* tickets;                        [segments][n_embd / kTicketCh], zero between calls
/// A ticket counts a segment's pairs done for kTicketCh channels; the block that completes it runs the finish, reading
/// the other blocks' rows (fenced before their counts), and zeroes the ticket. The sums are plain stores read back in a
/// fixed order, never atomics.
template<class F>
__device__ __forceinline__ void run_finishes(const F& f, const int* s_fin, int nfin, int tid, int nt) {
    __syncthreads();
    for (int i = 0; i < nfin; ++i) f.finish(s_fin[i] >> 8, (s_fin[i] & 255) * kTicketCh, tid, nt);
}

/// Down: work item blockIdx.x = tile * (n_embd / ROWS) + row block writes rows dst[e] [n_embd] (fp32) of its entries;
/// with `fin`, then counts them on its segments' tickets and runs the finishes it completes. A programmatic dependent of
/// gate/up: the plan and the first weights before its wait.
template<int TY, int K, class C, class F>
__global__ void __launch_bounds__(C::NT, 2) down_kernel(Plan P, NativeExpertLayout L, const uint16_t* __restrict__ H,
                                                        float* __restrict__ out, F f, bool fin) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int s_t[3];
    __shared__ int s_arow[C::TT];
    __shared__ const uint8_t* s_rowp[C::ROWS];
    __shared__ int s_fin[2 * C::TT];   // an item's runs x its tickets
    __shared__ int s_nfin;
    const int ncb = (int) L.n_embd / kTicketCh;
    const int tid = threadIdx.x;
    const int nblk = (int) L.n_embd / C::ROWS;
    const int tile = (int) blockIdx.x / nblk, rblk = (int) blockIdx.x - tile * nblk;
    if (tid < 32) find_tile<C::TT>(P, tile, s_t);   // the plan came before gate/up
    __syncthreads();
    const int gi = s_t[0];
    if (gi < 0) return;
    const int e0 = s_t[1], n = s_t[2], n0 = rblk * C::ROWS;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(P.ptr[gi]);
    for (int r = tid; r < C::ROWS; r += C::NT) s_rowp[r] = blob + L.down_off + (size_t) (n0 + r) * C::ROWB;
    if (tid < C::TT) s_arow[tid] = e0 + tid;
    __syncthreads();
    const uint64_t pol = evict_first_policy();
    // the first chunks' weights before the wait (constant data)
    for (int q = 0; q < C::NS - 1; ++q) {
        uint8_t* ws = smem + q * C::SLOT;
        for (int id = tid; id < C::ROWS * C::WP; id += C::NT) {
            const int r = id / C::WP, p = id - r * C::WP;
            const uint8_t* a = s_rowp[r] + q * C::CB;
            const uint8_t* a8 = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(a) & ~(uintptr_t) 7);
            cp_async8(ws + r * C::WSZ + 8 * p, a8 + 8 * p, pol);
        }
    }
    pdl_wait();   // gate/up's h
    pdl_launch_dependents();
    const float sd = expert_scale<TY>(blob, L, 2);
    const int lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
    const int ch = n0 + 16 * w + 2 * g;
    const auto run = [&](auto nbc) {
        constexpr int NB = decltype(nbc)::value;
        float acc[NB][4][4];
        item_loop<TY, K, false, C, NB>(smem, s_rowp, n, s_arow, H, acc, pol, true);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int t = nb * 8 + 2 * c + p;
                if (t >= n) continue;
                const float2 v = make_float2(__fmul_rn(chains(acc[nb], p), sd), __fmul_rn(chains(acc[nb], 2 + p), sd));
                *reinterpret_cast<float2*>(out + (size_t) P.dst[e0 + t] * L.n_embd + ch) = v;
            }
    };
    switch ((n + 7) >> 3) {
        case 1: run(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (C::TT >= 16) run(std::integral_constant<int, 2>{}); break;
        case 3: if constexpr (C::TT >= 24) run(std::integral_constant<int, 3>{}); break;
        default: if constexpr (C::TT >= 32) run(std::integral_constant<int, 4>{}); break;
    }
    if (!fin) return;
    // the tickets, a lane an entry of warp 0 (TT <= 32): the item's entries are in row order, so its segments come as
    // runs; each run's first lane counts it
    static_assert(C::TT <= 32, "an item's entries fit a warp");
    __threadfence();
    __syncthreads();
    if (tid < 32) {
        const int lane = tid;
        const int sg = lane < n ? f.seg(P.tok[e0 + lane]) : -1;
        const int before = __shfl_up_sync(0xffffffffu, sg, 1);
        const bool head = lane < n && (lane == 0 || before != sg);
        const unsigned heads = __ballot_sync(0xffffffffu, head);
        if (lane == 0) s_nfin = 0;
        __syncwarp();
        if (head) {
            const unsigned later = heads & ~((2u << lane) - 1u);
            const int m = (later ? __ffs((int) later) - 1 : n) - lane;
            const int tgt = f.target(sg);
            for (int cb = n0 / kTicketCh; cb < (n0 + C::ROWS) / kTicketCh; ++cb) {
                const int old = atomicAdd(&f.tickets[sg * ncb + cb], m);
                if (old + m == tgt) {
                    f.tickets[sg * ncb + cb] = 0;
                    s_fin[atomicAdd(&s_nfin, 1)] = sg << 8 | cb;
                }
            }
        }
        __syncwarp();
        if (lane == 0 && s_nfin) __threadfence();
    }
    __syncthreads();
    const int nf = s_nfin;
    if (nf) run_finishes(f, s_fin, nf, tid, C::NT);
#else
    (void) P; (void) L; (void) H; (void) out; (void) f; (void) fin;
    __trap();
#endif
}

/// Tiles a call of `pairs` entries over at most `groups` groups can have.
constexpr int max_tiles(int pairs, int groups, int tt) { return (pairs + tt - 1) / tt + (pairs < groups ? pairs : groups); }

}  // namespace basalt::kernels::xt
