// src/kernels/cuda/gu2.cuh - DEC_GU2=1: the grouped gate/up rows of the grid formats (IQ1_S, IQ2_XXS, IQ2_XS,
// IQ2_S, IQ3_XXS, IQ3_S), Q2_0 and IQ4_XS with n_embd 2560 or 5120, two rows a warp.  Included by iq_kernels.cu (its
// Split / Fmt / grid helpers).
//
// Bitwise native_gu_multi_kernel's: each lane sums the calls k = lane, lane + 32, ... of a row in that order (fused
// multiply-adds of the block scale product and the integer sum, as the contracted `s += apply`), then the same
// warp_sum.  A warp takes two rows, so each activation load serves both.  At n_embd 2560 the 80 calls of a row
// leave the third pass half idle: it runs row A's call on lanes 0..15 and row B's on lanes 16..31, and row B's
// scale and sum go to lane - 16 by shuffle before its add.  At 5120 a row is five whole passes.  The weight side
// differs only in how the same integers are made: the part's words are aligned 32-bit loads funnel-shifted to the
// 2-byte block layout, the grid table is in shared memory for every format, and a sign byte's masks come from
// prmt's sign-replicating mode instead of the SIMD compare and subtract.
#pragma once

namespace gu2 {

__device__ __forceinline__ uint32_t ldw(const uint8_t* p) { return __ldg((const unsigned int*) p); }

// the 32-bit value at the 2-byte aligned p
__device__ __forceinline__ uint32_t int_at(const uint8_t* p) {
    const uint8_t* a = (const uint8_t*) ((uintptr_t) p & ~(uintptr_t) 3);
    return __funnelshift_r(ldw(a), ldw(a + 4), ((uint32_t) (uintptr_t) p & 2u) << 3);
}
// the two 32-bit values at the 2-byte aligned p
__device__ __forceinline__ void int2_at(const uint8_t* p, uint32_t& v0, uint32_t& v1) {
    const uint8_t* a = (const uint8_t*) ((uintptr_t) p & ~(uintptr_t) 3);
    const uint32_t sh = ((uint32_t) (uintptr_t) p & 2u) << 3;
    const uint32_t w0 = ldw(a), w1 = ldw(a + 4), w2 = ldw(a + 8);
    v0 = __funnelshift_r(w0, w1, sh);
    v1 = __funnelshift_r(w1, w2, sh);
}

// each byte's top bit over the whole byte (__byte_perm takes 3-bit selectors only)
__device__ __forceinline__ uint32_t prmt_sign(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(r) : "r"(x));
    return r;
}
// sign byte s (bit j: value j negative): byte j of m0 all ones for bits 0..3, of m1 for bits 4..7
__device__ __forceinline__ void masks(uint32_t s, uint32_t& m0, uint32_t& m1) {
    m0 = prmt_sign((s & 0xfu) * 0x10204080u);
    m1 = prmt_sign(((s >> 4) & 0xfu) * 0x10204080u);
}
// 7 sign bits and their parity as the eighth (ksigns)
__device__ __forceinline__ void kmasks(uint32_t v7, uint32_t& m0, uint32_t& m1) {
    masks(v7 | (uint32_t) (__popc(v7) & 1) << 7, m0, m1);
}
// the bytes of g negated where m is all ones; grid bytes are nonzero, so no byte carries into the next
__device__ __forceinline__ int neg(uint32_t g, uint32_t m) { return (int) ((g ^ m) + (m & 0x01010101u)); }

__device__ __forceinline__ int dp8(const int (&g)[8], const block_q8_1* __restrict__ x) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = ggml_cuda_dp4a(g[j], get_int_b4(x->qs, j), s);
    return s;
}
__device__ __forceinline__ int dp8_ls2(const int (&g)[8], const block_q8_1* __restrict__ x, int ls0, int ls1) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) s0 = ggml_cuda_dp4a(g[j], get_int_b4(x->qs, j), s0);
#pragma unroll
    for (int j = 4; j < 8; ++j) s1 = ggml_cuda_dp4a(g[j], get_int_b4(x->qs, j), s1);
    return (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
}

// load(row, k, grid): call k (block k / 8, part k % 8) of a row as Split<TY>::load makes it; apply(w, x, d, f): the
// scale product d and the integer sum f of Split<TY>::apply (which returns d * f) against x, the call's q8_1 block.
template<int TY> struct Fmt2;

template<> struct Fmt2<19> {   // IQ1_S: 50-byte blocks (d, qs[32], qh[8] as 16-bit words)
    static constexpr int kGridWords = 2048;
    using W = Split<19>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 50;
        const int j = k & 7;
        const uint32_t qs = int_at(b + 2 + 4 * j);
        const uint32_t qh = __ldg((const unsigned short*) (b + 34 + 2 * j));
        W r;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t g = grid[((qs >> 8 * p) & 0xffu) | (((qh >> 3 * p) & 0x07u) << 8)];
            r.g[2 * p + 0] = (int) (g & 0x0F0F0F0Fu);
            r.g[2 * p + 1] = (int) ((g >> 4) & 0x0F0F0F0Fu);
        }
        r.ls = (int) ((qh >> 11) & 0x0Eu) + 1;
        r.delta = -1.0f + IQ1S_DELTA - (int) (qh & 0x8000u) * (2.0f * IQ1S_DELTA / 0x8000);
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    // Split<19>::apply's d1q * ds * (sumi + delta * sumy): the two products, then the sum as one fused add
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        int sumi = 0, sumy = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int u = get_int_b4(x->qs, j);
            sumi = ggml_cuda_dp4a(r.g[j], u, sumi);
            sumy = ggml_cuda_dp4a(u, 0x01010101, sumy);
        }
        d = __fmul_rn(r.dw * r.ls, __low2float(x->ds));
        f = __fmaf_rn(r.delta, (float) sumy, (float) sumi);
    }
};

template<> struct Fmt2<16> {   // IQ2_XXS: 66-byte blocks
    static constexpr int kGridWords = 512;
    using W = Split<16>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 66;
        uint32_t q, aux;
        int2_at(b + 2 + 8 * (k & 7), q, aux);
        const char* gb = (const char*) grid;
        W r;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint2 gp = *(const uint2*) (gb + ((q >> 8 * p << 3) & 0x7f8u));
            uint32_t m0, m1;
            kmasks((aux >> 7 * p) & 0x7fu, m0, m1);
            r.g[2 * p + 0] = neg(gp.x, m0);
            r.g[2 * p + 1] = neg(gp.y, m1);
        }
        r.ls = aux >> 27 | 1;
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        const int s = dp8(r.g, x) * r.ls / 8;
        d = r.dw * __low2float(x->ds);
        f = (float) s;
    }
};

template<> struct Fmt2<17> {   // IQ2_XS: 74-byte blocks
    static constexpr int kGridWords = 1024;
    using W = Split<17>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 74;
        const int j = k & 7;
        uint32_t q0, q1;
        int2_at(b + 2 + 8 * j, q0, q1);
        const char* gb = (const char*) grid;
        W r;
        const uint32_t sc = b[66 + j];
        r.ls0 = (int) (sc & 0x0Fu);
        r.ls1 = (int) (sc >> 4);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t q = ((p < 2 ? q0 : q1) >> 16 * (p & 1)) & 0xffffu;
            const uint2 gp = *(const uint2*) (gb + ((q << 3) & 0xff8u));
            uint32_t m0, m1;
            kmasks(q >> 9, m0, m1);
            r.g[2 * p + 0] = neg(gp.x, m0);
            r.g[2 * p + 1] = neg(gp.y, m1);
        }
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        const int s = dp8_ls2(r.g, x, r.ls0, r.ls1);
        d = r.dw * __low2float(x->ds);
        f = (float) s;
    }
};

template<> struct Fmt2<22> {   // IQ2_S: 82-byte blocks (d, 32 grid bytes, 32 sign bytes, qh[8], scales[8])
    static constexpr int kGridWords = 2048;
    using W = Split<22>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 82;
        const int j = k & 7;
        const uint32_t qs = int_at(b + 2 + 4 * j), sg = int_at(b + 34 + 4 * j);
        const uint32_t qh = b[66 + j], sc = b[74 + j];
        const uint2* g64 = (const uint2*) grid;
        W r;
        r.ls0 = (int) (sc & 0x0Fu);
        r.ls1 = (int) (sc >> 4);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint2 gp = g64[((qs >> 8 * p) & 0xffu) | ((qh << (8 - 2 * p)) & 0x300u)];
            uint32_t m0, m1;
            masks((sg >> 8 * p) & 0xffu, m0, m1);
            r.g[2 * p + 0] = neg(gp.x, m0);
            r.g[2 * p + 1] = neg(gp.y, m1);
        }
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        const int s = dp8_ls2(r.g, x, r.ls0, r.ls1);
        d = r.dw * __low2float(x->ds);
        f = (float) s;
    }
};

template<> struct Fmt2<18> {   // IQ3_XXS: 98-byte blocks (d, 64 grid bytes, 8 words of signs and scale)
    static constexpr int kGridWords = 256;
    using W = Split<18>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 98;
        const int j = k & 7;
        uint32_t q0, q1;
        int2_at(b + 2 + 8 * j, q0, q1);
        const uint32_t aux = int_at(b + 66 + 4 * j);
        const char* gb = (const char*) grid;
        W r;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t q = p < 2 ? q0 : q1, s = 16 * (p & 1);
            const uint32_t g0 = *(const uint32_t*) (gb + ((q >> s << 2) & 0x3fcu));
            const uint32_t g1 = *(const uint32_t*) (gb + ((q >> (s + 6)) & 0x3fcu));
            uint32_t m0, m1;
            kmasks((aux >> 7 * p) & 0x7fu, m0, m1);
            r.g[2 * p + 0] = neg(g0, m0);
            r.g[2 * p + 1] = neg(g1, m1);
        }
        r.ls = (int) (aux >> 28);
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        int s = dp8(r.g, x);
        s = (r.ls * s + s / 2) / 2;
        d = r.dw * __low2float(x->ds);
        f = (float) s;
    }
};

template<> struct Fmt2<21> {   // IQ3_S: 110-byte blocks (d, 64 grid bytes, qh[8], 32 sign bytes, scales[4])
    static constexpr int kGridWords = 512;
    using W = Split<21>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t* grid) {
        const uint8_t* b = row + (k >> 3) * 110;
        const int j = k & 7;
        uint32_t q0, q1;
        int2_at(b + 2 + 8 * j, q0, q1);
        const uint32_t qh = b[66 + j], sg = int_at(b + 74 + 4 * j), sc = b[106 + (j >> 1)];
        W r;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t q = p < 2 ? q0 : q1, s = 16 * (p & 1);
            const uint32_t g0 = grid[((q >> s) & 0xffu) | ((qh << (8 - 2 * p)) & 0x100u)];
            const uint32_t g1 = grid[((q >> (s + 8)) & 0xffu) | ((qh << (7 - 2 * p)) & 0x100u)];
            uint32_t m0, m1;
            masks((sg >> 8 * p) & 0xffu, m0, m1);
            r.g[2 * p + 0] = neg(g0, m0);
            r.g[2 * p + 1] = neg(g1, m1);
        }
        r.ls = 1 + 2 * (int) ((sc >> 4 * (j & 1)) & 0x0Fu);
        r.dw = __half2float(*(const __half*) b);
        return r;
    }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        const int s = dp8(r.g, x) * r.ls;
        d = r.dw * __low2float(x->ds);
        f = (float) s;
    }
};

template<> struct Fmt2<23> {   // IQ4_XS: 136-byte blocks of 256 values (d, scales_h, scales_l[4], qs[128]).  No grid:
                               // the 16-entry codebook is Split<23>'s table lookup, the sub-block scale its integer sum
    static constexpr int kGridWords = 4;   // no grid
    using W = Split<23>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t*) { return Split<23>::load(row, k >> 3, 4 * (k & 7)); }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(r.v[j].x, get_int_b4(x->qs, j), sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y, get_int_b4(x->qs, j + 4), sumi);
        }
        sumi *= r.ls - 32;
        d = r.dw * __low2float(x->ds);
        f = (float) sumi;
    }
};

template<> struct Fmt2<42> {   // Q2_0: 18-byte blocks of 64 values, two calls a block; Split<42>'s words
    static constexpr int kGridWords = 4;   // no grid
    using W = Split<42>::W;
    __device__ static W load(const uint8_t* row, int k, const uint32_t*) { return Split<42>::load(row, k >> 1, k & 1); }
    __device__ static void apply(const W& r, const block_q8_1* __restrict__ x, float& d, float& f) {
        int s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s = ggml_cuda_dp4a(get_int_b4(x->qs, 2 * j), r.qx[j], s);
            s = ggml_cuda_dp4a(get_int_b4(x->qs, 2 * j + 1), r.qy[j], s);
        }
        d = r.d2 * __low2float(x->ds);
        f = (float) s;
    }
};

template<int TY>
__device__ __forceinline__ const uint32_t* stage_grid(uint32_t* s, int tid) {
    if constexpr (TY == 42 || TY == 23) {   // Q2_0 and IQ4_XS: no grid to stage
        return s;
    } else if constexpr (TY == 21) {
        for (int i = tid; i < 512; i += 256) s[i] = iq3s_grid[i];
        __syncthreads();
        return s;
    } else {
        return stage_iq_grid<TY, true>(s, tid, 256);
    }
}

// a format's grid as stage_grid lays it out, read from the device table instead: a mixed layer's up rows, whose grid
// the block does not stage (the same words, so the same bits)
template<int TY>
__device__ __forceinline__ const uint32_t* table_grid() {
    if constexpr (TY == 16) return (const uint32_t*) iq2xxs_grid;
    else if constexpr (TY == 17) return (const uint32_t*) iq2xs_grid;
    else if constexpr (TY == 22) return (const uint32_t*) iq2s_grid;
    else if constexpr (TY == 18) return (const uint32_t*) iq3xxs_grid;
    else if constexpr (TY == 21) return (const uint32_t*) iq3s_grid;
    else return nullptr;
}

// rows r0, r0 + 1 (gate or up rows r of `dst`; row A at rowA, row B row_bytes after it) for the n <= NC entries
// from e: entry e + c's activation is xq + ent_tok[e + c] * KC (KC calls a row, one q8_1 block each: 80 or 160)
template<int TY, int NC, int KC>
__device__ __forceinline__ void rows2(const uint8_t* rowA, int row_bytes, int r, int e, int n,
                                      const int32_t* __restrict__ ent_tok, const block_q8_1* __restrict__ xq,
                                      float* __restrict__ dst, int n_ff, const uint32_t* grid, int lane) {
    using F = Fmt2<TY>;
    const uint8_t* rowB = rowA + row_bytes;
    const block_q8_1* xc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) xc[c] = xq + (size_t) ent_tok[e + min(c, n - 1)] * KC;
    float sA[NC], sB[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) sA[c] = sB[c] = 0.0f;
    static_assert(KC == 80 || KC == 160, "a row is 2560 or 5120 values");
#pragma unroll
    for (int it = 0; it < KC / 32; ++it) {
        const int k = lane + 32 * it;
        const typename F::W wa = F::load(rowA, k, grid);
        const typename F::W wb = F::load(rowB, k, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                float d, f;
                F::apply(wa, xc[c] + k, d, f);
                sA[c] = __fmaf_rn(d, f, sA[c]);
                F::apply(wb, xc[c] + k, d, f);
                sB[c] = __fmaf_rn(d, f, sB[c]);
            }
        }
    }
    if constexpr (KC % 32 != 0) {
        const int k2 = KC - 16 + (lane & 15);
        const typename F::W w2 = F::load(lane < 16 ? rowA : rowB, k2, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                float d, f;
                F::apply(w2, xc[c] + k2, d, f);
                const float dB = __shfl_down_sync(0xffffffffu, d, 16), fB = __shfl_down_sync(0xffffffffu, f, 16);
                if (lane < 16) {
                    sA[c] = __fmaf_rn(d, f, sA[c]);
                    sB[c] = __fmaf_rn(dB, fB, sB[c]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c < n) {
            sA[c] = warp_sum(sA[c]);
            sB[c] = warp_sum(sB[c]);
            if (lane == 2 * c) dst[(size_t) (e + c) * n_ff + r] = sA[c];
            if (lane == 2 * c + 1) dst[(size_t) (e + c) * n_ff + r + 1] = sB[c];
        }
    }
}

// gate/up rows row, row + 1 (0 .. 2 n_ff: both gate or both up rows) of the expert at `blob`, entries [e0, e1)
template<int TY, int KC = 80>
__device__ __forceinline__ void pair(const uint8_t* blob, int row, int e0, int e1, const int32_t* __restrict__ ent_tok,
                                     const block_q8_1* __restrict__ xq, const NativeExpertLayout& L,
                                     float* __restrict__ gate, float* __restrict__ up, const uint32_t* grid, int lane) {
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const int n_ff = (int) L.n_ff, rb = (int) (is_up ? L.up_row : L.gu_row);
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * rb;
    float* dst = is_up ? up : gate;
    for (int e = e0; e < e1;) {
        const int n = e1 - e;
        if (n == 1) {
            rows2<TY, 1, KC>(wr, rb, r, e, 1, ent_tok, xq, dst, n_ff, grid, lane);
            e += 1;
        } else if (n == 2) {
            rows2<TY, 2, KC>(wr, rb, r, e, 2, ent_tok, xq, dst, n_ff, grid, lane);
            e += 2;
        } else if (n == 3) {
            rows2<TY, 3, KC>(wr, rb, r, e, 3, ent_tok, xq, dst, n_ff, grid, lane);
            e += 3;
        } else {
            // 5120-value rows three entries at a time: the decode then fits kGu2MinBlocks' registers
            constexpr int kNC = KC == 160 ? 3 : 6;
            const int m = min(n, kNC);
            rows2<TY, kNC, KC>(wr, rb, r, e, m, ent_tok, xq, dst, n_ff, grid, lane);
            e += m;
        }
    }
}

// pair() for a layer with gate rows of TG and up rows of TU: the up rows read TU's grid from the device table
template<int TG, int TU, int KC = 80>
__device__ __forceinline__ void pair_gu(const uint8_t* blob, int row, int e0, int e1, const int32_t* __restrict__ ent_tok,
                                        const block_q8_1* __restrict__ xq, const NativeExpertLayout& L,
                                        float* __restrict__ gate, float* __restrict__ up, const uint32_t* grid, int lane) {
    if constexpr (TG != TU) {
        if (row >= L.n_ff) {
            pair<TU, KC>(blob, row, e0, e1, ent_tok, xq, L, gate, up, table_grid<TU>(), lane);
            return;
        }
    }
    pair<TG, KC>(blob, row, e0, e1, ent_tok, xq, L, gate, up, grid, lane);
}

}  // namespace gu2

template<int TG> inline constexpr bool kGu2 = TG == 16 || TG == 17 || TG == 18 || TG == 19 || TG == 21 || TG == 22;
// the grid formats and Q2_0 (the grouped launch and the peer kernel's Q2_0 build)
template<int TG> inline constexpr bool kGu2Q2 = kGu2<TG> || TG == 42;
constexpr int GU2_ROWS = 16;   // rows per block: two a warp
// three blocks an SM at 5120-value rows, where a format's decode fits the 80 registers that leaves (IQ2_S's and
// IQ3_S's would spill); the 2560-value rows as they were
template<int TG, int KC> inline constexpr int kGu2MinBlocks = KC == 160 && TG != 21 && TG != 22 ? 3 : 1;

// native_gu_multi_kernel's rows, GU2_ROWS a block; the same group stride.  TU: the up rows' type; KC: calls a row.
template<int TG, int TU = TG, int KC = 80>
__global__ void __launch_bounds__(256, kGu2MinBlocks<TG, KC>)
native_gu2_kernel(const unsigned long long* __restrict__ grp_ptr,
                  const int32_t* __restrict__ grp_start,
                  const int32_t* __restrict__ n_groups,
                  const int32_t* __restrict__ ent_tok,
                  const block_q8_1* __restrict__ xq,
                  NativeExpertLayout L,
                  float* __restrict__ gate,
                  float* __restrict__ up) {
    asm volatile("griddepcontrol.launch_dependents;");   // the down kernel may start (it waits for this grid)
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    __shared__ alignas(16) uint32_t s_grid_buf[gu2::Fmt2<TG>::kGridWords];
    const uint32_t* s_grid = gu2::stage_grid<TG>(s_grid_buf, threadIdx.x);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU2_ROWS + 2 * warp;          // rows row, row + 1: both gate or both up
    if (row >= 2 * L.n_ff) return;
    for (int g = blockIdx.y; g < ng; g += gridDim.y)
        gu2::pair_gu<TG, TU, KC>((const uint8_t*) grp_ptr[g], row, grp_start[g], grp_start[g + 1], ent_tok, xq, L, gate,
                                 up, s_grid, lane);
}

// whether native_gu2_kernel<.., KC> takes L's gate/up rows: KC calls a row (32 KC values), an even n_ff, 4-byte
// aligned rows
__host__ __device__ inline bool gu2_usable(const NativeExpertLayout& L, int kc = 80) {
    return L.n_embd == 32 * kc && L.n_ff % 2 == 0 && L.gu_row % 4 == 0 && L.up_row % 4 == 0 && L.up_off % 4 == 0;
}

// The layers whose gate and up rows differ in type, as (gate, up) pairs: the kernels are built for these mixes only.
// UD-IQ1_S: IQ2_S gate with IQ2_XXS up.
#define GU_MIX(X) X(22, 16)

// ---------------------------------------------------------------- DEC_GU2=2: a programmatic chain
// The SwiGLU pass and the down kernel launched as programmatic dependents (gate/up -> SwiGLU -> down), each grid
// letting the next one launch as soon as all its blocks have started: a down block prefetches its weight rows into
// L2 before it waits for the SwiGLU pass, so the down weights stream while the gate/up rows are still being read.
// Before its wait a kernel reads only the plan (written before the gate/up kernel) and the weights.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ void prefetch_l2(const uint8_t* p, int bytes, int tid) {
    for (int o = tid * 128; o < bytes; o += 256 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
}

// swiglu_q8_1_entries_kernel's body after the wait
__global__ void __launch_bounds__(256) swiglu_q8_1_entries_pdl_kernel(const float* __restrict__ gate,
                                                                      const float* __restrict__ up,
                                                                      const int32_t* __restrict__ grp_start,
                                                                      const int32_t* __restrict__ n_groups, int n_ff,
                                                                      block_q8_1* __restrict__ hq) {
    pdl_trigger();
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    pdl_wait();
    for (long long i = lo + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += (long long) gridDim.x * blockDim.x) {
        const float g = gate[i];
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
        q8_1_store(h, hq, i);
    }
}

template<typename... KArgs, typename... Args>
void launch_dependent(void (*kernel)(KArgs...), dim3 grid, cudaStream_t s, Args... args) {
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    (void) cudaLaunchKernelEx(&cfg, kernel, args...);   // an error stays for the caller's check
}
