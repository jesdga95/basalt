// src/prefill/q8_gemm.cu - see include/basalt/prefill/q8_gemm.hpp.
// Blackwell TMA + int8 tensor-core GEMM for block-32 Q8_0 matrices in planes:
// - Large shapes (>= 12 waves at 128x128): 16-warp unified 128x128 double-buffered TMA kernel (70 KiB SMEM, 128 regs, 0 spills).
// - Smaller / narrower shapes: 2-CTA/SM warp-specialized 64x64 TMA kernel (35 KiB SMEM, ldmatrix.x4 A & B, 0 spills).
// - Quantizer: vectorized 128-element warp quantizer (4 blocks of 32 floats per warp via float4/char4).
#include "basalt/prefill/q8_gemm.hpp"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "basalt/log.hpp"

namespace basalt::prefill::q8 {
namespace {

constexpr int BK = 128, ST = 2;
constexpr int MAX_SEG = 3;

template <int BT, int BR>
constexpr int smem_bytes() {
    return ST * BT * BK + ST * BR * BK + ST * 4 * BT * (int) sizeof(float) + ST * 4 * BR * (int) sizeof(__half) + 3 * ST * 8;
}

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill q8: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct alignas(64) TmaDesc {
    unsigned long long w[16];
};
static_assert(sizeof(TmaDesc) == sizeof(CUtensorMap), "a descriptor argument is the descriptor size");
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap lost its alignas: the build needs /Zc:__cplusplus");

struct Seg {
    const __half* w_scale;
    float* dst;
    int64_t ld;
    int rows;
    int w_sc_ld;
    int tile0;
};

struct Args {
    TmaDesc tm[1 + MAX_SEG];
    Seg seg[MAX_SEG];
    const float* a_scale;
    int nseg, ntok, tok_tiles, total_tiles, k_tiles, k_blocks, a_sc_ld, epi;
};

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1, unsigned a2,
                                       unsigned a3, unsigned b0, unsigned b1) {
    int z = 0;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}
__device__ __forceinline__ void mb_init(unsigned long long* b, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, unsigned phase) {
    asm volatile("{\n.reg .pred d;\nw_%=:\nmbarrier.try_wait.parity.shared::cta.b64 d, [%0], %1, %2;\n@d bra e_%=;\nbra w_%=;\ne_%=:\n}\n"
                 ::"r"(su(b)), "r"(phase), "r"(0x989680u) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(b)) : "memory");
}
__device__ __forceinline__ void mb_expect(unsigned long long* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_mb_arrive(unsigned long long* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(su(b)) : "memory");
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(su(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const TmaDesc* m, int k, int r, unsigned long long* b) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(su(dst)), "l"(m), "r"(k), "r"(r), "r"(su(b)) : "memory");
}
__device__ __forceinline__ int sw(int row, int byte) { return row * BK + (byte ^ ((row & 7) << 4)); }

__global__ __launch_bounds__(160, 2)
void gemm_64x64_ws_kernel(const __grid_constant__ Args p) {
    constexpr int BT = 64, BR = 64, WT = 2, WR = 2;
    constexpr int WTOK = 32, WROW = 32, MT = 2, MR = 4;
    constexpr int ACT_B = ST * BT * BK, W_B = ST * BR * BK;

    extern __shared__ __align__(1024) unsigned char raw[];
    unsigned char* act = raw;
    unsigned char* wt = raw + ACT_B;
    float* asc = reinterpret_cast<float*>(wt + W_B);
    __half* wsc = reinterpret_cast<__half*>(asc + ST * 4 * BT);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(wsc + ST * 4 * BR);
    unsigned long long* full_sc = full + ST;
    unsigned long long* empty = full_sc + ST;

    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            mb_init(full + s, 1);
            mb_init(full_sc + s, 32);
            mb_init(empty + s, WT * WR);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    constexpr int G = 8;
    const int gsz = G * p.tok_tiles;
    const int grp = (int) blockIdx.x / gsz;
    const int rem = (int) blockIdx.x - grp * gsz;
    const int rg = (grp + 1) * G <= p.total_tiles ? G : (p.total_tiles - grp * G);
    const int rt = grp * G + rem % rg;
    const int tt = rem / rg;
    int si = 0;
    for (int i = 1; i < p.nseg; ++i) if (rt >= p.seg[i].tile0) si = i;
    const Seg& seg = p.seg[si];
    const int row0 = (rt - seg.tile0) * BR;
    const int tok0 = tt * BT;
    const int full_kt = p.k_blocks >> 2;

    if (threadIdx.x < 32) {
        const int lane = (int) threadIdx.x;
        const int ks_a0 = lane >> 4, off_a0 = (lane & 15) * 4;
        const int ks_w = lane >> 3, off_w = (lane & 7) * 8;
        for (int kt = 0; kt < p.k_tiles; ++kt) {
            const int st = kt & 1;
            mb_wait(empty + st, 1u ^ ((kt >> 1) & 1u));
            if (lane == 0) {
                mb_expect(full + st, (BT + BR) * BK);
                tma2d(act + st * BT * BK, &p.tm[0], kt * BK, tok0, full + st);
                tma2d(wt + st * BR * BK, &p.tm[1 + si], kt * BK, row0, full + st);
            }
            const int kb = kt * 4;
            float* ad = asc + st * 4 * BT;
            __half* wd = wsc + st * 4 * BR;
            if (kt < full_kt) {
                cp16(ad + ks_a0 * BT + off_a0, p.a_scale + (int64_t) (kb + ks_a0) * p.a_sc_ld + tok0 + off_a0);
                cp16(ad + (ks_a0 + 2) * BT + off_a0, p.a_scale + (int64_t) (kb + ks_a0 + 2) * p.a_sc_ld + tok0 + off_a0);
                cp16(wd + ks_w * BR + off_w, seg.w_scale + (int64_t) (kb + ks_w) * seg.w_sc_ld + row0 + off_w);
            } else {
                if (kb + ks_a0 < p.k_blocks)
                    cp16(ad + ks_a0 * BT + off_a0, p.a_scale + (int64_t) (kb + ks_a0) * p.a_sc_ld + tok0 + off_a0);
                if (kb + ks_a0 + 2 < p.k_blocks)
                    cp16(ad + (ks_a0 + 2) * BT + off_a0, p.a_scale + (int64_t) (kb + ks_a0 + 2) * p.a_sc_ld + tok0 + off_a0);
                if (kb + ks_w < p.k_blocks)
                    cp16(wd + ks_w * BR + off_w, seg.w_scale + (int64_t) (kb + ks_w) * seg.w_sc_ld + row0 + off_w);
            }
            cp_mb_arrive(full_sc + st);
        }
        return;
    }

    const int tid = (int) threadIdx.x - 32;
    const int lane = tid & 31, warp = tid >> 5;
    const int wtk = warp / WR, wrw = warp % WR;
    const int am = lane >> 3;
    const int a_row = (lane & 7) + ((am & 1) << 3);
    const int a_col = (am >> 1) * 16;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) * 16;
    const int g = lane >> 2, c2 = 2 * (lane & 3);
    const int t_off = wtk * WTOK + g, r_off = wrw * WROW + c2;

    float acc[MT][MR][4] = {};

    const auto stage = [&]<int NKS>(int st, unsigned ph) {
        mb_wait(full + st, ph);
        mb_wait(full_sc + st, ph);
        const unsigned char* as = act + st * BT * BK;
        const unsigned char* ws = wt + st * BR * BK;
        const float* a_sc = asc + st * 4 * BT;
        const __half* w_sc = wsc + st * 4 * BR;
        unsigned af[2][MT][4], bf[2][MR][2];
        auto ld = [&](int ks, unsigned (&a)[MT][4], unsigned (&b)[MR][2]) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                ldsm4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], su(as + sw(wtk * WTOK + mt * 16 + a_row, ks * 32 + a_col)));
#pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                ldsm2(b[mr][0], b[mr][1], su(ws + sw(wrw * WROW + mr * 8 + b_row, ks * 32 + b_col)));
        };
        ld(0, af[0], bf[0]);
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            const int sl = ks & 1;
            if constexpr (NKS > 1) { if (ks + 1 < NKS) ld(ks + 1, af[sl ^ 1], bf[sl ^ 1]); }
            float2 db[MR];
            const __half* wr = w_sc + ks * BR + r_off;
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) db[mr] = __half22float2(*reinterpret_cast<const __half2*>(wr + mr * 8));
            const float* ar = a_sc + ks * BT + t_off;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float da0 = ar[mt * 16], da1 = ar[mt * 16 + 8];
#pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    int c0, c1, c2i, c3;
                    mma_s8(c0, c1, c2i, c3, af[sl][mt][0], af[sl][mt][1], af[sl][mt][2], af[sl][mt][3],
                           bf[sl][mr][0], bf[sl][mr][1]);
                    acc[mt][mr][0] = fmaf((float) c0, da0 * db[mr].x, acc[mt][mr][0]);
                    acc[mt][mr][1] = fmaf((float) c1, da0 * db[mr].y, acc[mt][mr][1]);
                    acc[mt][mr][2] = fmaf((float) c2i, da1 * db[mr].x, acc[mt][mr][2]);
                    acc[mt][mr][3] = fmaf((float) c3, da1 * db[mr].y, acc[mt][mr][3]);
                }
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive(empty + st);
    };

    for (int kt = 0; kt < full_kt; kt += 2) {
        const unsigned ph = (unsigned) ((kt / 2) & 1);
        stage.template operator()<4>(0, ph);
        if (kt + 1 < full_kt) stage.template operator()<4>(1, ph);
    }
    if (full_kt < p.k_tiles) {
        stage.template operator()<2>(full_kt & 1, (unsigned) ((full_kt / 2) & 1));
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tk = tok0 + t_off + mt * 16 + h * 8;
            if (tk >= p.ntok) continue;
            float* orow = seg.dst + (int64_t) tk * seg.ld;
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + r_off + mr * 8;
                if (r < seg.rows) {
                    float2* dst = reinterpret_cast<float2*>(orow + r);
                    if (p.epi == (int) Epilogue::Add) {
                        float2 v = *dst;
                        v.x += acc[mt][mr][2 * h];
                        v.y += acc[mt][mr][2 * h + 1];
                        *dst = v;
                    } else {
                        *dst = make_float2(acc[mt][mr][2 * h], acc[mt][mr][2 * h + 1]);
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(512, 1)
void gemm_128x128_u16_kernel(const __grid_constant__ Args p) {
    constexpr int BT = 128, BR = 128, WR = 4;
    constexpr int WTOK = 32, WROW = 32, MT = 2, MR = 4;
    constexpr int ACT_B = ST * BT * BK, W_B = ST * BR * BK;

    extern __shared__ __align__(1024) unsigned char raw[];
    unsigned char* act = raw;
    unsigned char* wt = raw + ACT_B;
    float* asc = reinterpret_cast<float*>(wt + W_B);
    __half* wsc = reinterpret_cast<__half*>(asc + ST * 4 * BT);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(wsc + ST * 4 * BR);
    unsigned long long* full_sc = full + ST;

    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            mb_init(full + s, 1);
            mb_init(full_sc + s, 64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    constexpr int G = 8;
    const int gsz = G * p.tok_tiles;
    const int grp = (int) blockIdx.x / gsz;
    const int rem = (int) blockIdx.x - grp * gsz;
    const int rg = (grp + 1) * G <= p.total_tiles ? G : (p.total_tiles - grp * G);
    const int rt = grp * G + rem % rg;
    const int tt = rem / rg;
    int si = 0;
    for (int i = 1; i < p.nseg; ++i) if (rt >= p.seg[i].tile0) si = i;
    const Seg& seg = p.seg[si];
    const int row0 = (rt - seg.tile0) * BR;
    const int tok0 = tt * BT;
    const int full_kt = p.k_blocks >> 2;

    const auto issue_stage = [&](int kt) {
        const int st = kt & 1;
        if (threadIdx.x == 0) {
            mb_expect(full + st, (BT + BR) * BK);
            tma2d(act + st * BT * BK, &p.tm[0], kt * BK, tok0, full + st);
            tma2d(wt + st * BR * BK, &p.tm[1 + si], kt * BK, row0, full + st);
        }
        if (threadIdx.x < 64) {
            const int tid = (int) threadIdx.x;
            const int kb = kt * 4;
            float* ad = asc + st * 4 * BT;
            __half* wd = wsc + st * 4 * BR;
            const int ks_a0 = tid >> 5, off_a = (tid & 31) * 4;
            const int ks_w  = tid >> 4, off_w = (tid & 15) * 8;
            if (kb + ks_a0 < p.k_blocks)
                cp16(ad + ks_a0 * BT + off_a, p.a_scale + (int64_t) (kb + ks_a0) * p.a_sc_ld + tok0 + off_a);
            if (kb + ks_a0 + 2 < p.k_blocks)
                cp16(ad + (ks_a0 + 2) * BT + off_a, p.a_scale + (int64_t) (kb + ks_a0 + 2) * p.a_sc_ld + tok0 + off_a);
            if (kb + ks_w < p.k_blocks)
                cp16(wd + ks_w * BR + off_w, seg.w_scale + (int64_t) (kb + ks_w) * seg.w_sc_ld + row0 + off_w);
            cp_mb_arrive(full_sc + st);
        }
    };

    const int tid = (int) threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int wtk = warp / WR, wrw = warp % WR;
    const int am = lane >> 3;
    const int a_row = (lane & 7) + ((am & 1) << 3);
    const int a_col = (am >> 1) * 16;
    const int g = lane >> 2, c2 = 2 * (lane & 3);
    const int t_off = wtk * WTOK + g, r_off = wrw * WROW + c2;

    float acc[MT][MR][4] = {};
    if (p.k_tiles > 0) issue_stage(0);

    const auto run_kt = [&]<int NKS>(int kt) {
        if (kt + 1 < p.k_tiles) issue_stage(kt + 1);
        const int st = kt & 1;
        const unsigned ph = (unsigned) ((kt >> 1) & 1);
        mb_wait(full + st, ph);
        mb_wait(full_sc + st, ph);
        const unsigned char* as = act + st * BT * BK;
        const unsigned char* ws = wt + st * BR * BK;
        const float* a_sc = asc + st * 4 * BT;
        const __half* w_sc = wsc + st * 4 * BR;
        unsigned af[2][MT][4], bf[2][MR][2];
        auto ld = [&](int ks, unsigned (&a)[MT][4], unsigned (&b)[MR][2]) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                ldsm4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], su(as + sw(wtk * WTOK + mt * 16 + a_row, ks * 32 + a_col)));
#pragma unroll
            for (int m2 = 0; m2 < MR; m2 += 2)
                ldsm4(b[m2][0], b[m2 + 1][0], b[m2][1], b[m2 + 1][1], su(ws + sw(wrw * WROW + m2 * 8 + a_row, ks * 32 + a_col)));
        };
        ld(0, af[0], bf[0]);
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            const int sl = ks & 1;
            if constexpr (NKS > 1) { if (ks + 1 < NKS) ld(ks + 1, af[sl ^ 1], bf[sl ^ 1]); }
            float2 db[MR];
            const __half* wr = w_sc + ks * BR + r_off;
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) db[mr] = __half22float2(*reinterpret_cast<const __half2*>(wr + mr * 8));
            const float* ar = a_sc + ks * BT + t_off;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float da0 = ar[mt * 16], da1 = ar[mt * 16 + 8];
#pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    int c0, c1, c2i, c3;
                    mma_s8(c0, c1, c2i, c3, af[sl][mt][0], af[sl][mt][1], af[sl][mt][2], af[sl][mt][3],
                           bf[sl][mr][0], bf[sl][mr][1]);
                    acc[mt][mr][0] = fmaf((float) c0, da0 * db[mr].x, acc[mt][mr][0]);
                    acc[mt][mr][1] = fmaf((float) c1, da0 * db[mr].y, acc[mt][mr][1]);
                    acc[mt][mr][2] = fmaf((float) c2i, da1 * db[mr].x, acc[mt][mr][2]);
                    acc[mt][mr][3] = fmaf((float) c3, da1 * db[mr].y, acc[mt][mr][3]);
                }
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    };

    for (int kt = 0; kt < full_kt; ++kt) run_kt.template operator()<4>(kt);
    if (full_kt < p.k_tiles) run_kt.template operator()<2>(full_kt);

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tk = tok0 + t_off + mt * 16 + h * 8;
            if (tk >= p.ntok) continue;
            float* orow = seg.dst + (int64_t) tk * seg.ld;
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + r_off + mr * 8;
                if (r < seg.rows) {
                    float2* dst = reinterpret_cast<float2*>(orow + r);
                    if (p.epi == (int) Epilogue::Add) {
                        float2 v = *dst;
                        v.x += acc[mt][mr][2 * h];
                        v.y += acc[mt][mr][2 * h + 1];
                        *dst = v;
                    } else {
                        *dst = make_float2(acc[mt][mr][2 * h], acc[mt][mr][2 * h + 1]);
                    }
                }
            }
        }
    }
}

__global__ void quantize_q8_planes_vec4_kernel(const float* __restrict__ x, int64_t rows, int64_t cols, int64_t ld,
                                               int64_t sc_ld, int8_t* __restrict__ codes, float* __restrict__ scales) {
    const int64_t nb = cols >> 5;
    const int64_t b0 = (((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4;
    const int lane = (int) (threadIdx.x & 31);
    const int sub = lane >> 3, idx = (lane & 7) * 4;
    const int64_t b = b0 + sub;
    if (b >= rows * nb) return;
    const int64_t r = b / nb, k = b - r * nb;
    const float4 v = *reinterpret_cast<const float4*>(x + r * ld + (k << 5) + idx);
    const unsigned mask = 0xffu << (sub * 8);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));
    const float d = amax * (1.0f / 127.0f), id = amax > 0.0f ? (127.0f / amax) : 0.0f;
    char4 q{(signed char) __float2int_rn(v.x * id), (signed char) __float2int_rn(v.y * id),
            (signed char) __float2int_rn(v.z * id), (signed char) __float2int_rn(v.w * id)};
    *reinterpret_cast<char4*>(codes + r * cols + (k << 5) + idx) = q;
    if ((lane & 7) == 0) scales[k * sc_ld + r] = d;
}

__global__ void quantize_q8_planes_scalar_kernel(const float* __restrict__ x, int64_t rows, int64_t cols, int64_t ld,
                                                 int64_t sc_ld, int8_t* __restrict__ codes, float* __restrict__ scales) {
    const int64_t nb = cols >> 5;
    const int64_t b = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= rows * nb) return;
    const int lane = (int) (threadIdx.x & 31);
    const int64_t r = b / nb, k = b - r * nb;
    const float v = x[r * ld + (k << 5) + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.0f / 127.0f), id = amax > 0.0f ? (127.0f / amax) : 0.0f;
    codes[r * cols + (k << 5) + lane] = (int8_t) __float2int_rn(v * id);
    if (lane == 0) scales[k * sc_ld + r] = d;
}

__global__ void transpose_scales_kernel(const uint16_t* __restrict__ src, int64_t rows, int64_t nb, int64_t dst_ld,
                                        uint16_t* __restrict__ dst) {
    const int64_t col = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t kb = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= dst_ld || kb >= nb) return;
    dst[kb * dst_ld + col] = col < rows ? src[col * nb + kb] : (uint16_t) 0;
}

size_t scale_off(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~size_t(15); }

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
    static const EncodeFn f = []() -> EncodeFn {
        void* fp = nullptr;
        cudaDriverEntryPointQueryResult st{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fp, 12000, cudaEnableDefault, &st) !=
                cudaSuccess ||
            !fp)
            return nullptr;
        return reinterpret_cast<EncodeFn>(fp);
    }();
    return f;
}

void tmap(TmaDesc* out, const void* base, int64_t rows, int64_t cols, int box_rows) {
    const EncodeFn f = encode_fn();
    CUtensorMap m{};
    const cuuint64_t dim[2] = {(cuuint64_t) cols, (cuuint64_t) rows}, str[1] = {(cuuint64_t) cols};
    const cuuint32_t box[2] = {(cuuint32_t) BK, (cuuint32_t) box_rows}, es[2] = {1, 1};
    const CUresult rc =
        f ? f(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, str, box, es,
              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
          : CUDA_ERROR_NOT_FOUND;
    if (rc != CUDA_SUCCESS) {
        basalt::log::error("basalt", "prefill q8: tensor map for [%lld x %lld] failed (rc=%d)", (long long) rows,
                           (long long) cols, (int) rc);
        std::exit(1);
    }
    std::memcpy(out, &m, sizeof m);
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) {
    return scale_off(rows, cols) + (size_t) (cols >> 5) * (size_t) act_scale_ld(rows) * sizeof(float);
}

float* act_scales(void* act, int64_t rows, int64_t cols) {
    return reinterpret_cast<float*>(static_cast<uint8_t*>(act) + scale_off(rows, cols));
}

size_t scale_plane_bytes(int64_t rows, int64_t cols) {
    return (size_t) (cols >> 5) * (size_t) scale_plane_ld(rows) * sizeof(uint16_t);
}

void transpose_scales(const uint16_t* row_major, int64_t rows, int64_t cols, uint16_t* col_major, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols % 32 != 0) {
        basalt::log::error("basalt", "prefill q8: transpose_scales cols %lld", (long long) cols);
        std::exit(1);
    }
    const int64_t nb = cols >> 5;
    const int64_t dst_ld = scale_plane_ld(rows);
    const dim3 blk(32, 8);
    const dim3 grd((unsigned) ((dst_ld + 31) / 32), (unsigned) ((nb + 7) / 8));
    transpose_scales_kernel<<<grd, blk, 0, (cudaStream_t) stream>>>(row_major, rows, nb, dst_ld, col_major);
    ck("transpose_scales");
}

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0) {
        basalt::log::error("basalt", "prefill q8: quantize cols %lld", (long long) cols);
        std::exit(1);
    }
    const int64_t nblocks = rows * (cols >> 5);
    if (((uintptr_t) x & 15) == 0 && (ld & 3) == 0 && (((uintptr_t) act) & 3) == 0) {
        const unsigned grid = (unsigned) (((nblocks + 3) / 4 + 7) / 8);
        quantize_q8_planes_vec4_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
            x, rows, cols, ld, act_scale_ld(rows), static_cast<int8_t*>(act), act_scales(act, rows, cols));
    } else {
        const unsigned grid = (unsigned) ((nblocks + 7) / 8);
        quantize_q8_planes_scalar_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
            x, rows, cols, ld, act_scale_ld(rows), static_cast<int8_t*>(act), act_scales(act, rows, cols));
    }
    ck("quantize");
}

void gemm_segmented(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi,
                    void* stream) {
    if (rows <= 0 || nseg < 1) return;
    if (nseg > MAX_SEG || w_cols % 64 != 0) {
        basalt::log::error("basalt", "prefill q8: %d segments of %lld columns", nseg, (long long) w_cols);
        std::exit(1);
    }
    static std::atomic<bool> configured[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !configured[dev].load(std::memory_order_acquire)) {
        cudaFuncSetAttribute(gemm_64x64_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<64, 64>());
        cudaFuncSetAttribute(gemm_128x128_u16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<128, 128>());
        configured[dev].store(true, std::memory_order_release);
    }
    int tiles_128 = 0;
    for (int i = 0; i < nseg; ++i) {
        const Segment& s = seg[i];
        if (s.rows <= 0 || s.rows % 2 != 0 || !s.w_scales_t || ((uintptr_t) s.w & 15) != 0 ||
            ((uintptr_t) s.w_scales_t & 15) != 0 || ((uintptr_t) s.dst & 7) != 0 || s.ld_dst % 2 != 0 ||
            s.ld_dst < s.rows) {
            basalt::log::error("basalt", "prefill q8: segment %d: [%lld x %lld] matrix, ld %lld", i, (long long) s.rows,
                               (long long) w_cols, (long long) s.ld_dst);
            std::exit(1);
        }
        tiles_128 += (int) ((s.rows + 127) / 128);
    }
    const int tok_128 = (int) ((rows + 127) / 128);
    const bool use_128 = (rows >= 1024 && tiles_128 >= 32 && tok_128 * tiles_128 >= 800);
    const int bt = use_128 ? 128 : 64;
    const int br = use_128 ? 128 : 64;

    Args p{};
    tmap(&p.tm[0], act, rows, w_cols, bt);
    int tiles = 0;
    for (int i = 0; i < nseg; ++i) {
        const Segment& s = seg[i];
        tmap(&p.tm[1 + i], s.w, s.rows, w_cols, br);
        p.seg[i] = {reinterpret_cast<const __half*>(s.w_scales_t), s.dst, s.ld_dst, (int) s.rows,
                    (int) scale_plane_ld(s.rows), tiles};
        tiles += (int) ((s.rows + br - 1) / br);
    }
    p.a_scale = reinterpret_cast<const float*>(static_cast<const uint8_t*>(act) + scale_off(rows, w_cols));
    p.nseg = nseg;
    p.ntok = (int) rows;
    p.tok_tiles = (int) ((rows + bt - 1) / bt);
    p.total_tiles = tiles;
    p.k_tiles = (int) ((w_cols + BK - 1) / BK);
    p.k_blocks = (int) (w_cols >> 5);
    p.a_sc_ld = (int) act_scale_ld(rows);
    p.epi = (int) epi;
    const unsigned grid = (unsigned) (p.tok_tiles * p.total_tiles);
    if (use_128) {
        gemm_128x128_u16_kernel<<<grid, 512, smem_bytes<128, 128>(), (cudaStream_t) stream>>>(p);
    } else {
        gemm_64x64_ws_kernel<<<grid, 160, smem_bytes<64, 64>(), (cudaStream_t) stream>>>(p);
    }
    ck("gemm");
}

void gemm(const int8_t* w, const uint16_t* w_scales_t, int64_t w_rows, int64_t w_cols, const void* act, int64_t rows,
          float* dst, int64_t ld_dst, Epilogue epi, void* stream) {
    const Segment s{w, w_rows, w_scales_t, dst, ld_dst};
    gemm_segmented(&s, 1, w_cols, act, rows, epi, stream);
}

}  // namespace basalt::prefill::q8
