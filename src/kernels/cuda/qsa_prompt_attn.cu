// src/kernels/cuda/qsa_prompt_attn.cu - see include/basalt/kernels/qsa_prompt_attn.hpp.
#include <cstdlib>
#include <cstring>
#include "basalt/kernels/qsa_prompt_attn.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// m16n8k16 with f16 inputs and FP32 accumulation.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values (served by `prompt_attn_i8_kernel`, not instantiated from
// this template). KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
// KV_MODE 4 (Q4_0 K and V, `--kv q4_0`): mode 1 with a scale per 32 values - each q4_0 block's codes enter as exact
// int8 (code - 8) and its fp16 scale is applied in FP32, as mode 1's are; the caller rotates q and un-rotates the
// output (kv_q4.hpp).
template <int KV_MODE>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1 || KV_MODE == 4, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = (KV_MODE == 1 || KV_MODE == 4) ? HD + 16 : HD + 8;
    static constexpr int NG = KV_MODE == 4 ? HD / QK4_0 : 4;      // scale groups per row (q4_0: 8 of 32, else 4 of 64)
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][NG];
    float vs[CH][NG];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ steps, int n_kv_heads,
                                                              int page_size, float scale_log2, float* __restrict__ attn,
                                                              int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        if constexpr (KV_MODE == 4) {
            // q4_0 K and V: one (K or V, cell, block) per item - the block's 16 code bytes (2-byte aligned: eight
            // 16-bit loads) become 32 exact int8 codes (element j in the low nibble of byte j, j + 16 in the high
            // one), its fp16 scale goes to ks/vs
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
            for (int i = t; i < 2 * CH * BLKS; i += THREADS) {
                const int kv = i / (CH * BLKS), rem = i % (CH * BLKS), c = rem / BLKS, b = rem % BLKS;
                const long long r = S.row[c];
                uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
                float d = 0.0f;
                if (r >= 0) {
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((kv == 0 ? p.k_q4 : p.v_q4) + r * BYTES) + b;
                    d = __half2float(__ushort_as_half(__ldg(&blk->d)));
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs);
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        w[j] = (uint32_t) __ldg(q16 + 2 * j) | ((uint32_t) __ldg(q16 + 2 * j + 1) << 16);
                    // per byte: the low nibble minus 8 is element j, the high one minus 8 element j + 16
                    lo = make_uint4(__vsub4(w[0] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[1] & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4(w[2] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[3] & 0x0F0F0F0Fu, 0x08080808u));
                    hi = make_uint4(__vsub4((w[0] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[1] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[2] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[3] >> 4) & 0x0F0F0F0Fu, 0x08080808u));
                }
                int8_t* dst = kv == 0 ? &S.k[c][b * QK4_0] : &S.v[c][b * QK4_0];
                reinterpret_cast<uint4*>(dst)[0] = lo;
                reinterpret_cast<uint4*>(dst)[1] = hi;
                (kv == 0 ? S.ks : S.vs)[c][b] = d;
            }
        } else {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per scale group (64 dims; q4_0's 32)
        constexpr int NG = Smem<KV_MODE>::NG, KPG = HD / 16 / NG;   // groups per row, 16-dim MMA steps per group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < KPG; ++kk) {
                    const int k0 = (g * KPG + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = fmaf(tg[0], s0, sc[0]);
                sc[1] = fmaf(tg[1], s1, sc[1]);
                sc[2] = fmaf(tg[2], s0, sc[2]);
                sc[3] = fmaf(tg[3], s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) { x[j] = S.s[r][sub * PER + j]; mx = fmaxf(mx, x[j]); }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w (q4_0: groups 2w and 2w+1, four n-tiles
        // each). A group's scale is folded into p relative to the chunk's largest magnitude (q4_0's scales are signed:
        // ggml's d = max / -8), times 2^14 (|p'| <= 2^14: inside FP16's range, its lo half out of the subnormals); the
        // chunk's sum is then added to the running one in FP32 with the factor taken back out
        {
            constexpr int GPW = NG / 4, JPG = 8 / GPW;   // scale groups per warp, n-tiles per group
            float vdown_g[GPW];
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int gi = 0; gi < GPW; ++gi) {
            const int vg = warp * GPW + gi;
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, fabsf(S.vs[c][vg]));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown_g[gi] = vmax * (1.0f / 16384.0f);
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][vg] * vup, w1 = S.vs[cA + 1][vg] * vup, w2 = S.vs[cB][vg] * vup,
                            w3 = S.vs[cB + 1][vg] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int jj = 0; jj < JPG; ++jj) {
                    const int j = gi * JPG + jj;
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[cA][d], S.v[cA + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float vdown = vdown_g[j / JPG];
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v3 (int8 KV; adapted from Infernix, Apache-2.0, see THIRD_PARTY_LICENSES): each warp owns whole 16-cell tiles (cells w, w+4, ... of the list), runs q.k and p.v over all
// 256 dims with a single FP16 MMA (FP32 accumulate) and keeps its own online softmax and output accumulators in
// registers. Warps never sync during the loop; one merge at the end combines the four partial softmaxes. Each warp
// gathers its own tiles with cp.async into a private double-buffered stage.
constexpr int TILE = 16;                   // cells per warp tile
constexpr int NWARP = THREADS / 32;
constexpr int QROW = HD * 2;               // bytes of one FP16 q row
constexpr int KROW = HD;                   // bytes of one int8 K or V row
constexpr int STAGE_BYTES = 2 * TILE * KROW + 2 * TILE * 8;   // K codes, V codes, K scales, V scales (4 halves a cell)
constexpr int OSTRIDE = HD + 16;           // merge row stride in floats

struct Smem3 {
    alignas(16) unsigned char q[16 * QROW];
    alignas(16) unsigned char stage[NWARP][2][STAGE_BYTES];
    float qmax[NWARP];
    float mw[NWARP][16];
    float lw[NWARP][16];
};
static_assert(NWARP * 12 * OSTRIDE * 4 <= NWARP * 2 * STAGE_BYTES, "merge rows overlay the stages");

// 16-byte chunk c of row r lives at c ^ (r & 7): the rows one ldmatrix reads hit distinct banks.
__device__ __forceinline__ int swz16(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
template <int BYTES>
__device__ __forceinline__ void cp_async_zfill(void* smem, const void* gmem, bool valid) {
    const unsigned sa = smem_addr(smem);
    if constexpr (BYTES == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 8 : 0));
    }
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}
__device__ __forceinline__ void cp_async_wait0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// MMA k position {2t, 2t+1 | 2t+8, 2t+9} of a 16-wide step takes dims {4t..4t+3} of a code row (what one
// ldmatrix lane receives), so q is stored in that order and the dot product is unchanged.
__device__ __forceinline__ int q_position(int d) {
    const int local = d & 15;
    return (d & ~15) + (((local & 2) != 0 ? 8 : 0) + 2 * (local >> 2) + (local & 1));
}

// Four int8 codes (dims d..d+3 of a K row) as the two FP16 B registers of one k step, widened exactly.
__device__ __forceinline__ void widen_codes(uint32_t codes, uint32_t& lo, uint32_t& hi) {
    const uint32_t biased = codes ^ 0x80808080u;
    const uint32_t e = __byte_perm(biased, 0x64646464u, 0x5140);
    const uint32_t o = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 off = __float2half2_rn(1152.0f);
    const __half2 l = __hsub2(*reinterpret_cast<const __half2*>(&e), off);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&o), off);
    lo = *reinterpret_cast<const uint32_t*>(&l);
    hi = *reinterpret_cast<const uint32_t*>(&h);
}

// One transposed-ldmatrix lane of an int8 [cell][dim] tile holds {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}: the
// FP16 B pairs {V[k][d], V[k+1][d]} (even) and {V[k][d+1], V[k+1][d+1]} (odd), each code widened exactly and
// multiplied once by its cell's scale (scales = {s_k, s_k+1}).
__device__ __forceinline__ void decode_v_pair(uint32_t codes, __half2 scales, uint32_t& even, uint32_t& odd) {
    const uint32_t biased = codes ^ 0x80808080u;
    const uint32_t e = __byte_perm(biased, 0x64646464u, 0x4240);
    const uint32_t o = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 off = __float2half2_rn(1152.0f);
    const __half2 ve = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&e), off), scales);
    const __half2 vo = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&o), off), scales);
    even = *reinterpret_cast<const uint32_t*>(&ve);
    odd = *reinterpret_cast<const uint32_t*>(&vo);
}

__global__ void __launch_bounds__(THREADS, 2) prompt_attn_i8_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem3& S = *reinterpret_cast<Smem3*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, lid = lane & 3;

    // q: power-of-two prescale over all 12 heads (keeps the FP16 products in range), then FP16 rows in k-step order;
    // rows 12..15 are zero.
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int r = warp; r < 16; r += NWARP) {
#pragma unroll
        for (int k = 0; k < HD / 32; ++k) {
            const int d = lane + 32 * k, e = q_position(d);
            const float v = r < G ? q[(size_t) r * HD + d] * qup : 0.0f;
            *reinterpret_cast<__half*>(S.q + swz16(r, QROW, e >> 3) + 2 * (e & 7)) = __float2half_rn(v);
        }
    }
    __syncthreads();

    const int n_tiles = (n + TILE - 1) / TILE;
    unsigned char* stg = &S.stage[warp][0][0];
    auto cell_of = [&](int tile) -> int {
        const int c = tile * TILE + (lane & 15);
        return c < n ? __ldg(ids + c) : -1;
    };
    // Gathers a tile's K and V codes and scales into stage s; cells past the list are zero-filled.
    auto issue = [&](int cell, int s) {
        long long r = -1;
        if (cell >= 0) {
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        unsigned char* k_s = stg + s * STAGE_BYTES;
        unsigned char* v_s = k_s + TILE * KROW;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = 2 * j + (lane >> 4), c = lane & 15;
            const long long rr = __shfl_sync(0xffffffffu, r, row);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + c * 16 : 0;
            cp_async_zfill<16>(k_s + swz16(row, KROW, c), p.k_q + off, ok);
            cp_async_zfill<16>(v_s + swz16(row, KROW, c), p.v_q + off, ok);
        }
        const bool ok = r >= 0;
        const int row = lane & 15;
        if (lane < 16)
            cp_async_zfill<8>(v_s + TILE * KROW + row * 8, ok ? (const void*) (p.k_scale + r * 4) : (const void*) p.k_scale, ok);
        else
            cp_async_zfill<8>(v_s + TILE * KROW + TILE * 8 + row * 8,
                              ok ? (const void*) (p.v_scale + r * 4) : (const void*) p.v_scale, ok);
    };

    float o[16][2][4];
#pragma unroll
    for (int b = 0; b < 16; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h) o[b][h][0] = o[b][h][1] = o[b][h][2] = o[b][h][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int my_tiles = warp < n_tiles ? (n_tiles - warp + NWARP - 1) / NWARP : 0;
    int cell_a = my_tiles > 0 ? cell_of(warp) : -1;
    int cell_b = my_tiles > 1 ? cell_of(warp + NWARP) : -1;
    if (my_tiles > 0) issue(cell_a, 0);
    cp_async_commit();
    cell_a = cell_b;
    cell_b = my_tiles > 2 ? cell_of(warp + 2 * NWARP) : -1;

    const int a_mat = lane >> 3, a_row = (lane & 7) + ((a_mat & 1) << 3);
    for (int j = 0; j < my_tiles; ++j) {
        const int st = j & 1, tile = warp + j * NWARP;
        if (j + 1 < my_tiles) issue(cell_a, st ^ 1);
        cp_async_commit();
        cell_a = cell_b;
        cell_b = j + 3 < my_tiles ? cell_of(warp + (j + 3) * NWARP) : -1;
        cp_async_wait1();
        __syncwarp();
        const unsigned char* k_s = stg + st * STAGE_BYTES;
        const unsigned char* v_s = k_s + TILE * KROW;
        const __half* ksc = reinterpret_cast<const __half*>(v_s + TILE * KROW);
        const __half* vsc = ksc + TILE * 4;

        // q.k: 16 k steps per cell pair; each 64-dim group's partial is scaled by that group's K scale
        float score[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            float acc[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int kb = g * 4 + kk;
                uint32_t a[4], rb[2];
                ldsm_x4(a, smem_addr(S.q + swz16(a_row, QROW, 2 * kb + (a_mat >> 1))));
                ldsm_x2(rb, smem_addr(k_s + swz16(lane & 15, KROW, kb)));
                uint32_t b[2];
                widen_codes(rb[0], b[0], b[1]);
                mma16816(acc[0], a, b);
                widen_codes(rb[1], b[0], b[1]);
                mma16816(acc[1], a, b);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int key = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ksc[key * 4 + g]), ks1 = __half2float(ksc[(key + 1) * 4 + g]);
                score[nt][0] = fmaf(ks0, acc[nt][0], score[nt][0]);
                score[nt][1] = fmaf(ks1, acc[nt][1], score[nt][1]);
                score[nt][2] = fmaf(ks0, acc[nt][2], score[nt][2]);
                score[nt][3] = fmaf(ks1, acc[nt][3], score[nt][3]);
            }
        }

        // online softmax over the tile (log2 domain); cells past the list are masked
        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int c = tile * TILE + nt * 8 + 2 * lid;
            score[nt][0] = c < n ? score[nt][0] * qdown : -INFINITY;
            score[nt][1] = c + 1 < n ? score[nt][1] * qdown : -INFINITY;
            score[nt][2] = c < n ? score[nt][2] * qdown : -INFINITY;
            score[nt][3] = c + 1 < n ? score[nt][3] * qdown : -INFINITY;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
#pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, x));
            bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, x));
        }
        const float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
        const float al0 = m0 == -INFINITY ? 0.0f : ex2(m0 - nm0), al1 = m1 == -INFINITY ? 0.0f : ex2(m1 - nm1);
        m0 = nm0;
        m1 = nm1;
        float pr[2][4], bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = ex2(score[nt][0] - nm0);
            pr[nt][1] = ex2(score[nt][1] - nm0);
            pr[nt][2] = ex2(score[nt][2] - nm1);
            pr[nt][3] = ex2(score[nt][3] - nm1);
            bl0 += pr[nt][0] + pr[nt][1];
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = fmaf(l0, al0, bl0);
        l1 = fmaf(l1, al1, bl1);
        if (!__all_sync(0xffffffffu, al0 == 1.0f && al1 == 1.0f)) {
#pragma unroll
            for (int b = 0; b < 16; ++b)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    o[b][h][0] *= al0;
                    o[b][h][1] *= al0;
                    o[b][h][2] *= al1;
                    o[b][h][3] *= al1;
                }
        }

        // p.v: B = code * V scale in FP16. A tile whose largest scale exceeds 256 decodes with scales divided by
        // 2^shift and p multiplied by 2^shift (exact), keeping every product finite.
        {
            const __half2 pair = *reinterpret_cast<const __half2*>(vsc + 2 * lane);
            float vmax = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
#pragma unroll
            for (int x = 16; x > 0; x >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, x));
            int shift = 0;
            while (shift < 14 && ldexpf(vmax, -shift) > 256.0f) ++shift;
            const float up = ldexpf(1.0f, shift);
            uint32_t pa[4];
            pa[0] = pack_h2(pr[0][0] * up, pr[0][1] * up);
            pa[1] = pack_h2(pr[0][2] * up, pr[0][3] * up);
            pa[2] = pack_h2(pr[1][0] * up, pr[1][1] * up);
            pa[3] = pack_h2(pr[1][2] * up, pr[1][3] * up);
            const __half2 down = __float2half2_rn(ldexpf(1.0f, -shift));
            const int v_row = (lane & 7) + (((lane >> 3) & 1) << 3), v_mat2 = lane >> 4;
            const int key = 2 * lid;
#pragma unroll
            for (int db = 0; db < 16; db += 2) {
                uint32_t r[4];
                ldsm_x4_t(r, smem_addr(v_s + swz16(v_row, KROW, db + v_mat2)));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g = (db + h) >> 2;
                    const __half2 s01 = __hmul2(__halves2half2(vsc[key * 4 + g], vsc[(key + 1) * 4 + g]), down);
                    const __half2 s89 = __hmul2(__halves2half2(vsc[(key + 8) * 4 + g], vsc[(key + 9) * 4 + g]), down);
                    uint32_t e0, o0, e1, o1;
                    decode_v_pair(r[2 * h], s01, e0, o0);
                    decode_v_pair(r[2 * h + 1], s89, e1, o1);
                    const uint32_t be[2] = {e0, e1}, bo[2] = {o0, o1};
                    mma16816(o[db + h][0], pa, be);
                    mma16816(o[db + h][1], pa, bo);
                }
            }
        }
        __syncwarp();
    }
    cp_async_wait0();

    // merge the four partial softmaxes: row max and sum through smem, then every warp's rows, rescaled
    if (lid == 0) {
        S.mw[warp][gid] = m0;
        S.mw[warp][gid + 8] = m1;
    }
#pragma unroll
    for (int x = 1; x < 4; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    if (lid == 0) {
        S.lw[warp][gid] = l0;
        S.lw[warp][gid + 8] = l1;
    }
    __syncthreads();
    float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
        M0 = fmaxf(M0, S.mw[w][gid]);
        M1 = fmaxf(M1, S.mw[w][gid + 8]);
    }
    const float f0 = m0 == -INFINITY ? 0.0f : ex2(m0 - M0), f1 = m1 == -INFINITY ? 0.0f : ex2(m1 - M1);
    float* mrg = reinterpret_cast<float*>(&S.stage[0][0][0]);
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const int d = 16 * b + 4 * lid;
        float* r0 = mrg + ((size_t) warp * G + gid) * OSTRIDE + d;
        *reinterpret_cast<float4*>(r0) =
            make_float4(o[b][0][0] * f0, o[b][1][0] * f0, o[b][0][1] * f0, o[b][1][1] * f0);
        if (gid + 8 < G) {
            float* r1 = mrg + ((size_t) warp * G + gid + 8) * OSTRIDE + d;
            *reinterpret_cast<float4*>(r1) =
                make_float4(o[b][0][2] * f1, o[b][1][2] * f1, o[b][0][3] * f1, o[b][1][3] * f1);
        }
    }
    __syncthreads();
    for (int i = t; i < G * HD / 4; i += THREADS) {
        const int r = i / (HD / 4), d = (i % (HD / 4)) * 4;
        float L = 0.0f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        float Mr = -INFINITY;
#pragma unroll
        for (int w = 0; w < NWARP; ++w) Mr = fmaxf(Mr, S.mw[w][r]);
#pragma unroll
        for (int w = 0; w < NWARP; ++w) {
            const float mw = S.mw[w][r];
            L += mw == -INFINITY ? 0.0f : S.lw[w][r] * ex2(mw - Mr);
            const float4 v = *reinterpret_cast<const float4*>(mrg + ((size_t) w * G + r) * OSTRIDE + d);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        *reinterpret_cast<float4*>(attn + (size_t) r * HD + d) =
            make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
    }
}

bool launch_i8(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static std::atomic<bool> attr[64];   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem3);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(prompt_attn_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev].store(true, std::memory_order_release);
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_i8_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "qsa_prompt_attn_batch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

template <int KV_MODE>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static std::atomic<bool> attr[64];   // per device, as above
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(prompt_attn_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev].store(true, std::memory_order_release);
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "qsa_prompt_attn_batch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}


}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    cudaStream_t st = (cudaStream_t) stream;
    if (pools.k_q4 != nullptr) {   // Q4_0 K and V (--kv q4_0): mode 4
        if (pools.v_q4 == nullptr) return false;
        return launch<4>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}

}  // namespace basalt::kernels
