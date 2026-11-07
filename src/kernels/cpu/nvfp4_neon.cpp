// src/kernels/cpu/nvfp4_neon.cpp - NVFP4 expert rows against Q8_0 activations, several tokens at once, in NEON with
// the dot product.
//
// The ggml dot (ggml_vec_dot_nvfp4_q8_0 in ggml_cpu_dots_neon.c) is single-token: every token and row re-decodes the
// codes and the four UE4M3 scales of each block, and regroups the Q8_0 values to the codes' order.  Here each
// token's Q8_0 values are regrouped once per call, a block of codes and scales is decoded once per row, and every
// token applies it with four dot products, a pairwise add and an fma.  Per token and row the arithmetic is the ggml
// dot's: the same int32 sums per sub-block, the same fp32 products d_y * scale, the same accumulator and the same
// final reduction, so results are bitwise equal.
#include "basalt/kernels/cpu/nvfp4_neon.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <arm_neon.h>

#include <bit>
#include <cmath>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

// UE4M3 scale as the ggml dot decodes it (nvfp4_scale in ggml_cpu_dots_neon.c), one entry per 7-bit code
struct ScaleLut {
    float v[128];
    constexpr ScaleLut() : v{} {
        for (uint32_t x = 0; x < 128; ++x) {
            const uint32_t e = x >> 3, m = x & 7;
            if (x == 0 || x == 0x7F) v[x] = 0.0f;
            else if (e == 0) v[x] = (float) m * (1.0f / 1024.0f);
            else v[x] = std::bit_cast<float>((uint32_t) (((e + 119) << 23) | (m << 20)));
        }
    }
};
constexpr ScaleLut kLut{};

constexpr int kMaxN = 16384;
constexpr int kMaxT = 4;   // tokens per pass

// One token's Q8_0 values in the codes' order, four vectors per NVFP4 block (the ggml dot's q8_lo_0, q8_hi_0,
// q8_lo_1, q8_hi_1), and its two Q8_0 scales per block as {d0, d0, d1, d1}.
struct TokenBuf {
    int8x16_t q[kMaxN / QK_NVFP4 * 4];
    float32x4_t d[kMaxN / QK_NVFP4];
};

TokenBuf* token_bufs() {
    alignas(16) static thread_local TokenBuf buf[kMaxT];
    return buf;
}

template <int NT>
inline void regroup(const block_q8_0* const* y, int nb, TokenBuf* tb) {
    for (int t = 0; t < NT; ++t)
        for (int ib = 0; ib < nb; ++ib) {
            const block_q8_0* b = y[t] + 2 * ib;
            const int8x16_t a0 = vld1q_s8(b[0].qs), a1 = vld1q_s8(b[0].qs + 16);
            const int8x16_t c0 = vld1q_s8(b[1].qs), c1 = vld1q_s8(b[1].qs + 16);
            int8x16_t* q = tb[t].q + 4 * ib;
            q[0] = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
            q[1] = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
            q[2] = vcombine_s8(vget_low_s8(c0), vget_low_s8(c1));
            q[3] = vcombine_s8(vget_high_s8(c0), vget_high_s8(c1));
            const uint64_t h0 = b[0].d, h1 = b[1].d;
            tb[t].d[ib] = vcvt_f32_f16(vreinterpret_f16_u16(vcreate_u16(h0 | h0 << 16 | h1 << 32 | h1 << 48)));
        }
}

// R rows (each its own block pointer) against NT regrouped tokens: res[k][t]
template <int R, int NT>
inline void rows_dot(const block_nvfp4* const* w, int nb, const TokenBuf* tb, float (*res)[kMaxT]) {
    const int8x16_t values = vld1q_s8(kvalues_fp4);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) acc[k][t] = vdupq_n_f32(0.0f);
    for (int ib = 0; ib < nb; ++ib) {
        int8x16_t wl0[R], wh0[R], wl1[R], wh1[R];
        float32x4_t ws[R];
        for (int k = 0; k < R; ++k) {
            const block_nvfp4* x = w[k] + ib;
            const uint8x16_t b0 = vld1q_u8(x->qs), b1 = vld1q_u8(x->qs + 16);
            wl0[k] = vqtbl1q_s8(values, vandq_u8(b0, m4b));
            wh0[k] = vqtbl1q_s8(values, vshrq_n_u8(b0, 4));
            wl1[k] = vqtbl1q_s8(values, vandq_u8(b1, m4b));
            wh1[k] = vqtbl1q_s8(values, vshrq_n_u8(b1, 4));
            uint32_t d;
            std::memcpy(&d, x->d, 4);
            float32x4_t s = vdupq_n_f32(kLut.v[d & 0x7F]);
            s = vsetq_lane_f32(kLut.v[(d >> 8) & 0x7F], s, 1);
            s = vsetq_lane_f32(kLut.v[(d >> 16) & 0x7F], s, 2);
            s = vsetq_lane_f32(kLut.v[(d >> 24) & 0x7F], s, 3);
            ws[k] = s;
        }
        for (int t = 0; t < NT; ++t) {
            const int8x16_t* q = tb[t].q + 4 * ib;
            const int8x16_t ql0 = q[0], qh0 = q[1], ql1 = q[2], qh1 = q[3];
            const float32x4_t dy = tb[t].d[ib];
            for (int k = 0; k < R; ++k) {
                const int32x4_t p0 = vdotq_s32(vdotq_s32(zero, wl0[k], ql0), wh0[k], qh0);
                const int32x4_t p1 = vdotq_s32(vdotq_s32(zero, wl1[k], ql1), wh1[k], qh1);
                acc[k][t] = vfmaq_f32(acc[k][t], vcvtq_f32_s32(vpaddq_s32(p0, p1)), vmulq_f32(dy, ws[k]));
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) res[k][t] = vaddvq_f32(acc[k][t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, const void* const* act,
             float* const* ff, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int nb = n / QK_NVFP4;
    TokenBuf* tb = token_bufs();
    regroup<NT>(y, nb, tb);
    float res[2][kMaxT];
    for (int r = r0; r < r1; ++r) {
        const block_nvfp4* w[2] = {(const block_nvfp4*) (blob + (size_t) r * gu_row),
                                   (const block_nvfp4*) (blob + up_off + (size_t) r * gu_row)};
        rows_dot<2, NT>(w, nb, tb, res);
        for (int t = 0; t < NT; ++t) {
            const float g = res[0][t] * sg, u = res[1][t] * su;
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

template <int NT>
void down_rows(const uint8_t* w0, size_t row_bytes, int n, float sd, const void* const* act, float* const* out, int r0,
               int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int nb = n / QK_NVFP4;
    TokenBuf* tb = token_bufs();
    regroup<NT>(y, nb, tb);
    float res[2][kMaxT];
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const block_nvfp4* w[2] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes),
                                   (const block_nvfp4*) (w0 + (size_t) (r + 1) * row_bytes)};
        rows_dot<2, NT>(w, nb, tb, res);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = res[0][t] * sd;
            out[t][r + 1] = res[1][t] * sd;
        }
    }
    if (r < r1) {
        const block_nvfp4* w[1] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes)};
        rows_dot<1, NT>(w, nb, tb, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t] * sd;
    }
}

}  // namespace

bool nvfp4_neon_fits(int n_embd, int n_ff) noexcept {
    return n_embd % QK_NVFP4 == 0 && n_ff % QK_NVFP4 == 0 && n_embd <= kMaxN && n_ff <= kMaxN;
}

void nvfp4_neon_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                        const void* const* act, int nt, float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const void* const* a = act + t0;
        float* const* o = ff + t0;
        switch (nt - t0 < kMaxT ? nt - t0 : kMaxT) {
            case 1: gu_rows<1>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            default: gu_rows<4>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
        }
    }
}

void nvfp4_neon_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt,
                          float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const void* const* a = act + t0;
        float* const* o = out + t0;
        switch (nt - t0 < kMaxT ? nt - t0 : kMaxT) {
            case 1: down_rows<1>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 2: down_rows<2>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 3: down_rows<3>(w, row_bytes, n, sd, a, o, r0, r1); break;
            default: down_rows<4>(w, row_bytes, n, sd, a, o, r0, r1); break;
        }
    }
}

}  // namespace basalt::kernels::cpu
