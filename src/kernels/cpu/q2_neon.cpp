// src/kernels/cpu/q2_neon.cpp - the Q2_0 expert rows, the activation quantizer and the router lookahead's dot on
// Arm64 NEON with the dot product.
//
// The arithmetic is the x86 kernels' (q2_avx2_rows.inc): codes 0..3 against the int8 activation per 32-value chunk,
// times the weight scale and the chunk scale, minus the weight scale times the chunk's `hx` (the -1 code offset).
// The integer sums are exact; the float accumulator has four lanes here against eight and sixteen there, so a row's
// last bits are this file's own. A token's row is computed by the same instructions in the same order whatever the
// window width. The quantizer is the scalar rule, bit for bit.
#include "basalt/kernels/cpu/neon.hpp"

#include "basalt/platform/compiler.hpp"

#include <cmath>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

// byte i/4 of the input in lane i, and the right shift that brings code i%4 down
alignas(16) constexpr uint8_t kRep[4][16] = {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
                                             {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7},
                                             {8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11},
                                             {12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15}};
alignas(16) constexpr int8_t kShift[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, 16 per vector
inline void unpack64(const uint8_t* codes, int8x16_t c[4]) {
    const uint8x16_t b = vld1q_u8(codes);
    const int8x16_t sh = vld1q_s8(kShift);
    const uint8x16_t m3 = vdupq_n_u8(3);
    for (int g = 0; g < 4; ++g)
        c[g] = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(b, vld1q_u8(kRep[g])), sh), m3));
}

// hxs[t * MAXC / 2 + b]: the block's two chunk corrections summed, once per call instead of once per row
template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, const float* hxs, int nblocks, float* res) {
    float32x4_t acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = vdupq_n_f32(0.f);
        corr[t] = 0.f;
    }
    const int32x4_t zero = vdupq_n_s32(0);
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        int8x16_t c[4];
        unpack64(blk + 2, c);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const int32x4_t s0 = vdotq_s32(vdotq_s32(zero, c[0], vld1q_s8(q)), c[1], vld1q_s8(q + 16));
            const int32x4_t s1 = vdotq_s32(vdotq_s32(zero, c[2], vld1q_s8(q + 32)), c[3], vld1q_s8(q + 48));
            acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(s0), d * a[t]->scale[2 * b]);
            acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(s1), d * a[t]->scale[2 * b + 1]);
            corr[t] += d * hxs[t * (MAXC / 2) + b];
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = vaddvq_f32(acc[t]) - corr[t];
}

// an ActQ holds at most MAXC chunks, so nblocks <= MAXC / 2
template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float hxs[NT * (MAXC / 2)];
    for (int t = 0; t < NT; ++t)
        for (int b = 0; b < nblocks; ++b) hxs[t * (MAXC / 2) + b] = a[t]->hx[2 * b] + a[t]->hx[2 * b + 1];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, hxs, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void q2_0_gguf_rows_multi_neon(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_neon(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void act_quant_q8_1_neon(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const float32x4_t half = vdupq_n_f32(0.5f), mhalf = vdupq_n_f32(-0.5f), zero = vdupq_n_f32(0.f);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float32x4_t v[8];
        float32x4_t m = vdupq_n_f32(0.f);
        for (int i = 0; i < 8; ++i) {
            v[i] = vld1q_f32(xb + 4 * i);
            m = vmaxq_f32(m, vabsq_f32(v[i]));
        }
        const float amax = vmaxvq_f32(m);
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32x4_t sum = vdupq_n_s32(0);
        int8_t* q = a.q + k * QKA;
        for (int i = 0; i < 8; i += 4) {
            int32x4_t qi[4];
            for (int j = 0; j < 4; ++j) {
                // t + copysign(0.5, t), then truncation: round half away from zero (the scalar rule, not fcvtas on t,
                // which would round the same but is kept out so the two steps stay the scalar loop's two steps)
                const float32x4_t t = vmulq_n_f32(v[i + j], inv);
                const float32x4_t r = vaddq_f32(t, vbslq_f32(vcgeq_f32(t, zero), half, mhalf));
                qi[j] = vminq_s32(vmaxq_s32(vcvtq_s32_f32(r), lo), hi);
                sum = vaddq_s32(sum, qi[j]);
            }
            const int16x8_t h0 = vcombine_s16(vmovn_s32(qi[0]), vmovn_s32(qi[1]));
            const int16x8_t h1 = vcombine_s16(vmovn_s32(qi[2]), vmovn_s32(qi[3]));
            vst1q_s8(q + 4 * i, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
        }
        const int32_t total = vaddvq_s32(sum);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

void router_q4_dot_neon(const uint8_t* row, const float* scales, int nb, const uint8_t* xq, int nt, float* out) {
    const uint8x16_t lo4 = vdupq_n_u8(0x0f);
    const int8x16_t off = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = vdupq_n_f32(0.f);
    for (int b = 0; b < nb; ++b) {
        const uint8x16_t q = vld1q_u8(row + b * 16);
        const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, lo4)), off);   // values 0..15
        const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), off);   // values 16..31
        const float sw = scales[b];
        for (int t = 0; t < nt; ++t) {
            const uint8_t* blk = xq + ((size_t) t * nb + b) * 36;   // block_q8_1: half d, half s, 32 x int8
            const int32x4_t p = vdotq_s32(vdotq_s32(zero, wl, vld1q_s8((const int8_t*) blk + 4)), wh,
                                          vld1q_s8((const int8_t*) blk + 20));
            acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(p), sw * h2f(blk));
        }
    }
    for (int t = 0; t < nt; ++t) out[t] = vaddvq_f32(acc[t]);
}

}  // namespace basalt::kernels::cpu
