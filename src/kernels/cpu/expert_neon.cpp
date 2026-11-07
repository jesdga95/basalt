// src/kernels/cpu/expert_neon.cpp - the activation quantizer and the Q2_0 rows of expert.hpp on Arm64: NEON with the
// dot product.
//
// The x86-64 build is expert.cpp, which says why each piece is shaped as it is.  The contract is the same: the
// activation quantizer does the scalar loop's operations per element, and a row is, per 32-value chunk,
// d_w * (d_x * sum(c * xhat) - d_x * sum(xhat)) with the codes c = 0..3 against the int8 activations (SDOT), accumulated
// in float and reduced once per row.  A block's codes are unpacked once for every token of a multi-token call, and each
// token sees the single-token operations in the same order, so a token's rows do not depend on the batch.
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <arm_neon.h>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

/// fp16 -> fp32, written out (see expert.cpp).
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

/// The 64 codes of one block (16 bytes, code j in bits 2(j&3) of byte j/4): c[m] lane k holds code 4k + m.
inline void unpack64(const uint8_t* codes, int8x16_t c[4]) {
    const uint8x16_t b = vld1q_u8(codes);
    const uint8x16_t m3 = vdupq_n_u8(3);
    c[0] = vreinterpretq_s8_u8(vandq_u8(b, m3));
    c[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 2), m3));
    c[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 4), m3));
    c[3] = vreinterpretq_s8_u8(vshrq_n_u8(b, 6));
}

/// sum(c * xhat) of one block against its 64 activations, de-interleaved to the codes' order: lanes 0-1 hold the
/// first 32-value chunk, lanes 2-3 the second.
inline int32x4_t block_dot(const int8x16_t c[4], const int8_t* xq) {
    const int8x16x4_t a = vld4q_s8(xq);   // a.val[m] lane k = xq[4k + m]
    int32x4_t p = vdotq_s32(vdupq_n_s32(0), c[0], a.val[0]);
    p = vdotq_s32(p, c[1], a.val[1]);
    p = vdotq_s32(p, c[2], a.val[2]);
    return vdotq_s32(p, c[3], a.val[3]);
}

/// A row of GGUF Q2_0 blocks: the fp16 scale, then 16 code bytes, 18 bytes per 64 weights.
struct GgufRow {
    const uint8_t* row;
    const uint8_t* code(int b) const { return row + (size_t) b * 18 + 2; }
    float d(int b) const { return h2f(row + (size_t) b * 18); }
};

template <int NT, class Row>
inline void row_dot_multi(const Row& r, const ActQ* const* a, int nblocks, float* res) {
    float32x4_t acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = vdupq_n_f32(0.f);
        corr[t] = 0.f;
    }
    for (int b = 0; b < nblocks; ++b) {
        int8x16_t c[4];
        unpack64(r.code(b), c);
        const float dw = r.d(b);
        for (int t = 0; t < NT; ++t) {
            const int32x4_t p = block_dot(c, a[t]->q + b * QK);
            const float32x4_t sv = vcombine_f32(vdup_n_f32(dw * a[t]->scale[2 * b]), vdup_n_f32(dw * a[t]->scale[2 * b + 1]));
            acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(p), sv);
            corr[t] += dw * a[t]->hx[2 * b] + dw * a[t]->hx[2 * b + 1];
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = vaddvq_f32(acc[t]) - corr[t];
}

template <int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(GgufRow{w + (size_t) r * row_bytes}, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

const char* CpuFeatures::reason() const { return usable() ? "ok" : "missing the Armv8.2 dot product"; }

CpuFeatures cpu_features() {
    CpuFeatures f;
    f.arm_dotprod = cpu_neon_dot_ok();
    return f;
}

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    // The scalar loop of expert.cpp, four lanes at a time: max of |x|, one multiply, +-0.5 away from zero,
    // truncation, clamp.
    const float32x4_t half = vdupq_n_f32(0.5f), mhalf = vdupq_n_f32(-0.5f), zero = vdupq_n_f32(0.f);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float32x4_t v[8];
        for (int j = 0; j < 8; ++j) v[j] = vld1q_f32(xb + 4 * j);
        float32x4_t m = vabsq_f32(v[0]);
        for (int j = 1; j < 8; ++j) m = vmaxq_f32(m, vabsq_f32(v[j]));
        const float amax = vmaxvq_f32(m);
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32x4_t sum = vdupq_n_s32(0);
        for (int j = 0; j < 8; j += 2) {
            int32x4_t q[2];
            for (int h = 0; h < 2; ++h) {
                const float32x4_t t = vmulq_n_f32(v[j + h], inv);
                const float32x4_t r = vaddq_f32(t, vbslq_f32(vcgeq_f32(t, zero), half, mhalf));
                q[h] = vminq_s32(vmaxq_s32(vcvtq_s32_f32(r), lo), hi);
                sum = vaddq_s32(sum, q[h]);
            }
            vst1_s8(a.q + k * QKA + 4 * j, vmovn_s16(vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]))));
        }
        const int32_t isum = vaddvq_s32(sum);
        a.scale[k] = s;
        a.sum[k] = isum;
        a.hx[k] = s * (float) isum;
    }
}

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
    }
}

}  // namespace basalt::kernels::cpu
