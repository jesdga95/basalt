// src/kernels/cpu/expert_neon.cpp - the Q2_0 expert kernels of expert.hpp on Arm64: NEON with the dot product.
//
// The x86-64 build is expert.cpp, which says why each piece is shaped as it is.  The contract is the same: the
// activation quantizer does the scalar loop's operations per element, and a row is, per 32-value chunk,
// d_w * (d_x * sum(c * xhat) - d_x * sum(xhat)) with the codes c = 0..3 against the int8 activations (SDOT), accumulated
// in float and reduced once per row.  A block's codes are unpacked once for every token of a multi-token call, and each
// token sees the single-token operations in the same order, so a token's rows do not depend on the batch.
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/log.hpp"

#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

/// A row of the expert blob: its codes contiguous, its fp16 scales in their own array.
struct BlobRow {
    const uint8_t* codes;
    const uint8_t* scales;
    const uint8_t* code(int b) const { return codes + (size_t) b * 16; }
    float d(int b) const { return h2f(scales + 2 * (size_t) b); }
};

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

inline float row_dot(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    const ActQ* pa = &a;
    float r;
    row_dot_multi<1>(BlobRow{codes, scales}, &pa, nblocks, &r);
    return r;
}

inline BlobRow gu_row(const uint8_t* blob, int r) {
    return BlobRow{blob + O_GU_CODES + (size_t) r * ROW_GU, blob + O_GU_SCALES + (size_t) r * SC_GU * 2};
}

inline BlobRow d_row(const uint8_t* blob, int r) {
    return BlobRow{blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2};
}

template <int NT>
void gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(gu_row(blob, 2 * r), a1, SC_GU, g);
        row_dot_multi<NT>(gu_row(blob, 2 * r + 1), a1, SC_GU, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
void down_rows_multi(const uint8_t* blob, const ActQ* const* a2, float* const* out, int r0, int r1) {
    float o[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(d_row(blob, r), a2, SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
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

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (f.usable()) return;
    basalt::log::error("basalt", "this CPU cannot run the expert kernel: %s.", f.reason());
    std::exit(1);
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

void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws) {
    act_quant_q8_1(x, H, ws.a1);
    s2_expert_vnni_q(blob, ws.a1, out, ws);
}

void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    s2_expert_gu_rows(blob, a1, ws.ff, 0, FF);
    act_quant_q8_1(ws.ff, FF, ws.a2);
    s2_expert_down_rows(blob, ws.a2, out, 0, H);
}

void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const BlobRow gr = gu_row(blob, 2 * r), ur = gu_row(blob, 2 * r + 1);
        const float g = row_dot(gr.codes, gr.scales, a1, SC_GU);
        const float u = row_dot(ur.codes, ur.scales, a1, SC_GU);
        // SiLU on the GATE, multiplied by up.
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

void s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const BlobRow dr = d_row(blob, r);
        out[r] = row_dot(dr.codes, dr.scales, a2, SC_D);
    }
}

void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                             int r1) {
    switch (n_tokens) {
        case 1: gu_rows_multi<1>(blob, a1, ff, r0, r1); break;
        case 2: gu_rows_multi<2>(blob, a1, ff, r0, r1); break;
        case 3: gu_rows_multi<3>(blob, a1, ff, r0, r1); break;
        case 4: gu_rows_multi<4>(blob, a1, ff, r0, r1); break;
        case 5: gu_rows_multi<5>(blob, a1, ff, r0, r1); break;
        case 6: gu_rows_multi<6>(blob, a1, ff, r0, r1); break;
        case 7: gu_rows_multi<7>(blob, a1, ff, r0, r1); break;
        default: gu_rows_multi<8>(blob, a1, ff, r0, r1); break;
    }
}

void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                               int r1) {
    switch (n_tokens) {
        case 1: down_rows_multi<1>(blob, a2, out, r0, r1); break;
        case 2: down_rows_multi<2>(blob, a2, out, r0, r1); break;
        case 3: down_rows_multi<3>(blob, a2, out, r0, r1); break;
        case 4: down_rows_multi<4>(blob, a2, out, r0, r1); break;
        case 5: down_rows_multi<5>(blob, a2, out, r0, r1); break;
        case 6: down_rows_multi<6>(blob, a2, out, r0, r1); break;
        case 7: down_rows_multi<7>(blob, a2, out, r0, r1); break;
        default: down_rows_multi<8>(blob, a2, out, r0, r1); break;
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

void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                          ExpertScratchMulti& ws) {
    if (n_tokens < 1 || n_tokens > MAXT) {
        for (int t = 0; t < n_tokens; ++t) s2_expert_vnni_q(blob, *a1[t], out[t], ws.single);
        return;
    }
    float* ff[MAXT];
    const ActQ* a2[MAXT];
    for (int t = 0; t < n_tokens; ++t) ff[t] = ws.ff[t];
    s2_expert_gu_rows_multi(blob, a1, n_tokens, ff, 0, FF);
    for (int t = 0; t < n_tokens; ++t) {
        act_quant_q8_1(ws.ff[t], FF, ws.a2[t]);
        a2[t] = &ws.a2[t];
    }
    s2_expert_down_rows_multi(blob, a2, n_tokens, out, 0, H);
}

void s2_expert_scalar(const uint8_t* blob, const float* x_in, float* out, bool quant_acts) {
    // The oracle of expert.cpp, unchanged: both stages consume the kernel's int8 activations when `quant_acts`.
    float xq[H];
    const float* x = x_in;
    if (quant_acts) {
        ActQ a1;
        act_quant_q8_1(x_in, H, a1);
        for (int i = 0; i < H; ++i) xq[i] = a1.scale[i / QKA] * (float) a1.q[i];
        x = xq;
    }
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = h2f(gs + 2 * b), du = h2f(us + 2 * b);
            for (int j = 0; j < QK; ++j) {
                const int o = b * QK + j;
                sg += (float) (((gc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * dg * x[o];
                su += (float) (((uc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * du * x[o];
            }
        }
        ff[r] = (sg / (1.f + std::exp(-sg))) * su;
    }
    if (quant_acts) {
        ActQ a2;
        act_quant_q8_1(ff, FF, a2);
        for (int i = 0; i < FF; ++i) ff[i] = a2.scale[i / QKA] * (float) a2.q[i];
    }
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = h2f(ds + 2 * b);
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * d * ff[b * QK + j];
        }
        out[r] = acc;
    }
}

}  // namespace basalt::kernels::cpu
