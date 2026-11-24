// SPDX-License-Identifier: MIT
// MXFP8 (E4M3 codes, UE8M0 sub-block scales) row . Q8_0 row on NEON: Basalt's own, the Arm64 counterpart of the MXFP8
// rows in ggml_cpu_dots.c. Included by ggml_cpu_dots_neon.c.
//
// Every row here gives the bits of the table-lookup scalar row below. An E4M3 code is its fp16 bit pattern shifted,
// ((c & 0x7F) << 7) | ((c & 0x80) << 8), which is the fp16 of value / 256 (the subnormals included), so one fcvt decodes
// four weights; the 2^8 goes back in exactly. E4M3's NaN code (0x7F, 0xFF) is 0 in the table, so its seven low bits are
// cleared first. A sub-block's four lane sums run over j = lane, lane + 4, ... in turn (one fused multiply-add per
// step: the products are exact in fp32), combine as (l0 + l1) + (l2 + l3), and (fp16 scale * UE8M0 scale) * sum is added
// to the row sum, sub-block by sub-block.

static const float k_mxfp8_e4m3[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -0.0f
};

static inline float mxfp8_ue8m0_to_fp32(uint8_t u) {
    if (u == 0xFFu) return 0.0f;
    const uint32_t bits = u == 0u ? 0x00400000u : ((uint32_t) u << 23);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// The scalar row, the table-lookup reference of the NEON rows below.  Each sub-block's four lane sums run over
// j = lane, lane + 4, ... in turn (the products are exact in fp32, so a fused and a separate multiply-add agree),
// they combine as (l0 + l1) + (l2 + l3), and (fp16 scale * UE8M0 scale) * sum is added to the row sum, sub-block
// by sub-block.  The build gives it its own name (CMakeLists.txt) so the unit test can call it beside the vector row.
void ggml_vec_dot_mxfp8_q8_0_scalar(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy);
void ggml_vec_dot_mxfp8_q8_0_scalar(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    assert(n % QK_MXFP8 == 0);
    const block_mxfp8 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    const int nb = n / QK_MXFP8;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        for (int sb = 0; sb < QK_MXFP8 / QK_MXFP8_SUB; ++sb) {
            const block_q8_0 * yb = y + 4 * ib + sb;
            const uint8_t * q = x[ib].qs + sb * QK_MXFP8_SUB;
            const int8_t * a = yb->qs;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (int j = 0; j < QK_MXFP8_SUB; j += 4) {
                d0 += k_mxfp8_e4m3[q[j + 0]] * (float) a[j + 0];
                d1 += k_mxfp8_e4m3[q[j + 1]] * (float) a[j + 1];
                d2 += k_mxfp8_e4m3[q[j + 2]] * (float) a[j + 2];
                d3 += k_mxfp8_e4m3[q[j + 3]] * (float) a[j + 3];
            }
            sumf += GGML_CPU_FP16_TO_FP32(yb->d) * mxfp8_ue8m0_to_fp32(x[ib].d[sb]) * ((d0 + d1) + (d2 + d3));
        }
    }
    *s = sumf;
}

#define MXFP8_PREP_BLOCK 144
#define MXFP8_MAX_TOK 8

// The activation of a row set is converted once ("prepped"): per 128 values, the four Q8_0 blocks' values as 128 floats
// in order, then the four fp16 scales as floats, padded to MXFP8_PREP_BLOCK floats.
static inline void mxfp8_prep_block(const block_q8_0 * yb, float * o) {
    for (int sb = 0; sb < 4; ++sb) {
        for (int h = 0; h < 2; ++h) {
            const int8x16_t q = vld1q_s8(yb[sb].qs + 16 * h);
            const int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
            float * f = o + 32 * sb + 16 * h;
            vst1q_f32(f + 0,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
            vst1q_f32(f + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
            vst1q_f32(f + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
            vst1q_f32(f + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
        }
        o[128 + sb] = GGML_CPU_FP16_TO_FP32(yb[sb].d);
    }
    for (int i = 132; i < MXFP8_PREP_BLOCK; ++i) o[i] = 0.0f;
}

// Eight E4M3 codes as floats: [0] the first four, [1] the next four.
static inline void mxfp8_decode8(const uint8_t * q, float32x4_t * w) {
    const uint8x8_t c = vld1_u8(q);
    const uint8x8_t nan = vceq_u8(vand_u8(c, vdup_n_u8(0x7F)), vdup_n_u8(0x7F));
    const uint16x8_t u = vmovl_u8(vbsl_u8(nan, vand_u8(c, vdup_n_u8(0x80)), c));
    const uint16x8_t h = vorrq_u16(vshlq_n_u16(vandq_u16(u, vdupq_n_u16(0x7F)), 7), vshlq_n_u16(vandq_u16(u, vdupq_n_u16(0x80)), 8));
    w[0] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))), 256.0f);
    w[1] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))), 256.0f);
}

// A sub-block's sum: the lane sums of its 32 weights against 32 prepped activation floats, combined as the scalar row does.
static inline float mxfp8_sub_sum(const float32x4_t * w, const float * a) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int k = 0; k < 8; ++k) acc = vfmaq_f32(acc, w[k], vld1q_f32(a + 4 * k));
    return (vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1)) + (vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3));
}

// The prepped activation of one Q8_0 row (n a multiple of 128): n / 128 * MXFP8_PREP_BLOCK floats.
void ggml_mxfp8_q8_0_prep(int n, const void * GGML_RESTRICT vy, float * GGML_RESTRICT out);
void ggml_mxfp8_q8_0_prep(int n, const void * GGML_RESTRICT vy, float * GGML_RESTRICT out) {
    assert(n % QK_MXFP8 == 0);
    const block_q8_0 * y = vy;
    for (int ib = 0; ib < n / QK_MXFP8; ++ib) mxfp8_prep_block(y + 4 * ib, out + (size_t) ib * MXFP8_PREP_BLOCK);
}

// out[t][r] for rows r in [r0, r1) of the weight rows `stride` bytes apart, `nt` <= MXFP8_MAX_TOK prepped activations.
// A weight sub-block is decoded once and meets every token's prepped values: a token's result is the single-token
// dot's bits whatever the number of tokens.
void ggml_mxfp8_q8_0_rows(int n, const void * GGML_RESTRICT vw, size_t stride, const float * const * prep, int nt, float * const * out, int r0, int r1);
void ggml_mxfp8_q8_0_rows(int n, const void * GGML_RESTRICT vw, size_t stride, const float * const * prep, int nt, float * const * out, int r0, int r1) {
    assert(n % QK_MXFP8 == 0 && nt <= MXFP8_MAX_TOK);
    const int nb = n / QK_MXFP8;
    for (int r = r0; r < r1; ++r) {
        const block_mxfp8 * x = (const block_mxfp8 *) ((const uint8_t *) vw + (size_t) r * stride);
        float sumf[MXFP8_MAX_TOK] = {0};
        for (int ib = 0; ib < nb; ++ib) {
            for (int sb = 0; sb < 4; ++sb) {
                float32x4_t w[8];
                for (int k = 0; k < 4; ++k) mxfp8_decode8(x[ib].qs + 32 * sb + 8 * k, w + 2 * k);
                const float ue = mxfp8_ue8m0_to_fp32(x[ib].d[sb]);
                for (int t = 0; t < nt; ++t) {
                    const float * p = prep[t] + (size_t) ib * MXFP8_PREP_BLOCK;
                    sumf[t] += p[128 + sb] * ue * mxfp8_sub_sum(w, p + 32 * sb);
                }
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = sumf[t];
    }
}

static void ggml_vec_dot_mxfp8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP8 == 0);
    const block_mxfp8 * x = vx;
    const block_q8_0 * y = vy;
    const int nb = n / QK_MXFP8;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        float p[MXFP8_PREP_BLOCK];
        mxfp8_prep_block(y + 4 * ib, p);
        for (int sb = 0; sb < 4; ++sb) {
            float32x4_t w[8];
            for (int k = 0; k < 4; ++k) mxfp8_decode8(x[ib].qs + 32 * sb + 8 * k, w + 2 * k);
            sumf += p[128 + sb] * mxfp8_ue8m0_to_fp32(x[ib].d[sb]) * mxfp8_sub_sum(w, p + 32 * sb);
        }
    }
    *s = sumf;
}
