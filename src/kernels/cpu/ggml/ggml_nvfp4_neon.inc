// SPDX-License-Identifier: MIT
// NVFP4 row . Q8_0 row on NEON with the dot product: Basalt's own, the Arm64 counterpart of the NVFP4 kernel in
// ggml_cpu_dots.c.  Included by ggml_cpu_dots_neon.c.

// UE4M3 scale as ggml decodes it (half the value, kvalues_fp4 being doubled; 0x7F as 0; bit 7 ignored), built from
// its bits: exact, and the same as dequantize_row_nvfp4.
static inline float nvfp4_scale(uint8_t x) {
    const uint32_t v = x & 0x7F;
    if (v == 0 || v == 0x7F) return 0.0f;
    const uint32_t e = v >> 3, m = v & 7;
    if (e == 0) return (float) m * (1.0f / 1024.0f);
    const uint32_t bits = ((e + 119) << 23) | (m << 20);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// NVFP4 x Q8_0: a 64-value block meets two Q8_0 blocks; sub-block s (16 values) has scale d[s] and Q8_0 block s/2.
// The codes of sub-block s are qs[8s..8s+7], low nibbles first.  Integer sums per sub-block, scaled in float: the
// accumulator's four lanes are the four sub-blocks.
static void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);
    static_assert(QK_NVFP4 == 2 * QK8_0, "an NVFP4 block spans two Q8_0 blocks");

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0  * GGML_RESTRICT y = vy;
    const int nb = n / QK_NVFP4;

    const int8x16_t values = vld1q_s8(kvalues_fp4);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int ib = 0; ib < nb; ++ib) {
        const uint8x16_t q01 = vld1q_u8(x[ib].qs);        // the code bytes of sub-blocks 0 and 1, eight each
        const uint8x16_t q23 = vld1q_u8(x[ib].qs + 16);   // and of 2 and 3
        // per half: [values 0..7 of one sub-block] from the low nibbles, [values 8..15] from the high ones
        const int8x16_t lo01 = vqtbl1q_s8(values, vandq_u8(q01, m4b));
        const int8x16_t hi01 = vqtbl1q_s8(values, vshrq_n_u8(q01, 4));
        const int8x16_t lo23 = vqtbl1q_s8(values, vandq_u8(q23, m4b));
        const int8x16_t hi23 = vqtbl1q_s8(values, vshrq_n_u8(q23, 4));
        const int8x16_t w0 = vcombine_s8(vget_low_s8(lo01), vget_low_s8(hi01));
        const int8x16_t w1 = vcombine_s8(vget_high_s8(lo01), vget_high_s8(hi01));
        const int8x16_t w2 = vcombine_s8(vget_low_s8(lo23), vget_low_s8(hi23));
        const int8x16_t w3 = vcombine_s8(vget_high_s8(lo23), vget_high_s8(hi23));
        const int32x4_t p0 = vdotq_s32(zero, w0, vld1q_s8(y[2*ib].qs));
        const int32x4_t p1 = vdotq_s32(zero, w1, vld1q_s8(y[2*ib].qs + 16));
        const int32x4_t p2 = vdotq_s32(zero, w2, vld1q_s8(y[2*ib + 1].qs));
        const int32x4_t p3 = vdotq_s32(zero, w3, vld1q_s8(y[2*ib + 1].qs + 16));
        // one integer per sub-block, in order
        const int32x4_t sumi = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        const float d0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d), d1 = GGML_CPU_FP16_TO_FP32(y[2*ib + 1].d);
        const float sc[4] = {d0 * nvfp4_scale(x[ib].d[0]), d0 * nvfp4_scale(x[ib].d[1]),
                             d1 * nvfp4_scale(x[ib].d[2]), d1 * nvfp4_scale(x[ib].d[3])};
        acc = vfmaq_f32(acc, vcvtq_f32_s32(sumi), vld1q_f32(sc));
    }
    *s = vaddvq_f32(acc);
}
