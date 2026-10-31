// src/kernels/cpu/nvfp4_avx2_rows.inc - NVFP4 expert rows against Q8_0 activations, several tokens at once, in 256-bit
// lanes.
//
// Included by nvfp4_avx2.cpp (BASALT_NVFP4_VNNI_INT8 0: `vpsignb` + `vpmaddubsw` + `vpmaddwd`) and nvfp4_avx2_vnni.cpp
// (BASALT_NVFP4_VNNI_INT8 1: AVX-VNNI-INT8 `vpdpbssd` directly on the signed bytes), which name the exported functions
// through BASALT_NVFP4_EXPORT.
//
// The ggml dot (ggml_vec_dot_nvfp4_q8_0, AVX2 branch) is single-token: every token re-decodes the codes, the sign
// pattern and the four UE4M3 scales of each block.  Here a block is decoded once and every token applies it.  Per token
// and row the arithmetic is the ggml dot's: the same int32 sums per sub-block, the same fp32 products d_y * scale, the
// same two accumulators and the same final reduction, so results are bitwise equal.

namespace basalt::kernels::cpu {
namespace {

// UE4M3 scale as ggml's dot decodes it (nvfp4_scale in ggml_cpu_dots.c), one entry per 7-bit code.  Built at compile
// time: this file is compiled with vector extensions, and a static constructor here would run before the CPU is checked.
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
constexpr int kMaxT = 8;

// Pre-unpacked Q8_0 activation pair for one NVFP4 block (64 weights = 2 Q8_0 blocks): sub-blocks (0, 2) in ya,
// sub-blocks (1, 3) in yb, and Q8_0 scales (d0 on lanes 0..3, d1 on lanes 4..7) in d.
struct alignas(32) TokBlk {
    __m256i ya, yb;
    __m256 d;
};

TokBlk* tok_buf() {
    alignas(32) static thread_local TokBlk buf[kMaxT * kMaxN / 64];
    return buf;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// the same reduction order as ggml's hsum_float_8
inline float hsum8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// One decoded weight block: sub-blocks (0, 2) and (1, 3) and the matching scales.
struct Blk {
#if !BASALT_NVFP4_VNNI_INT8
    __m256i aa, ab;
#endif
    __m256i wa, wb;
    __m256 sa, sb;
};

inline Blk decode(const block_nvfp4* x) {
    const __m128i v128 = _mm_loadu_si128((const __m128i*) kvalues_fp4);
    const __m256i values = _mm256_inserti128_si256(_mm256_castsi128_si256(v128), v128, 1);
    const __m256i m4b = _mm256_set1_epi8(0x0f);
    const __m256i q = _mm256_loadu_si256((const __m256i*) x->qs);
    const __m256i lo = _mm256_and_si256(q, m4b);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4b);
    Blk b;
    b.wa = _mm256_shuffle_epi8(values, _mm256_unpacklo_epi64(lo, hi));
    b.wb = _mm256_shuffle_epi8(values, _mm256_unpackhi_epi64(lo, hi));
#if !BASALT_NVFP4_VNNI_INT8
    b.aa = _mm256_sign_epi8(b.wa, b.wa);
    b.ab = _mm256_sign_epi8(b.wb, b.wb);
#endif
    const float s0 = kLut.v[x->d[0] & 0x7F], s1 = kLut.v[x->d[1] & 0x7F], s2 = kLut.v[x->d[2] & 0x7F],
                s3 = kLut.v[x->d[3] & 0x7F];
    b.sa = _mm256_setr_ps(s0, s0, s0, s0, s2, s2, s2, s2);
    b.sb = _mm256_setr_ps(s1, s1, s1, s1, s3, s3, s3, s3);
    return b;
}

template <int NT>
inline void prep_tokens(const block_q8_0* const* y, int nb, TokBlk* tb) {
    for (int ib = 0; ib < nb; ++ib) {
        for (int t = 0; t < NT; ++t) {
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][2 * ib].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][2 * ib + 1].qs);
            TokBlk& dst = tb[(size_t) ib * NT + t];
            dst.ya = _mm256_permute2x128_si256(y0, y1, 0x20);
            dst.yb = _mm256_permute2x128_si256(y0, y1, 0x31);
            dst.d = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(h2f(y[t][2 * ib].d))),
                                         _mm_set1_ps(h2f(y[t][2 * ib + 1].d)), 1);
        }
    }
}

// R rows (each its own block pointer) against NT tokens: res[k][t].
template <int R, int NT>
inline void rows_dot(const block_nvfp4* const* w, int nb, const TokBlk* tb, float (*res)[kMaxT]) {
#if !BASALT_NVFP4_VNNI_INT8
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    __m256 acc1[R][NT], acc2[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) acc1[k][t] = acc2[k][t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        Blk b[R];
        for (int k = 0; k < R; ++k) b[k] = decode(w[k] + ib);
        const TokBlk* tk = tb + (size_t) ib * NT;
        for (int t = 0; t < NT; ++t) {
            const __m256i ya = tk[t].ya;
            const __m256i yb = tk[t].yb;
            const __m256 d = tk[t].d;
            for (int k = 0; k < R; ++k) {
#if BASALT_NVFP4_VNNI_INT8
                const __m256i pa = _mm256_dpbssd_epi32(_mm256_setzero_si256(), b[k].wa, ya);
                const __m256i pb = _mm256_dpbssd_epi32(_mm256_setzero_si256(), b[k].wb, yb);
#else
                const __m256i pa = _mm256_madd_epi16(_mm256_maddubs_epi16(b[k].aa, _mm256_sign_epi8(ya, b[k].wa)), ones);
                const __m256i pb = _mm256_madd_epi16(_mm256_maddubs_epi16(b[k].ab, _mm256_sign_epi8(yb, b[k].wb)), ones);
#endif
                acc1[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(d, b[k].sa), _mm256_cvtepi32_ps(pa), acc1[k][t]);
                acc2[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(d, b[k].sb), _mm256_cvtepi32_ps(pb), acc2[k][t]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) res[k][t] = hsum8(_mm256_add_ps(acc1[k][t], acc2[k][t]));
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, const void* const* act,
             float* const* ff, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int nb = n / QK_NVFP4;
    TokBlk* tb = tok_buf();
    prep_tokens<NT>(y, nb, tb);
    float res[2][kMaxT];
    for (int r = r0; r < r1; ++r) {
        const block_nvfp4* w[2] = {(const block_nvfp4*) (blob + (size_t) r * gu_row),
                                   (const block_nvfp4*) (blob + up_off + (size_t) r * gu_row)};
        if constexpr (NT <= 2) {
            rows_dot<2, NT>(w, nb, tb, res);
        } else {
            rows_dot<1, NT>(&w[0], nb, tb, &res[0]);
            rows_dot<1, NT>(&w[1], nb, tb, &res[1]);
        }
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
    TokBlk* tb = tok_buf();
    prep_tokens<NT>(y, nb, tb);
    float res[2][kMaxT];
    int r = r0;
    if constexpr (NT <= 2) {
        for (; r + 2 <= r1; r += 2) {
            const block_nvfp4* w[2] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes),
                                       (const block_nvfp4*) (w0 + (size_t) (r + 1) * row_bytes)};
            rows_dot<2, NT>(w, nb, tb, res);
            for (int t = 0; t < NT; ++t) {
                out[t][r] = res[0][t] * sd;
                out[t][r + 1] = res[1][t] * sd;
            }
        }
    }
    for (; r < r1; ++r) {
        const block_nvfp4* w[1] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes)};
        rows_dot<1, NT>(w, nb, tb, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t] * sd;
    }
}

}  // namespace

void BASALT_NVFP4_EXPORT(nvfp4_256_gu_rows)(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg,
                                            float su, const void* const* act, int nt, float* const* ff, int r0,
                                            int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = ff + t0;
        switch (k) {
            case 1: gu_rows<1>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
            case 8: gu_rows<8>(blob, gu_row, up_off, n, sg, su, a, o, r0, r1); break;
        }
    }
}

void BASALT_NVFP4_EXPORT(nvfp4_256_down_rows)(const uint8_t* w, size_t row_bytes, int n, float sd,
                                              const void* const* act, int nt, float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = out + t0;
        switch (k) {
            case 1: down_rows<1>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 2: down_rows<2>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 3: down_rows<3>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 4: down_rows<4>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 5: down_rows<5>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 6: down_rows<6>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 7: down_rows<7>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 8: down_rows<8>(w, row_bytes, n, sd, a, o, r0, r1); break;
        }
    }
}

}  // namespace basalt::kernels::cpu

