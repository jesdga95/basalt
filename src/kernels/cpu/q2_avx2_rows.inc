// src/kernels/cpu/q2_avx2_rows.inc - the Q2_0 expert rows in 256-bit lanes, several tokens at once.
//
// Included by q2_avx2.cpp (BASALT_Q2_VNNI 0: `maddubs` and `madd`) and q2_avx2_vnni.cpp (BASALT_Q2_VNNI 1: the AVX-VNNI
// instruction that does both), which name the exported function through BASALT_Q2_EXPORT.  The arithmetic is the
// AVX-512 kernels': codes 0..3 against the int8 activation per 32-value chunk, times the weight scale and the chunk
// scale, minus the weight scale times the chunk's `hx` (the -1 code offset).  The four-wise integer sums are exact on
// both builds (codes 0..3, activations -127..127), so the two give the same bits.

namespace basalt::kernels::cpu {
namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, two 32-byte vectors
inline void unpack64(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));  // values 0..31
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));  // values 32..63
}

// Four-wise sums of u8 x s8 products into int32 lanes.
inline __m256i q2_dot4(__m256i u8, __m256i s8) {
#if BASALT_Q2_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
#endif
}

// hxs[t * MAXC / 2 + b]: the block's two chunk corrections summed, once per call instead of once per row
template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, const float* hxs, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const __m256i s0 = q2_dot4(lo, _mm256_loadu_si256((const __m256i*) q));
            const __m256i s1 = q2_dot4(hi, _mm256_loadu_si256((const __m256i*) (q + 32)));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * hxs[t * (MAXC / 2) + b];
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
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

void BASALT_Q2_EXPORT(q2_0_gguf_rows_multi_avx2)(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a,
                                                 int nt, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                BASALT_Q2_EXPORT(q2_0_gguf_rows_multi_avx2)(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

}  // namespace basalt::kernels::cpu
