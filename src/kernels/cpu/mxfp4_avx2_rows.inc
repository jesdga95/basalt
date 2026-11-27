// src/kernels/cpu/mxfp4_avx2_rows.inc - MXFP4 expert rows against Q8_0 activations, several tokens at once, in 256-bit
// lanes. Included by mxfp4_avx2.cpp (BASALT_MXFP4_VNNI 0) and mxfp4_avx2_vnni.cpp (BASALT_MXFP4_VNNI 1: vpdpbusd).
//
// Each pair of block_mxfp4 (64 values = 2 Q8_0 blocks) is decoded ONCE for all NT tokens. Per token and row the
// arithmetic matches ggml_vec_dot_mxfp4_q8_0's AVX2 loop bit-for-bit: same int32 sums per block, same d_y * e8m0_half
// scale, same two accumulators (accum1, accum2), same final hsum8 reduction.

namespace basalt::kernels::cpu {
namespace {

struct E8M0HalfLut {
    float v[256];
    constexpr E8M0HalfLut() : v{} {
        for (uint32_t x = 0; x < 256; ++x) {
            const uint32_t bits = x < 2 ? (0x00200000u << x) : ((x - 1u) << 23);
            v[x] = std::bit_cast<float>(bits);
        }
    }
};
constexpr E8M0HalfLut kE8m0Lut{};

constexpr int kMaxN = 16384;
constexpr int kMaxT = 8;

// Pre-unpacked Q8_0 pair for 64 values (2 block_q8_0) per token.
struct alignas(32) TokPair {
    __m256i y0, y1;
    float d0, d1;
};

TokPair* tok_buf() {
    alignas(32) static thread_local TokPair buf[kMaxT * kMaxN / 64];
    return buf;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

inline float hsum8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

struct BlkPair {
    __m256i w0, w1;   // signed doubled E2M1 weights [-12..12]
    __m256i a0, a1;   // abs(w0), abs(w1) in [0..12] (unsigned for vpmaddubsw / vpdpbusd)
    float s0, s1;     // E8M0 half-scales
};

inline BlkPair decode_pair(const block_mxfp4* x, const __m128i values128, const __m128i m4b) {
    const __m128i q0 = _mm_loadu_si128((const __m128i*) x[0].qs);
    const __m128i q1 = _mm_loadu_si128((const __m128i*) x[1].qs);
    const __m128i lo0 = _mm_shuffle_epi8(values128, _mm_and_si128(q0, m4b));
    const __m128i hi0 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q0, 4), m4b));
    const __m128i lo1 = _mm_shuffle_epi8(values128, _mm_and_si128(q1, m4b));
    const __m128i hi1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q1, 4), m4b));
    BlkPair b;
    b.w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo0), hi0, 1);
    b.w1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo1), hi1, 1);
    b.a0 = _mm256_sign_epi8(b.w0, b.w0);
    b.a1 = _mm256_sign_epi8(b.w1, b.w1);
    b.s0 = kE8m0Lut.v[x[0].e];
    b.s1 = kE8m0Lut.v[x[1].e];
    return b;
}

template <int NT>
inline void prep_tokens(const block_q8_0* const* y, int npairs, TokPair* tb) {
    for (int ip = 0; ip < npairs; ++ip) {
        for (int t = 0; t < NT; ++t) {
            TokPair& dst = tb[(size_t) ip * NT + t];
            dst.y0 = _mm256_loadu_si256((const __m256i*) y[t][2 * ip + 0].qs);
            dst.y1 = _mm256_loadu_si256((const __m256i*) y[t][2 * ip + 1].qs);
            dst.d0 = h2f(y[t][2 * ip + 0].d);
            dst.d1 = h2f(y[t][2 * ip + 1].d);
        }
    }
}

template <int R, int NT>
inline void rows_dot(const block_mxfp4* const* w, int npairs, const TokPair* tb, float (*res)[kMaxT]) {
    const __m128i values128 = _mm_loadu_si128((const __m128i*) kvalues_mxfp4);
    const __m128i m4b = _mm_set1_epi8(0x0f);
#if !BASALT_MXFP4_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    __m256 acc1[R][NT], acc2[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) acc1[k][t] = acc2[k][t] = _mm256_setzero_ps();

    for (int ip = 0; ip < npairs; ++ip) {
        BlkPair b[R];
        for (int k = 0; k < R; ++k) b[k] = decode_pair(w[k] + 2 * ip, values128, m4b);
        const TokPair* tk = tb + (size_t) ip * NT;
        for (int t = 0; t < NT; ++t) {
            const __m256i y0 = tk[t].y0;
            const __m256i y1 = tk[t].y1;
            const float d0 = tk[t].d0;
            const float d1 = tk[t].d1;
            for (int k = 0; k < R; ++k) {
                const __m256i sy0 = _mm256_sign_epi8(y0, b[k].w0);
                const __m256i sy1 = _mm256_sign_epi8(y1, b[k].w1);
#if BASALT_MXFP4_VNNI
                const __m256i p0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), b[k].a0, sy0);
                const __m256i p1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), b[k].a1, sy1);
#else
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(b[k].a0, sy0), ones);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(b[k].a1, sy1), ones);
#endif
                acc1[k][t] = _mm256_fmadd_ps(_mm256_set1_ps(d0 * b[k].s0), _mm256_cvtepi32_ps(p0), acc1[k][t]);
                acc2[k][t] = _mm256_fmadd_ps(_mm256_set1_ps(d1 * b[k].s1), _mm256_cvtepi32_ps(p1), acc2[k][t]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) res[k][t] = hsum8(_mm256_add_ps(acc1[k][t], acc2[k][t]));
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu, const void* const* act,
             float* const* ff, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int npairs = n / (2 * QK_MXFP4);
    TokPair* tb = tok_buf();
    prep_tokens<NT>(y, npairs, tb);
    float res[2][kMaxT];
    for (int r = r0; r < r1; ++r) {
        const block_mxfp4* w[2] = {(const block_mxfp4*) (blob + (size_t) r * gu_row),
                                   (const block_mxfp4*) (blob + up_off + (size_t) r * gu_row)};
        if constexpr (NT <= 2) {
            rows_dot<2, NT>(w, npairs, tb, res);
        } else {
            rows_dot<1, NT>(&w[0], npairs, tb, &res[0]);
            rows_dot<1, NT>(&w[1], npairs, tb, &res[1]);
        }
        for (int t = 0; t < NT; ++t) ff[t][r] = swiglu(glu, res[0][t], res[1][t]);
    }
}

template <int NT>
void down_rows(const uint8_t* w0, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int npairs = n / (2 * QK_MXFP4);
    TokPair* tb = tok_buf();
    prep_tokens<NT>(y, npairs, tb);
    float res[2][kMaxT];
    int r = r0;
    if constexpr (NT <= 2) {
        for (; r + 2 <= r1; r += 2) {
            const block_mxfp4* w[2] = {(const block_mxfp4*) (w0 + (size_t) r * row_bytes),
                                       (const block_mxfp4*) (w0 + (size_t) (r + 1) * row_bytes)};
            rows_dot<2, NT>(w, npairs, tb, res);
            for (int t = 0; t < NT; ++t) {
                out[t][r] = res[0][t];
                out[t][r + 1] = res[1][t];
            }
        }
    }
    for (; r < r1; ++r) {
        const block_mxfp4* w[1] = {(const block_mxfp4*) (w0 + (size_t) r * row_bytes)};
        rows_dot<1, NT>(w, npairs, tb, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t];
    }
}

}  // namespace

void BASALT_MXFP4_EXPORT(mxfp4_256_gu_rows)(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                                            const void* const* act, int nt, float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = ff + t0;
        switch (k) {
            case 1: gu_rows<1>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
            case 8: gu_rows<8>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
        }
    }
}

void BASALT_MXFP4_EXPORT(mxfp4_256_down_rows)(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                                              float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = out + t0;
        switch (k) {
            case 1: down_rows<1>(w, row_bytes, n, a, o, r0, r1); break;
            case 2: down_rows<2>(w, row_bytes, n, a, o, r0, r1); break;
            case 3: down_rows<3>(w, row_bytes, n, a, o, r0, r1); break;
            case 4: down_rows<4>(w, row_bytes, n, a, o, r0, r1); break;
            case 5: down_rows<5>(w, row_bytes, n, a, o, r0, r1); break;
            case 6: down_rows<6>(w, row_bytes, n, a, o, r0, r1); break;
            case 7: down_rows<7>(w, row_bytes, n, a, o, r0, r1); break;
            case 8: down_rows<8>(w, row_bytes, n, a, o, r0, r1); break;
        }
    }
}

}  // namespace basalt::kernels::cpu
