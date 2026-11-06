// src/kernels/cpu/iq_avx512.cpp - the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// IQ1_S (19) has a single-token kernel, bitwise ggml's dot; so have IQ2_XXS, IQ2_XS and IQ1_M (29), for limited-SwiGLU
// layers, whose output stays ggml's.
// Multi-token formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).
//
// The sign masks of IQ2_XXS, IQ2_XS and IQ3_XXS are built from their 7-bit indices with BMI2 (pdep, and pext for
// IQ2_XS) and a byte parity rather than eight table loads.  IQ3_S and IQ2_S, whose grid index takes extra high bits,
// look the grid up with one gather; the grids indexed by a plain byte keep scalar loads, which are faster there.
#include "basalt/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/platform/compiler.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// the index of the lowest set bit of a nonzero mask: the compiler's own scan
inline int lowest_bit(unsigned m) { return platform::ctz32(m); }

// software prefetch distance in bytes: the rows are streamed from DRAM, so ask for them a few blocks ahead
constexpr int prefetch_ahead = 2048;

// ksigns_iq2xs[i] for eight 7-bit sign indices at once: the table is i | parity(i) << 7.  `x` holds the indices back to
// back (index l in bits 7l..7l+6); byte l of the result is sign byte l, as the table lookups place it.
inline uint64_t ksigns8(uint64_t x) {
    const uint64_t s = _pdep_u64(x, 0x7F7F7F7F7F7F7F7Full);
    uint64_t p = s ^ (s >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return s | ((p & 0x0101010101010101ull) << 7);
}

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = ksigns8((uint64_t) (a1 & 0x0FFFFFFFu) | ((uint64_t) (b1 & 0x0FFFFFFFu) << 28));
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        // the top 7 bits of each u16 are its sign index
        const uint64_t s = ksigns8(_pext_u64(u64(q), 0xFE00FE00FE00FE00ull) | (_pext_u64(u64(q + 8), 0xFE00FE00FE00FE00ull) << 28));
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        // grid index l: qs[l] | (2 high bits l of qh) << 8, eight lookups in one gather
        const int h = (int) b[66 + 2 * j] | ((int) b[66 + 2 * j + 1] << 8);
        const __m256i hb = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(h), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
                                            _mm256_set1_epi32(3));
        const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), _mm256_slli_epi32(hb, 8));
        g = _mm512_i32gather_epi64(idx, (const void*) iq2s_grid, 8);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = ksigns8((uint64_t) (a & 0x0FFFFFFFu) | ((uint64_t) (c & 0x0FFFFFFFu) << 28));
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // grid index l: q[l] | (high bit l of qh) << 8, sixteen lookups in one gather
        const int h = (int) b[66 + 2 * j] | ((int) b[66 + 2 * j + 1] << 8);
        const __m512i hb = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(h),
                                                              _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
                                            _mm512_set1_epi32(1));
        const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), _mm512_slli_epi32(hb, 8));
        g = _mm512_i32gather_epi32(idx, (const void*) iq3s_grid, 4);
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};


// ---- IQ1_S (19), one token: ggml's AVX2 dot with the 64-value steps widened to 512 bits.  The integer sums are
// exact and folded to ggml's eight int32 lanes, and the float accumulation keeps ggml's order, so the result
// is bit for bit ggml's single-token dot.
inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

inline float iq1s_row_dot(const uint8_t* row, int nb, const block_q8_K* y) {
    // lane i of the 32 grid indices of a block takes its high bits from qh word i/4, shifted by 3 * (i % 4)
    const __m512i rep = _mm512_set_epi16(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1,
                                         0, 0, 0, 0);
    const __m512i sh = _mm512_set1_epi64(0x0009000600030000LL);
    const __m512i c0 = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i ones = _mm512_set1_epi16(1);
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * sizeof(block_iq1_s);
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        const __m128i qh = _mm_loadu_si128((const __m128i*) (blk + 34));
        const __m512i hv = _mm512_permutexvar_epi16(rep, _mm512_castsi128_si512(qh));
        const __m512i hi = _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(hv, sh), _mm512_set1_epi16(7)), 8);
        const __m512i idx16 = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2))), hi);
        // per group scale 2 * ((qh >> 12) & 7) + 1 (int16 lanes 0..7) and the sign of the delta term
        const __m128i ls = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), _mm_set1_epi16(7)), 1), _mm_set1_epi16(1));
        const __m128i sg = _mm_srai_epi16(qh, 15);
        const __m128i lss = _mm_sub_epi16(_mm_xor_si128(ls, sg), sg);
        const __m512i ls512 = _mm512_castsi128_si512(ls);
        __m512i acc = zero;
        alignas(64) uint16_t ix[32];
        _mm512_store_si512((void*) ix, idx16);
        for (int j = 0; j < 4; ++j) {
            const uint16_t* q = ix + 8 * j;
            const __m512i g = _mm512_set_epi64((long long) iq1s_grid[q[7]], (long long) iq1s_grid[q[6]], (long long) iq1s_grid[q[5]],
                                               (long long) iq1s_grid[q[4]], (long long) iq1s_grid[q[3]], (long long) iq1s_grid[q[2]],
                                               (long long) iq1s_grid[q[1]], (long long) iq1s_grid[q[0]]);
            const __m512i yv = _mm512_loadu_si512((const void*) (y[i].qs + 64 * j));
            const __m512i ys = _mm512_mask_sub_epi8(yv, _mm512_movepi8_mask(g), zero, yv);
            const __m512i dot = _mm512_maddubs_epi16(_mm512_abs_epi8(g), ys);
            const __m512i lsj = _mm512_permutexvar_epi16(_mm512_add_epi16(c0, _mm512_set1_epi16((short) (2 * j))), ls512);
            // (the permute index: lanes 0-15 -> word 2j, lanes 16-31 -> word 2j + 1)
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(dot, lsj));
        }
        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
        // the delta term: sum over the 8 groups of bsums pair * sign * scale (exact integers)
        const __m256i pairs = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[i].bsums), _mm512_castsi512_si256(ones));
        __m256i t = _mm256_mullo_epi32(pairs, _mm256_cvtepi16_epi32(lss));
        __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
        t4 = _mm_add_epi32(t4, _mm_unpackhi_epi64(t4, t4));
        const int sumi1 = _mm_cvtsi128_si32(_mm_add_epi32(t4, _mm_shuffle_epi32(t4, 1)));
        const float d = y[i].d * h2f(u16(blk));
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }
    return hsum8(accum) + IQ1S_DELTA * accum1;
}

// ---- IQ2_XXS, IQ2_XS, one token: integer sums folded to ggml's eight int32 lanes, its float accumulation order, so the
// result is bit for bit ggml's single-token dot.
template <int TY>
inline float row_dot_exact(const uint8_t* row, int nb, const block_q8_K* y) {
    __m256 accum = _mm256_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        __m512i acc = zero;
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            const __m512i yv = _mm512_loadu_si512((const void*) (y[i].qs + 64 * j));
            const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
        }
        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
        accum = _mm256_fmadd_ps(_mm256_set1_ps(h2f(u16(blk)) * y[i].d), _mm256_cvtepi32_ps(sumi), accum);
    }
    return Fmt<TY>::K * hsum8(accum);
}

// ---- IQ1_M (29), one token, bit for bit ggml's dot: the same folding, with the grid-shift term in its own lanes.
inline float iq1m_row_dot(const uint8_t* row, int nb, const block_q8_K* y) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one8 = _mm512_set1_epi8(1);
    __m256 accum1 = _mm256_setzero_ps(), accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const block_iq1_m* blk = (const block_iq1_m*) (row + (size_t) i * sizeof(block_iq1_m));
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        uint16_t sc[4];
        std::memcpy(sc, blk->scales, 8);
        iq1m_scale_t scale;
        scale.u16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        __m512i acc1 = zero, acc2 = zero;
        for (int j = 0; j < 4; ++j) {
            const uint32_t qh = u32(blk->qh + 4 * j);
            const uint8_t* qs = blk->qs + 8 * j;
            // grid index k: qs[k] | (low 3 bits of qh nibble k) << 8; the nibble's top bit is the group's shift sign
            const uint64_t i0 = _pdep_u64(u32(qs), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh & 0x7777u, 0x0f000f000f000f00ULL);
            const uint64_t i1 = _pdep_u64(u32(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64((qh >> 16) & 0x7777u, 0x0f000f000f000f00ULL);
            const __m512i g = _mm512_set_epi64((long long) iq1s_grid[i1 >> 48], (long long) iq1s_grid[(i1 >> 32) & 0xffff],
                                               (long long) iq1s_grid[(i1 >> 16) & 0xffff], (long long) iq1s_grid[i1 & 0xffff],
                                               (long long) iq1s_grid[i0 >> 48], (long long) iq1s_grid[(i0 >> 32) & 0xffff],
                                               (long long) iq1s_grid[(i0 >> 16) & 0xffff], (long long) iq1s_grid[i0 & 0xffff]);
            const __m512i yv = _mm512_loadu_si512((const void*) (y[i].qs + 64 * j));
            const __m512i ys = _mm512_mask_sub_epi8(yv, _mm512_movepi8_mask(g), zero, yv);
            const __m512i dot = _mm512_maddubs_epi16(_mm512_abs_epi8(g), ys);
            const __mmask64 ds = _cvtu64_mask64(_pdep_u64(_pext_u32(qh, 0x88888888u), 0x0101010101010101ULL) * 0xff);
            const __m512i dd = _mm512_maddubs_epi16(one8, _mm512_mask_sub_epi8(yv, ds, zero, yv));
            const int w = sc[j];
            const __m512i ls = scales4(2 * (w & 7) + 1, 2 * ((w >> 3) & 7) + 1, 2 * ((w >> 6) & 7) + 1, 2 * ((w >> 9) & 7) + 1);
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(dot, ls));
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(dd, ls));
        }
        const __m256 d = _mm256_set1_ps(y[i].d * h2f(scale.u16));
        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_add_epi32(_mm512_castsi512_si256(acc1), _mm512_extracti64x4_epi64(acc1, 1))), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_add_epi32(_mm512_castsi512_si256(acc2), _mm512_extracti64x4_epi64(acc2, 1))), accum2);
    }
    return hsum8(accum1) + IQ1M_DELTA * hsum8(accum2);
}

inline float exact_dot(int type, const uint8_t* row, int nb, const block_q8_K* y) {
    switch (type) {
        case 16: return row_dot_exact<16>(row, nb, y);
        case 17: return row_dot_exact<17>(row, nb, y);
        default: return iq1m_row_dot(row, nb, y);
    }
}

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

}  // namespace

void quantize_q8_K_avx512(const float* x, void* dst, int64_t k) {
    block_q8_K* y = (block_q8_K*) dst;
    const int64_t nb = k / QK_K;
    const __m512i top = _mm512_set1_epi32(127);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m512 v[QK_K / 16];
        __m512 am = _mm512_setzero_ps();
        for (int j = 0; j < QK_K / 16; ++j) {
            v[j] = _mm512_loadu_ps(x + 16 * j);
            am = _mm512_max_ps(am, _mm512_abs_ps(v[j]));
        }
        const float amax = _mm512_reduce_max_ps(am);
        if (amax == 0.f) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        // the signed value of the first element of largest magnitude fixes the scale
        const __m512 amv = _mm512_set1_ps(amax);
        float mx = 0.f;
        for (int j = 0; j < QK_K / 16; ++j) {
            const __mmask16 m = _mm512_cmp_ps_mask(_mm512_abs_ps(v[j]), amv, _CMP_EQ_OQ);
            if (m != 0) { mx = x[16 * j + lowest_bit((unsigned) m)]; break; }
        }
        const float iscale = -127.f / mx;
        const __m512 is = _mm512_set1_ps(iscale);
        for (int j = 0; j < QK_K / 16; ++j) {
            const __m512i q = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(is, v[j])), top);
            _mm_storeu_si128((__m128i*) (y[i].qs + 16 * j), _mm512_cvtepi32_epi8(q));
            y[i].bsums[j] = (int16_t) _mm512_reduce_add_epi32(q);
        }
        y[i].d = 1 / iscale;
    }
}

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

bool iq1s_512_fits(int n) noexcept { return n > 0 && n % QK_K == 0; }

void iq1s_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled,
                      float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * gu_row;
        const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t];
            float g = iq1s_row_dot(gr, nb, y), u = iq1s_row_dot(ur, nb, y);
            if (scaled) {
                g *= sg;
                u *= su;
            }
            if (limit > 0.f) {
                g = std::fmin(g, limit);
                u = std::fmax(std::fmin(u, limit), -limit);
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void iq1s_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1) {
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) out[t][r] = iq1s_row_dot(w + (size_t) r * row_bytes, nb, (const block_q8_K*) act[t]);
}

bool iq_exact_512_supported(int type) noexcept { return type == 16 || type == 17 || type == 29; }

void iq_exact_512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                          bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * gu_row;
        const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t];
            float g = exact_dot(type, gr, nb, y), u = exact_dot(type, ur, nb, y);
            if (scaled) {
                g *= sg;
                u *= su;
            }
            if (limit > 0.f) {
                g = std::fmin(g, limit);
                u = std::fmax(std::fmin(u, limit), -limit);
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void iq_exact_512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                       float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) out[t][r] = exact_dot(type, w + (size_t) r * row_bytes, nb, (const block_q8_K*) act[t]);
}

}  // namespace basalt::kernels::cpu
