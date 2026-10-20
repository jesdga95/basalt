// src/kernels/cpu/tq_avx512.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7 / TQP5 / TQP6 / TQP7) on AVX-512
// F/BW/VL/VNNI (tq_cpu.hpp).  Reached behind cpu_avx512_ok() only.
//
// TQ2_T / TQK, per 128-weight block: the AVX2 kernel's decode (scalar states, scalar 8-byte state-table loads, one
// ZMM of 32 int16 weights per Q8_0 sub-block); per token four VPDPWSSD against the unpacked int16 activations, an
// unpack/add tree to the four exact sub-block sums S[k] in one XMM, and the scalar kernel's float steps lane for lane.
// The weights are decoded once per block for all the tokens of the group.  A gathered decode (VPERMB states,
// VPGATHERQQ rows) is slower: gathers are slow on the CPUs this targets.
// TQP (pair trellis): the states and their points in vector registers.  The 8K stream bytes are one masked load;
// VPERMB lays out the 8 lanes' 64-bit windows (lane l = bytes (7 - l)K .. +7, circular), two VPMULTISHIFTQB cut the
// 64 15-bit states from them, VPMULLD + VPSRLD hash them to codebook points, stored in weight order; the int16 pairs
// are then read from the 8 KiB point table by broadcast loads blended into YMMs (no gather, no scalar multiply).  The
// points are worked out a block ahead of their table reads, so those reads never wait on the stores just made.  The
// per-token part is the TQ kernels'.  The same weights and the same integer and float steps as the AVX2 and scalar
// kernels, so the same bits.
#include "basalt/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace basalt::kernels::cpu::tq_detail {
namespace {

// four ZMM of 16 int32 partial sums -> [S0, S1, S2, S3] (exact integer adds)
inline __m128i reduce4(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i a = _mm512_add_epi32(_mm512_unpacklo_epi32(p0, p1), _mm512_unpackhi_epi32(p0, p1));
    const __m512i b = _mm512_add_epi32(_mm512_unpacklo_epi32(p2, p3), _mm512_unpackhi_epi32(p2, p3));
    const __m512i c = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
    // c: per 128-bit lane [p0, p1, p2, p3] partial sums; add the four lanes
    const __m256i d = _mm256_add_epi32(_mm512_castsi512_si256(c), _mm512_extracti64x4_epi64(c, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
}

// four state-table rows (8 bytes each) into one YMM: scalar loads, broadcast and blended in; two make a ZMM
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

template <int KB>
inline __m512i rows8(const long long* tab, const uint8_t* qs, uint64_t wr, int t0) {
    const __m256i lo = rows4(tab, tq_state<KB>(qs, wr, t0), tq_state<KB>(qs, wr, t0 + 1), tq_state<KB>(qs, wr, t0 + 2),
                             tq_state<KB>(qs, wr, t0 + 3));
    const __m256i hi = rows4(tab, tq_state<KB>(qs, wr, t0 + 4), tq_state<KB>(qs, wr, t0 + 5),
                             tq_state<KB>(qs, wr, t0 + 6), tq_state<KB>(qs, wr, t0 + 7));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

template <int KB>
void rows_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    const long long* tab = (const long long*) tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            // states 8k .. 8k+7 -> weights 32k .. 32k+31 = Q8_0 sub-block k
            const __m512i w0 = rows8<KB>(tab, qs, wr, 0), w1 = rows8<KB>(tab, qs, wr, 8);
            const __m512i w2 = rows8<KB>(tab, qs, wr, 16), w3 = rows8<KB>(tab, qs, wr, 24);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m512i* q = (const __m512i*) (a[t]->q + 128 * i);
                const __m512i z = _mm512_setzero_si512();
                const __m128i s = reduce4(_mm512_dpwssd_epi32(z, w0, _mm512_load_si512(q + 0)),
                                          _mm512_dpwssd_epi32(z, w1, _mm512_load_si512(q + 1)),
                                          _mm512_dpwssd_epi32(z, w2, _mm512_load_si512(q + 2)),
                                          _mm512_dpwssd_epi32(z, w3, _mm512_load_si512(q + 3)));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

// TQP: VPERMB indices of the 8 lane windows (qword l = stream bytes (7 - l)K + b mod 8K, b = 0..7)
template <int K>
inline __m512i tqp_window_index() {
    alignas(64) uint8_t idx[64];
    for (int l = 0; l < 8; ++l)
        for (int b = 0; b < 8; ++b) idx[8 * l + b] = (uint8_t) (((7 - l) * K + b) % (8 * K));
    return _mm512_load_si512(idx);
}
// VPMULTISHIFTQB controls: word m of every qword = the 16 window bits from (7 - j)K, j = j0 + m
template <int K>
inline __m512i tqp_state_shifts(int j0) {
    uint64_t q = 0;
    for (int m = 0; m < 4; ++m) {
        const int o = (7 - (j0 + m)) * K;
        q |= (uint64_t) (o & 63) << (16 * m) | (uint64_t) ((o + 8) & 63) << (16 * m + 8);
    }
    return _mm512_set1_epi64((long long) q);
}

// eight table entries into one YMM: broadcast loads blended in (no gather: slow on Zen)
inline __m256i points8(const int* cb, const uint32_t* p) {
    __m256i v = _mm256_set1_epi32(cb[p[0]]);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[1]]), 0x02);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[2]]), 0x04);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[3]]), 0x08);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[4]]), 0x10);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[5]]), 0x20);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[6]]), 0x40);
    return _mm256_blend_epi32(v, _mm256_set1_epi32(cb[p[7]]), 0x80);
}

template <int K>
void rows_p(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * K;
    const int* cb = (const int*) tq_pair_table();
    const __m512i win_idx = tqp_window_index<K>();
    const __m512i sh_lo = tqp_state_shifts<K>(0), sh_hi = tqp_state_shifts<K>(4);
    const __m512i mask15 = _mm512_set1_epi16(0x7fff), hash = _mm512_set1_epi32((int) 0x9e3779b1u);
    // lanes 2k and 2k+1 next to each other: 128-bit lanes [lo.k, hi.k, lo.k+1, hi.k+1]
    const __m512i pair01 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    const __m512i pair23 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    constexpr __mmask64 qs_mask = (8 * K == 64) ? ~0ull : ((1ull << (8 * K)) - 1);
    // block g's 64 codebook points, in weight order, into pt
    auto points = [&](const uint8_t* blk, uint32_t* pt) {
        const __m512i win = _mm512_permutexvar_epi8(win_idx, _mm512_maskz_loadu_epi8(qs_mask, blk + 2));
        const __m512i s_lo = _mm512_and_si512(_mm512_multishift_epi64_epi8(sh_lo, win), mask15);   // steps 0..3
        const __m512i s_hi = _mm512_and_si512(_mm512_multishift_epi64_epi8(sh_hi, win), mask15);   // steps 4..7
        const __m512i ev = _mm512_unpacklo_epi64(s_lo, s_hi);   // 128-bit lane m: lane 2m's 8 states
        const __m512i od = _mm512_unpackhi_epi64(s_lo, s_hi);   // lane 2m + 1's
        const __m512i p01 = _mm512_permutex2var_epi64(ev, pair01, od), p23 = _mm512_permutex2var_epi64(ev, pair23, od);
        const __m256i st[4] = { _mm512_castsi512_si256(p01), _mm512_extracti64x4_epi64(p01, 1),
                                _mm512_castsi512_si256(p23), _mm512_extracti64x4_epi64(p23, 1) };
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (int k = 0; k < 4; ++k)
            _mm512_store_si512(pt + 16 * k, _mm512_srli_epi32(_mm512_mullo_epi32(_mm512_cvtepu16_epi32(st[k]), hash), 21));
    };
    // the rows' blocks are contiguous: one stream, each block's points worked out a block ahead of its dot products
    // (their table reads then load from stores that have long left the store queue)
    const uint8_t* base = w + (size_t) r0 * row_bytes;
    const int total = (r1 - r0) * nb;
    alignas(64) uint32_t pt[2][64];
    if (total > 0) points(base, pt[0]);
    __m128 acc[kTqMaxTokens];
    for (int g = 0; g < total; ++g) {
        const int i = g % nb;
        if (i == 0)
            for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        const uint8_t* blk = base + (size_t) g * bsz;
        if (g + 1 < total) points(blk + bsz, pt[(g + 1) & 1]);
        const uint32_t* p = pt[g & 1];
        __m512i wv[4];   // wv[k] = weights 32k .. 32k+31 (lanes 2k, 2k+1)
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (int k = 0; k < 4; ++k)
            wv[k] = _mm512_inserti64x4(_mm512_castsi256_si512(points8(cb, p + 16 * k)), points8(cb, p + 16 * k + 8), 1);
        const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
        for (int t = 0; t < nt; ++t) {
            const __m512i* q = (const __m512i*) (a[t]->q + 128 * i);
            const __m512i z = _mm512_setzero_si512();
            const __m128i s = reduce4(_mm512_dpwssd_epi32(z, wv[0], _mm512_load_si512(q + 0)),
                                      _mm512_dpwssd_epi32(z, wv[1], _mm512_load_si512(q + 1)),
                                      _mm512_dpwssd_epi32(z, wv[2], _mm512_load_si512(q + 2)),
                                      _mm512_dpwssd_epi32(z, wv[3], _mm512_load_si512(q + 3)));
            const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
            acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
        }
        if (i == nb - 1) {
            const int r = r0 + g / nb;
            for (int t = 0; t < nt; ++t) {
                alignas(16) float v[4];
                _mm_store_ps(v, acc[t]);
                out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
            }
        }
    }
}

}  // namespace

void tq_rows_avx512(int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
    switch (type) {
        case 144: rows_t<8>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 145: rows_t<6>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 146: rows_t<7>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 147: rows_p<5>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 148: rows_p<6>(w, row_bytes, n, a, nt, out, r0, r1); break;
        default: rows_p<7>(w, row_bytes, n, a, nt, out, r0, r1); break;
    }
}

}  // namespace basalt::kernels::cpu::tq_detail
