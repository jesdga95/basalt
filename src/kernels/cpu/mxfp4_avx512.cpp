// src/kernels/cpu/mxfp4_avx512.cpp - MXFP4 expert rows against Q8_0 activations, several tokens at once, in 512-bit
// lanes (AVX-512 VNNI).
//
// Each pair of block_mxfp4 (64 values = 2 Q8_0 blocks) is decoded once into one 512-bit vector for all NT tokens;
// every token applies the sign mask with `vpand`/`vpsubb` (`_mm512_mask_sub_epi8`) and accumulates both 32-value
// blocks with one `vpdpbusd` and one `vfmadd231ps`.  Low 256 bits hold ggml's `accum1`, high 256 bits hold `accum2`,
// and the final reduction folds them with the same `hsum_float_8(accum1 + accum2)` as `ggml_vec_dot_mxfp4_q8_0`,
// so every output float is bitwise identical to ggml's single-token dot.
#include "basalt/kernels/cpu/mxfp4_avx512.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"
#include "basalt/platform/compiler.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

// E8M0 half-scales as ggml's dot decodes them (GGML_CPU_E8M0_TO_FP32_HALF). Built at compile time so no static
// constructor runs in an AVX-512 translation unit before CPUID probing.
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

// Pre-packed Q8_0 pair for 64 values (2 block_q8_0) per token: low 256 bits = block 2*ip, high 256 bits = 2*ip + 1.
struct alignas(64) TokPair512 {
    __m512i a;
    __m512 d;
};

TokPair512* tok_buf() {
    alignas(64) static thread_local TokPair512 buf[kMaxT * kMaxN / 64];
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

// One decoded pair of block_mxfp4 (64 values): |w| bytes in [0..12], 64-bit sign mask, and 16-lane E8M0 half-scales.
struct BlkPair512 {
    __m512i aw;
    __mmask64 m;
    __m512 ws;
};

inline BlkPair512 decode_pair(const block_mxfp4* x, const __m512i values, const __m256i m4b) {
    const __m128i q0 = _mm_loadu_si128((const __m128i*) x[0].qs);
    const __m128i q1 = _mm_loadu_si128((const __m128i*) x[1].qs);
    const __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(q0), q1, 1);
    const __m256i lo = _mm256_and_si256(q, m4b);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4b);
    // [lo0 | lo1 | hi0 | hi1] -> [lo0 | hi0 | lo1 | hi1]
    const __m512i lh = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    const __m512i codes = _mm512_shuffle_i64x2(lh, lh, _MM_SHUFFLE(3, 1, 2, 0));
    const __m512i w = _mm512_shuffle_epi8(values, codes);
    BlkPair512 b;
    b.aw = _mm512_abs_epi8(w);
    b.m = _mm512_movepi8_mask(w);
    const __m256 s0 = _mm256_set1_ps(kE8m0Lut.v[x[0].e]);
    const __m256 s1 = _mm256_set1_ps(kE8m0Lut.v[x[1].e]);
    b.ws = _mm512_insertf32x8(_mm512_castps256_ps512(s0), s1, 1);
    return b;
}

template <int NT>
inline void prep_tokens(const block_q8_0* const* y, int npairs, TokPair512* tb) {
    for (int ip = 0; ip < npairs; ++ip) {
        for (int t = 0; t < NT; ++t) {
            TokPair512& dst = tb[(size_t) ip * NT + t];
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][2 * ip + 0].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][2 * ip + 1].qs);
            dst.a = _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1);
            const __m256 d0 = _mm256_set1_ps(h2f(y[t][2 * ip + 0].d));
            const __m256 d1 = _mm256_set1_ps(h2f(y[t][2 * ip + 1].d));
            dst.d = _mm512_insertf32x8(_mm512_castps256_ps512(d0), d1, 1);
        }
    }
}

// R rows against NT tokens: res[k][t]. Two rows x up to 8 tokens keep all 16 accumulators in ZMM registers.
template <int R, int NT>
inline void rows_dot(const block_mxfp4* const* w, int npairs, const TokPair512* tb, float (*res)[kMaxT]) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_mxfp4));
    const __m256i m4b = _mm256_set1_epi8(0x0f);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) acc[k][t] = _mm512_setzero_ps();

    for (int ip = 0; ip < npairs; ++ip) {
        BlkPair512 b[R];
        for (int k = 0; k < R; ++k) {
            if constexpr (platform::kPreferMemberwiseVectorAssignment) {
                const BlkPair512 decoded = decode_pair(w[k] + 2 * ip, values, m4b);
                b[k].aw = decoded.aw;
                b[k].m = decoded.m;
                b[k].ws = decoded.ws;
            } else {
                b[k] = decode_pair(w[k] + 2 * ip, values, m4b);
            }
        }
        const TokPair512* tk = tb + (size_t) ip * NT;
        for (int t = 0; t < NT; ++t) {
            const __m512i a = tk[t].a;
            const __m512 d = tk[t].d;
            for (int k = 0; k < R; ++k) {
                const __m512i sa = _mm512_mask_sub_epi8(a, b[k].m, zero, a);
                const __m512i p = _mm512_dpbusd_epi32(zero, b[k].aw, sa);
                acc[k][t] = _mm512_fmadd_ps(_mm512_mul_ps(d, b[k].ws), _mm512_cvtepi32_ps(p), acc[k][t]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t)
            res[k][t] = hsum8(_mm256_add_ps(_mm512_castps512_ps256(acc[k][t]), _mm512_extractf32x8_ps(acc[k][t], 1)));
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu, const void* const* act,
             float* const* ff, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int npairs = n / (2 * QK_MXFP4);
    TokPair512* tb = tok_buf();
    prep_tokens<NT>(y, npairs, tb);
    float res[2][kMaxT];
    for (int r = r0; r < r1; ++r) {
        const block_mxfp4* w[2] = {(const block_mxfp4*) (blob + (size_t) r * gu_row),
                                   (const block_mxfp4*) (blob + up_off + (size_t) r * gu_row)};
        rows_dot<2, NT>(w, npairs, tb, res);
        for (int t = 0; t < NT; ++t) ff[t][r] = swiglu(glu, res[0][t], res[1][t]);
    }
}

template <int NT>
void down_rows(const uint8_t* w0, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int npairs = n / (2 * QK_MXFP4);
    TokPair512* tb = tok_buf();
    prep_tokens<NT>(y, npairs, tb);
    float res[2][kMaxT];
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const block_mxfp4* w[2] = {(const block_mxfp4*) (w0 + (size_t) r * row_bytes),
                                   (const block_mxfp4*) (w0 + (size_t) (r + 1) * row_bytes)};
        rows_dot<2, NT>(w, npairs, tb, res);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = res[0][t];
            out[t][r + 1] = res[1][t];
        }
    }
    if (r < r1) {
        const block_mxfp4* w[1] = {(const block_mxfp4*) (w0 + (size_t) r * row_bytes)};
        rows_dot<1, NT>(w, npairs, tb, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t];
    }
}

}  // namespace

bool mxfp4_512_fits(int n_embd, int n_ff) noexcept {
    return n_embd > 0 && n_ff > 0 && n_embd % (2 * QK_MXFP4) == 0 && n_ff % (2 * QK_MXFP4) == 0 && n_embd <= kMaxN &&
           n_ff <= kMaxN;
}

void mxfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                       const void* const* act, int nt, float* const* ff, int r0, int r1) {
    if (nt <= 0 || r0 >= r1) return;
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
            default: gu_rows<8>(blob, gu_row, up_off, n, glu, a, o, r0, r1); break;
        }
    }
}

void mxfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                         int r0, int r1) {
    if (nt <= 0 || r0 >= r1) return;
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
            default: down_rows<8>(w, row_bytes, n, a, o, r0, r1); break;
        }
    }
}

}  // namespace basalt::kernels::cpu
