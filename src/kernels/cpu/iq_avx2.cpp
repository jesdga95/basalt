// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once (iq_avx2_rows.inc),
// the activation quantizer, and the choice between this file's rows and the AVX-VNNI build of the same code.

#include "basalt/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/platform/compiler.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "basalt/kernels/cpu/expert_layout.hpp"

#define BASALT_IQ_VNNI 0
#define BASALT_IQ_EXPORT(name) name##_base
#include "iq_avx2_rows.inc"

namespace basalt::kernels::cpu {

// ggml's quantize_row_q8_K, bit for bit: the signed value of the first element of largest magnitude fixes the
// scale, codes are rounded to nearest-even (ggml's magic-number rounding) and capped at 127, and each 16 codes
// are summed.  An all-zero block writes d = 0 and zero codes and sums.
void quantize_q8_K_avx2(const float* x, void* dst, int64_t k) {
    block_q8_K* y = (block_q8_K*) dst;
    const int64_t nb = k / QK_K;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i top = _mm256_set1_epi32(127), ones = _mm256_set1_epi16(1);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 v[QK_K / 8];
        __m256 am = _mm256_setzero_ps();
        for (int j = 0; j < QK_K / 8; ++j) {
            v[j] = _mm256_loadu_ps(x + 8 * j);
            am = _mm256_max_ps(am, _mm256_and_ps(v[j], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        if (amax == 0.f) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const __m256 amv = _mm256_set1_ps(amax);
        float mx = 0.f;
        for (int j = 0; j < QK_K / 8; ++j) {
            const int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(v[j], absmask), amv, _CMP_EQ_OQ));
            if (m != 0) { mx = x[8 * j + lowest_bit((unsigned) m)]; break; }
        }
        const float iscale = -127.f / mx;
        const __m256 is = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K / 16; ++j) {
            const __m256i q0 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(is, v[2 * j])), top);
            const __m256i q1 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(is, v[2 * j + 1])), top);
            // 16 codes in order as int16, then as bytes
            const __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
            _mm_storeu_si128((__m128i*) (y[i].qs + 16 * j),
                             _mm_packs_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1)));
            const __m256i s = _mm256_madd_epi16(p, ones);
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            y[i].bsums[j] = (int16_t) _mm_cvtsi128_si32(s4);
        }
        y[i].d = 1 / iscale;
    }
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq256_gu_rows_vnni(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
    else iq256_gu_rows_base(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq256_rows_vnni(type, w, row_bytes, n, act, nt, out, r0, r1);
    else iq256_rows_base(type, w, row_bytes, n, act, nt, out, r0, r1);
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq4nl256_down_rows_vnni(w, row_bytes, n, hq, nt, out, r0, r1);
    else iq4nl256_down_rows_base(w, row_bytes, n, hq, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu
