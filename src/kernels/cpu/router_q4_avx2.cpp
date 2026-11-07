// src/kernels/cpu/router_q4_avx2.cpp - see router_q4.hpp. AVX2 with FMA and F16C.
#include "basalt/kernels/cpu/router_q4.hpp"

#include <cstddef>
#include <cstring>
#include <immintrin.h>

namespace basalt::kernels::cpu {

void router_q4_logits(const uint8_t* rows, const float* scales, int n_embd, int n_expert, int e0, int e1,
                      const uint8_t* xq, int nt, float* logits, const std::atomic<bool>& stop) {
    const int NB = n_embd / 32;
    const __m128i lo4 = _mm_set1_epi8(0x0f), off = _mm_set1_epi8(8);
    for (int e = e0; e < e1; ++e) {
        if (stop.load(std::memory_order_relaxed)) return;
        __m256 acc[kRouterQ4MaxT];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        const uint8_t* row = rows + (size_t) e * (n_embd / 2);
        for (int b = 0; b < NB; ++b) {
            const __m128i q = _mm_loadu_si128((const __m128i*) (row + b * 16));
            const __m128i ql = _mm_sub_epi8(_mm_and_si128(q, lo4), off);
            const __m128i qh = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), lo4), off);
            const __m256i wl = _mm256_cvtepi8_epi16(ql);
            const __m256i wh = _mm256_cvtepi8_epi16(qh);
            const float sw = scales[(size_t) e * NB + b];
            for (int t = 0; t < nt; ++t) {
                const uint8_t* blk = xq + ((size_t) t * NB + b) * 36;   // block_q8_1: half d, half s, 32 x int8
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const __m256i x = _mm256_loadu_si256((const __m256i*) (blk + 4));
                const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wl, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x))),
                                                   _mm256_madd_epi16(wh, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1))));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh)))), acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            logits[(size_t) t * n_expert + e] = _mm_cvtss_f32(v);
        }
    }
}

}  // namespace basalt::kernels::cpu
