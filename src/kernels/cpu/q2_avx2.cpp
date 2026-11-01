// src/kernels/cpu/q2_avx2.cpp - the Q2_0 expert rows and the activation quantizer for CPUs
// without AVX-512.
//
// Compiled with AVX2 only, so nothing here can fault on those CPUs.  The rows are q2_avx2_rows.inc, built here with
// `maddubs` and `madd` and in q2_avx2_vnni.cpp with the AVX-VNNI instruction that does both; this file picks one at
// run time.  The quantizer is the scalar rule, bit for bit.
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

#define BASALT_Q2_VNNI 0
#define BASALT_Q2_EXPORT(name) name##_base
#include "q2_avx2_rows.inc"

namespace basalt::kernels::cpu {

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) q2_0_gguf_rows_multi_avx2_vnni(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2_base(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), zero = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m256 v[4];
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(xb + 8 * i);
            m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m256 vinv = _mm256_set1_ps(inv);
        __m256i sum = _mm256_setzero_si256();
        alignas(32) int32_t qi[QKA];
        for (int i = 0; i < 4; ++i) {
            const __m256 t = _mm256_mul_ps(v[i], vinv);
            const __m256 r = _mm256_add_ps(t, _mm256_blendv_ps(mhalf, half, _mm256_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m256i q = _mm256_cvttps_epi32(r);
            q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
            sum = _mm256_add_epi32(sum, q);
            _mm256_store_si256((__m256i*) (qi + 8 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

}  // namespace basalt::kernels::cpu
