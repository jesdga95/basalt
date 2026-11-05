// src/kernels/cpu/expert.cpp - the Q2_0 expert kernel.
//
// See include/basalt/kernels/cpu/expert.hpp first; it says why each piece is shaped this way.
#include "basalt/kernels/cpu/expert.hpp"

#include <immintrin.h>
#include "basalt/platform/compiler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "basalt/log.hpp"

namespace basalt::kernels::cpu {
namespace {

/// fp16 -> fp32, written out rather than using `_cvtsh_ss`, because the F16C intrinsic's behaviour on
/// subnormals is the one place the two can differ and the scales in this artifact are small.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

/// 2-bit unpack via VPMULTISHIFTQB.
///
/// ARGUMENT ORDER IS (control, data), determined empirically: the Intel guide states
/// (a=data, b=control), and following it produces wrong codes.  Within each 64-bit lane, output byte i is `(u16 >> 2i) & 0xFF`, so AND 3 yields the
/// eight 2-bit codes in order.
///
/// The result is the 32 codes of ONE ACTIVATION CHUNK (4 qword lanes), which is the unit the Q8_1 scale is
/// defined on - not the 64-weight block.
inline __m256i unpack_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadl_epi64((const __m128i*) codes);   // 8 bytes -> 4 u16 in the low half
    const __m256i lanes = _mm256_cvtepu16_epi64(packed);              // 4 qwords, 8 codes each
    const __m256i ctrl = _mm256_set1_epi64x((long long) 0x0E0C0A0806040200ULL);
    return _mm256_and_si256(_mm256_multishift_epi64_epi8(ctrl, lanes), _mm256_set1_epi8(3));
}

/// 32 codes against 32 int8 activations. Codes 0 contribute nothing to `sum(c*xhat)`, so the zero lanes the
/// narrow load leaves behind are harmless.
inline __m256i block_dot(const uint8_t* codes, const int8_t* xq) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), unpack_q2_0(codes),
                               _mm256_load_si256((const __m256i*) xq));
}

inline float hsum_ps(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    return _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(s, s), s));
}

// ================================ the row dot in 512-bit lanes ================================
//
// A block is ONE unpack (16 code bytes -> 64 codes), ONE VNNI dot over 64 activations, ONE convert and ONE FMA whose
// per-chunk scales come from a permute of a vector of scale products built for eight blocks at a time; the
// weight-independent correction is one FMA per eight blocks.  The verify window and plain decode both use this
// kernel, so speculative decode reproduces plain decode exactly.

inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) codes);          // 8 u16 = 64 codes
    const __m512i lanes = _mm512_cvtepu16_epi64(packed);                    // 8 qwords, 8 codes each
    const __m512i ctrl = _mm512_set1_epi64((long long) 0x0E0C0A0806040200ULL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, lanes), _mm512_set1_epi8(3));
}

// For blocks [b0, b0+nb) (nb <= 8): p[2i+h] = d_{b0+i} * xscale[2(b0+i)+h], dd[2i+h] = d_{b0+i}.
inline void scales8(const uint8_t* wscales, const ActQ& a, int b0, int nb, __m512& p, __m512& dd) {
    const __mmask8 m8 = nb >= 8 ? (__mmask8) 0xFF : (__mmask8) ((1u << nb) - 1u);
    const __m128i h = _mm_maskz_loadu_epi16(m8, wscales + 2 * b0);
    const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(h));
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    dd = _mm512_permutexvar_ps(dup, d8);
    const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
    dd = _mm512_maskz_mov_ps(m16, dd);
    p = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a.scale + 2 * b0));
}

inline float row_dot_z(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    __m512 acc = _mm512_setzero_ps(), corr = _mm512_setzero_ps();
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        __m512 p, dd;
        scales8(scales, a, b0, nb, p, dd);
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        corr = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a.hx + 2 * b0), corr);
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unpack64_q2_0(codes + b * 16),
                                                    _mm512_load_si512((const void*) (a.q + b * QK)));
            const __m512 sv = _mm512_permutexvar_ps(_mm512_add_epi32(base, _mm512_set1_epi32(2 * i)), p);
            acc = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), acc);
        }
    }
    return _mm512_reduce_add_ps(acc) - _mm512_reduce_add_ps(corr);
}

template<int NT>
inline void row_dot_multi_z(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                            float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            __m512 dd;
            scales8(scales, *a[t], b0, nb, p[t], dd);
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(codes + b * 16);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}

/// One weight row against a quantized activation.
///
///     sum_j (c_j - 1) d_w xhat_j d_x  ==  d_w * ( d_x*sum(c*xhat) - d_x*sum(xhat) )
///
/// split per 32-chunk because `d_x` changes at the chunk boundary while `d_w` does not.
///
/// The accumulator stays a FLOAT VECTOR for the whole row and is reduced once at the end.  Reducing per
/// 64-weight block (two 8-lane horizontal sums: a store plus 16 scalar adds) cost far more than the dot
/// products themselves - it made the kernel compute-bound at 21 GB/s instead of DRAM-bound at 32.
inline float row_dot(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    return row_dot_z(codes, scales, a, nblocks);
}

/// `row_dot` for NT activations at once: the codes of each block are unpacked once, and each token's
/// accumulator sees exactly the operations `row_dot` would apply to it, in the same order.
template<int NT>
inline void row_dot_multi(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                          float* res) {
    row_dot_multi_z<NT>(codes, scales, a, nblocks, res);
}

}  // namespace

const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    // Named individually: "AVX-512 not supported" sends a user looking for a new CPU when the machine may have
    // AVX-512F and be missing only VNNI, which is a much narrower and more explicable gap.
    static char buf[160];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s", avx512f ? "" : "AVX512F ",
                  avx512bw ? "" : "AVX512BW ", avx512vl ? "" : "AVX512VL ",
                  avx512_vnni ? "" : "AVX512-VNNI ", avx512_vbmi ? "" : "AVX512-VBMI");
    return buf;
}

CpuFeatures cpu_features() {
    CpuFeatures f;
    int reg[4] = {0, 0, 0, 0};
    unsigned r[4] = {0, 0, 0, 0};
    platform::cpuid(0, 0, r);
    if (r[0] < 7) return f;
    platform::cpuid(7, 0, r);
    for (int i = 0; i < 4; ++i) reg[i] = (int) r[i];
    const unsigned ebx = (unsigned) reg[1], ecx = (unsigned) reg[2];
    f.avx512f = (ebx >> 16) & 1u;
    f.avx512bw = (ebx >> 30) & 1u;
    f.avx512vl = (ebx >> 31) & 1u;
    f.avx512_vnni = (ecx >> 11) & 1u;
    f.avx512_vbmi = (ecx >> 1) & 1u;
    return f;
}

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    // AVX-512, the same operations per element as the scalar loop below (max of |x|, one multiply,
    // +-0.5 away from zero, truncation, clamp), so the result is bitwise the scalar one.  The scalar loop is a
    // visible cost per speculative round.
    {
        const __m512 half = _mm512_set1_ps(0.5f), mhalf = _mm512_set1_ps(-0.5f), zero = _mm512_setzero_ps();
        const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
        const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (int k = 0; k < a.nchunks; ++k) {
            const float* xb = x + k * QKA;
            const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
            const float s = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = s > 0.f ? 1.f / s : 0.f;
            const __m512 vinv = _mm512_set1_ps(inv);
            const __m512 t0 = _mm512_mul_ps(x0, vinv), t1 = _mm512_mul_ps(x1, vinv);
            const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t0, zero, _CMP_GE_OQ), mhalf, half));
            const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t1, zero, _CMP_GE_OQ), mhalf, half));
            __m512i v0 = _mm512_cvttps_epi32(r0), v1 = _mm512_cvttps_epi32(r1);
            v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lo), hi);
            v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lo), hi);
            _mm_storeu_si128((__m128i*) (a.q + k * QKA), _mm512_cvtepi32_epi8(v0));
            _mm_storeu_si128((__m128i*) (a.q + k * QKA + 16), _mm512_cvtepi32_epi8(v1));
            const int32_t sum = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
            a.scale[k] = s;
            a.sum[k] = sum;
            a.hx[k] = s * (float) sum;
        }
        return;
    }
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            // **`std::lround` IS A FUNCTION CALL.**  It respects the current rounding mode, so a compiler may not inline it
            // to a single instruction and emits a call per element, which is pure overhead on the CPU expert path.
            //
            // `t + copysign(0.5, t)` is EXACTLY `lround`'s rule - round half AWAY FROM ZERO - and it is
            // branchless and vectorizable.  It is NOT `_mm256_round_ps`, which rounds half to EVEN and would
            // quietly change the activation on the ties.  The cast is safe because `|t| <= 127` by construction
            // (`s = amax/127`, so `|x_j|/s <= 127`), well inside int32.
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

namespace {
}  // namespace

// ================================ Q2_0 rows in the GGUF block layout ================================
//
// The IQ model files keep their down projections in Q2_0 GGUF blocks (fp16 d, then 16 code bytes: 18 bytes per
// 64 weights, interleaved), which ggml-cpu computes with a scalar loop on x86.  This is `row_dot_multi_z` with
// the block stride of the GGUF layout: the same unpack, the same VNNI dot, the same correction.
namespace {
template<int NT>
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        alignas(16) uint16_t sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nb; ++i) std::memcpy(&sc[i], row + (size_t) (b0 + i) * 18, 2);
        const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_load_si128((const __m128i*) sc)));
        const __m512 dd = _mm512_maskz_mov_ps(m16, _mm512_permutexvar_ps(dup, d8));
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            p[t] = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->scale + 2 * b0));
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(row + (size_t) b * 18 + 2);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}
template<int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}
}  // namespace

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
    }
}

}  // namespace basalt::kernels::cpu
