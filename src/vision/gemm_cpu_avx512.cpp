// src/vision/gemm_cpu_avx512.cpp - the bf16 product on AVX-512 BF16: x and w as bf16 pairs, fp32 accumulation.
//
// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512bf16; callers check cpu_has_bf16() first.
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "gemm_cpu.hpp"

namespace basalt::vision {

namespace {

// rows [m0, m0 + rows) of x against 4 weight rows: acc[i][j] over K
inline void tile4(const uint16_t* x, const uint16_t* w, float* y, int m0, int rows, int n0, int ncols, int N, int K) {
    __m512 acc[4][4];
    for (auto& r : acc) for (auto& a : r) a = _mm512_setzero_ps();
    const uint16_t* wr[4];
    for (int j = 0; j < 4; ++j) wr[j] = w + (size_t) (n0 + std::min(j, ncols - 1)) * K;
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m512bh wv[4];
        for (int j = 0; j < 4; ++j) wv[j] = (__m512bh) _mm512_loadu_si512(wr[j] + k);
        for (int i = 0; i < rows; ++i) {
            const __m512bh xv = (__m512bh) _mm512_loadu_si512(x + (size_t) (m0 + i) * K + k);
            for (int j = 0; j < 4; ++j) acc[i][j] = _mm512_dpbf16_ps(acc[i][j], xv, wv[j]);
        }
    }
    if (k < K) {
        const __mmask32 mk = (__mmask32) ((1ull << (K - k)) - 1);
        __m512bh wv[4];
        for (int j = 0; j < 4; ++j) wv[j] = (__m512bh) _mm512_maskz_loadu_epi16(mk, wr[j] + k);
        for (int i = 0; i < rows; ++i) {
            const __m512bh xv = (__m512bh) _mm512_maskz_loadu_epi16(mk, x + (size_t) (m0 + i) * K + k);
            for (int j = 0; j < 4; ++j) acc[i][j] = _mm512_dpbf16_ps(acc[i][j], xv, wv[j]);
        }
    }
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < ncols; ++j) y[(size_t) (m0 + i) * N + n0 + j] = _mm512_reduce_add_ps(acc[i][j]);
}

}  // namespace

#if defined(_MSC_VER)
bool cpu_has_bf16() {
    // The same readings GCC's run-time probe takes: the OS has to save the AVX-512 state, and leaf 7 subleaf 0
    // names AVX512BW (EBX bit 30) and AVX512_BF16 (ECX bit 5).
    unsigned r[4];
    { int x[4]; __cpuidex(x, 0, 0); if ((unsigned) x[0] < 7u) return false; }
    { int x[4]; __cpuidex(x, 1, 0); for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i]; }
    if (!((r[2] >> 27) & 1u)) return false;                 // OSXSAVE
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6) != 0xE6) return false;                // the OS saves the AVX-512 state
    { int x[4]; __cpuidex(x, 7, 0); for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i]; }
    return ((r[1] >> 30) & 1u) != 0 && ((r[2] >> 5) & 1u) != 0;   // AVX512BW, AVX512_BF16
}
#else
bool cpu_has_bf16() { return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw"); }
#endif

void gemm_bf16_avx512(const float* X, const uint16_t* W, float* Y, int M, int N, int K, int threads) {
    std::vector<uint16_t> xb((size_t) M * K);
    {   // x to bf16, round to nearest even
        const size_t total = xb.size();
        size_t i = 0;
        for (; i + 16 <= total; i += 16)
            _mm256_storeu_si256((__m256i*) &xb[i], (__m256i) _mm512_cvtneps_pbh(_mm512_loadu_ps(X + i)));
        for (; i < total; ++i) {
            __m128bh r = _mm_cvtneps_pbh(_mm_set1_ps(X[i]));
            xb[i] = (uint16_t) _mm_extract_epi16((__m128i) r, 0);
        }
    }
    const int mblocks = (M + 31) / 32, ntiles = (N + 3) / 4;
    threads = std::max(1, threads);
    const bool by_m = mblocks >= 2 * threads;
    const int units = by_m ? mblocks : ntiles;
    threads = std::min(threads, units);
    auto work = [&](int a, int b) {
        if (by_m) {
            for (int mb = a; mb < b; ++mb) {
                const int m_lo = mb * 32, m_hi = std::min(M, m_lo + 32);
                for (int t = 0; t < ntiles; ++t)
                    for (int m0 = m_lo; m0 < m_hi; m0 += 4)
                        tile4(xb.data(), W, Y, m0, std::min(4, m_hi - m0), t * 4, std::min(4, N - t * 4), N, K);
            }
        } else {
            for (int t = a; t < b; ++t)
                for (int m0 = 0; m0 < M; m0 += 4) tile4(xb.data(), W, Y, m0, std::min(4, M - m0), t * 4, std::min(4, N - t * 4), N, K);
        }
    };
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        const int a = (int) ((int64_t) units * t / threads), b = (int) ((int64_t) units * (t + 1) / threads);
        if (threads == 1) work(a, b);
        else ts.emplace_back(work, a, b);
    }
    for (auto& t : ts) t.join();
}

}  // namespace basalt::vision
