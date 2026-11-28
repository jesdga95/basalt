// src/vision/gemm_cpu.cpp - see gemm_cpu.hpp.  AVX2 + FMA: a 4 x 4 register tile, bf16 widened by a shift.
#include "gemm_cpu.hpp"

#include "basalt/platform/compiler.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

namespace basalt::vision {

namespace {

inline float bf16_round(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float hsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

// Y rows [m0,m0+rows) x columns [n0, n0+4): W rows as fp32 in wf (4 x K)
void tile(const float* X, const float* wf, float* Y, int m0, int rows, int n0, int ncols, int N, int K) {
    __m256 acc[4][4];
    for (auto& r : acc) for (auto& a : r) a = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 w[4];
        for (int j = 0; j < 4; ++j) w[j] = _mm256_loadu_ps(wf + (size_t) j * K + k);
        for (int i = 0; i < rows; ++i) {
            const __m256 xv = _mm256_loadu_ps(X + (size_t) (m0 + i) * K + k);
            for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_fmadd_ps(xv, w[j], acc[i][j]);
        }
    }
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < ncols; ++j) {
            float s = hsum(acc[i][j]);
            for (int kk = k; kk < K; ++kk) s += X[(size_t) (m0 + i) * K + kk] * wf[(size_t) j * K + kk];
            Y[(size_t) (m0 + i) * N + n0 + j] = s;
        }
}

template <class LoadRow>
void run(const float* X, float* Y, int M, int N, int K, int threads, LoadRow load) {
    threads = std::max(1, std::min(threads, (N + 3) / 4));
    auto work = [&](int a, int b) {   // column tiles [a, b)
        std::vector<float> wf((size_t) 4 * K);
        for (int t = a; t < b; ++t) {
            const int n0 = t * 4, nc = std::min(4, N - n0);
            for (int j = 0; j < 4; ++j) {
                if (j < nc) load(n0 + j, wf.data() + (size_t) j * K);
                else std::fill_n(wf.data() + (size_t) j * K, K, 0.0f);
            }
            for (int m0 = 0; m0 < M; m0 += 4) tile(X, wf.data(), Y, m0, std::min(4, M - m0), n0, nc, N, K);
        }
    };
    const int tiles = (N + 3) / 4;
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        const int a = (int) ((int64_t) tiles * t / threads), b = (int) ((int64_t) tiles * (t + 1) / threads);
        if (threads == 1) work(a, b);
        else ts.emplace_back(work, a, b);
    }
    for (auto& t : ts) t.join();
}

}  // namespace

// Probed outside the AVX-512 translation unit so hosts without AVX-512 never enter EVEX-compiled code.
bool cpu_has_bf16() { return basalt::platform::cpu_has_avx512_bf16(); }

void gemm_bf16w(const float* X, const uint16_t* W, float* Y, int M, int N, int K, int threads, bool round_x) {
    if (M <= 0 || N <= 0 || K < 0) return;
    static const bool has_bf16 = cpu_has_bf16();
    if (round_x && has_bf16) { gemm_bf16_avx512(X, W, Y, M, N, K, threads); return; }
    std::vector<float> xr;
    if (round_x) {
        xr.resize((size_t) M * K);
        for (size_t i = 0; i < xr.size(); ++i) xr[i] = bf16_round(X[i]);
        X = xr.data();
    }
    run(X, Y, M, N, K, threads, [&](int n, float* dst) {
        const uint16_t* w = W + (size_t) n * K;
        for (int k = 0; k < K; ++k) {
            const uint32_t u = (uint32_t) w[k] << 16;
            std::memcpy(dst + k, &u, 4);
        }
    });
}

void gemm_f32w(const float* X, const float* W, float* Y, int M, int N, int K, int threads) {
    if (M <= 0 || N <= 0 || K < 0) return;
    run(X, Y, M, N, K, threads, [&](int n, float* dst) { std::memcpy(dst, W + (size_t) n * K, sizeof(float) * K); });
}

}  // namespace basalt::vision
