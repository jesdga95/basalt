// src/vision/gemm_cpu_neon.cpp - see gemm_cpu.hpp.  The Arm64 build of gemm_cpu.cpp: NEON with FMA, a 2 x 4 tile.
//
// Each tile entry keeps two four-lane accumulators, values k..k+3 and k+4..k+7 of every eight, and folds them the way
// the AVX2 file folds its eight lanes (low half plus high half, then pairs): the same products meet in the same
// order, so a product here has the bits the AVX2 build gives it. A 2 x 4 tile holds 16 accumulators + 8 weight + 2 x
// vectors (26 <= 32 NEON registers), avoiding register spills in the inner loop.
#include "gemm_cpu.hpp"

#include "basalt/platform/compiler.hpp"

#include <algorithm>
#include <cstring>
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

// (lo + hi), then lanes 0+2 and 1+3, then their sum: the AVX2 file's hsum
inline float hsum(float32x4_t lo, float32x4_t hi) {
    const float32x4_t a = vaddq_f32(lo, hi);
    const float32x2_t p = vadd_f32(vget_low_f32(a), vget_high_f32(a));
    return vget_lane_f32(p, 0) + vget_lane_f32(p, 1);
}

// Y rows [m0, m0 + ROWS) x columns [n0, n0 + ncols): 4 weight row pointers in wrows
template <int ROWS>
void tile(const float* X, const float* const wrows[4], float* Y, int m0, int n0, int ncols, int N, int K) {
    float32x4_t lo[ROWS][4], hi[ROWS][4];
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < 4; ++j) lo[i][j] = hi[i][j] = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        float32x4_t w0[4], w1[4];
        for (int j = 0; j < 4; ++j) {
            w0[j] = vld1q_f32(wrows[j] + k);
            w1[j] = vld1q_f32(wrows[j] + k + 4);
        }
        for (int i = 0; i < ROWS; ++i) {
            const float32x4_t x0 = vld1q_f32(X + (size_t) (m0 + i) * K + k);
            const float32x4_t x1 = vld1q_f32(X + (size_t) (m0 + i) * K + k + 4);
            for (int j = 0; j < 4; ++j) {
                lo[i][j] = vfmaq_f32(lo[i][j], x0, w0[j]);
                hi[i][j] = vfmaq_f32(hi[i][j], x1, w1[j]);
            }
        }
    }
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < ncols; ++j) {
            float s = hsum(lo[i][j], hi[i][j]);
            for (int kk = k; kk < K; ++kk) s += X[(size_t) (m0 + i) * K + kk] * wrows[j][kk];
            Y[(size_t) (m0 + i) * N + n0 + j] = s;
        }
}

void run_col_tile(const float* X, const float* const wrows[4], float* Y, int M, int n0, int nc, int N, int K) {
    int m0 = 0;
    for (; m0 + 2 <= M; m0 += 2) tile<2>(X, wrows, Y, m0, n0, nc, N, K);
    if (m0 < M) tile<1>(X, wrows, Y, m0, n0, nc, N, K);
}

template <class Work>
void dispatch_tiles(int N, int threads, Work work) {
    const int tiles = (N + 3) / 4;
    threads = std::max(1, std::min(threads, tiles));
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        const int a = (int) ((int64_t) tiles * t / threads), b = (int) ((int64_t) tiles * (t + 1) / threads);
        if (threads == 1) work(a, b);
        else ts.emplace_back(work, a, b);
    }
    for (auto& t : ts) t.join();
}

}  // namespace

// AVX-512 BF16 is x86's: the bf16 product always takes the fp32 path below.
bool cpu_has_bf16() { return false; }
void gemm_bf16_avx512(const float* X, const uint16_t* W, float* Y, int M, int N, int K, int threads) {
    gemm_bf16w(X, W, Y, M, N, K, threads, true);
}

void gemm_bf16w(const float* X, const uint16_t* W, float* Y, int M, int N, int K, int threads, bool round_x) {
    if (M <= 0 || N <= 0 || K < 0) return;
    std::vector<float> xr;
    if (round_x) {
        const size_t total = (size_t) M * K;
        xr.resize(total);
        const uint32x4_t v7fff = vdupq_n_u32(0x7fffu);
        const uint32x4_t vone = vdupq_n_u32(1u);
        const uint32x4_t vmask = vdupq_n_u32(0xffff0000u);
        size_t i = 0;
        for (; i + 4 <= total; i += 4) {
            uint32x4_t u = vreinterpretq_u32_f32(vld1q_f32(X + i));
            u = vandq_u32(vaddq_u32(vaddq_u32(u, v7fff), vandq_u32(vshrq_n_u32(u, 16), vone)), vmask);
            vst1q_f32(xr.data() + i, vreinterpretq_f32_u32(u));
        }
        for (; i < total; ++i) xr[i] = bf16_round(X[i]);
        X = xr.data();
    }
    dispatch_tiles(N, threads, [&](int a, int b) {
        std::vector<float> wf((size_t) 4 * K);
        for (int t = a; t < b; ++t) {
            const int n0 = t * 4, nc = std::min(4, N - n0);
            const float* wrows[4] = {};
            for (int j = 0; j < nc; ++j) {
                const uint16_t* w = W + (size_t) (n0 + j) * K;
                float* dst = wf.data() + (size_t) j * K;
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    const uint16x8_t v = vld1q_u16(w + k);
                    vst1q_f32(dst + k, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
                    vst1q_f32(dst + k + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16)));
                }
                for (; k < K; ++k) {
                    const uint32_t u = (uint32_t) w[k] << 16;
                    std::memcpy(dst + k, &u, 4);
                }
                wrows[j] = dst;
            }
            const float* last_row = wf.data() + (size_t) (nc - 1) * K;
            for (int j = nc; j < 4; ++j) wrows[j] = last_row;
            run_col_tile(X, wrows, Y, M, n0, nc, N, K);
        }
    });
}

void gemm_f32w(const float* X, const float* W, float* Y, int M, int N, int K, int threads) {
    if (M <= 0 || N <= 0 || K < 0) return;
    dispatch_tiles(N, threads, [&](int a, int b) {
        for (int t = a; t < b; ++t) {
            const int n0 = t * 4, nc = std::min(4, N - n0);
            const float* wrows[4];
            for (int j = 0; j < 4; ++j) wrows[j] = W + (size_t) (n0 + std::min(j, nc - 1)) * K;
            run_col_tile(X, wrows, Y, M, n0, nc, N, K);
        }
    });
}

}  // namespace basalt::vision
