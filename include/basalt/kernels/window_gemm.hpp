// include/basalt/kernels/window_gemm.hpp - the weight products of a decode window: up to 8 activation rows against a
// Q8_0 or NVFP4 matrix, read once at memory speed.
//
//   window_quantize   the window's 8 activation rows as int8 codes per 32 values with an fp32 scale (amax / 127)
//   window_gemm       y[t][r] = sum_k W[r][k] x[t][k] for t < T, from those codes
//
// Every product is computed for all 8 rows on int8 tensor cores (mma m16n8k32 for Q8_0, m16n8k16 per NVFP4 sub-block)
// and a row's sums never meet another row's, so a row's bits do not depend on T or on what the other rows hold: a drafted
// window gives the same rows as plain decoding.  The K range is split over four warps per block in a fixed pattern, each
// warp's partial summed in block order, the four partials then added in warp order: deterministic.
//
// NVFP4 values are the doubled E2M1 grid (integers -12..12) with each 16-value sub-block's UE4M3 scale halved, as ggml
// stores them; the matrix's FP32 scale is not applied here.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kWinRows = 8;

/// The quantized window: q [8][K] int8 codes then d [8][K / 32] fp32 scales, in one buffer of window_act_bytes(K).
std::size_t window_act_bytes(int64_t K);

/// Rows 0..7 of x (row t at x + t * ld_x, K values) quantized into `act` (window_act_bytes(K) bytes).
void window_quantize(const float* x, int64_t ld_x, int64_t K, void* act, void* stream);

/// Whether window_gemm takes this matrix: ggml type 8 (Q8_0) or 40 (NVFP4), K a multiple of 256, rows a multiple of 16,
/// the rows 16-byte aligned.
bool window_gemm_supported(int type, int64_t rows, int64_t K, const void* w);

/// y[t * ld_y + r] for t < T (1..8) and r < rows: the window's rows (quantized by window_quantize for this K) times W.
void window_gemm(int type, const void* w, int64_t rows, int64_t K, const void* act, float* y, int64_t ld_y, int T,
                 void* stream);

}  // namespace basalt::kernels
