// include/basalt/kernels/fp8_window.hpp - the weight products of a decode window against an FP8 E4M3 matrix: up to 8
// activation rows, the matrix read once at memory speed.
//
// Activations: fp16, each row scaled by a power of two so its largest value sits in [2^14, 2^15) (pack); the power comes
// back out after the product, exactly.  Weights: E4M3 codes converted to fp16 in registers (exact).  Products: mma
// m16n8k16 (fp16 in, fp32 accumulate) with the window's 8 rows as the 8 columns.  A column's sums never meet another
// column's, and the order of a row's sums (chunks of 256 values in order within a K slice, the slices added in slice
// order) depends only on the matrix's shape: a row's bits do not depend on T or on what the other rows hold, so a
// drafted window gives the same rows as plain decoding.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::fp8w {

constexpr int kRows = 8;   // rows a window holds

/// Bytes of a window's packed activations for K values per row: [8][K] fp16, then 8 fp32 output factors (2^-e).
std::size_t act_bytes(int64_t K);

/// Rows 0..T-1 of x (row t at x + t * ld_x, K values, K a multiple of 256) packed into `act`.
void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream);

/// The device scratch for split-K sums, sized with workspace_need over every matrix a model runs.
struct Workspace {
    float* partial = nullptr;    // [slices][8][rows] of the product in flight
    int32_t* tickets = nullptr;  // per row tile, back to 0 after each product
    std::size_t partial_floats = 0, tickets_n = 0;
};
void workspace_need(int64_t rows, int64_t K, std::size_t& partial_floats, std::size_t& tickets);

enum class Epilogue { Store, Add };
/// y[t * ld_y + r] (=, or += with Add) = scale * sum_k W[r][k] x[t][k] for t < T (1..8), r < rows: W [rows][K] E4M3
/// codes (16-byte aligned, K a multiple of 256), x from `act` (packed for this K), `scale` a device fp32 scalar or null.
void gemm(const uint8_t* w, int64_t rows, int64_t K, const void* act, const float* scale, float* y, int64_t ld_y,
          Epilogue epi, int T, const Workspace& ws, void* stream);

}  // namespace basalt::kernels::fp8w
