// include/basalt/kernels/window_gemm.hpp - the weight products of a decode window: up to 16 activation rows against an
// FP8 E4M3 matrix, an NVFP4 matrix in planes (nvfp4_planes.hpp) or a BF16 matrix, the matrix read once at memory speed.
//
// Activations: bf16 rows.  Weights: E4M3 codes, or E2M1 codes times their UE4M3 group scale, converted to bf16 in registers
// (both exact).  Products: mma m16n8k16 (bf16 in, fp32 accumulate) with the window's rows as the columns, 8 per mma.  A
// column's sums never meet another column's, and the order of a row's sums (fixed chunks of K in order within a K slice,
// the slices added in slice order) depends only on the shapes of the call's matrices: a row's bits do not depend on T or
// on what the other rows hold, so a drafted or batched window gives the same rows as plain decoding.
//
// Products start as programmatic dependents inside a PdlScope (pdl.hpp): each stages its first weights before it waits
// for the kernel before it.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::wgemm {

constexpr int kRows = 16;   // rows a window holds

enum class Format {
    F8,      // E4M3 codes [rows][K]
    Nvfp4,   // planes (nvfp4_planes.hpp)
    Bf16,    // [rows][K]
    Q8,      // Q8_0 in planes: int8 codes [rows][K], then fp16 scales [rows][K / 32] (q8_planes); a code times its scale
             // rounds to bf16
};

/// Q8_0 blocks (rows x K, ggml's 34-byte blocks) into Q8 planes, the same bytes, `planes` distinct from `blocks`.
void q8_planes(const void* blocks, int64_t rows, int64_t K, void* planes, void* stream);

/// Bytes of a window's activations for K values per row: [16][K] bf16.
std::size_t act_bytes(int64_t K);

/// Rows 0..T-1 of x (row t at x + t * ld_x, K values) as bf16 rows in `act`.
void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream);
/// The RMS norm of rows 0..T-1 of x ([T][K], weights w, any offset folded) packed into `act`; with `xn` the normed fp32
/// rows too (bitwise rms_norm_rows).
void norm_pack(const float* x, const float* w, float eps, int64_t K, int T, void* act, float* xn, void* stream);
/// The device scratch for split-K sums, sized with workspace_need over every call a model makes.
struct Workspace {
    float* partial = nullptr;    // [slices][16][rows] of the product in flight
    int32_t* tickets = nullptr;  // per row tile, back to 0 after each product
    std::size_t partial_floats = 0, tickets_n = 0;
};

/// One matrix of a call: W [rows][K], its output y[t * ld_y + r], `scale` a device fp32 scalar or null.
struct Matrix {
    const uint8_t* w = nullptr;
    int64_t rows = 0;
    const float* scale = nullptr;
    float* y = nullptr;
    int64_t ld_y = 0;
    const uint8_t* scales = nullptr;   // NVFP4, Q8: the scale plane when `rows` are the first rows of a larger matrix (null: after
                                       // the codes of these rows)
};

enum class Epilogue {
    Store,    // y = scale * W x
    Add,      // y += scale * W x
    SwiGLU,   // two matrices of the same rows (gate, up): m[0].y = silu(s_g gate x) * (s_u up x)
};

constexpr int kMaxMatrices = 3;

/// One call: up to three matrices of the same K and format against the same activations, one launch.
struct Call {
    Format fmt = Format::F8;
    int64_t K = 0;
    Matrix m[kMaxMatrices];
    int n = 1;
    Epilogue epi = Epilogue::Store;
    void* act_out = nullptr;   // SwiGLU: h also as bf16 rows for the next product ([16][rows], act_bytes(rows)); m[0].y may be null
};

/// The workspace a call needs (its shape only).
void workspace_need(const Call& c, std::size_t& partial_floats, std::size_t& tickets);

/// Runs the call for rows 0..T-1 (1..16) of `act` (packed for c.K).  F8: K a multiple of 128, W 16-byte aligned.  NVFP4:
/// W in planes, rows a multiple of 128, K a multiple of 128.
void gemm(const Call& c, const void* act, int T, const Workspace& ws, void* stream);

}  // namespace basalt::kernels::wgemm
