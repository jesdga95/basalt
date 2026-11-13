// include/basalt/kernels/window_gemm.hpp - the weight products of a decode window: up to 16 activation rows against an
// FP8 E4M3 matrix or an NVFP4 matrix in planes (nvfp4_planes.hpp), the matrix read once at memory speed.
//
// Activations: fp16, each row scaled by a power of two so its largest value sits in [2^14, 2^15) (pack); the power comes
// back out after the product, exactly.  Weights: E4M3 codes, or E2M1 codes times their UE4M3 group scale, converted to
// fp16 in registers (both exact).  Products: mma m16n8k16 (fp16 in, fp32 accumulate) with the window's rows as the
// columns, 8 per mma.  A column's sums never meet another column's, and the order of a row's sums (fixed chunks of K in
// order within a K slice, the slices added in slice order) depends only on the shapes of the call's matrices: a row's
// bits do not depend on T or on what the other rows hold, so a drafted or batched window gives the same rows as plain
// decoding.
//
// Products start as programmatic dependents inside a PdlScope (pdl.hpp): each stages its first weights before it waits
// for the kernel before it.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::wgemm {

constexpr int kRows = 16;   // rows a window holds

enum class Format { F8, Nvfp4 };

/// Bytes of a window's packed activations for K values per row: [16][K] fp16, then 16 fp32 output factors (2^-e).
std::size_t act_bytes(int64_t K);

/// Rows 0..T-1 of x (row t at x + t * ld_x, K values, K a multiple of 128) packed into `act`.
void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream);
/// The RMS norm of rows 0..T-1 of x ([T][K], weights w, any offset folded) packed into `act`; with `xn` the normed fp32
/// rows too.  Bitwise rms_norm_rows followed by pack.
void norm_pack(const float* x, const float* w, float eps, int64_t K, int T, void* act, float* xn, void* stream);
/// A gated attention layer's output rows: attn [T][n_head][256] times sigmoid of each head's gate (q_full [T][n_head][512],
/// the second half), packed into `act`.  Bitwise attn_gate followed by pack.
void gate_pack(const float* attn, const float* q_full, int T, int n_head, void* act, void* stream);

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
};

/// The workspace a call needs (its shape only).
void workspace_need(const Call& c, std::size_t& partial_floats, std::size_t& tickets);

/// Runs the call for rows 0..T-1 (1..16) of `act` (packed for c.K).  F8: K a multiple of 128, W 16-byte aligned.  NVFP4:
/// W in planes, rows a multiple of 128, K a multiple of 128.
void gemm(const Call& c, const void* act, int T, const Workspace& ws, void* stream);

}  // namespace basalt::kernels::wgemm
