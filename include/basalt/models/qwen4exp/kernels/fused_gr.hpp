// include/basalt/models/qwen4exp/kernels/fused_gr.hpp - the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstdint>

namespace basalt::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

/// The same read on the matrices' 8-bit copies (hc_fused.hpp's hc8_view) in three kernels (norm,
/// down + inject, up); a token's outputs come from the same operations whatever n_tok.  `rs` and `lo` are written as
/// above; R' goes to R_out (when apply) before the down kernel reads it.  False, with nothing enqueued, when w_down has
/// no copy.  fused_gr_read, fused_gr_read_multi and gr_read take this path for such matrices.
bool hc8_gr_read_multi(const FusedGrArgs* a, int n_tok, int64_t n_embd, int64_t hc, void* stream);
/// The multi read's variants, all computing every output with the plain read's operations in its order, so bitwise
/// the plain read's and the single-token read's: plain (the norm one block per token, the down projection on 41 blocks), split (the norm
/// one block per token and stream, then the plain down projection) and staged (split's norm, and the down
/// projection's activations staged ahead by cp.async, two half-stream tiles in flight, bank-conflict-free).
/// `fused_gr_check` runs all of them and the single-token read on the current card with random weights and inputs
/// (1..8 tokens, with and without the pending write) and from then on uses there the newest one that agrees with the
/// plain read bit for bit.  It runs once per card (Verifier::init calls it) and prints which one runs.  On a card it
/// has not checked, `fused_gr_variant` is the plain read.
void fused_gr_check();
int fused_gr_variant();

}  // namespace basalt::kernels
