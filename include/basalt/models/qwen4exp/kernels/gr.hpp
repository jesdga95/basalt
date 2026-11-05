// include/basalt/models/qwen4exp/kernels/gr.hpp - the gated residual (hyper-connection).
//
// Transcribed from `llama_model_qwen4exp::graph::build_hc_mix` / `build_hc_combine`.
//
// The residual is a STACK of `hc` streams, each `n_embd` wide - not one vector with a gate.  `gr_read`
// collapses the stack to the one vector the mixer consumes and produces a per-stream injection; `gr_write`
// puts the mixer's output back into every stream, weighted per stream.
//
// WEIGHTS ARE BF16, stored as the high 16 bits of an f32 in a `uint16_t`.  That is not a shortcut: every one
// of these tensors has source type BF16, so the dense section's 32-bit promotion is lossless padding and
// re-rounding to 16 bits costs nothing.  It is also worth a large share of the dense weights, which is the
// difference between them fitting in VRAM beside the expert cache and not fitting at all.
//
// The projections are the BF16 MMVF: BF16 weights multiply FP32 activations, as the single-token BF16 MMVF of the
// reference CUDA backend does.  The CPU oracle and some batched CUDA paths instead round activations to BF16.
// Activation precision is therefore fixed by the oracle backend and batch geometry, not inferred from weights.
#pragma once

#include <cstdint>
#include <cstddef>

namespace basalt::kernels {

/// Geometry of the hyper-connection.  The real model: n_embd = 2560, hc = 4, hc_lr = 320.
struct GrShapes {
    int64_t n_embd = 0;   ///< width of one residual stream
    int64_t hc = 0;       ///< number of streams in the stack
    int64_t hc_lr = 0;    ///< rank of the down/up bottleneck
};

/// Scratch for `gr_read`.  EXPLICIT and caller-owned, because the token path must make ZERO allocations:
/// a hidden `cudaMalloc` on first use would satisfy every functional test and break graph capture.
///
/// **IT MUST BE DEVICE MEMORY.**  `gr_workspace_init` takes a raw pointer and does no checking, so handing it
/// a host buffer produces device kernels writing to a truncated host address - an illegal access whose
/// reported fault address is the low 32 bits of a host pointer, which reads as a wild pointer rather than as
/// the mistake it is.  That cost one round in `gr_parity`;
///
///   xn    hc*n_embd floats   the activated streams
///   gated hc*n_embd floats   xn * sigmoid(gate), before the mean over streams
///   lo    hc_lr     floats   SiLU bottleneck
///
/// `cudaMalloc(gr_workspace_bytes(s))`, then `gr_workspace_init(s, ptr, ws)` to carve it.
struct GrWorkspace {
    float* xn = nullptr;
    float* gated = nullptr;
    float* lo = nullptr;
    /// Set by `gr_workspace_init`.  `gr_read` CHECKS it, so an under-sized buffer is a loud failure at the
    /// first call instead of silent corruption.
    size_t bytes = 0;
};

/// Carves `base` (which must be DEVICE memory, or null to size only) into `out` and returns the bytes needed.
/// 16-byte aligned between regions.
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out);
inline size_t gr_workspace_bytes(const GrShapes& s) { GrWorkspace w; return gr_workspace_init(s, nullptr, w); }

/// `build_hc_mix`.  ONE token: `R` is (hc, n_embd) with n_embd fastest, i.e. stream c starts at c*n_embd.
///
///   w_norm   (hc*n_embd)          per-stream RMSNorm gamma, STORED AS (1 + w) by the converter
///   w_down   (hc_lr, hc*n_embd)   MANIFEST LAYOUT - row k is contiguous, `w_down[k*hc*n_embd + i]`
///   w_up     (hc*n_embd, hc_lr)   MANIFEST LAYOUT - row i is contiguous, `w_up[i*hc_lr + k]`
///   w_inject (hc, hc*n_embd)      null for the FINAL mixer, which has no write-back to gate
///
///   mixed    (n_embd)   = mean over streams of  xn * sigmoid(lo @ w_up.T)
///   inject   (hc)       = xn @ w_inject.T, or untouched when w_inject is null
///
/// `lo = silu((xn @ w_down.T) / hc)` - the division by hc is INSIDE the silu, which is easy to read
/// past and changes the result materially.  The activations stay FP32: xn and lo feed the projections unrounded.
///
/// **NO PERMUTATION: both matrices are taken exactly as the manifest stores them**, so the loader does nothing
/// to them.
void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream);

/// `build_hc_combine`.  `R_out[i] = R[i] + block_out[d] * w[c]` with `w[c] = 2*sigmoid(inject[c]/hc)`.
///
/// The `2*sigmoid` is what CENTRES the gate on 1, so a zero injection is a plain residual add - the source
/// comment says so outright and `gr_write`'s test asserts it.  Note the block output is added to EVERY stream
/// identically; only the weight is per-stream.
void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream);

}  // namespace basalt::kernels
