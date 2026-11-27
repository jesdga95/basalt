// src/models/glm5next/kernels/kda.hpp - see kda.cu: one recurrent KDA (Kimi Delta Attention) step for one token.
//
// KDA is the gated delta rule with a PER-CHANNEL key decay: Gated DeltaNet's scalar per-head alpha becomes a vector,
// one decay per key channel (the a-side projection has head_k_dim outputs per head). Same state (the matrix every
// other linear-attention model carries), same read; only the per-step decay differs. The recurrence, per head:
//
//   h        <- h .* g                      (g[j] in (0,1): exp of the log decay, caller computes it)
//   v'       <- v - h^T k                   (the value minus what the state already remembers)
//   u        <- beta * v'
//   h        <- h + k u^T
//   o        <- h q                         (the read, AFTER the update; the gated RMS norm and o_proj are outside)
//
// The state layout is value-major rows: state[head][i][j] (row i = value channel, contiguous over the key channels
// j), so one thread owns one row: its two dot products, its rank-1 update and its output value are all local.
#pragma once

#include <cuda_runtime.h>

namespace basalt::models::glm5next {

/// One recurrent KDA step for `n_heads` heads at once, updating `state` in place.
///   state [n_heads][dv][dk] f32  in/out
///   q     [n_heads][dk] f32      (L2-normalized and scaled by the caller)
///   k     [n_heads][dk] f32      (L2-normalized)
///   v     [n_heads][dv] f32
///   g     [n_heads][dk] f32      the per-channel decay FACTOR exp(log_decay), in (0,1)
///   beta  [n_heads] f32
///   out   [n_heads][dv] f32      the read o, before the gated RMS norm
void kda_step(const float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int n_heads, int dk, int dv, cudaStream_t stream);

}  // namespace basalt::models::glm5next
