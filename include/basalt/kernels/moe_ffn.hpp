// include/basalt/kernels/moe_ffn.hpp - the routed-expert FFN of a dense-stack layer: a softmax router picks `n_used` of
// `n_expert` SwiGLU experts per token, a shared expert (an ordinary FFN the caller runs) is gated by a sigmoid, and the
// layer's output joins the BF16 residual.
//
// An expert is one blob [gate | up | down] of NVFP4 blocks (64 values in 36 bytes: four UE4M3 group scales, then 32 code
// bytes where group s keeps value j in the low nibble of byte 8 s + j and value j + 8 in the high one) and an FP32 tail
// {s_gate, s_up, s_down, 0}; gate and up are [ff][hidden] rows, down [hidden][ff].  Rows are tokens, FP32 unless said.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::moe {

/// The expert tables of one layer: `n_expert` blobs `blob` bytes apart from `base` (device).
struct Experts {
    const uint8_t* base = nullptr;
    size_t blob = 0;
    int n_expert = 0, hidden = 0, ff = 0;
};

/// Where an expert's FP32 tail {s_gate, s_up, s_down, 0} lies in its blob, and the blob's bytes (the tail rounded up to a
/// multiple of 144: MMQ strides experts in whole 36-byte blocks, 16-byte aligned) for these widths (multiples of 64).
size_t tail_offset(int hidden, int ff);
size_t blob_bytes(int hidden, int ff);

/// logits[t][r] = x_t . w_r for the `rows` rows of w (BF16 [rows][hidden]: the router's, then the shared expert's gate), x
/// FP32 [T][hidden]; hidden 2048.  Rows of logits are `ld` floats apart.
void router(const float* x, const uint16_t* w, int T, int hidden, int rows, float* logits, int ld, void* stream);

/// x [rows][cols] rounded to INT8 (q) with a scale per 16 values (s [rows][cols / 16]), each group's values in the order the
/// experts' code nibbles come: v0 v8 v1 v9, then the same pairs up to v7 v15.  cols a multiple of 16.
void quantize(const float* x, int rows, int cols, int8_t* q, float* s, void* stream);

/// For each of T tokens (rows of `logits`, `ld` floats apart; the router's `n_expert` outputs come first): softmax over the
/// `n_expert` logits, the `n_used` largest (the lower expert first on a tie) as
/// idx[t][k] with their probabilities renormalised to sum 1 (the sum clamped to 6.1e-5 below) as wgt[t][k], k in descending
/// order of probability.
void top_k(const float* logits, int ld, int T, int n_expert, int n_used, int32_t* idx, float* wgt, void* stream);

/// The pairs p = t * n_used + k of T tokens' choices idx [T][n_used] grouped by expert, for the grouped products of a prompt
/// chunk: bounds [n_expert + 1] (the first entry of each expert's run), and for entry i its token row tok[i], its pair
/// dst[i] and its expert exp[i].  The entries of an expert are in no fixed order.  T * n_used at most 32768.
void group(const int32_t* idx, int T, int n_used, int n_expert, int32_t* bounds, int32_t* tok, int32_t* dst, int32_t* exp, void* stream);

/// h[p][r] = silu(gate_e r . x_t) * (up_e r . x_t) for the pair p = t * n_used + k, e = idx[p], r < ff.  x_t as quantize()
/// gave it (xq [T][hidden], xs [T][hidden / 16]).
void gate_up(const Experts& e, const int8_t* xq, const float* xs, const int32_t* idx, int T, int n_used, float* h, void* stream);

/// y[p][c] = down_e c . h_p for every pair p (h_p rounded to INT8 as quantize() does).  h [T * n_used][ff], y [T * n_used][hidden].
void down(const Experts& e, const float* h, const int32_t* idx, int T, int n_used, float* y, void* stream);

/// resid[t][c] (BF16) = resid[t][c] + sum_k wgt[t][k] y[t * n_used + k][c] + sigmoid(sgate[t * ld]) shared[t][c], in fp32,
/// rounded once.
void combine(uint16_t* resid, const float* y, const float* wgt, const float* sgate, int ld, const float* shared, int T, int n_used,
             int hidden, void* stream);

}  // namespace basalt::kernels::moe
