// include/basalt/models/qwen4exp/kernels/s_gemv.hpp - the S-family GEMV, host-callable.
//
// ONE kernel for S2, S4 and S8, because the pack format gives them ONE decode:
//
//     value = cb[code] * scale + offset
//
// with a per-TYPE codebook, code bias, group size and offset presence.  That uniformity is the point of the
// canonical form - 13 source types collapse to three code widths and a handful of attributes - so the kernel
// takes those attributes as ARGUMENTS rather than being specialised per source type.  The manifest already
// carries them per tensor.
#pragma once

#include <cstdint>

namespace basalt::kernels {

// Which table turns a code into a number.  `Affine` is `code + bias`; `Iq4Nl` is the non-linear
// `kvalues_iq4nl`, which no bias can express - the reason the canonical form carries a codebook at all.
enum class Codebook : int { Affine = 0, Iq4Nl = 1 };

struct SForm {
    int code_bits = 2;              // 2, 4 or 8
    int code_bias = -1;             // subtracted from the code IN THE INTEGER DOMAIN before the multiply
    int group_elems = 64;           // elements per scale (and per offset, when present)
    Codebook codebook = Codebook::Affine;
    bool has_offset = false;        // only the K-quants with a min (Q4_K, Q5_K) have one

    /// WHICH ACTIVATION THIS WEIGHT'S `vec_dot_type` IS: 0 = Q8_0, 1 = Q8_K.
    ///
    /// **IT IS AN ATTRIBUTE OF THE CANONICAL FORM AND IT CANNOT BE DERIVED FROM THE OTHERS.**  Q5_0 and Q5_K
    /// are both 8-bit with bias -16 (only `has_offset` differs); IQ4_NL and IQ4_XS are both 4-bit with the
    /// IQ4NL codebook and no offset, and differ in NOTHING this struct otherwise holds.  So it is CARRIED from
    /// the manifest's `source_type` rather than reconstructed - see the activation contract and the
    /// `act_kind` column of the dense index.
    int act_kind = 0;
};

// y[o] = sum_i fp16(x[i]) * W[i][o].  GGUF's ne[0] varies fastest, so a tensor is [n_in, n_out] with n_in
// CONTIGUOUS: row o is a contiguous run of n_in/group_elems groups.
//
// `codes`  n_out * (n_in / (8/code_bits)) bytes
// `scales` n_out * (n_in / group_elems) floats
// `offset` the same count as scales, or nullptr when `has_offset` is false
// `y`      n_out floats
void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form);

// The same computation with the ROW SPLIT ACROSS THREADS: one block of `threads_per_row` threads per output
// row, each accumulating a strided subset, reduced through shared memory.
//
// WHY BOTH EXIST.  `s_gemv` gives one thread per output row, so its parallelism is the OUTPUT WIDTH - and the
// expert gate/up shape is [2560 x 640], which is 640 threads over 48 SMs, 13 per SM.  That
// orientation runs several times slower than [640 x 2560] with identical weights.  This variant
// makes the parallelism independent of the output width.
//
// It is NOT bit-identical to `s_gemv`: the partial sums are added in a different order, so the parity test
// carries a tolerance for exactly this reason.  Both are kept and both are tested, because the naive one is
// the reference the split one is checked against.
void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row);

/// THE SAME GEMV OVER **Q8_K** ACTIVATIONS, which is the contract for every K-quant and IQ-quant weight.
///
/// the activation contract settles this: `ggml_mul_mat` converts the activation to the weight's
/// `vec_dot_type`, and for Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS that type is Q8_K - NOT fp16.  The two differ by
/// about 0.6-0.9% per GEMV against a 1e-3 tolerance, and the K-quants are the attention projections and
/// `ssm_out`, i.e. the numerically sensitive dense weights.
///
/// `x` is the block_q8_K layout, 292 bytes per 256 elements: `{ f32 d ; int8_t qs[256] ; int16_t bsums[16] }`.
/// `bsums` is NOT read here - it exists for ggml's AVX2 path - but the STRIDE includes it, so a buffer
/// produced by `quantize_q8_K` can be passed straight in.
///
/// The activation is dequantized on the fly (`d * qs[i]`), one multiply per element, exactly as the fp16 path
/// does one `__half2float` per element.  The weight decode is the SAME code as `s_gemv` - one implementation,
/// because a second decode of the canonical form is a second thing to get wrong.
void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

/// The row-split form, for output widths too small to fill the machine.
///
/// **THERE IS NO `threads_per_row` HERE ON PURPOSE.**  This form is WARP per output row, which is what the
/// shape wants, so there is no such knob to tune.  `s_gemv_split` DOES take threads_per_row
/// and DOES use it, if a caller wants to tune that shape.
void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

/// THE SAME, OVER A **Q8_0** ACTIVATION - `block_q8_0`, 34 bytes per 32 elements, from `quantize_q8_0`.
///
/// **THIS IS THE KERNEL THAT WAS MISSING, AND ITS ABSENCE WAS DESCRIBED INCORRECTLY FOR SEVERAL ROUNDS.**  The
/// legacy block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL) have `vec_dot_type` Q8_0, not Q8_K.  Every one of them
/// in this pack except `ffn_down_shexp` is `Q2_0`, whose attributes happen to BE S2's (2 bits, group 64,
/// bias -1) - so `s2_gemv_q8`, which hardcodes those, covered them by coincidence.  `ffn_down_shexp` is
/// IQ4_NL/Q4_0/Q5_0/Q8_0 in every layer with **group 32 and bias -16/0/-8**, and its `n_in` is **640**.
///
/// 640 is a multiple of 32 and NOT of 256, so **Q8_K is structurally impossible for it rather than merely
/// unimplemented** - which is what the "Q8_K is not implemented" note in `shared_expert.hpp` was really
/// pointing at.
///
/// `n_in` must be a multiple of 32.  `group_elems` must be a power of two and a multiple of four, the same two
/// requirements `s_gemv_q8k_split` has and for the same reasons.
void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

}  // namespace basalt::kernels
