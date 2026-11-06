// include/basalt/kernels/cpu/iq_avx512.hpp - AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) against Q8_K activations (ggml's block_q8_K).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// ggml's quantize_row_q8_K for `k` floats (a multiple of 256), bit for bit.
void quantize_q8_K_avx512(const float* x, void* dst, int64_t k);
bool iq512_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// IQ1_S (type 19) rows, bit for bit ggml's single-token dot.  n is the row length (a multiple of 256).
bool iq1s_512_fits(int n) noexcept;
/// ff[t][r] = silu(g) * u for g, u the gate and up dots; `scaled` multiplies g, u by sg, su first; limit > 0 caps g
/// above and clamps u to +-limit (the same expressions as the ggml path).
void iq1s_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled,
                      float limit, const void* const* act, int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t] for IQ1_S rows.
void iq1s_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1);

/// IQ2_XXS (16), IQ2_XS (17) and IQ1_M (29) rows, one token's dot at a time, bit for bit ggml's single-token dot.
bool iq_exact_512_supported(int ggml_type) noexcept;
/// As iq1s_512_gu_rows, for a supported type.
void iq_exact_512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                          bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], for a supported type.
void iq_exact_512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                       float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
