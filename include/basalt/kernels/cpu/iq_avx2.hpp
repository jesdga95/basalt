// include/basalt/kernels/cpu/iq_avx2.hpp - AVX-2 multi-token dot products for the i-quant expert
// formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) against Q8_K activations (ggml's block_q8_K).
//
// The AVX-512 kernel's scheme on the CPUs without it: the weights of a
// 32-value chunk are decoded once per verify window and every token applies them with five instructions.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// ggml's quantize_row_q8_K for k floats (a multiple of 256), bit for bit.
void quantize_q8_K_avx2(const float* x, void* dst, int64_t k);
bool iq256_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The same two with the rows' arithmetic spelled out, for the tests: `_base` sums the int16 products with `madd` and `add`,
/// `_vnni` with the one AVX-VNNI instruction that is exactly that (callable only where cpu_avxvnni_ok()). The functions
/// above and below take `_vnni` where the CPU has it; every output bit is the same.
void iq256_gu_rows_base(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        int nt, float* const* ff, int r0, int r1);
void iq256_gu_rows_vnni(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        int nt, float* const* ff, int r0, int r1);
void iq256_rows_base(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                     float* const* out, int r0, int r1);
void iq256_rows_vnni(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                     float* const* out, int r0, int r1);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0).
/// IQ4_NL is a 32-value-block format, so this does not go through iq256_rows (QK_K blocks, Q8_K acts).
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);

void iq4nl256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                             float* const* out, int r0, int r1);
void iq4nl256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                             float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
