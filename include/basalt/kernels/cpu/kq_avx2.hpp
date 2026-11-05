// include/basalt/kernels/cpu/kq_avx2.hpp - AVX-2 multi-token dot products for Unsloth UD-Q4_K_XL's expert formats
// (Q4_K gate/up against Q8_K activations; Q5_1 down against Q8_1, Q8_0 down against Q8_0), bit-exact against
// ggml-cpu's AVX2 vec_dot for every token.  See src/kernels/cpu/kq_avx2.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Q4_K (12), Q5_1 (7), Q8_0 (8).
bool kq256_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The same for `nt` <= 8 tokens (x: nt rows of cols, cols % 8 == 0), out[t * rows + r].
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

}  // namespace basalt::kernels::cpu
