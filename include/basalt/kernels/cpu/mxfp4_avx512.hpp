// include/basalt/kernels/cpu/mxfp4_avx512.hpp - AVX-512 VNNI multi-token dot products for MXFP4 expert rows against
// Q8_0 activations, bitwise equal per token to ggml's single-token MXFP4 dot.
#pragma once

#include <cstddef>
#include <cstdint>

#include "basalt/kernels/cpu/swiglu.hpp"

namespace basalt::kernels::cpu {

/// Whether a layer's geometry fits these kernels (rows a multiple of 64 values, at most 16384).
bool mxfp4_512_fits(int n_embd, int n_ff) noexcept;

/// ff[t][r] = swiglu(glu, gate_r . a[t], up_r . a[t]) for rows [r0, r1); gate rows at blob, up at blob + up_off.
void mxfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                       const void* const* act, int nt, float* const* ff, int r0, int r1);

/// out[t][r] = w_r . a[t] for rows [r0, r1).
void mxfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                         int r0, int r1);

}  // namespace basalt::kernels::cpu
