// include/basalt/kernels/cpu/nvfp4_avx512.hpp - AVX-512 VNNI multi-token dot products for NVFP4 expert rows against
// Q8_0 activations, bitwise equal per token to ggml's single-token NVFP4 dot on the same CPU.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Whether a layer's geometry fits these kernels (rows a multiple of 64 values, at most 16384).
bool nvfp4_512_fits(int n_embd, int n_ff) noexcept;
/// ff[t][r] = silu(g) * u with g = sg * gate_r . a[t], u = su * up_r . a[t], rows [r0, r1); gate rows at blob, up at
/// blob + up_off.  `limit` > 0: g capped above at it and u clamped to [-limit, limit] first (the limited SwiGLU).
void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit,
                       const void* const* act, int nt, float* const* ff, int r0, int r1);
/// out[t][r] = sd * (w_r . a[t]), rows [r0, r1).
void nvfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt,
                         float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
