// include/basalt/kernels/cpu/nvfp4_neon.hpp - NEON multi-token dot products for NVFP4 expert rows against Q8_0
// activations, bitwise equal per token to the NEON ggml NVFP4 dot (ggml_cpu_dots_neon.c). Arm64 builds only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Whether a layer's geometry fits these kernels (rows a multiple of 64 values, at most 16384).
bool nvfp4_neon_fits(int n_embd, int n_ff) noexcept;
/// ff[t][r] = silu(sg * gate_r . a[t]) * (su * up_r . a[t]), rows [r0, r1); gate rows at blob, up at blob + up_off.
void nvfp4_neon_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                        const void* const* act, int nt, float* const* ff, int r0, int r1);
/// out[t][r] = sd * (w_r . a[t]), rows [r0, r1).
void nvfp4_neon_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt,
                          float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
