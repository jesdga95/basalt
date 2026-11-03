// include/basalt/kernels/cpu/neon.hpp - the Arm64 (NEON with the dot product) CPU kernels.
//
// The Arm64 counterparts of the AVX2 / AVX-512 kernels: the same contracts, each file summing in its own lane order,
// so a row's low bits differ from an x86 build's within the tolerances the parity tests state. A token's row rounds
// the same alone as in a window of several, on every kernel here. Built and called on Arm64 only.
#pragma once

#include "basalt/kernels/cpu/expert.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

// cpu_neon_ok() (expert_layout.hpp) says whether this CPU runs them.


/// Q2_0 rows [r0, r1) in the GGUF block layout against `nt` activations (q2_0_gguf_rows_multi's contract).
void q2_0_gguf_rows_multi_neon(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1);
/// act_quant_q8_1's contract: the scalar rule (round half away from zero), bit for bit.
void act_quant_q8_1_neon(const float* x, int n, ActQ& a);

/// The router lookahead's dot (decode/verify/cpu_warm.cpp): Q4_0-style rows (16 nibble bytes and one FP32 scale per
/// 32 values) against `nt` block_q8_1 activation rows of `nb` blocks; out[t] = row . x[t].
void router_q4_dot_neon(const uint8_t* row, const float* scales, int nb, const uint8_t* xq, int nt, float* out);

}  // namespace basalt::kernels::cpu
