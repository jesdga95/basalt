// include/basalt/kernels/cpu/router_q4.hpp - a layer's router scored on the CPU from its 4-bit codes, for the CPU
// tier's cache warm-up (src/core/verify/cpu_warm.cpp). A prediction only. Built as AVX2 on x86-64
// (router_q4_avx2.cpp) and as NEON on Arm64 (router_q4_neon.cpp).
#pragma once

#include <atomic>
#include <cstdint>

namespace basalt::kernels::cpu {

/// The most tokens one call scores.
inline constexpr int kRouterQ4MaxT = 8;

/// Router rows [e0, e1) of one layer against `nt` (at most kRouterQ4MaxT) block_q8_1 activations `xq` (36 bytes per 32
/// values, token after token): logits[t * n_expert + e]. `rows` holds n_embd / 2 bytes per expert: per 32 values,
/// value i in the low nibble of byte i and value i + 16 in its high nibble, each the code plus 8; `scales` one float
/// per 32 values. Returns between rows once `stop` is set.
void router_q4_logits(const uint8_t* rows, const float* scales, int n_embd, int n_expert, int e0, int e1,
                      const uint8_t* xq, int nt, float* logits, const std::atomic<bool>& stop);

}  // namespace basalt::kernels::cpu
