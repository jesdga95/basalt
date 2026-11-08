// src/models/qwen4exp/kernels/dec_moe_peer_q0_m1.cu - the peer window kernel, Q2 = false, MIX = true (see dec_moe_peer.hpp).
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.cuh"
#include "dec_moe_peer.hpp"

namespace basalt::kernels {
const void* dm_peer_fn_q0_m1() { return (const void*) dm_peer_kernel<false, true>; }
}  // namespace basalt::kernels
