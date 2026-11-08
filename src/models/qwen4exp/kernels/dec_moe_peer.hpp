// src/models/qwen4exp/kernels/dec_moe_peer.hpp - the peer window kernel's four variants, one TU each (dec_moe_peer_*.cu): each returns its
// kernel's address for dec_moe.cu to launch.  The name carries the variant: q2 = Q2_0 gate/up rows, mix = the GDN mixer split's item.
#pragma once

namespace basalt::kernels {
const void* dm_peer_fn_q0_m0();
const void* dm_peer_fn_q0_m1();
const void* dm_peer_fn_q1_m0();
const void* dm_peer_fn_q1_m1();
}  // namespace basalt::kernels
