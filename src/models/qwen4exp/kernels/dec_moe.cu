// src/models/qwen4exp/kernels/dec_moe.cu - the decode MoE tail and the persistent experts, on the shared expert kernels' device code.
// The peer window kernel is not here: its four variants take minutes each to compile, so each has a TU of its own
// (dec_moe_peer_*.cu) and this one reaches it through dec_moe_peer.hpp.
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.hpp"
#include "dec_moe_kernels.cuh"   // route, stage, wait and the tests
#include "dec_moe_host.cuh"   // the launchers
#include "pk_moe.cuh"   // persistent experts + combine, on the same device code
