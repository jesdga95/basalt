// src/models/qwen4exp/kernels/dec_moe.cu - the decode MoE tail and the persistent experts, on the shared expert kernels' device code.
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.cuh"   // the peer window
#include "dec_moe_kernels.cuh"   // route, stage, wait and the tests
#include "dec_moe_host.cuh"   // the launchers
#include "pk_moe.cuh"   // persistent experts + combine, on the same device code
