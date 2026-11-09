// src/models/qwen4exp/kernels/dec_moe_prelude.cuh - what every dec_moe TU (dec_moe.cu and the dec_moe_peer_*.cu) starts from: the shared expert
// kernels' device code and the peer mixer's recurrence, then dec_moe.cuh on top of them.
#pragma once
#define IQ_DEV_ONLY
#include "../../../kernels/cuda/iq_kernels.cu"
#undef IQ_DEV_ONLY

#define FZ_DEV_ONLY
#include "dec_gdn_rec.cu"   // the peer mixer item's recurrence (device code only)
#undef FZ_DEV_ONLY
#include "dec_moe.cuh"   // DEC_MOE: the decode MoE tail's device code, on the shared row dots
