// src/models/qwen4exp/kernels/dec_moe.cu - the decode MoE tail and the persistent experts, on the shared expert kernels' device code.
#define IQ_DEV_ONLY
#include "../../../kernels/cuda/iq_kernels.cu"
#undef IQ_DEV_ONLY

#define FZ_DEV_ONLY
#include "dec_gdn_rec.cu"   // the peer mixer item's recurrence (device code only)
#undef FZ_DEV_ONLY
#include "dec_moe.cuh"   // DEC_MOE: the decode MoE tail's kernels, on the shared row dots
#include "pk_moe.cuh"   // persistent experts + combine, on the same device code
