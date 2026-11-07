// include/basalt/models/qwen4exp/prefill/moe_fused_iq.hpp - the fused int8 prompt experts (moe_fused.hpp) for the native GGUF
// experts (an expert is its raw GGUF slices [gate rows | up rows | down rows], formats per layer in
// expert_layout().fmt), used wherever available (devices with int8 MMA; others use MMQ).
//
// The grouping on the GPU (fused::group), the per-expert pointers (fused::Batch: a cache slot or a ring slot, no
// gather) and the launch shape are the Q2_0 path's.  What differs is the load stage: the i-quant blocks are decoded to
// int8 in shared memory before the mma (codebook and sign lookups, as llama.cpp's MMQ load_tiles does), with the
// block's scale per 32 (or per 16: IQ2_XS, IQ2_S, multiplied at m16n8k16) beside them.  The activations and H are
// int8 per 32 values in their natural order here (quantize_act_native), not the Q2_0 path's permuted one.
//
// Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL - every layer of the IQ2_XS,
// IQ3_XXS and IQ3_S packs but the IQ1_M ones (which MMQ does not cover either).  Other layers keep MMQ.
#pragma once

#include "basalt/models/qwen4exp/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::fused {

/// One layer's native expert geometry (cpu::NativeFmt's fields the kernels read; n_embd 2560, n_ff 640).
struct NativeGeom {
    int gu_type = -1, d_type = -1;    ///< ggml types
    size_t gu_row = 0, d_row = 0;     ///< bytes per gate/up row and per down row
    size_t up_off = 0, down_off = 0;  ///< inside the blob
};

/// enabled() (int8 MMA), and the kernels cover this gate/up and down pair on this device.
bool native_supported(int gu_type, int d_type);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)): int8 per 32 values in natural order, the native kernels' form.
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// experts() for native blobs: b.blob[e - e0] is the native blob of expert e (gate at 0, up at g.up_off, down at
/// g.down_off; 2-byte aligned).  `xa` from quantize_act_native; H in `ha` (act_bytes(n, 640)); down into `dm` at the
/// rows of `group`.
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);

/// experts_native() runs IQ3_XXS gate/up and IQ4_NL down on the v2 kernels
/// (raw weight bytes by cp.async, decoded straight into the mma fragments, a work item over up to 128 routed rows);
/// the other formats keep the kernels above.  Same math, rounded in another order.
/// Tests: native_v2_override(1 / 0) forces v2 on / off, -1 returns to the variable.
void native_v2_override(int mode);
/// v2 is wanted (the variable or the override) and its kernels run on the current device.
bool native_v2_ready();

}  // namespace basalt::prefill::fused
