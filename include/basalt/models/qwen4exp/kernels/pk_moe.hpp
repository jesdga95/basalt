// include/basalt/models/qwen4exp/kernels/pk_moe.hpp - the routed experts of one decode layer as ONE persistent launch.
//
// From the router's ids and weights to the layer's MoE output: every block plans the window's groups itself, runs
// gate/up items, then (group by group, as each group's rows are complete) the SwiGLU, the down items and, for the
// last block to finish a 64-column span, the weighted combine of that span.  Every value is computed by the same
// device code as the launch sequence it replaces (dec_moe_glue, dec_moe_combine), so the output has the same bits.
// All routed experts are VRAM-resident here (no peer or CPU rows).
#pragma once

#include "basalt/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// Device words the persistent launch uses (zero before the first launch; each launch leaves them zero).
constexpr int kPkSyncWords = 256;

struct PkMoeArgs {
    const int32_t* ids = nullptr;                ///< device: n_tok * k routed experts
    const float* weights = nullptr;              ///< device: n_tok * k router weights
    const unsigned long long* ptab = nullptr;    ///< device: this layer's [n_expert] blob address or 0
    int n_tok = 0, k = 0, n_expert = 0;
    const void* xq = nullptr;                    ///< device: n_tok q8_1 rows
    const float* shared = nullptr;               ///< device: n_tok * n_embd shared-expert rows added last, or null
    float* out = nullptr;                        ///< device: n_tok * n_embd
    float* parts = nullptr;                      ///< device scratch: n_tok * k * n_embd
    unsigned int* sync = nullptr;                ///< device: kPkSyncWords zeroed words
    /// the shared rows are final once sync[1] is nonzero (pk_moe_shared_done); false: they are final at launch
    bool shared_flag = false;
    unsigned long long* trace = nullptr;         ///< device: PK_T_SLOTS words of stage stamps, or null
};

/// Stage-stamp slots of PkMoeArgs::trace (device nanosecond clock; *_MIN slots start at ~0, the rest at 0).
enum PkTrace {
    PK_T_ENTRY_MIN, PK_T_ENTRY_MAX, PK_T_WAIT_MIN, PK_T_WAIT_MAX, PK_T_PLAN_MIN, PK_T_PLAN_MAX, PK_T_GU_FIRST, PK_T_GU_LAST,
    PK_T_GU_END_MIN, PK_T_GU_BUSY, PK_T_GU_ITEMS, PK_T_DOWN_FIRST, PK_T_DOWN_LAST, PK_T_DOWN_BUSY, PK_T_DOWN_ITEMS,
    PK_T_SWIGLU_WAIT, PK_T_SHARED_SEEN, PK_T_COMBINE_END, PK_T_BLOCK_END, PK_T_BLOCK_END_MIN, PK_T_SLOTS
};

/// Whether pk_moe_experts covers L.
bool pk_moe_ok(const NativeExpertLayout& L);
/// Bytes of scratch for pk_moe_experts (gate, up and the SwiGLU rows of up to kDmCap entries).
size_t pk_moe_scratch_bytes(int n_ff);
/// Launches the persistent experts + combine on `stream` as a programmatic dependent of the kernel before it (a
/// PdlScope applies).  Needs co-resident blocks: it leaves an eighth of the SMs to other streams.
void pk_moe_experts(const NativeExpertLayout& L, const PkMoeArgs& a, void* scratch, void* stream);
/// One-thread kernel: marks the shared rows final (the end of the shared expert's chain on its stream).
void pk_moe_shared_done(unsigned int* sync, void* stream);

}  // namespace basalt::kernels
