// include/basalt/kernels/shared_expert.hpp - the shared expert, host-callable.
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU ON GATE
//     h = h @ down_shexp.T
//     g = sigmoid(x @ gate_inp_shexp)                     one SCALAR per token
//     return h * g[:, None]
//
// The result is ADDED to the routed output, not weighted against it.  Shape note: `n_ff` for the shared expert
// matches the routed experts' width (640) in this artifact.
//
// The SForm arguments describe canonical planes for the three expert projections and their historical
// CPU-derived activation images (Q8_0 or Q8_K). The default scalar gate separately consumes BF16 activations.
// The opt-in native BF16 gate instead consumes the original F32 activation and uses the pinned CUDA MMVF
// contract. Optional native projection weights select the pinned CUDA Q8_1 MMVQ contract independently.
#pragma once

#include <cstdint>

#include "basalt/kernels/s_gemv.hpp"

namespace basalt::kernels {

/// Optional native GGUF projections. Each supported type with nonnull data
/// replaces only that canonical projection; absent or unsupported entries fall
/// back independently. Any active entry selects pinned FP32 SwiGLU arithmetic.
/// Caller owns device weights and Q8_1 scratch for the largest active input
/// (n_embd for gate/up, n_ff for down), sized by native_q8_1_bytes. All storage
/// must survive graph execution; use one ordered nonnull stream with no overlap.
struct NativeSharedWeights {
    int gate_type = -1, up_type = -1, down_type = -1;
    const void* gate_data = nullptr;
    const void* up_data = nullptr;
    const void* down_data = nullptr;
    void* q8_1 = nullptr;
};

/// Bytes of caller-owned scratch `shared_expert` needs.  **THE KERNEL ALLOCATES NOTHING**:
/// `cudaMalloc` is ILLEGAL INSIDE A CAPTURE, and the token path must make no allocations.
uint64_t shared_expert_scratch_bytes(int64_t n_ff);

/// `x_q8_0` and `x_q8k` are the TWO QUANTIZED IMAGES of the same activation, and which one a projection uses is
/// decided per weight by its `SForm::act_kind`.  There is no fp16 activation here any more.
///
/// **THE ACTIVATION IS NOT FP16.**  `ffn_down_shexp` is IQ4_NL/Q4_0/Q5_0/Q8_0/Q2_0 in every layer - **no K-quant at all** - so its contract is
/// Q8_0; and its `n_in` is 640, which is not a multiple of 256, so **Q8_K is structurally impossible for it**.
/// The gate/up weights are Q2_0 on 21 and 13 layers (Q8_0 again) and K-quants on the rest.
///
/// x_f32 is required (the gate and the native gate/up projections read it).
/// Native down reads the unrounded F32 SwiGLU intermediate directly. No allocation occurs.
///
/// `s_gemv_q8_0_split` serves the Q8_0 half and `s2_gemv_q8` the Q2_0 half.
void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32 = nullptr,
                   const NativeSharedWeights* native = nullptr);

/// Plan v0.3 P6: the shared expert for `n_tok` <= 8 tokens (a verify window) with all three projections native:
/// multi-column MMVQ, so the weights are read once; every token is bitwise `shared_expert` on that token.
/// `x` (n_tok, n_embd) f32, `gate`/`up`
/// (n_tok, n_ff) scratch, `g` n_tok floats, `out` (n_tok, n_embd).  `nw.q8_1` must hold n_tok columns of n_embd.
void shared_expert_multi(int n_tok, const float* x, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream);

/// shared_expert_multi's projections alone, for a caller that has x's q8_1 rows (native_quantize_q8_1's) in
/// `nw.q8_1` and applies the per-token gate itself: `out` = down(swiglu(gate, up)), bitwise shared_expert_multi's
/// before its gate.  With `stream2` (and `join`, an event) the up projection runs there beside the gate projection;
/// stream2 must already follow everything `nw.q8_1`'s rows wait for.
void shared_expert_multi_rows(int n_tok, const NativeSharedWeights& nw, float* gate, float* up, float* out,
                              int64_t n_embd, int64_t n_ff, void* stream, void* stream2 = nullptr,
                              void* join = nullptr);

/// The MoE block's final combination:
///
///     y = sum over the k routed experts of  w[k] * parts[k]  +  shared
///
/// `parts` is (k, n_embd) contiguous - one row per SELECTED expert, in the router's own order - and `shared`
/// may be null for a layer with no shared expert.
///
/// TWO THINGS A READER GETS WRONG HERE, and they are why this is not inlined at the call site:
///
///   * The ROUTED outputs are router-WEIGHTED; the SHARED output is ADDED PLAIN.  Not renormalised against it; weighting the shared term too is a
///     one-line change that produces a plausible number.
///   * The weights come from `router_top10` ALREADY renormalised (they sum to 1 over the selected experts).
///     Normalising again here would be a second, different normalisation, and it is invisible whenever the
///     router's weights happen to sum near 1 anyway.
///
/// One thread per output element, k accumulations each; k is 10 at the real geometry.  This is where the CPU
/// expert pool's outputs are summed: `ExpertPool` carries a `weight` per job and deliberately does NOT apply
/// it, because the sum has to happen once over all k and not once per worker.
void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream);

}  // namespace basalt::kernels
