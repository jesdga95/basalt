// include/basalt/kernels/router.hpp - the MoE router: scores from a gate matrix, the top-k of the biased scores, the chosen
// unbiased scores normalised.
//
// All activations are fp32 and device pointers; a call takes T tokens (any T >= 1) and the result for a token does not depend
// on T or on the token's place in the window. `stream` may be null (the call then synchronises).
#pragma once

#include "basalt/kernels/route_lossy.hpp"

#include <cstdint>

namespace basalt::kernels {

enum class RouterGate : int { SqrtSoftplus = 0, Sigmoid = 1 };

struct RouterArgs {
    int n_expert = 384;               ///< multiple of 32, at most 512
    int n_embd = 5120;                ///< multiple of 128
    int k = 6;                        ///< at most 16
    float weight_scale = 1.5f;        ///< multiplies the normalised weights
    RouterGate gate = RouterGate::SqrtSoftplus;
    /// --route-prior (opt-in, changes the output): above 0, an expert `held` holds is chosen on s + bias + `prior`; the
    /// weights stay the unbiased s of the chosen, normalised as above.  0 = the plain router, bit for bit.
    float prior = 0.0f;
    ResidentTabs held;
};

/// One router selection call with fixed-id and score-selected rows mixed per token.
struct RouterMixedArgs {
    int n_expert = 256;               ///< multiple of 32, at most 512
    int k = 6;                        ///< at most 16
    float weight_scale = 1.5f;
    RouterGate gate = RouterGate::SqrtSoftplus;
    const int32_t* fixed_ids = nullptr;       ///< device [T][k]; direct fixed rows
    const int32_t* fixed_table = nullptr;     ///< device [rows][k]; alternative source for fixed rows
    const int32_t* fixed_rows = nullptr;      ///< device [T]; row indices into fixed_table when fixed_mask is set
    int fixed_table_rows = 0;                 ///< bounds for fixed_rows; validated by the caller
    const uint8_t* fixed_mask = nullptr;      ///< device [T]; nonzero rows use fixed_ids or fixed_table and ignore biases
    const uint8_t* alternate_mask = nullptr;  ///< device [T]; nonzero dynamic rows use alternate_bias
    const float* bias = nullptr;              ///< device [n_expert]; null means zero bias for ordinary dynamic rows
    const float* alternate_bias = nullptr;    ///< device [n_expert]; optional selection bias for alternate rows
};

/// logits = gate . x (fp32, gate row-major [n_expert][n_embd]); s = the gate function of the logit; the top-k of s + bias
/// are chosen (the lower index wins a tie); weights are the unbiased s of those experts, divided by their sum plus
/// 1e-20, times weight_scale. `logits` is a [T][n_expert] output (workspace); ids and weights are [T][k].
void router_logits(const float* x, const float* gate, int n_tokens, const RouterArgs& a, float* logits, void* stream);
void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream);

/// Same as `router_logits` / `router` with BF16 gate weights (exact bit-for-bit match to fp32 weights whose low 16 bits
/// are zero, at half the VRAM and half the memory traffic).
void router_logits_bf16(const float* x, const uint16_t* gate, int n_tokens, const RouterArgs& a, float* logits, void* stream);
void router_bf16(const float* x, const uint16_t* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
                 int32_t* ids, float* weights, void* stream);

/// router's choice from logits computed another way (a prompt chunk's scores as one product): the same selection and
/// weights `router` makes from the same logits.
void router_select(const float* logits, const float* bias, int n_tokens, const RouterArgs& a, int32_t* ids, float* weights,
                   void* stream);

/// Fixed rows copy prescribed [T][k] expert ids or gather a [fixed_table_rows][k] row selected by fixed_rows[t], then weight
/// them from unbiased gate scores. Other rows choose top-k from scores plus `bias`, or `alternate_bias` when
/// alternate_mask[t] is nonzero. A null `bias` means zero ordinary bias. Both paths normalize selected unbiased scores and
/// multiply by weight_scale. Fixed ids must be in range and unique; the model validates table contents and row indices before
/// dispatch. `logits` is [T][n_expert]; masks, indices, and ids are device pointers.
void router_select_mixed(const float* logits, int n_tokens, const RouterMixedArgs& a, int32_t* ids, float* weights,
                         void* stream);

}  // namespace basalt::kernels
