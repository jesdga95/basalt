// include/basalt/kernels/router.hpp - the MoE router: scores from a gate matrix, the top-k of the biased scores, the chosen
// unbiased scores normalised.
//
// All activations are fp32 and device pointers; a call takes T tokens (any T >= 1) and the result for a token does not depend
// on T or on the token's place in the window. `stream` may be null (the call then synchronises).
#pragma once

#include <cstdint>

namespace basalt::kernels {

enum class RouterGate : int { SqrtSoftplus = 0, Sigmoid = 1 };

struct RouterArgs {
    int n_expert = 384;               ///< multiple of 32, at most 512
    int n_embd = 5120;                ///< multiple of 128
    int k = 6;                        ///< at most 16
    float weight_scale = 1.5f;        ///< multiplies the normalised weights
    RouterGate gate = RouterGate::SqrtSoftplus;
};

/// logits = gate . x (fp32, gate row-major [n_expert][n_embd]); s = the gate function of the logit; the top-k of s + bias
/// are chosen (the lower index wins a tie); weights are the unbiased s of those experts, divided by their sum plus
/// 1e-20, times weight_scale. `logits` is a [T][n_expert] output (workspace); ids and weights are [T][k].
void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream);

}  // namespace basalt::kernels
