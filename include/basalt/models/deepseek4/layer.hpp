#pragma once

#include "basalt/kernels/mhc.hpp"
#include "basalt/models/deepseek4/attention.hpp"
#include "basalt/models/deepseek4/router.hpp"

namespace basalt::models::deepseek4 {

struct AttentionLayerWeights {
    kernels::HcArgs hc;
    const float* hc_fn = nullptr;
    const float* hc_scale = nullptr;
    const float* hc_base = nullptr;
    const float* norm = nullptr;
    AttentionStepWeights attention;
};

struct AttentionLayerScratch {
    float* hc_workspace = nullptr;
    float* coefficients = nullptr;
    float* collapsed = nullptr;
    float* normalized = nullptr;
    AttentionStepScratch attention;
};

/// Runs one V4 attention sublayer for one token. `streams` and `next_streams` are [4][n_embd].
/// The output does not alias the input; all scratch and cache storage belongs to the caller.
void attention_layer_step(const float* streams, const AttentionLayerWeights& weights, const kernels::MqaRopeTable& rope,
                         float* window_ring, kernels::SparseMqaArgs selection, const AttentionLayerScratch& scratch,
                         float* next_streams, void* stream);

struct FfnRouteWeights {
    kernels::HcArgs hc;
    const float* hc_fn = nullptr;
    const float* hc_scale = nullptr;
    const float* hc_base = nullptr;
    const float* norm = nullptr;
    const float* gate = nullptr;
    kernels::RouterMixedArgs routing;
};

struct FfnRouteScratch {
    float* hc_workspace = nullptr;
    float* coefficients = nullptr;
    float* collapsed = nullptr;
    float* normalized = nullptr;
    float* logits = nullptr;
    int32_t* ids = nullptr;
    float* weights = nullptr;
};

/// Collapses and normalizes [T][4][n_embd] streams, then computes V4's hash/image-aware top-6 FFN routes.
/// `normalized`, `ids`, and `weights` become the inputs to the shared and routed experts; all storage is caller-owned.
void prepare_ffn_routes(const float* streams, int n_tokens, const FfnRouteWeights& weights,
                        const FfnRouteScratch& scratch, void* stream);

}  // namespace basalt::models::deepseek4
