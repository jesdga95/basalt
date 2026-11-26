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

struct SharedExpertWeights {
    int gate_type = -1;
    int up_type = -1;
    int down_type = -1;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    int n_embd = 0;
    int n_ff = 0;
    float swiglu_limit = 0.f;
};

struct SharedExpertScratch {
    void* input_q8 = nullptr;
    void* hidden_q8 = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* hidden = nullptr;
};

struct RoutedExpertWeights {
    int gate_up_type = -1;
    int down_type = -1;
    int n_embd = 0;
    int n_ff = 0;
    float swiglu_limit = 10.f;
};

struct RoutedExpertPlan {
    const unsigned long long* group_ptr = nullptr;
    const int32_t* group_start = nullptr;
    const int32_t* group_count = nullptr;
    const int32_t* entry_dst = nullptr;
    const int32_t* entry_token = nullptr;
    int cap_groups = 0;
    int cap_entries = 0;
};

struct RoutedExpertScratch {
    void* input_q8 = nullptr;
    void* expert = nullptr;
    float* expert_rows = nullptr;
};

/// Collapses and normalizes [T][4][n_embd] streams, then computes V4's hash/image-aware top-6 FFN routes.
/// `normalized`, `ids`, and `weights` become the inputs to the shared and routed experts; all storage is caller-owned.
void prepare_ffn_routes(const float* streams, int n_tokens, const FfnRouteWeights& weights,
                        const FfnRouteScratch& scratch, void* stream);

/// Adds the shared and top-6 routed expert results, then expands them through FFN mHC into [T][4][n_embd].
/// `shared_output` is consumed in place; all other inputs remain unchanged.
void finish_ffn_layer(const float* streams, const float* coefficients, float* shared_output,
                      const float* routed_output, int n_tokens, int n_embd, float* next_streams, void* stream);

/// Runs the checkpoint's shared SwiGLU expert on normalized [T][n_embd] rows, T=1..8.
/// The returned [T][n_embd] output is the unweighted branch added to routed-expert output.
void shared_expert_step(const float* normalized, int n_tokens, const SharedExpertWeights& weights,
                        const SharedExpertScratch& scratch, float* output, void* stream);

/// Runs the grouped routed-expert plan for top-6 routes and sums rows by the router weights.
/// `entry_dst` indexes [token][route], and `entry_token` indexes normalized input rows.
void routed_expert_step(const float* normalized, const float* route_weights, int n_tokens,
                        const RoutedExpertWeights& weights, const RoutedExpertPlan& plan,
                        const RoutedExpertScratch& scratch, float* output, void* stream);

}  // namespace basalt::models::deepseek4
