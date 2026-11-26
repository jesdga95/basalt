#include "basalt/models/deepseek4/layer.hpp"

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/row_norm.hpp"

#include "shared_expert.hpp"
#include "routed_expert.hpp"

#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek4 {

void prepare_attention_layer(const float* streams, int n_tokens, const AttentionLayerWeights& weights,
                             const AttentionLayerScratch& scratch, void* stream) {
    const int n_embd = weights.hc.n_embd;
    if (!streams || !weights.hc_fn || !weights.hc_scale || !weights.hc_base || !weights.norm ||
        !scratch.hc_workspace || !scratch.coefficients || !scratch.collapsed || !scratch.normalized ||
        weights.hc.sinkhorn_iters <= 0 || n_embd <= 0 || weights.attention.q.n_embd != n_embd ||
        n_tokens <= 0 || n_tokens > 8)
        throw std::invalid_argument("deepseek4 attention layer prepare: invalid weights, state, or workspace");

    kernels::hc_mixes(streams, weights.hc_fn, weights.hc_scale, weights.hc_base, n_tokens, weights.hc,
                      scratch.hc_workspace, scratch.coefficients, stream);
    kernels::hc_pre_rms_norm(streams, scratch.coefficients, kernels::kHcCoef, weights.norm, n_tokens, n_embd,
                             weights.hc.rms_eps, scratch.collapsed, scratch.normalized, stream);
}

void finish_attention_layer(const float* streams, const float* coefficients, const float* attention_output,
                            int n_tokens, int n_embd, float* next_streams, void* stream) {
    if (!streams || !coefficients || !attention_output || !next_streams || n_tokens <= 0 || n_tokens > 8 ||
        n_embd <= 0 || next_streams == streams || next_streams == coefficients ||
        next_streams == attention_output)
        throw std::invalid_argument("deepseek4 attention layer finish: invalid state or output");
    kernels::hc_post(attention_output, streams, coefficients, kernels::kHcCoef, n_tokens, n_embd, next_streams, stream);
}

void attention_layer_step(const float* streams, const AttentionLayerWeights& weights,
                          const kernels::MqaRopeTable& rope, float* window_ring,
                          kernels::SparseMqaArgs selection, const AttentionLayerScratch& scratch,
                          float* next_streams, void* stream) {
    const int n_embd = weights.hc.n_embd;
    const int n_tokens = selection.n_tokens;
    if (!streams || !next_streams || streams == next_streams || !weights.hc_fn || !weights.hc_scale ||
        !weights.hc_base || !weights.norm || !window_ring || !scratch.hc_workspace || !scratch.coefficients ||
        !scratch.collapsed || !scratch.normalized || !scratch.attention.output || weights.hc.sinkhorn_iters <= 0 ||
        n_embd <= 0 || weights.attention.q.n_embd != n_embd || selection.n_tokens <= 0 || selection.n_tokens > 8)
        throw std::invalid_argument("deepseek4 attention layer: invalid weights, state, or workspace");

    prepare_attention_layer(streams, n_tokens, weights, scratch, stream);
    attention_step(scratch.normalized, weights.attention, rope, window_ring, selection, scratch.attention, stream);
    finish_attention_layer(streams, scratch.coefficients, scratch.attention.output, n_tokens, n_embd,
                           next_streams, stream);
}

void prepare_ffn_routes(const float* streams, int n_tokens, const FfnRouteWeights& weights,
                        const FfnRouteScratch& scratch, void* stream) {
    const int n_embd = weights.hc.n_embd;
    if (!streams || n_tokens <= 0 || n_embd <= 0 || !weights.hc_fn || !weights.hc_scale || !weights.hc_base ||
        !weights.norm || !weights.gate || !scratch.hc_workspace || !scratch.coefficients || !scratch.collapsed ||
        !scratch.normalized || !scratch.logits || !scratch.ids || !scratch.weights || weights.hc.sinkhorn_iters <= 0 ||
        weights.routing.k != 6 || weights.routing.n_expert <= 0)
        throw std::invalid_argument("deepseek4 FFN route preparation: invalid weights, state, or workspace");

    kernels::hc_mixes(streams, weights.hc_fn, weights.hc_scale, weights.hc_base, n_tokens, weights.hc,
                      scratch.hc_workspace, scratch.coefficients, stream);
    kernels::hc_pre_rms_norm(streams, scratch.coefficients, kernels::kHcCoef, weights.norm, n_tokens, n_embd,
                             weights.hc.rms_eps, scratch.collapsed, scratch.normalized, stream);
    router_step(scratch.normalized, weights.gate, n_tokens, n_embd, weights.routing, scratch.logits,
                scratch.ids, scratch.weights, stream);
}

void finish_ffn_layer(const float* streams, const float* coefficients, float* shared_output,
                      const float* routed_output, int n_tokens, int n_embd, float* next_streams, void* stream) {
    if (!streams || !coefficients || !shared_output || !routed_output || !next_streams ||
        n_tokens <= 0 || n_embd <= 0 || streams == shared_output || streams == routed_output ||
        shared_output == routed_output || coefficients == shared_output || next_streams == streams ||
        next_streams == coefficients ||
        next_streams == shared_output || next_streams == routed_output)
        throw std::invalid_argument("deepseek4 FFN layer finish: invalid state or output");
    kernels::add_scaled(shared_output, routed_output, nullptr, (int64_t) n_tokens * n_embd, stream);
    kernels::hc_post(shared_output, streams, coefficients, kernels::kHcCoef, n_tokens, n_embd, next_streams, stream);
}

void shared_expert_step(const float* normalized, int n_tokens, const SharedExpertWeights& weights,
                        const SharedExpertScratch& scratch, float* output, void* stream) {
    if (!normalized || !output || normalized == output || n_tokens <= 0 || n_tokens > 8 || weights.n_embd <= 0 ||
        weights.n_ff <= 0 || !weights.gate || !weights.up || !weights.down || !scratch.input_q8 ||
        !scratch.hidden_q8 || !scratch.gate || !scratch.up || !scratch.hidden || !stream ||
        !(weights.swiglu_limit > 0.f) ||
        !std::isfinite(weights.swiglu_limit) || weights.n_embd % 32 != 0 || weights.n_ff % 32 != 0 ||
        !kernels::native_mmvq_supported(weights.gate_type) || !kernels::native_mmvq_supported(weights.up_type) ||
        !kernels::native_mmvq_supported(weights.down_type))
        throw std::invalid_argument("deepseek4 shared expert: invalid weights, geometry, or workspace");

    (void) kernels::native_mmvq_weight_bytes(weights.gate_type, weights.n_embd, weights.n_ff);
    (void) kernels::native_mmvq_weight_bytes(weights.up_type, weights.n_embd, weights.n_ff);
    (void) kernels::native_mmvq_weight_bytes(weights.down_type, weights.n_ff, weights.n_embd);
    kernels::native_quantize_q8_1(normalized, scratch.input_q8, weights.n_embd, n_tokens, stream);
    kernels::native_mmvq(weights.gate_type, weights.gate, scratch.input_q8, scratch.gate, weights.n_embd, weights.n_ff,
                         n_tokens, stream);
    kernels::native_mmvq(weights.up_type, weights.up, scratch.input_q8, scratch.up, weights.n_embd, weights.n_ff,
                         n_tokens, stream);
    swiglu_clamped(scratch.gate, scratch.up, weights.n_ff, weights.swiglu_limit, scratch.hidden, n_tokens, stream);
    kernels::native_quantize_q8_1(scratch.hidden, scratch.hidden_q8, weights.n_ff, n_tokens, stream);
    kernels::native_mmvq(weights.down_type, weights.down, scratch.hidden_q8, output, weights.n_ff, weights.n_embd,
                         n_tokens, stream);
}

void routed_expert_step(const float* normalized, const float* route_weights, int n_tokens,
                        const RoutedExpertWeights& weights, const RoutedExpertPlan& plan,
                        const RoutedExpertScratch& scratch, float* output, void* stream) {
    constexpr int K = 6;
    if (!normalized || !route_weights || !output || normalized == output || n_tokens <= 0 || n_tokens > 8 ||
        weights.n_embd <= 0 || weights.n_ff <= 0 || weights.n_embd % 32 != 0 || weights.n_ff % 32 != 0 ||
        !std::isfinite(weights.swiglu_limit) || !(weights.swiglu_limit > 0.f) || !plan.group_ptr ||
        !plan.group_start || !plan.group_count || !plan.entry_dst || !plan.entry_token || plan.cap_groups <= 0 ||
        plan.cap_groups > n_tokens * K || plan.cap_entries < n_tokens * K || !scratch.input_q8 || !scratch.expert ||
        !scratch.expert_rows || !stream ||
        !kernels::native_expert_supported(weights.gate_up_type, weights.down_type, weights.n_embd, weights.n_ff))
        throw std::invalid_argument("deepseek4 routed expert: invalid weights, plan, or workspace");

    const kernels::NativeExpertLayout layout = kernels::native_expert_layout(
        weights.gate_up_type, weights.down_type, weights.n_embd, weights.n_ff, weights.swiglu_limit);
    kernels::native_quantize_q8_1(normalized, scratch.input_q8, weights.n_embd, n_tokens, stream);
    kernels::native_expert_grouped_clamped(layout, weights.swiglu_limit, plan.group_ptr, plan.group_start,
        plan.group_count, plan.entry_dst, plan.entry_token, plan.cap_groups, plan.cap_entries, scratch.input_q8,
        scratch.expert, scratch.expert_rows, stream);
    weighted_route_sum(scratch.expert_rows, route_weights, n_tokens, weights.n_embd, output, stream);
}

}  // namespace basalt::models::deepseek4
