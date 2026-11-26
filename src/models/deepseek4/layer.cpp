#include "basalt/models/deepseek4/layer.hpp"

#include "basalt/kernels/row_norm.hpp"

#include <stdexcept>

namespace basalt::models::deepseek4 {

void attention_layer_step(const float* streams, const AttentionLayerWeights& weights,
                          const kernels::MqaRopeTable& rope, float* window_ring,
                          kernels::SparseMqaArgs selection, const AttentionLayerScratch& scratch,
                          float* next_streams, void* stream) {
    const int n_embd = weights.hc.n_embd;
    if (!streams || !next_streams || streams == next_streams || !weights.hc_fn || !weights.hc_scale ||
        !weights.hc_base || !weights.norm || !window_ring || !scratch.hc_workspace || !scratch.coefficients ||
        !scratch.collapsed || !scratch.normalized || !scratch.attention.output || weights.hc.sinkhorn_iters <= 0 ||
        n_embd <= 0 || weights.attention.q.n_embd != n_embd || selection.n_tokens != 1)
        throw std::invalid_argument("deepseek4 attention layer: invalid weights, state, or workspace");

    kernels::hc_mixes(streams, weights.hc_fn, weights.hc_scale, weights.hc_base, 1, weights.hc,
                      scratch.hc_workspace, scratch.coefficients, stream);
    kernels::hc_pre_rms_norm(streams, scratch.coefficients, kernels::kHcCoef, weights.norm, 1, n_embd,
                             weights.hc.rms_eps, scratch.collapsed, scratch.normalized, stream);
    attention_step(scratch.normalized, weights.attention, rope, window_ring, selection, scratch.attention, stream);
    kernels::hc_post(scratch.attention.output, streams, scratch.coefficients, kernels::kHcCoef, 1, n_embd,
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

}  // namespace basalt::models::deepseek4
