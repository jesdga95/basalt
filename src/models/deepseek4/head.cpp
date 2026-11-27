#include "basalt/models/deepseek4/head.hpp"

#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/mhc.hpp"

#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek4 {

void model_head_step(const float* streams, int n_tokens, const HeadWeights& weights, const HeadScratch& scratch,
                     float* logits, void* stream) {
    if (!streams || !weights.hc_fn || !weights.hc_scale || !weights.hc_base || !weights.norm || !weights.output ||
        !scratch.hc_workspace || !scratch.coefficients || !scratch.collapsed || !scratch.normalized || !logits ||
        n_tokens <= 0 || n_tokens > 8 || weights.n_embd <= 0 || weights.n_vocab <= 0 ||
        !(weights.rms_eps > 0.f) || !std::isfinite(weights.rms_eps) || !(weights.hc_eps > 0.f) ||
        !std::isfinite(weights.hc_eps))
        throw std::invalid_argument("deepseek4 model head: invalid weights, geometry, state, or workspace");

    kernels::HcPlainArgs hc;
    hc.n_embd = weights.n_embd;
    hc.rms_eps = weights.rms_eps;
    hc.hc_eps = weights.hc_eps;
    kernels::hc_mixes_plain(streams, weights.hc_fn, weights.hc_scale, weights.hc_base, n_tokens, 4, hc,
                            scratch.hc_workspace, scratch.coefficients, stream);
    kernels::hc_pre(streams, scratch.coefficients, kernels::hc_plain_coef_stride(4), n_tokens, weights.n_embd,
                    scratch.collapsed, stream);
    logits_step(scratch.collapsed, weights.norm, weights.output, n_tokens, weights.n_embd, weights.n_vocab,
                weights.rms_eps, scratch.normalized, logits, stream);
}

void logits_step(const float* hidden, const float* norm, const uint16_t* output_weight, int n_tokens,
                 int n_embd, int n_vocab, float rms_eps, float* normalized, float* logits, void* stream) {
    const void* h = hidden;
    const void* n = norm;
    const void* w = output_weight;
    const void* z = normalized;
    const void* y = logits;
    if (!hidden || !norm || !output_weight || !normalized || !logits || !stream || n_tokens <= 0 || n_tokens > 8 ||
        n_embd <= 0 || (n_embd & 1) || n_vocab <= 0 || !(rms_eps > 0.f) || !std::isfinite(rms_eps) ||
        h == n || h == w || h == z || h == y || n == w || n == z || n == y || w == z || w == y || z == y)
        throw std::invalid_argument("deepseek4 logits: invalid weights, geometry, state, or workspace");

    kernels::rms_norm_rows(hidden, norm, normalized, n_tokens, n_embd, rms_eps, stream);
    kernels::bf16_gemv_fp32_mmvf_multi(normalized, n_embd, output_weight, logits, n_vocab, n_embd,
                                       n_vocab, n_tokens, stream);
}

}  // namespace basalt::models::deepseek4
