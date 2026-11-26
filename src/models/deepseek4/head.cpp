#include "basalt/models/deepseek4/head.hpp"

#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"

#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek4 {

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
