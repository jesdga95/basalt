#pragma once

#include <cstdint>

namespace basalt::models::deepseek4 {

struct HeadWeights {
    const float* hc_fn = nullptr;
    const float* hc_scale = nullptr;
    const float* hc_base = nullptr;
    const float* norm = nullptr;
    const uint16_t* output = nullptr;
    int n_embd = 0;
    int n_vocab = 0;
    float rms_eps = 0.f;
    float hc_eps = 0.f;
};

struct HeadScratch {
    float* hc_workspace = nullptr;
    float* coefficients = nullptr;
    float* collapsed = nullptr;
    float* normalized = nullptr;
};

/// Collapses final [T][4][n_embd] streams through V4's learned head mHC, applies RMSNorm and emits logits.
/// All workspace is caller-owned and sized for 1..8 tokens.
void model_head_step(const float* streams, int n_tokens, const HeadWeights& weights, const HeadScratch& scratch,
                     float* logits, void* stream);

/// Applies final RMSNorm and the checkpoint's BF16 output matrix to 1..8 hidden rows.
/// `normalized` and `logits` are caller-owned [T][n_embd] and [T][n_vocab] workspaces.
void logits_step(const float* hidden, const float* norm, const uint16_t* output_weight, int n_tokens,
                 int n_embd, int n_vocab, float rms_eps, float* normalized, float* logits, void* stream);

}  // namespace basalt::models::deepseek4
