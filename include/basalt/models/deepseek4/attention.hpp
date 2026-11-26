#pragma once

#include "basalt/kernels/sparse_mqa.hpp"

#include <cstdint>
#include <cstddef>

namespace basalt::models::deepseek4 {

/// Applies the source V4 KV boundary: BF16 RMSNorm, forward RoPE, UE8M0 E4M3 simulation for non-RoPE lanes,
/// and BF16 rounding. `projected` and `out` are [n_tokens][512] F32 containers.
void prepare_attention_kv(const float* projected, const float* norm, float rms_eps, const int32_t* positions,
                          const kernels::MqaRopeTable& rope, int n_tokens, float* out, void* stream);

/// Projects source `wkv` `[512][4096]` with native quantized matvec, then applies the KV boundary.
/// The caller provides Q8_1 input scratch sized by `attention_kv_scratch_bytes` and F32 `[T][512]` workspace.
size_t attention_kv_scratch_bytes(int n_tokens);
void project_attention_kv(const float* x, int wkv_type, const void* wkv, const float* norm, float rms_eps,
                          const int32_t* positions, const kernels::MqaRopeTable& rope, int n_tokens,
                          void* q8_scratch, float* projected, float* out, void* stream);

/// Writes prepared `[T][512]` rows to the sliding cache slot `position % 128`; ring slots must be unique within the call.
void cache_attention_window(float* ring, const float* rows, const int32_t* positions, int n_tokens, void* stream);

struct AttentionStepWeights {
    kernels::SparseMqaQWeights q;
    int kv_type = -1;
    const void* kv = nullptr;
    const float* kv_norm = nullptr;
    const float* sinks = nullptr;
    int output_a_type = -1;
    const void* output_a = nullptr;
    int output_b_type = -1;
    const void* output_b = nullptr;
};

struct AttentionStepScratch {
    void* input_q8 = nullptr;
    float* q_latent = nullptr;
    void* q_latent_q8 = nullptr;
    float* q_heads = nullptr;
    float* kv_projected = nullptr;
    float* kv_row = nullptr;
    float* mqa_output = nullptr;
    float* mqa_workspace = nullptr;
    void* output_q8 = nullptr;
    float* output_latent = nullptr;
    float* output = nullptr;
};

/// One token's attention projection, sliding cache update, sparse MQA and grouped output projection. `selection` supplies
/// device positions, window bounds and optional preselected compressed rows for the same token; its other inputs are set
/// from `weights`, `window_ring` and `rope`. Caller owns all device storage.
void attention_step(const float* x, const AttentionStepWeights& weights, const kernels::MqaRopeTable& rope,
                   float* window_ring, kernels::SparseMqaArgs selection, const AttentionStepScratch& scratch,
                   void* stream);

}  // namespace basalt::models::deepseek4
