#pragma once

#include "basalt/models/deepseek4/cache_rows.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek4 {

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kIndexerTopK = 512;
constexpr int kIndexerWindow = 8;
constexpr int kIndexerEmbedding = 4096;
constexpr int kIndexerQueryRank = 1024;

/// Workspace for indexer_select() over at most max_rows compressed keys.
size_t indexer_select_workspace_bytes(int max_rows);

/// Workspace for indexer_project_q8_0() for up to max_tokens rows.
size_t indexer_project_workspace_bytes(int max_tokens);

/// Source indexer projections. qr is [T][1024], x is [T][4096], the query matrix is Q8_0 [8192][1024], and
/// the head-weight matrix is BF16 [64][4096].
/// rope_angle is [T][32] radians after the configured RoPE/YaRN scaling. The latent is BF16-rounded, quantized through
/// Basalt's Q8_1 activation path, then multiplied by the pack's Q8_0 query matrix. q_out is [T][64][128] after the
/// source BF16/FP4/rotary/Hadamard boundaries, and head_weights is [T][64] after source score scaling. w_proj is BF16.
/// The workspace size is returned by indexer_project_workspace_bytes(max_tokens); pointers are device pointers and the
/// stream must be non-null.
void indexer_project_q8_0(const void* w_q, const uint16_t* w_proj, const float* qr, const float* x,
                          const float* rope_angle, int n_tokens, float* q_out, float* head_weights,
                          void* workspace, size_t workspace_bytes, void* stream);

void indexer_round_bf16(const float* qr, int n_tokens, float* qr_bf16, void* stream);
void indexer_project_post(const uint16_t* w_proj, const float* q_raw, const float* x,
                          const float* rope_angle, int n_tokens, float* w_raw, float* q_out,
                          float* head_weights, void* stream);

/// Scores compressed indexer keys and writes each query's top rows in descending score order, with lower row ids first
/// on ties. q is [n_tokens][64][128], head_weights is [n_tokens][64] and already includes the source scale
/// 1/sqrt(64*128), keys is [n_comp][idx_row_bytes(fmt)] (E2M1 codes with a power-of-two scale per 32, or fp16). The
/// queries are rounded the same way the keys are before scoring. The output is [n_tokens][512], padded with -1;
/// n_selected is [n_tokens].
/// For query p=start_pos+t, only compressed rows [0, (p+1)/4) are visible. The ids are relative to keys; the MLA layer
/// adds its sliding-window offset when it builds the combined attention row list. Workspace is device memory sized by
/// indexer_select_workspace_bytes(n_comp). All tensors are device pointers; stream may be null.
void indexer_select(const float* q, const float* head_weights, const void* keys, int n_comp, int start_pos,
                    int n_tokens, int32_t* out_ids, int32_t* n_selected, void* workspace, size_t workspace_bytes,
                    CacheFormat fmt, void* stream);

}  // namespace basalt::models::deepseek4
