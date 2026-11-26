#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek4 {

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kIndexerTopK = 512;
constexpr int kIndexerWindow = 8;

/// Workspace for indexer_select() over at most max_rows compressed keys.
size_t indexer_select_workspace_bytes(int max_rows);

/// Scores compressed indexer keys and writes each query's top rows in descending score order, with lower row ids first
/// on ties. q is [n_tokens][64][128], head_weights is [n_tokens][64] and already includes the source scale
/// 1/sqrt(64*128), keys is [n_comp][128]. The output is [n_tokens][512], padded with -1; n_selected is [n_tokens].
/// For query p=start_pos+t, only compressed rows [0, (p+1)/4) are visible. The ids are relative to keys; the MLA layer
/// adds its sliding-window offset when it builds the combined attention row list. Workspace is device memory sized by
/// indexer_select_workspace_bytes(n_comp). All tensors are device pointers; stream may be null.
void indexer_select(const float* q, const float* head_weights, const float* keys, int n_comp, int start_pos,
                    int n_tokens, int32_t* out_ids, int32_t* n_selected, void* workspace, size_t workspace_bytes,
                    void* stream);

}  // namespace basalt::models::deepseek4
