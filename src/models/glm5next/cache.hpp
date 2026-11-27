// src/models/glm5next/cache.hpp - sizing for the three caches a glm5-next session keeps per sequence.
//
// The memory is derived wholly from the pack geometry (no per-tensor shapes, so it holds for the real pack). There are
// three caches; the recurrent one is context-independent (linear attention), the other two grow with the context:
//
//   KDA  (linear-attn layers): a per-head delta-rule matrix h [n_head][head_dim][head_dim] plus a causal-conv window of
//        (conv_kernel-1) positions over the 3 q|k|v streams of n_head*head_dim channels - constant per sequence.
//   MLA  (sparse layers): one compressed latent row per cached token per layer, kv_lora wide (nope: no rope half, and MQA
//        so all heads share the row).
//   DSA  (the indexer over the sparse layers): the indexer scores POOLS, not tokens. Every `kpool` consecutive tokens
//        collapse into one pool, and idx_score reads every pool's pooled key as one contiguous [n_pool][idx_dim]
//        array - so the cache holds one pooled key per pool (context/kpool of them per layer), not one row per token.
//        What is per token is only the not-yet-pooled remainder: the incomplete pool holds at most kpool-1 tokens, and
//        each contributes its indexer key and its compressor gate.
//
// (glm5-next.cpp: n_embd_head_kda, "the MLA cache holds the compressed latent" = n_lora_kv, and the k-pool's
// set_input_kpool args pool_cells / pool_idxs / new_pool_rep.)
#pragma once

#include "models/glm5next/pack.hpp"

#include <cstdint>

namespace basalt::models::glm5next {

/// An un-pooled token's cached indexer rows: its LayerNormed key and its compressor gate, both idx_dim wide.
constexpr int kIndexerPendingRows = 2;

struct CacheDims {
    int n_kda = 0, n_mla = 0, n_idx = 0;   // model blocks of each kind (the drafter is excluded); n_idx = the indexer layers
    int n_head = 0, head_dim = 0, conv = 0, kv_lora = 0, idx_dim = 0, kpool = 0;
    /// The drafter's own MLA layers: one per next-n block, 0 when the pack has no drafter. Its block attends over
    /// latents of its own - the model's are computed from other inputs - at the same positions, so the same
    /// token-indexed rows have to exist beside the model's.
    int n_mla_draft = 0;

    uint64_t kda_state_elems = 0;    // delta-rule matrices h, all KDA layers, one sequence
    uint64_t kda_conv_elems = 0;     // causal-conv windows, all KDA layers, one sequence
    uint64_t mla_latent_elems = 0;   // compressed latents, all MLA layers, one token
    uint64_t draft_latent_elems = 0;  // ... the drafter's own, all its MLA layers, one token
    uint64_t idx_pooled_elems = 0;   // pooled indexer keys, all indexer layers, one POOL
    uint64_t idx_pending_elems = 0;  // the incomplete pool's keys and gates, all indexer layers, one sequence

    /// The pools a `context`-token sequence can fill. A trailing partial pool is not scored, so it is not reserved.
    uint64_t n_pools(uint64_t context) const { return kpool > 0 ? context / (uint64_t) kpool : 0; }
};

/// Sizes from the model geometry. The indexer cache exists on the sparse (MLA) layers, so n_idx = n_mla.
CacheDims cache_dims(const Geometry& g);

/// The fp32 bytes the three caches take for `context` tokens across `n_seq` sequences (0 when the geometry has none).
uint64_t cache_bytes(const CacheDims& d, uint64_t context, int n_seq);

}  // namespace basalt::models::glm5next
