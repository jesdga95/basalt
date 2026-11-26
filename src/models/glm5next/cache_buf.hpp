// src/models/glm5next/cache_buf.hpp - the allocated three caches, sized from CacheDims and laid out for the kernels.
//
// One GlmCaches holds n_seq sequences × `context` tokens of:
//   KDA recurrent state  - one [n_head*head_dim*head_dim] block per KDA layer per sequence (kda_step's `state`);
//   KDA conv window       - one [3*(conv-1)*n_head*head_dim] block per KDA layer per sequence (kda_front's `hist`);
//   MLA latent cache      - one [context][kv_lora] block per MLA layer per sequence (mla_sparse's `latent`, row = token);
//   DSA pooled keys       - one [n_pools][idx_dim] block per indexer layer per sequence (idx_score's `pooled`, one row
//                           per pool, contiguous over the pools, which is what that kernel reads);
//   DSA pending rows      - one [2][kpool-1][idx_dim] block per indexer layer per sequence: the keys, then the gates, of
//                           the tokens of the pool still being filled.
// The recurrent state and conv window are zeroed on alloc and by reset() (a fresh sequence starts empty, and a
// reused session has to be told to start over). Layout is sequence-major,
// then layer, so the kernels get a contiguous block per (sequence, layer) and step by a token stride inside it.
#pragma once

#include "models/glm5next/cache.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace basalt::models::glm5next {

class GlmCaches {
public:
    GlmCaches() = default;
    ~GlmCaches();
    GlmCaches(const GlmCaches&) = delete;
    GlmCaches& operator=(const GlmCaches&) = delete;

    /// Allocates all four buffers for `dims` at `context` tokens over `n_seq` sequences and zeroes the recurrent state.
    bool alloc(const CacheDims& dims, uint64_t context, int n_seq, std::string& err);
    void free();

    /// Starts sequence `seq` over: zeroes its KDA recurrent state and conv window, the only caches a pass reads
    /// before writing them. The MLA latents and the indexer's pool need nothing, because every row of theirs a
    /// window reads was written by that window's own tokens. A session lives across conversations, so a request that
    /// starts at position 0 has to ask for this: without it the new conversation's first tokens run on the state the
    /// previous one left behind.
    void reset(int seq, cudaStream_t stream);

    bool empty() const { return state_ == nullptr; }
    uint64_t bytes() const { return bytes_; }
    uint64_t context() const { return context_; }
    int n_seq() const { return n_seq_; }

    /// kda_layer indexes the KDA blocks [0,n_kda); mla_layer / idx_layer the sparse blocks [0,n_mla).
    float* kda_state(int seq, int kda_layer) const;
    float* kda_conv(int seq, int kda_layer) const;
    /// Token t of the sequence is at row t (stride kv_lora floats).
    float* mla_latent(int seq, int mla_layer) const;
    /// The drafter block's own latents, laid out the same way, so the model's indexer's cells address both. Null when
    /// the pack has no drafter.
    float* draft_latent(int seq, int draft_mla_layer) const;
    /// Pool p of the sequence is at row p (stride idx_dim floats).
    float* idx_pooled(int seq, int idx_layer) const;
    /// The pool being filled: its keys at slot s (stride idx_dim), its gates kIndexerPendingRows/2 slots further.
    float* idx_pending(int seq, int idx_layer) const;

private:
    float* state_ = nullptr;
    float* conv_ = nullptr;
    float* latent_ = nullptr;
    float* dlat_ = nullptr;
    float* pooled_ = nullptr;
    float* pend_ = nullptr;
    CacheDims dims_;
    uint64_t context_ = 0;
    int n_seq_ = 0;
    uint64_t bytes_ = 0;
    size_t state_stride_ = 0, conv_stride_ = 0, mla_stride_ = 0, dlat_stride_ = 0, pooled_stride_ = 0, pend_stride_ = 0;   // floats per (sequence, layer)
};

}  // namespace basalt::models::glm5next
