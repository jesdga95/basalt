// src/models/glm5next/kernels/kpool.hpp - see kpool.cu: the DSA indexer's key pooling (kpool). The indexer cache keeps
// one row per token; every `kpool` consecutive tokens collapse into one POOL row that the indexer scores and the
// sparse MLA selects. The pooled key is a per-channel softmax over the pool's gate scores biased by a learned
// per-slot position embedding (the model's only positional signal for the indexer - the architecture is nope):
//
//   prob[s][d] = softmax_s( gate[s][d] + ape[s][d] )                    (softmax over the pool's slots, per channel)
//   pooled[d]  = sum_s prob[s][d] * key[s][d]
//
// (llama.cpp glm5-next build_kpool_select: soft_max over the kpool axis of gate + ape, then the weighted sum of the
// keys). All buffers fp32 device pointers; a call takes `n_pools` pools completed by one window and one pool's result
// does not depend on the others. `stream` may be null (the call then synchronises).
#pragma once

#include <cuda_runtime.h>

namespace basalt::models::glm5next {

/// The model's pool: kpool = 4 slots of head_dim = 128.
///   key  [n_pools][kpool][head_dim] the pool's per-token indexer keys (LayerNormed indexer.attn_k outputs), oldest first
///   gate [n_pools][kpool][head_dim] the pool's per-token gate scores (indexer_compressor_gate outputs)
///   ape  [kpool][head_dim]          indexer_compressor_ape.weight, GGUF dims [head_dim, kpool] - channel d of slot s
///                                   sits at ape[s * head_dim + d]
///   out  [n_pools][head_dim]        the pooled keys the indexer scores
/// kpool <= kPoolMax.
constexpr int kPoolMax = 8;

void kpool_compress(const float* key, const float* gate, const float* ape, float* out, int n_pools, int kpool,
                    int head_dim, cudaStream_t stream);

}  // namespace basalt::models::glm5next
