// src/models/deepseek41/attn_batch.hpp - attention of a chunk of prompt tokens for one layer.
#pragma once

#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

/// The per-layer caches the decode path and the batch path share (owned by the decode ops).
struct AttnCaches {
    CacheFormat format = CacheFormat::Fp8;       ///< the rows' format (--kv)
    std::vector<uint8_t*>* ring = nullptr;       ///< window rows [kWindowSlots][win_row_bytes] per layer
    std::vector<uint8_t*>* comp_kv = nullptr;    ///< compressed rows of the kv-source layers
    std::vector<uint8_t*>* comp_ik = nullptr;    ///< their index keys
    std::vector<float*>* st_kv = nullptr;        ///< partial-group compressor state
    std::vector<float*>* st_sc = nullptr;
    const float* freq[2] = {nullptr, nullptr};   ///< rotary tables of the two layer kinds
    const RopeTable* rope = nullptr;             ///< [2]
};

class AttnBatch {
public:
    virtual ~AttnBatch() = default;
    /// Attention of `layer` over T tokens at pos0.. (x: normed input [T][n_embd]; out: [T][n_embd]). Layers run in order
    /// 0..n-1 for one chunk; the layer-0 call starts a chunk.
    virtual bool run(int layer, int64_t pos0, int n_tokens, const float* x, float* out, void* stream, std::string& err) = 0;
    /// Only the compressor of a kv-source layer over T tokens: their compressed latents and index keys.
    virtual bool compress_only(int layer, int64_t pos0, int n_tokens, const float* x, void* stream, std::string& err) = 0;
};

/// The workspaces of a batch of up to `max_tokens` tokens per call. They are a prompt's workspaces: opened and
/// released with the others, and counted with them (attn_batch_fixed_bytes + attn_batch_token_bytes per token).
std::unique_ptr<AttnBatch> make_attn_batch(const DsModel& model, const AttnCaches& caches, int max_tokens, int max_positions,
                                           std::string& err);
uint64_t attn_batch_fixed_bytes(CacheFormat format);
uint64_t attn_batch_token_bytes();

}  // namespace basalt::models::deepseek41
