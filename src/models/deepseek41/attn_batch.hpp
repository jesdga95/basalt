// src/models/deepseek41/attn_batch.hpp - attention of a chunk of prompt tokens for one layer.
#pragma once

#include "basalt/models/deepseek41/attn.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

/// The most prompt tokens one batch call takes.
constexpr int kMaxChunk = 16384;

/// The per-layer caches the decode path and the batch path share (owned by the decode ops).
struct AttnCaches {
    std::vector<uint16_t*>* ring = nullptr;      ///< fp16 [kWindowSlots][512] per layer
    std::vector<uint16_t*>* comp_kv = nullptr;   ///< compressed latents of the kv-source layers
    std::vector<uint16_t*>* comp_ik = nullptr;   ///< their index keys
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

std::unique_ptr<AttnBatch> make_attn_batch(const DsModel& model, const AttnCaches& caches, int max_tokens, int max_positions,
                                           std::string& err);

}  // namespace basalt::models::deepseek41
