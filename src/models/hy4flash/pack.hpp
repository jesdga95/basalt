// src/models/hy4flash/pack.hpp - the Hy4-flash .basalt pack as the loader sees it: geometry, the tensor table, the experts
// layout and the Engram table, all from the pack's meta section (tools/convert/arch/hy4flash.py writes it).
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0, n_ff = 0, n_dense = 0;
    int q_lora = 0, kv_lora = 0, qk_dim = 0, v_dim = 0, rope_dim = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_shared = 0, n_latent = 0;
    float expert_scale = 1.f, rms_eps = 1e-5f, hc_eps = 1e-6f, hc_magnitude = 2.f;
    int hc_mult = 4;
    int window = 512;
    float rope_base = 1e7f, rope_base_swa = 1e4f;
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0, idx_pool = 0;
    bool idx_select_tail = true;
    int64_t context_length = 0;
    std::vector<int> heads;          ///< per layer
    std::vector<bool> swa;           ///< per layer: a sliding-window layer
    std::vector<bool> indexed;       ///< per layer: owns an indexer (the sparse layers)
    std::vector<float> clamp_exp;    ///< per layer: the routed experts' SwiGLU limit
    int head_dim() const { return qk_dim; }
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;   ///< GGUF order, innermost first
    uint64_t offset = 0;           ///< absolute file offset
    uint64_t bytes = 0;
};

struct ExpertLayer {
    bool present = false;
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section
};

/// The Engram hash constants and table.
struct EngramInfo {
    bool present = false;
    uint64_t file_offset = 0;      ///< absolute offset of row 0
    uint64_t rows = 0;
    uint32_t row_bytes = 96;       ///< F8_E4M3 values, one byte each
    int row_values = 96;
    std::vector<int> layers;
    int n_gram = 3, n_head = 8, pad_token_id = 0;
    std::vector<uint64_t> mult, primes, offsets;
    std::vector<int32_t> token_map;
    int n_cols() const { return (n_gram - 1) * n_head; }
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;   ///< per layer
    core::FileRange experts_range;
    EngramInfo engram;
    core::FileRange vision;   ///< the vision/mmproj section (empty when the pack carries none)

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const {
        return find("blk." + std::to_string(layer) + "." + suffix);
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not a Hy4-flash pack, is malformed or has a layout this
/// engine does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::hy4flash
