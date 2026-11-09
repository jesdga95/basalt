// src/models/deepseek41/pack.hpp - the DeepSeek-V4.1 .basalt pack as the loader sees it: geometry, the tensor
// table, the experts layout and the Engram hash constants, all from the pack's meta section.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/models/deepseek41/kernels.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, head_dim = 0, rope_dim = 0, q_lora = 0, window = 0;
    int o_groups = 0, o_lora = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_shared = 0;
    float weight_scale = 1.0f, rms_eps = 1e-20f, hc_eps = 1e-6f;
    int sinkhorn_iters = 20;
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    float rope_base = 10000.f, comp_rope_base = 160000.f;
    /// The longest context the model was extended to, and the YaRN extension of the compressed layers' rotation.
    int64_t context_length = 0;
    float yarn_factor = 16.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f;
    int yarn_orig_ctx = 65536;
    std::vector<float> clamp_exp, clamp_shexp;   ///< per layer
    std::vector<int> compress_ratio;             ///< per layer
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;   ///< GGUF order, innermost first
    uint64_t offset = 0;           ///< absolute file offset
    uint64_t bytes = 0;
};

struct ExpertLayer {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section
};

struct EngramInfo {
    bool present = false;
    uint64_t file_offset = 0;      ///< absolute offset of row 0
    uint64_t rows = 0;
    int type = 20;                 ///< ggml type of the rows: IQ4_NL (20) or MXFP4 (39)
    uint32_t row_bytes = 144;
    int row_values = 256;
    std::vector<int> layers;       ///< model layers with an Engram
    EngramHashParams hash;
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;
    core::FileRange experts_range;
    EngramInfo engram;
    core::FileRange vision;   ///< the vision/mmproj section (empty when the pack carries none)
    core::FileRange dspark;   ///< the dspark/gguf section (empty when the pack carries none)

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const {
        return find("blk." + std::to_string(layer) + "." + suffix);
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not a DeepSeek-V4.1 pack or is malformed.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::deepseek41
