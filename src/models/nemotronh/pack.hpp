// src/models/nemotronh/pack.hpp - the Nemotron-H .basalt pack as the loader sees it: geometry, the tensor table and the
// experts layout, from the pack's meta section (tools/convert/checkpoint_nemotronh.py writes it).
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::nemotronh {

enum class Kind : char { Mamba = 'M', Attention = '*', Moe = 'E' };

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    int ssm_heads = 0, ssm_head_dim = 0, ssm_state = 0, ssm_groups = 0, ssm_conv = 0, ssm_inner = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_ff_shared = 0, n_latent = 0;
    float expert_scale = 1.f, rms_eps = 1e-5f;
    int64_t context_length = 0;
    std::vector<Kind> kinds;   ///< per layer
    int conv_dim() const { return ssm_inner + 2 * ssm_groups * ssm_state; }
    int proj_width() const { return 2 * ssm_inner + 2 * ssm_groups * ssm_state + ssm_heads; }   ///< [z | xBC | dt]
    int count(Kind k) const {
        int n = 0;
        for (Kind x : kinds) n += x == k;
        return n;
    }
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

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;   ///< per layer (MoE layers present)
    core::FileRange experts_range;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const { return find("blk." + std::to_string(layer) + "." + suffix); }
};

/// Opens the pack and fills `p`. False with `err` when it is not a Nemotron-H pack, is malformed or has a layout this
/// engine does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::nemotronh
