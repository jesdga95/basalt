#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::deepseek4 {

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, head_dim = 0, rope_dim = 0, q_lora = 0, window = 0;
    int o_groups = 0, o_lora = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_shared = 0;
    int hc_mult = 0, sinkhorn_iters = 0, hash_layers = 0;
    int n_head_kv = 0, router_func = -1, output_embd = 0;
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    bool route_norm = false;
    float rms_eps = 0.f, hc_eps = 0.f, route_scale = 0.f;
    float rope_base = 0.f, comp_rope_base = 0.f, yarn_factor = 0.f;
    float yarn_beta_fast = 0.f, yarn_beta_slow = 0.f;
    int64_t context_length = 0;
    int yarn_orig_ctx = 0;
    std::vector<int> compress_ratio;
};

struct TensorRef {
    int type = 0;
    std::vector<uint64_t> shape;
    uint64_t offset = 0, bytes = 0;
};

struct ExpertLayer {
    int gate_up_type = 0, down_type = 0;
    uint64_t offset = 0, bytes = 0;
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;
    core::FileRange vision;

    const TensorRef* find(const std::string& name) const {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const {
        return find("blk." + std::to_string(layer) + "." + suffix);
    }
};

/// Opens and validates a `deepseek4` Basalt pack, including the per-layer routed-expert layout.
bool open_pack(const std::string& path, Pack& pack, std::string& err);
bool validate_supported_geometry(const Geometry& geometry, std::string& err);
/// Read one validated six-expert token-to-expert table into host memory for the model session.
bool load_hash_route_table(const Pack& pack, int layer, std::vector<int32_t>& ids, std::string& err);

/// Parses a JSON compression schedule without truncating fractional, non-finite, or out-of-range values.
bool parse_compression_ratios(const core::Json& value, std::vector<int>& ratios, std::string& err);

}  // namespace basalt::models::deepseek4
