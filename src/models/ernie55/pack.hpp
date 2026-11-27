// src/models/ernie55/pack.hpp - ERNIE-5.5-Flash .basalt pack geometry, tensor table, and MXFP4 experts layout.
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::ernie55 {

struct Geometry {
    int n_layers = 0, n_dense = 1, n_embd = 0, n_ff = 0, n_vocab = 0;
    int64_t context_length = 0;
    float rms_eps = 1e-6f;
    int window = 128;
    std::vector<int> compress_ratios;   ///< > 0: HCA with this ratio (128); <= 0: MLA-DSA

    // HCA geometry
    int hca_heads = 64, hca_kv_heads = 1, hca_head_dim = 512, hca_v_dim = 512;
    int hca_q_lora = 1024, hca_kv_lora = 512, hca_o_groups = 8, hca_o_lora = 1024;
    int hca_rope_dim = 64;
    float rope_theta = 8000000.f, comp_rope_theta = 160000.f;
    float yarn_factor = 16.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f;
    int yarn_orig_ctx = 65536;

    // MLA-DSA geometry
    int mla_heads = 64, mla_q_lora = 1024, mla_kv_lora = 512;
    int mla_qk_nope = 192, mla_qk_rope = 64, mla_v_dim = 256;
    int idx_heads = 64, idx_dim = 128, idx_top_k = 2048;

    // Latent MoE geometry
    int n_expert = 512, n_used = 10, n_shared = 1, n_ff_exp = 2048, n_latent = 2048;
    bool norm_topk_prob = true;
    float gate_softcap = 4.0f, up_softcap = 25.0f;

    // mHC geometry
    int hc_mult = 4, sinkhorn_iters = 20;
    float hc_eps = 1e-6f;

    bool has_flashindexer = false;

    bool is_dsa(int l) const { return l >= 0 && l < (int) compress_ratios.size() && compress_ratios[(size_t) l] <= 0; }
    int num_dsa_layers() const {
        int c = 0;
        for (int r : compress_ratios) c += (r <= 0);
        return c;
    }
    int num_hca_layers() const { return n_layers - num_dsa_layers(); }
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

struct DsparkPack {
    bool present = false;
    int n_stages = 0;
    int block_size = 7;
    int noise_token_id = 1024;
    int markov_rank = 256;
    int window = 128;
    float rope_theta = 8000000.f;
    std::vector<int> target_layers;
    std::map<std::string, TensorRef> tensors;   ///< offsets are absolute file offsets
    uint64_t experts_offset = 0;                ///< absolute file offset of stage MXFP4 blobs
    uint64_t experts_bytes = 0;
    int n_expert = 0;
    uint64_t blob_bytes = 0;
    uint64_t stage_stride = 0;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int stage, const char* suffix) const {
        return find("blk." + std::to_string(stage) + "." + suffix);
    }
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;   ///< indexed by layer 0..n_layers-1
    core::FileRange experts_range;
    DsparkPack dspark;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const {
        return find("blk." + std::to_string(layer) + "." + suffix);
    }
};

/// Opens an ERNIE-5.5 .basalt pack and fills `p`. False with `err` on mismatch or corruption.
bool open_pack(const std::string& path, Pack& p, std::string& err);

/// Checks whether every F32 element in `[off, off + bytes)` of `fd` has zero low 16 bits (an exact BF16 value).
bool f32_range_is_bf16(int fd, uint64_t off, uint64_t bytes, bool& exact, std::string& err);

}  // namespace basalt::models::ernie55
