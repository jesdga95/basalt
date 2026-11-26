// src/models/glm5next/pack.hpp - the glm5-next .basalt pack as the loader sees it: geometry, the tensor table and the
// routed-experts layout, all read from the pack's meta section. The meta.geometry object carries the whole `glm5-next.*`
// GGUF block with the prefix stripped, so the keys read as "block_count", "attention.indexer.kpool", and so on. The
// drafter (next-n) block's tensors live under the mtp/* sections, laid out like the main ones; the first
// leading_dense_block_count model layers have no routed experts. There is no Engram in this architecture.
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::glm5next {

struct Geometry {
    int n_layers = 0;          ///< blocks incl. the drafter
    int nextn = 0;             ///< the drafter's next-n block count (model layers = n_layers - nextn)
    int leading_dense = 0;     ///< model layers up front with a dense MLP instead of routed experts
    int n_embd = 0, n_vocab = 0, n_head = 0;
    int qk_dim = 0, v_dim = 0;         ///< MLA head key/value width (nope: no rope half)
    int q_lora = 0, kv_lora = 0;       ///< MLA low-rank query / compressed key-value widths
    int kda_head_dim = 0, conv = 0;    ///< KDA head width and conv kernel
    float gate_lower_bound = 0.f;      ///< KDA decay gate lower bound
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;  ///< the DSA indexer
    int n_expert = 0, n_used = 0, n_shared = 0, n_ff_exp = 0;
    int n_ff = 0;                    ///< the dense lead MLP's hidden width (the layers before leading_dense)
    int n_ff_shexp = 0;              ///< a shared expert's hidden width (expert_shared_feed_forward_length; = n_ff_exp when the pack is silent)
    float expert_scale = 2.5f;         ///< routed-expert weight renormalisation scale
    int hc_count = 4;                  ///< hyper-connection streams
    float hc_eps = 1e-6f, rms_eps = 1e-5f;
    float ln_eps = 1e-5f;          ///< attention.layer_norm_epsilon: the indexer key LayerNorm's epsilon
    std::vector<float> clamp_exp, clamp_shexp;   ///< per block: the SwiGLU clamp of the routed / shared experts (0: none)
    int sinkhorn_iters = 20;
    /// The drafter's indexer: true (what the released pack states) selects with the model's own indexer state, false
    /// wants one of the block's own. Read at load so the load says which; nothing runs the block yet.
    bool index_share_mtp = true;
    int64_t context_length = 0;
    std::vector<int> kinds;            ///< per block: 0 linear attention (KDA), 1 sparse MLA

    int n_model_layers() const { return n_layers - nextn; }
    bool is_mla(int layer) const { return layer >= 0 && layer < (int) kinds.size() && kinds[layer] != 0; }
    /// The shared experts' hidden width: the pack's own when it states one apart, the routed experts' otherwise.
    int shared_ffn() const { return n_ff_shexp > 0 ? n_ff_shexp : n_ff_exp; }
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
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section, and ONE expert's blob (the layer's region is `n_expert` of them)
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;         ///< by model layer; empty entries for the dense lead layers
    core::FileRange experts_range;            ///< the main experts section
    core::FileRange mtp_experts_range;        ///< the drafter's experts section (empty when there is no drafter)
    std::vector<ExpertLayer> mtp_experts;     ///< the drafter's per-layer expert layout
    core::FileRange mtp_dense_range;          ///< the drafter's float-tensor pool (empty when there is no drafter)
    /// The drafter's pool tensors, by name, read from `mtp/dense.index`. The pack's tensor table lists only the
    /// drafter's quantized tensors, whose bytes come from the `gguf` section like every other one, so its norms, its
    /// router gate and its index projection are named nowhere else. `shape` is the index's own [ne0, ne1]: the pool
    /// flattens every axis past the first, which is how the products read them anyway.
    std::map<std::string, TensorRef> mtp_tensors;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const {
        return find("blk." + std::to_string(layer) + "." + suffix);
    }
    const TensorRef* find_mtp(const std::string& n) const {
        auto it = mtp_tensors.find(n);
        return it == mtp_tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find_mtp(int layer, const char* suffix) const {
        return find_mtp("blk." + std::to_string(layer) + "." + suffix);
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not a glm5-next pack or is malformed.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::glm5next
