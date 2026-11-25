// src/models/gemma4/pack.hpp - the Gemma 4 .basalt pack as the loader sees it: geometry from the meta's gguf_kv, the
// tensor table, the experts layout (tools/convert/checkpoint_gemma4.py writes it).
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/models/gemma4/kernels.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

/// The assistant drafter's settings (`gemma4.assistant.*`), present when the pack carries it.
struct DrafterGeometry {
    bool present = false;
    int n_layers = 0, n_embd = 0, n_ff = 0, n_head = 0, draft_tokens = 0;
    std::vector<bool> swa;
};

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0, n_ff = 0;
    int n_head = 0, n_kv_swa = 0, n_kv_full = 0, hd_swa = 0, hd_full = 0, window = 0;
    bool k_eq_v = false;
    float rms_eps = 1e-6f, rope_base = 1e6f, rope_base_swa = 1e4f, partial_rotary = 1.f, softcap = 0.f, embed_scale = 1.f;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;   ///< 0 experts: a dense model
    int image_token = -1;
    int image_cells = 0;                          ///< the rows a picture becomes at most (vision_config.default_output_length)
    int64_t context_length = 0;
    std::vector<bool> swa;                        ///< per layer: a sliding-window layer
    DrafterGeometry drafter;

    bool moe() const { return n_expert > 0; }
    /// Layer l's attention shape: a window layer rotates every pair; a full layer the partial factor's share, and with
    /// k_eq_v keeps one cache.
    LayerShape shape(int l) const {
        LayerShape s;
        const bool w = swa[(size_t) l];
        s.n_head = n_head;
        s.n_kv = w ? n_kv_swa : n_kv_full;
        s.hd = w ? hd_swa : hd_full;
        s.rot = w ? s.hd / 2 : (int) (partial_rotary * (float) s.hd / 2.f);
        s.window = w ? window : 0;
        s.one = !w && k_eq_v;
        return s;
    }
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;     ///< GGUF order, innermost first
    uint64_t offset = 0;             ///< absolute file offset
    uint64_t bytes = 0;
};

struct ExpertLayer {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;      ///< per layer (empty for a dense model)
    core::FileRange experts_range;
    core::FileRange vision;                ///< the vision/mmproj section (empty when the pack carries none)

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not a Gemma 4 pack, is malformed or has a layout this engine
/// does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::gemma4
