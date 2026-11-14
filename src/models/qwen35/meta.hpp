// src/models/qwen35/meta.hpp - what a qwen35 pack's metadata says about the model: free of the runtime, so a CPU test
// can check it without linking the model.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/dense/model.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::qwen35 {

constexpr int kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBf16 = 30, kTypeNvfp4 = 40, kTypeF8 = 44, kTypeNvfp4Planes = 45;

/// Why `<base>.weight`, of tensor type `type` and rows x cols, is not a matrix the dense kernels read; empty when it is.
inline std::string linear_type_error(const std::string& base, uint32_t type, int64_t rows, int64_t cols) {
    if (type == kTypeNvfp4)
        return "qwen35: " + base + ".weight is NVFP4 in ggml blocks, from a pack older than NVFP4_PLANES: re-convert the "
               "pack (tools/convert/convert.py --checkpoint, docs/model-file.md)";
    if (type != kTypeQ8_0 && type != kTypeNvfp4Planes && type != kTypeF8)
        return "qwen35: " + base + ".weight is " + ggml_type_name(type) + "; the dense kernels read Q8_0, NVFP4_PLANES and F8_E4M3";
    if (type == kTypeNvfp4Planes && !kernels::nvfp4p::fits(rows, cols))
        return "qwen35: " + base + ".weight: NVFP4_PLANES of " + std::to_string(rows) + " x " + std::to_string(cols) +
               " (planes need rows in 128s and 64-value groups)";
    if (type == kTypeF8 && cols % 256 != 0)
        return "qwen35: " + base + ".weight: F8_E4M3 rows of " + std::to_string(cols) + " values (the kernels read multiples of 256)";
    return {};
}

inline const MetaValue* key(const GgufFile& g, const char* suffix) { return g.get(std::string("qwen35.") + suffix); }
inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = key(g, suffix);
    return v ? v->u : 0;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "qwen35") return "architecture is '" + arch->s + "', this adapter requires 'qwen35'";
    for (const char* k : {"block_count", "embedding_length", "feed_forward_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length", "rope.dimension_count", "ssm.state_size",
                          "ssm.group_count", "ssm.time_step_rank", "ssm.conv_kernel", "full_attention_interval"})
        if (key_u(g, k) == 0) return std::string("missing or zero qwen35.") + k;
    if (!g.get("tokenizer.ggml.tokens")) return "missing tokenizer.ggml.tokens";
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = "qwen35";
    d.n_layers = (int64_t) key_u(g, "block_count");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

/// The dense stack's geometry from the metadata; empty `err` when it is whole.
inline dense::Geometry geometry(const GgufFile& g, std::string& err) {
    dense::Geometry o;
    o.n_layers = (int) key_u(g, "block_count");
    o.n_embd = (int) key_u(g, "embedding_length");
    o.n_ff = (int) key_u(g, "feed_forward_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) o.n_vocab = (int) t->count;
    o.n_head = (int) key_u(g, "attention.head_count");
    o.n_kv = (int) key_u(g, "attention.head_count_kv");
    o.head_dim = (int) key_u(g, "attention.key_length");
    o.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* e = key(g, "attention.layer_norm_rms_epsilon")) o.eps = (float) e->num();
    o.rope.rot = (int) key_u(g, "rope.dimension_count");
    if (const MetaValue* f = key(g, "rope.freq_base")) o.rope.theta = (float) f->num();
    // interleaved M-RoPE: pair i takes h when i % 3 == 1 and i < 3 sections[1], w when i % 3 == 2 and i < 3 sections[2]
    int sec[3] = {0, 0, 0};
    if (const MetaValue* s = key(g, "rope.dimension_sections"))
        for (size_t i = 0; i < 3 && i < s->items.size(); ++i) sec[i] = (int) s->items[i].u;
    if (sec[0] + sec[1] + sec[2] != o.rope.rot / 2) err = "rope.dimension_sections do not cover the rotary width";
    for (int i = 0; i < o.rope.rot / 2 && i < 64; ++i)
        o.rope.axis[i] = (int8_t) (i % 3 == 1 && i < 3 * sec[1] ? 1 : i % 3 == 2 && i < 3 * sec[2] ? 2 : 0);
    o.gdn_state = (int) key_u(g, "ssm.state_size");
    o.gdn_k_heads = (int) key_u(g, "ssm.group_count");
    o.gdn_v_heads = (int) key_u(g, "ssm.time_step_rank");
    o.gdn_conv = (int) key_u(g, "ssm.conv_kernel");
    o.gdn_gate = kernels::GdnGate::Silu;
    const int interval = (int) key_u(g, "full_attention_interval");
    const MetaValue* rec = key(g, "attention.recurrent_layers");
    for (int il = 0; il < o.n_layers; ++il) {
        const bool recurrent = rec && rec->items.size() == (size_t) o.n_layers ? rec->items[(size_t) il].u != 0
                                                                                : (il + 1) % interval != 0;
        o.kinds.push_back(recurrent ? dense::Mixer::Gdn : dense::Mixer::Attention);
    }
    if (err.empty()) err = o.check();
    return o;
}

}  // namespace basalt::models::qwen35
