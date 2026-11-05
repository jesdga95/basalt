// src/models/qwen4exp/meta.hpp - what the file's metadata says about the model: free of the runtime, so a CPU test can
// check it without linking the model.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"

#include <string>

namespace basalt::models::qwen4exp {

inline std::string validate_meta(const GgufFile& g) {
    // keys the model must carry; a zero expectation checks presence only (pruned variants ship fewer experts)
    struct Req {
        const char* key;
        uint64_t want;
    };
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "qwen4exp") return "architecture is '" + arch->s + "', this adapter requires 'qwen4exp'";
    const core::ModelGeometry def;
    const Req reqs[] = {
        {"qwen4exp.block_count", (uint64_t) def.n_layers},
        {"qwen4exp.embedding_length", (uint64_t) def.n_embd},
        {"qwen4exp.expert_count", 0},
        {"qwen4exp.expert_used_count", 0},
        {"qwen4exp.attention.head_count", (uint64_t) def.n_head},
        {"qwen4exp.attention.head_count_kv", (uint64_t) def.n_head_kv},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (r.want && v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    const core::ModelGeometry def;
    model::ModelDesc d;
    d.arch = "qwen4exp";
    d.n_layers = def.n_layers;
    d.n_embd = def.n_embd;
    d.n_expert = def.n_expert;
    d.n_expert_used = 10;
    d.n_ff_exp = def.n_ff;
    d.n_vocab = 248320;
    if (const MetaValue* v = g.get("qwen4exp.expert_count")) d.n_expert = (int64_t) v->u;
    if (const MetaValue* v = g.get("qwen4exp.expert_used_count")) d.n_expert_used = (int64_t) v->u;
    return d;
}

}  // namespace basalt::models::qwen4exp
