// src/models/deepseek41/meta.hpp - what the file's metadata says about the model: free of the runtime, so a CPU test can
// check it without linking the model.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::deepseek41 {

inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = g.get(std::string("deepseek41.") + suffix);
    return v ? v->u : 0;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "deepseek41") return "architecture is '" + arch->s + "', this adapter requires 'deepseek41'";
    for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count",
                          "expert_feed_forward_length"})
        if (key_u(g, k) == 0) return std::string("missing or zero deepseek41.") + k;
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = "deepseek41";
    d.n_layers = (int64_t) key_u(g, "block_count");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.n_expert = (int64_t) key_u(g, "expert_count");
    d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
    d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
    d.n_expert_shared = (int64_t) key_u(g, "expert_shared_count");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

}  // namespace basalt::models::deepseek41
