// src/models/hy4flash/meta.hpp - what the file's metadata says about the model: free of the runtime, so a CPU test can
// check it without linking the model.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::hy4flash {

/// The architecture name the GGUF metadata (and so the pack's `gguf_kv`) carries.
inline constexpr const char* kGgufArch = "hy_v4_flash";
/// The id of the model (the converter's NVFP4 packs carry it in `general.architecture`).
inline constexpr const char* kArch = "hy4flash";

inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = g.get(std::string(kGgufArch) + "." + suffix);   // the keys keep the GGUF's prefix in both
    return v ? v->u : 0;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kGgufArch && arch->s != kArch)
        return "architecture is '" + arch->s + "', this adapter requires '" + kArch + "' (or '" + kGgufArch + "', the test pack's)";
    for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
                          "attention.kv_lora_rank", "expert_latent_length"})
        if (key_u(g, k) == 0) return std::string("missing or zero ") + kGgufArch + "." + k;
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = "hy4flash";
    d.n_layers = (int64_t) key_u(g, "block_count") - (int64_t) key_u(g, "nextn_predict_layers");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.n_expert = (int64_t) key_u(g, "expert_count");
    d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
    d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
    d.n_expert_shared = (int64_t) key_u(g, "expert_shared_count");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

}  // namespace basalt::models::hy4flash
