// src/models/glm5next/meta.hpp - what the file's metadata says about the model: free of the runtime, so a CPU test can
// check it without linking the model.
//
// The rules are the ones tools/convert/arch/glm5next.py applies to the source GGUF, kept worded the same way so a pack
// built from a source that the converter refused says the same thing as the refusal. The converter refuses such a source
// before it writes anything, so these are the guard for a pack that was edited, truncated or built by something else.
//
// The block count includes the drafter: of the 46 blocks of GLM-5.3-Flash, blocks 0..44 run as the model and block 45 is
// the next-n layer, whose tensors the pack keeps under mtp/*. So `n_layers` of the description is the model's own blocks.
// The numbers checked here are the release's: 46 blocks, 4096 wide, 288 experts with the router keeping 8 (plus one
// shared expert), 2048 wide, 34 KDA and 12 sparse-MLA layers, 3 dense blocks up front, key pooling 4, four hyper-connection
// streams, a 1M context. What the metadata does not state is not checked here either: the experts' quantized types come
// from the tensors, which the session reads.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>
#include <vector>

namespace basalt::models::glm5next {

inline constexpr const char* kArch = "glm5next";        ///< the Basalt name, `meta.arch` of the pack
inline constexpr const char* kGgufArch = "glm5-next";   ///< the GGUF's own architecture, which the pack's gguf_kv keeps

/// The metadata keys keep the GGUF's prefix, in the pack and in the source file alike.
inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = g.get(std::string(kGgufArch) + "." + suffix);
    return v ? v->u : 0;
}

/// The attention of each block, in order: 0 is linear attention (KDA), 1 is sparse MLA with an indexer. The GGUF spells
/// it as the number of key/value heads, which is 0 for a KDA layer and 1 for an MLA one; an empty result means the key
/// is missing or is not an array.
inline std::vector<uint64_t> layer_kinds(const GgufFile& g) {
    const MetaValue* v = g.get(std::string(kGgufArch) + ".attention.head_count_kv");
    if (v == nullptr || v->type != MetaType::ARRAY) return {};
    std::vector<uint64_t> kinds;
    kinds.reserve(v->items.size());
    for (const MetaValue& x : v->items) kinds.push_back(x.u);
    return kinds;
}

/// The drafter's blocks: the `nextn_predict_layers` blocks that close the model.
inline std::vector<int> drafter_layers(const GgufFile& g) {
    const int64_t n = (int64_t) key_u(g, "block_count");
    const int64_t k = (int64_t) key_u(g, "nextn_predict_layers");
    std::vector<int> out;
    for (int64_t l = n - k; k > 0 && l < n; ++l) out.push_back((int) l);
    return out;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArch && arch->s != kGgufArch)
        return "architecture is '" + arch->s + "', this adapter requires '" + kArch + "' (or '" + kGgufArch +
               "', the name the pack's gguf_kv keeps)";
    for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count",
                          "expert_feed_forward_length", "kda.head_dim", "ssm.conv_kernel", "attention.indexer.kpool",
                          "hyper_connection.count"})
        if (key_u(g, k) == 0) return std::string("missing or zero glm5-next.") + k;
    const uint64_t n = key_u(g, "block_count");
    const std::vector<uint64_t> kinds = layer_kinds(g);
    if (kinds.size() != n)
        return "attention.head_count_kv lists " + std::to_string(kinds.size()) + " layers, the model has " +
               std::to_string(n);
    for (const uint64_t k : kinds)
        if (k > 1) return "attention.head_count_kv holds " + std::to_string(k) + "; a layer is linear-attention (0) "
                          "or sparse MLA (1)";
    const uint64_t dense = key_u(g, "leading_dense_block_count");
    if (dense >= n) return "leading_dense_block_count is " + std::to_string(dense) + " of " + std::to_string(n) + " layers";
    if (key_u(g, "expert_used_count") > key_u(g, "expert_count"))
        return "the router keeps " + std::to_string(key_u(g, "expert_used_count")) + " of " +
               std::to_string(key_u(g, "expert_count")) + " experts";
    // The drafter is the last block, and there is exactly one: the pack's mtp sections hold it, so a source whose next-n
    // layer sits elsewhere would be packed against the wrong tensors.
    const std::vector<int> mtp = drafter_layers(g);
    if (mtp.empty()) return "the file has no next-n block (the drafter)";
    if (mtp.size() != 1 || mtp[0] != (int) n - 1)
        return "the drafter is block " + std::to_string(mtp.front()) + "; expected the last block, " +
               std::to_string(n - 1);
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = kArch;
    d.n_layers = (int64_t) key_u(g, "block_count") - (int64_t) key_u(g, "nextn_predict_layers");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.n_expert = (int64_t) key_u(g, "expert_count");
    d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
    d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
    d.n_expert_shared = (int64_t) key_u(g, "expert_shared_count");   // the release has one, 2048 wide
    d.context_length = (int64_t) key_u(g, "context_length");         // 1048576 on GLM-5.3-Flash
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

}  // namespace basalt::models::glm5next
