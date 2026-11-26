#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::deepseek4 {

inline double key_num(const GgufFile& file, const char* suffix, double fallback = 0.0) {
    const MetaValue* value = file.get(std::string("deepseek4.") + suffix);
    return value && value->is_num() ? value->num() : fallback;
}

inline std::string validate_meta(const GgufFile& file) {
    const MetaValue* arch = file.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING) return "missing or invalid general.architecture";
    if (arch->s != "deepseek4") return "architecture is '" + arch->s + "', this adapter requires 'deepseek4'";

    for (const char* key : {"block_count", "context_length", "embedding_length", "attention.head_count",
                            "attention.head_count_kv", "attention.key_length", "attention.q_lora_rank",
                            "attention.output_group_count", "attention.output_lora_rank", "rope.dimension_count",
                            "expert_count", "expert_used_count", "expert_feed_forward_length", "expert_shared_count",
                            "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
                            "hyper_connection.count", "hash_layer_count"}) {
        if (key_num(file, key) <= 0.0) return std::string("missing or zero deepseek4.") + key;
    }
    const MetaValue* norm = file.get("deepseek4.expert_weights_norm");
    if (!norm || norm->type != MetaType::BOOL || norm->u != 1 ||
        key_num(file, "expert_used_count") != 6 || key_num(file, "hash_layer_count") != 3 ||
        key_num(file, "expert_weights_scale") != 1.5)
        return "deepseek4 routing metadata is outside the supported top-6 hash-routing contract";

    const int layers = (int) key_num(file, "block_count");
    const double experts = key_num(file, "expert_count");
    if (key_num(file, "expert_used_count") > experts || key_num(file, "hash_layer_count") > layers)
        return "deepseek4 metadata has an invalid expert or hash-layer count";

    const MetaValue* ratios = file.get("deepseek4.attention.compress_ratios");
    if (!ratios || ratios->type != MetaType::ARRAY || ratios->count != (uint64_t) layers + 3 ||
        ratios->items.size() != ratios->count) return "deepseek4.attention.compress_ratios must cover trunk and three MTP slots";
    for (int i = 0; i < layers; ++i) {
        const uint64_t ratio = ratios->items[(size_t) i].u;
        if (ratio != 0 && ratio != 4 && ratio != 128)
            return "unsupported compression ratio at deepseek4 layer " + std::to_string(i);
    }
    for (size_t i = (size_t) layers; i < ratios->items.size(); ++i)
        if (ratios->items[i].u != 0) return "MTP compression schedule must be zero for this runtime";

    if (key_num(file, "hyper_connection.count") != 4 || key_num(file, "attention.head_count_kv") != 1 ||
        key_num(file, "rope.freq_base") <= 0.0 || key_num(file, "attention.compress_rope_freq_base") <= 0.0 ||
        key_num(file, "rope.scaling.factor") < 1.0 || key_num(file, "rope.scaling.original_context_length") <= 0.0)
        return "deepseek4 metadata is outside the supported V4 geometry";
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& file) {
    model::ModelDesc desc;
    desc.arch = "deepseek4";
    desc.n_layers = (int64_t) key_num(file, "block_count");
    desc.n_embd = (int64_t) key_num(file, "embedding_length");
    desc.n_expert = (int64_t) key_num(file, "expert_count");
    desc.n_expert_used = (int64_t) key_num(file, "expert_used_count");
    desc.n_ff_exp = (int64_t) key_num(file, "expert_feed_forward_length");
    desc.n_expert_shared = (int64_t) key_num(file, "expert_shared_count");
    desc.context_length = (int64_t) key_num(file, "context_length");
    if (const MetaValue* vocab = file.get("tokenizer.ggml.tokens")) desc.n_vocab = (int64_t) vocab->count;
    return desc;
}

}  // namespace basalt::models::deepseek4
