#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <cmath>
#include <string>

namespace basalt::models::deepseek4 {

inline double key_num(const GgufFile& file, const char* suffix, double fallback = 0.0) {
    const MetaValue* value = file.get(std::string("deepseek4.") + suffix);
    if (!value || !value->is_num()) return fallback;
    const double number = value->num();
    return std::isfinite(number) ? number : fallback;
}

inline bool integer_meta_type(MetaType type) {
    return type == MetaType::U8 || type == MetaType::I8 || type == MetaType::U16 || type == MetaType::I16 ||
           type == MetaType::U32 || type == MetaType::I32 || type == MetaType::U64 || type == MetaType::I64;
}

inline std::string validate_compression_ratios(const MetaValue* ratios, int layers) {
    if (!ratios || ratios->type != MetaType::ARRAY || !integer_meta_type(ratios->elem) ||
        ratios->count != (uint64_t) layers + 3 || ratios->items.size() != ratios->count)
        return "deepseek4.attention.compress_ratios must be an integer array covering trunk and three MTP slots";
    for (int i = 0; i < layers; ++i) {
        const MetaValue& item = ratios->items[(size_t) i];
        const uint64_t ratio = item.u;
        if (item.type != ratios->elem || (ratio != 0 && ratio != 4 && ratio != 128))
            return "unsupported or non-integer compression ratio at deepseek4 layer " + std::to_string(i);
    }
    for (size_t i = (size_t) layers; i < ratios->items.size(); ++i) {
        if (ratios->items[i].type != ratios->elem || ratios->items[i].u != 0)
            return "deepseek4 MTP compression schedule must contain integer zeros";
    }
    return {};
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
        key_num(file, "expert_gating_func") != 4 || key_num(file, "expert_weights_scale") != 1.5)
        return "deepseek4 routing metadata is outside the supported top-6 hash-routing contract";

    const int layers = (int) key_num(file, "block_count");
    const double experts = key_num(file, "expert_count");
    if (key_num(file, "expert_used_count") > experts || key_num(file, "hash_layer_count") > layers)
        return "deepseek4 metadata has an invalid expert or hash-layer count";

    if (std::string schedule_err = validate_compression_ratios(file.get("deepseek4.attention.compress_ratios"), layers);
        !schedule_err.empty()) return schedule_err;

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
