// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard and DeepSeek
// geometry from `deepseek41.*` metadata. Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <cstdio>
#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt;
    test::register_meta_adapters();
    check(model::find_adapter("qwen4exp") && model::find_adapter("deepseek41") && model::find_adapter("deepseek4"),
          "every metadata-only adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "basalt_adapter_registry_test";
    std::filesystem::create_directories(dir);
    using namespace fixture;

    write(dir / "other.gguf", {str("general.architecture", "llama")}, {});
    check(model::check_architecture(GgufFile((dir / "other.gguf").string())).find("no model adapter") != std::string::npos,
          "an unregistered architecture is refused by name");

    write(dir / "qwen.gguf", {str("general.architecture", "qwen4exp")}, {});
    check(model::check_architecture(GgufFile((dir / "qwen.gguf").string())).find("block_count") != std::string::npos,
          "qwen4exp without its keys is refused");

    const std::vector<Kv> ds = {str("general.architecture", "deepseek41"), u32("deepseek41.block_count", 43),
                                u32("deepseek41.embedding_length", 5120), u32("deepseek41.expert_count", 384),
                                u32("deepseek41.expert_used_count", 6),
                                u32("deepseek41.expert_feed_forward_length", 2048)};
    write(dir / "ds.gguf", ds, {});
    const GgufFile g((dir / "ds.gguf").string());
    check(model::check_architecture(g).empty(), "deepseek41 with its keys validates");
    std::string err;
    const model::ModelAdapter* a = model::adapter_for(g, err);
    check(a != nullptr && std::string(a->arch()) == "deepseek41", "adapter_for picks deepseek41");
    if (a) {
        const model::ModelDesc d = a->describe(g);
        check(d.n_layers == 43 && d.n_embd == 5120 && d.n_expert == 384 && d.n_expert_used == 6 && d.n_ff_exp == 2048,
              "describe() reads the geometry from the metadata");
    }
    // DeepSeek-V4: every numeric key validate_meta reads, the top-6 hash-routing contract, and a 43+3 schedule
    std::vector<uint32_t> ratios{0u, 0u};
    for (int i = 2; i < 43; ++i) ratios.push_back(i % 2 == 0 ? 4u : 128u);
    ratios.insert(ratios.end(), {0u, 0u, 0u});
    const std::vector<Kv> d4 = {
        str("general.architecture", "deepseek4"), str("tokenizer.ggml.pre", "joyai-llm"),
        u32("deepseek4.block_count", 43), u32("deepseek4.context_length", 1048576),
        u32("deepseek4.embedding_length", 4096), u32("deepseek4.attention.head_count", 64),
        u32("deepseek4.attention.head_count_kv", 1), u32("deepseek4.attention.key_length", 512),
        u32("deepseek4.attention.q_lora_rank", 1024), u32("deepseek4.attention.sliding_window", 128),
        u32("deepseek4.attention.output_group_count", 8), u32("deepseek4.attention.output_lora_rank", 1024),
        u32("deepseek4.rope.dimension_count", 64), u32("deepseek4.rope.scaling.original_context_length", 65536),
        u32("deepseek4.expert_count", 256), u32("deepseek4.expert_used_count", 6),
        u32("deepseek4.expert_shared_count", 1), u32("deepseek4.expert_feed_forward_length", 2048),
        u32("deepseek4.expert_gating_func", 4), u32("deepseek4.hyper_connection.count", 4),
        u32("deepseek4.hyper_connection.sinkhorn_iterations", 20), u32("deepseek4.embedding_length_out", 16384),
        u32("deepseek4.hash_layer_count", 3), u32("deepseek4.attention.indexer.head_count", 64),
        u32("deepseek4.attention.indexer.key_length", 128), u32("deepseek4.attention.indexer.top_k", 512),
        f32("deepseek4.attention.layer_norm_rms_epsilon", 1e-20), f32("deepseek4.attention.compress_rope_freq_base", 160000.0),
        f32("deepseek4.rope.freq_base", 10000.0), f32("deepseek4.rope.scaling.factor", 16.0),
        f32("deepseek4.rope.scaling.yarn_beta_fast", 32.0), f32("deepseek4.rope.scaling.yarn_beta_slow", 1.0),
        f32("deepseek4.expert_weights_scale", 1.5), f32("deepseek4.hyper_connection.epsilon", 1e-6),
        boolean("deepseek4.expert_weights_norm", true),
        u32_array("deepseek4.attention.compress_ratios", ratios),
    };
    write(dir / "ds4.gguf", d4, {});
    const GgufFile g4((dir / "ds4.gguf").string());
    check(model::check_architecture(g4).empty(), "deepseek4 with its keys validates");
    const model::ModelAdapter* a4 = model::adapter_for(g4, err);
    check(a4 != nullptr && std::string(a4->arch()) == "deepseek4", "adapter_for picks deepseek4");
    if (a4) {
        const model::ModelDesc d = a4->describe(g4);
        check(d.n_layers == 43 && d.n_embd == 4096 && d.n_expert == 256 && d.n_expert_used == 6 &&
                  d.n_ff_exp == 2048 && d.n_expert_shared == 1 && d.context_length == 1048576,
              "deepseek4 describe() reads the V4 geometry");
    }
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
