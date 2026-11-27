// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard, and the
// geometry a model reads from its own metadata (DeepSeek's `deepseek41.*`, GLM's `glm5-next.*`, including the rules that
// refuse a file whose metadata contradicts itself). Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <algorithm>
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
    check(model::find_adapter("qwen4exp") && model::find_adapter("deepseek41"), "both adapters registered");
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
    // GLM-5.3-Flash: 46 blocks of which the last is the drafter. These are the release's numbers, read from the header of
    // UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf; only the interleaving of the layer kinds is the fixture's
    // (the header gives 34 KDA and 12 sparse-MLA layers, not where each sits).
    std::vector<uint64_t> kinds(46, 0);
    for (int l = 3; l < 45; l += 4) kinds[l] = 1;
    kinds[45] = 1;
    const std::vector<Kv> glm = {str("general.architecture", "glm5-next"), u32("glm5-next.block_count", 46),
                                 u32("glm5-next.embedding_length", 4096), u32("glm5-next.expert_count", 288),
                                 u32("glm5-next.expert_used_count", 8), u32("glm5-next.expert_feed_forward_length", 2048),
                                 u32("glm5-next.expert_shared_count", 1), u32("glm5-next.context_length", 1048576),
                                 u32("glm5-next.kda.head_dim", 128), u32("glm5-next.ssm.conv_kernel", 4),
                                 u32("glm5-next.attention.indexer.kpool", 4), u32("glm5-next.hyper_connection.count", 4),
                                 u32("glm5-next.leading_dense_block_count", 3),
                                 u32("glm5-next.nextn_predict_layers", 1), u32arr("glm5-next.attention.head_count_kv", kinds)};
    write(dir / "glm.gguf", glm, {});
    const GgufFile gg((dir / "glm.gguf").string());
    check(model::check_architecture(gg).empty(), "glm5next with its keys validates");
    const model::ModelAdapter* ga = model::adapter_for(gg, err);
    check(ga != nullptr, "adapter_for picks glm5next");
    if (ga) {
        const model::ModelDesc d = ga->describe(gg);
        check(d.n_layers == 45 && d.n_embd == 4096 && d.n_expert == 288 && d.n_expert_used == 8 && d.n_ff_exp == 2048 &&
                  d.n_expert_shared == 1 && d.context_length == 1048576,
              "describe() reads the release's geometry and leaves the drafter out of the model's layers");
    }
    check(models::glm5next::layer_kinds(gg).size() == 46 && models::glm5next::drafter_layers(gg) == std::vector<int>{45},
          "the layer kinds and the drafter block come from the metadata");
    std::vector<Kv> pack_named = glm;
    pack_named.front().s = models::glm5next::kArch;   // a pack written under the Basalt name
    write(dir / "glm-pack.gguf", pack_named, {});
    check(model::check_architecture(GgufFile((dir / "glm-pack.gguf").string())).empty(),
          "the same metadata validates under either name");
    struct Bad {
        const char* what;      ///< the metadata that is wrong
        const char* want;      ///< what the refusal must say
        std::vector<Kv> patch;  ///< keys that replace the ones above
        const char* erase;     ///< a key to drop instead
    };
    const std::vector<Bad> bad = {
        {"a layer kind that is neither 0 nor 1", "linear-attention (0)",
         {u32arr("glm5-next.attention.head_count_kv", std::vector<uint64_t>(46, 2))}, nullptr},
        {"fewer layer kinds than blocks", "attention.head_count_kv lists 45",
         {u32arr("glm5-next.attention.head_count_kv", std::vector<uint64_t>(45, 0))}, nullptr},
        {"no layer kinds at all", "attention.head_count_kv lists 0", {}, "glm5-next.attention.head_count_kv"},
        {"a dense-block count past the last layer", "leading_dense_block_count",
         {u32("glm5-next.leading_dense_block_count", 46)}, nullptr},
        {"a router that keeps more experts than there are", "router keeps", {u32("glm5-next.expert_used_count", 289)}, nullptr},
        {"no drafter block", "no next-n block", {u32("glm5-next.nextn_predict_layers", 0)}, nullptr},
        {"a drafter that is not the last block", "expected the last block", {u32("glm5-next.nextn_predict_layers", 2)}, nullptr},
        {"no indexer key pooling", "glm5-next.attention.indexer.kpool", {u32("glm5-next.attention.indexer.kpool", 0)}, nullptr},
        {"no hyper-connection streams", "glm5-next.hyper_connection.count", {}, "glm5-next.hyper_connection.count"},
    };
    for (size_t i = 0; i < bad.size(); ++i) {
        const Bad& b = bad[i];
        std::vector<Kv> kv = glm;
        for (const Kv& k : b.patch)
            kv.erase(std::remove_if(kv.begin(), kv.end(), [&](const Kv& x) { return x.key == k.key; }), kv.end());
        if (b.erase)
            kv.erase(std::remove_if(kv.begin(), kv.end(), [&](const Kv& x) { return x.key == b.erase; }), kv.end());
        for (const Kv& k : b.patch) kv.push_back(k);
        const std::string file = "glm-bad-" + std::to_string(i) + ".gguf";
        write(dir / file, kv, {});
        const GgufFile g2((dir / file).string());
        const std::string e = model::check_architecture(g2);
        check(e.find(b.want) != std::string::npos, b.what);
        if (e.find(b.want) == std::string::npos) std::printf("     (said: %s)\n", e.c_str());
    }
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
