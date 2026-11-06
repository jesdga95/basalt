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
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
