// Hy4-flash adapter: registers the architecture, validates and describes the file from its `hy_v4_flash.*` metadata, and
// loads the pack into a Hy4Model.
#include "models/hy4flash/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "models/hy4flash/dfly.hpp"   // DFLY HOOK
#include "models/hy4flash/meta.hpp"
#include "models/hy4flash/model.hpp"

namespace basalt::models::hy4flash {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    explicit Adapter(const char* name) : name_(name) {}
    const char* arch() const override { return name_; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<Hy4Model>(); }

    std::vector<model::SideTableSpec> side_tables(const model::ModelDesc&) const override {
        return {{model::SideTableSpec::Kind::Engram, "per_layer_token_embd.weight", 96, 0}};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "hy_v4"}; }

    // <tool_calls><tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call></tool_calls>
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<tool_calls><tool_call>";
        f.call_close = "</tool_call></tool_calls>";
        f.fn_mid = "";
        f.param_open = "<arg_key>";
        f.param_mid_text = "</arg_key><arg_value>";
        f.param_mid_value = "</arg_key><arg_value>";
        f.param_close = "</arg_value>";
        f.text_end = "</arg_value>";
        f.trigger = "<tool_calls>";
        return f;
    }

    PreTokenizer pretokenizer() const override { return {"hy_v4", pretok_hy4}; }
    int max_expert_cards() const override { return kMaxSecondCards; }
    const char* model_name() const override { return "Hy4-flash"; }

    // --kv (the sparse layers' latent rows, bf16 by default or fp8) and, DFLY HOOK, --spec and its companions as DeepSeek's
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") {
            o.kv = args.next("--kv");
            return true;
        }
        return dfly_parse_flag(a, args, o);
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "bf16" && o.kv != "fp8") {
            basalt::log::error("basalt hy4flash", "--kv %s: the latent caches are bf16 (the default) or fp8", o.kv.c_str());
            return 2;
        }
        dfly_finish_flags(o);
        return std::nullopt;
    }

    // the end-of-turn token is read from the pack's vocabulary (<hy_end> is 120025)
    model::TokenNames token_names() const override { return {{"<hy_end>"}, "<hy_User>", "", "", ""}; }

private:
    const char* name_;
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() {
    // The test pack made from the AngelSlim GGUF keeps that file's `general.architecture` (hy_v4_flash) in its gguf_kv, which is
    // what the registry looks an adapter up by; the same adapter answers to it. Packs written with hy4flash need no alias.
    if (model::find_adapter(kGgufArch) == nullptr) model::register_adapter(std::make_unique<Adapter>(kGgufArch));
    return std::make_unique<Adapter>("hy4flash");
}

}  // namespace basalt::models::hy4flash
