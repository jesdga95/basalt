// Flash-Next adapter. Decode, prefill and the drafter stay on the engine's own paths (core/verify, prefill,
// mtp); this adapter owns the architecture guard, the geometry and the side-table/tokenizer descriptions.
#include "models/qwen4exp/adapter.hpp"
#include "models/qwen4exp/meta.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "unicode_flags.hpp"
#include "basalt/models/qwen4exp/options.hpp"

namespace basalt::models::qwen4exp {
namespace {

constexpr const char* kPrefix = "qwen4exp.";

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "qwen4exp"; }

    std::string validate(const GgufFile& g) const override { return validate_meta(g); }

    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    model::Result load(const model::LoadRequest&, std::unique_ptr<model::LoadedModel>&) const override {
        return {model::Status::NotImplemented, "Flash-Next runs its own start"};
    }

    std::optional<int> run(program::CommonOptions& o, std::vector<program::GpuCard>& cards) const override {
        return run_engine(static_cast<Options&>(o), cards);
    }

    int max_expert_cards() const override { return kernels::kDmMaxX; }

    std::vector<model::SideTableSpec> side_tables(const model::ModelDesc&) const override {
        return {{model::SideTableSpec::Kind::Ple, "per_layer_token_embd.weight", 0, 0}};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "qwen"}; }

    PreTokenizer pretokenizer() const override { return {"qwen35", tokenizer::pretok_qwen35}; }

    model::TokenNames token_names() const override {
        return {{"<|endoftext|>", "<|im_end|>"}, "<|im_start|>", "<|image_pad|>"};
    }

    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<tool_call>\n";
        f.call_close = "\n</tool_call>";
        f.fn_open = "<function=";
        f.fn_mid = ">\n";
        f.fn_close = "</function>";
        f.param_open = "<parameter=";
        f.param_mid_text = f.param_mid_value = ">\n";
        f.param_close = "\n</parameter>\n";
        f.text_end = "</parameter>";
        f.trigger = "<tool_call>";
        return f;
    }

    std::unique_ptr<program::CommonOptions> make_options() const override { return std::make_unique<Options>(); }
    bool parse_flag(const std::string& name, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        return qwen4exp::parse_flag(name, args, static_cast<Options&>(o), code);
    }
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        return qwen4exp::finish_flags(static_cast<Options&>(o));
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::qwen4exp
