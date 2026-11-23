// Nemotron-H adapter: registers the architecture, validates and describes the file from its `nemotronh.*` metadata, and
// loads the pack into a NemotronModel.
#include "models/nemotronh/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/nemotronh/meta.hpp"
#include "models/nemotronh/model.hpp"

namespace basalt::models::nemotronh {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<NemotronModel>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "nemotronh"}; }
    // the checkpoint's chat template asks for Qwen's form: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...
    program::grammar::ToolForm tool_form() const override { return program::qwen_tool_form(); }
    PreTokenizer pretokenizer() const override { return {"nemotronh", pretok_nemotronh}; }
    int max_expert_cards() const override { return 0; }
    const char* model_name() const override { return "Nemotron-3.5-Super"; }
    int quality_window() const override { return 16; }

    // --kv: the attention caches are fp16
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") {
            o.kv = args.next("--kv");
            return true;
        }
        return false;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16") {
            basalt::log::error("basalt nemotronh", "--kv %s: the attention caches are fp16", o.kv.c_str());
            return 2;
        }
        return std::nullopt;
    }

    // <|im_end|> (and </s>) end a reply, <|im_start|> opens a turn, `assistant` names the model's own; no pictures yet
    model::TokenNames token_names() const override { return {{"</s>", "<|im_end|>"}, "<|im_start|>", "", "", "assistant"}; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::nemotronh
