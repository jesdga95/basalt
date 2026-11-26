// GLM-5.3-Flash adapter: registers the architecture, validates and describes the file from its `glm5-next.*` metadata,
// and loads the pack into a GlmModel.
//
// The text-level answers are the release's own, taken from its tokenizer rather than guessed at, so they are recorded
// here with their source:
//   - generation_config.json's eos_token_id [154820, 154827, 154829] is `<|endoftext|>`, `<|user|>`, `<|observation|>`
//     in that vocabulary: the end of a reply, and the two role tokens that open the next turn when the model stops
//     speaking. `<|assistant|>` (154828) is the token the template writes to open the model's own turn, so it is the
//     turn token and no role word follows it. The image and video spans the template emits are
//     `<|begin_of_image|><|image|><|end_of_image|>` and the video equivalent, so the pad is `<|image|>` / `<|video|>`.
//   - the tool call is `<tool_call>` NAME `<arg_key>` PARAM `</arg_key>` `<arg_value>` VALUE `</arg_value>`
//     `</tool_call>`, the form the model is instructed to write in the release's chat_template.jinja and whose every
//     marker is a single token in the vocabulary (154843..154850).
//   - the pre-tokenizer is glm4: program/glm_forms.hpp records that the checkpoint's tokenizer.ggml.pre is glm4 for
//     GLM-5.3-Flash, and src/program/tokenizer/pretok_glm4.cpp implements it.
// The server renders the conversation from the pack's own chat_template.jinja (serve/engine/modelfile.py), so
// chat_format names the model to the server side and selects nothing in the engine.
#include "models/glm5next/adapter.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/glm_forms.hpp"
#include "basalt/program/grammar/tool_form.hpp"
#include "basalt/program/options.hpp"
#include "models/glm5next/meta.hpp"
#include "models/glm5next/model.hpp"

#include <cstdlib>
#include <memory>

namespace basalt::models::glm5next {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    explicit Adapter(const char* name) : name_(name) {}

    const char* arch() const override { return name_; }

    std::string validate(const GgufFile& g) const override { return validate_meta(g); }

    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<GlmModel>(); }

    const char* model_name() const override { return "GLM-5.3-Flash"; }

    // The window's rows are the start's call: --spec 0 or 1 decodes one row at a time, --spec 2 asks the pack's next-n
    // block for a draft beside the token the model already picked (the block holds one, so more than 2 is that).
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else if (a == "--spec-oracle") o.spec_oracle = args.next("--spec-oracle");
        else if (a == "--diag-teacher") o.diag_teacher = true;
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = args.next("--draft-log");
        else return false;
        return true;
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "glm5next"}; }

    // The main card and one peer that holds experts: core::ExpertTier is the model's peer driver, so a second card is
    // dealt a block of layers' homes and computes their rows with the grouped kernels this build also compiles for
    // sm_89 (-DPEER_ADA=ON). There is no prompt stage and no layer split to deal.
    program::GpuModes gpu_modes() const override { return {2, 0, 0, true}; }

    PreTokenizer pretokenizer() const override { return {"glm4", program::tokenizer::pretok_glm4}; }

    model::TokenNames token_names() const override {
        return {{"<|endoftext|>", "<|user|>", "<|observation|>"}, "<|assistant|>", "<|image|>", "<|video|>", ""};
    }

    // The name follows `<tool_call>` with nothing between them, so it is delimited by the `<arg_key>` that opens the
    // first parameter; a parameter's name ends at the `<arg_value>` that opens its value, and both string and
    // non-string values are written the same way. GLM writes one `<tool_call> .. </tool_call>` block per call.
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<tool_call>";
        f.call_close = "</tool_call>";
        f.fn_open = "";
        f.fn_mid = "<arg_key>";
        f.fn_close = "";
        f.param_open = "";
        f.param_mid_text = f.param_mid_value = "</arg_key><arg_value>";
        f.param_close = "</arg_value>";
        f.text_end = "</arg_value>";
        f.trigger = "<tool_call>";
        return f;
    }

private:
    const char* name_;
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() {
    // A pack converted from the release's GGUF keeps that file's `general.architecture` (glm5-next) in its gguf_kv,
    // which is what the registry looks an adapter up by; the same adapter answers to it. Packs this converter writes
    // carry kArch and need no alias.
    if (model::find_adapter(kGgufArch) == nullptr) model::register_adapter(std::make_unique<Adapter>(kGgufArch));
    return std::make_unique<Adapter>(kArch);
}

}  // namespace basalt::models::glm5next
