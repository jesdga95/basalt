#include "models/deepseek4/adapter.hpp"

#include "basalt/models/deepseek4/tokenizer.hpp"
#include "models/deepseek4/meta.hpp"

namespace basalt::models::deepseek4 {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "deepseek4"; }
    std::string validate(const GgufFile& file) const override { return validate_meta(file); }
    model::ModelDesc describe(const GgufFile& file) const override { return describe_meta(file); }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "deepseek"}; }

    program::grammar::ToolForm tool_form() const override {
        const std::string dsml = "｜DSML｜";
        program::grammar::ToolForm form;
        form.call_open = "<" + dsml + "tool_calls>\n";
        form.call_close = "\n</" + dsml + "tool_calls>";
        form.fn_open = "<" + dsml + "invoke name=\"";
        form.fn_mid = "\">\n";
        form.fn_close = "</" + dsml + "invoke>";
        form.param_open = "<" + dsml + "parameter name=\"";
        form.param_mid_text = "\" string=\"true\">";
        form.param_mid_value = "\" string=\"false\">";
        form.param_close = "</" + dsml + "parameter>\n";
        form.text_end = "</" + dsml + "parameter>";
        form.calls_share_block = true;
        form.trigger = "<" + dsml + "tool_calls>";
        return form;
    }

    PreTokenizer pretokenizer() const override { return {"joyai-llm", pretok_deepseek4}; }
    program::GpuModes gpu_modes() const override { return {}; }

    model::TokenNames token_names() const override {
        return {{"<｜end▁of▁sentence｜>"}, "<｜User｜>", "<｜deepseek_image｜>", "", "<｜Assistant｜>"};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::deepseek4
