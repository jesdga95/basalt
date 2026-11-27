#include "models/deepseek4/adapter.hpp"

#include "basalt/program/options.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/deepseek4/tokenizer.hpp"
#include "models/deepseek4/meta.hpp"
#include "models/deepseek4/model.hpp"

#include <cstdlib>

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
    int max_slots() const override { return 8; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        if (a == "--diag-rows") {
            o.diag_rows = std::atoi(args.next("--diag-rows"));
            if (o.diag_rows < 1 || o.diag_rows > 8192) code = 2;
            return true;
        }
        if (a != "--diag-tier-frozen") return false;
        o.diag_tier_frozen = true;
        return true;
    }

    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<DsModel>(); }

    model::TokenNames token_names() const override {
        return {{"<｜end▁of▁sentence｜>"}, "<｜User｜>", "<｜deepseek_image｜>", "", "<｜Assistant｜>"};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::deepseek4
