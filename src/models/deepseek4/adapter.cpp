#include "models/deepseek4/adapter.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/model/route_flags.hpp"
#include "basalt/models/deepseek4/tokenizer.hpp"
#include "models/deepseek4/context_policy.hpp"
#include "models/deepseek4/dspark.hpp"
#include "models/deepseek4/meta.hpp"
#include "models/deepseek4/model.hpp"

#include <cstdlib>
#include <vector>

namespace basalt::models::deepseek4 {
namespace {

// --route-prior alone: the DeepSeek line's measured default, on the biased score its router ranks by.
constexpr float kRoutePriorBare = 0.1f;

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
    const char* model_name() const override { return "DeepSeek-V4-Flash-Vision-Exp"; }
    // Its caches are F32, it takes no --kv format: a community row runs the model's own.
    const char* community_kv() const override { return ""; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        if (model::parse_route_flag(a, args, o.route, code)) {}
        else if (a == "--spec") {
            o.spec = std::atoi(args.next("--spec"));
            o.spec_given = true;
            if (o.spec < 0 || o.spec > 8) code = 2;
        } else if (a == "--spec-oracle") {
            o.spec_oracle = args.next("--spec-oracle");
        } else if (a == "--draft-log") {
            basalt::core::diagnostics().draft_log = args.next("--draft-log");
        } else if (a == "--diag-rows") {
            o.diag_rows = std::atoi(args.next("--diag-rows"));
            if (o.diag_rows < 1 || o.diag_rows > 8192) code = 2;
        } else if (a == "--diag-tier-frozen") {
            o.diag_tier_frozen = true;
        } else return false;
        return true;
    }

    // A context the pack refuses is refused with the flags, before anything loads; the memory check needs the card.
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        core::ModelFile file;
        std::string err;
        const bool opened = !o.model.empty() && file.open(o.model, err);
        const bool drafter = opened && file.find("dspark") != nullptr;
        // The pack's drafter offers a whole block unless --spec caps it; a token file drafts in its place.
        if (!o.spec_given) {
            o.spec = drafter ? kDsparkBlock + 1 : 0;
            if (opened && !drafter) basalt::log::info("basalt deepseek4", "no dspark section: plain decode");
        }
        if (o.spec >= 2 && o.spec_oracle.empty() && !drafter) {
            basalt::log::warn("basalt deepseek4", "--spec %d off: the pack has no dspark section", o.spec);
            o.spec = 0;
        }
        const bool meta = opened && file.load_meta(err);
        const core::Json* kv = meta ? file.meta().get("gguf_kv") : nullptr;
        const core::Json* geo = meta ? file.meta().get("geometry") : nullptr;
        const core::Json* used = geo ? geo->get("expert_used_count") : nullptr;
        if (const std::optional<int> code =
                model::finish_route_flags(o, used ? (int) used->num() : 0, kRoutePriorBare, "basalt deepseek4"))
            return code;
        if (!meta) return std::nullopt;
        const core::Json* native = kv ? kv->get("deepseek4.context_length") : nullptr;
        const core::Json* layers = geo ? geo->get("block_count") : nullptr;
        const core::Json* schedule = kv ? kv->get("deepseek4.attention.compress_ratios") : nullptr;
        if (!native || !layers || !schedule || !schedule->is_array() || schedule->count < layers->u64())
            return std::nullopt;   // the load validates the metadata it could not read here
        std::vector<int> ratios;
        for (uint64_t i = 0; i < layers->u64() && i < schedule->items.size(); ++i)
            ratios.push_back((int) schedule->items[(size_t) i].u64());
        const ctxp::Plan plan = ctxp::plan_context(o.max_context, (int64_t) native->u64(), UINT64_MAX, ratios);
        if (plan.error.empty()) return std::nullopt;
        basalt::log::error("basalt deepseek4", "%s", plan.error.c_str());
        return 2;
    }

    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<DsModel>(); }

    model::TokenNames token_names() const override {
        return {{"<｜end▁of▁sentence｜>"}, "<｜User｜>", "<｜deepseek_image｜>", "", "<｜Assistant｜>"};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::deepseek4
