// DeepSeek-V4.1-Flash adapter: registers the architecture, validates and describes the file from its
// `deepseek41.*` metadata, and loads the pack into a DsModel.
#include "models/deepseek41/adapter.hpp"

#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "models/deepseek41/context_policy.hpp"
#include "models/deepseek41/meta.hpp"
#include "models/deepseek41/model.hpp"

#include <cstdint>
#include <cstdlib>

namespace basalt::models::deepseek41 {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "deepseek41"; }

    std::string validate(const GgufFile& g) const override { return validate_meta(g); }

    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "deepseek41: no model path"};
        auto m = std::make_unique<DsModel>();
        std::string err;
        if (!m->load(req.paths.front(), err)) return {model::Status::Error, "deepseek41: " + err};
        if (!m->resolve_context(req.max_context, err)) return {model::Status::Error, err};
        m->set_diag_dump(req.diag_dump);
        m->set_diag_tier(req.diag_tier_ram, req.diag_tier_vram);
        m->set_tiers(req);
        if (!req.peer_devices.empty()) m->set_peer_device(req.peer_devices.front());
        if (req.peer_devices.size() > 1) basalt::log::warn("basalt deepseek41", "one second card holds experts; later cards idle");
        out = std::move(m);
        return {};
    }

    std::vector<model::SideTableSpec> side_tables(const model::ModelDesc&) const override {
        // the Engram table; its row geometry comes from the pack's meta.engram
        return {{model::SideTableSpec::Kind::Engram, "per_layer_token_embd.weight", 256, 0}};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "deepseek"}; }

    // <｜DSML｜ calls>, <｜DSML｜ invoke name="N">, <｜DSML｜ parameter name="P" string="true|false">
    program::grammar::ToolForm tool_form() const override {
        const std::string dsml = "\xef\xbd\x9c" "DSML\xef\xbd\x9c";
        program::grammar::ToolForm f;
        f.call_open = "<" + dsml + " calls>\n";
        f.call_close = "\n</" + dsml + " calls>";
        f.fn_open = "<" + dsml + " invoke name=\"";
        f.fn_mid = "\">\n";
        f.fn_close = "</" + dsml + " invoke>";
        f.param_open = "<" + dsml + " parameter name=\"";
        f.param_mid_text = "\" string=\"true\">";
        f.param_mid_value = "\" string=\"false\">";
        f.param_close = "</" + dsml + " parameter>\n";
        f.text_end = "</" + dsml + " parameter>";
        f.calls_share_block = true;
        f.trigger = "<" + dsml + " calls>";
        return f;
    }

    PreTokenizer pretokenizer() const override { return {"joyai-llm", pretok_joyai}; }

    int max_expert_cards() const override { return 1; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (a == "--no-prompt-skip") o.no_prompt_skip = true;
        else if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else if (a == "--spec-oracle") o.spec_oracle = args.next("--spec-oracle");
        else if (a == "--draft-experts") o.draft_experts = std::atoi(args.next("--draft-experts"));
        else if (a == "--diag-drafter-gpu") o.drafter_device = std::atoi(args.next("--diag-drafter-gpu"));
        else if (a == "--diag-teacher") o.diag_teacher = true;
        else if (a == "--diag-tier-frozen") o.diag_tier_frozen = true;
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = args.next("--draft-log");
        else return false;
        return true;
    }

    // A context the pack refuses is refused with the flags, before anything loads; the memory check needs the card.
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (o.draft_experts < 0 || o.draft_experts > 128 || o.draft_experts % 32 != 0) {
            basalt::log::error("basalt deepseek41", "--draft-experts %d: 32, 64, 96 or 128, or 0 for all", o.draft_experts);
            return 2;
        }
        core::ModelFile file;
        std::string err;
        const bool opened = !o.model.empty() && file.open(o.model, err);
        const bool drafter = opened && file.find("dspark/gguf") != nullptr;
        // the pack's drafter runs in windows of 3 unless --spec says otherwise; a token file drafts in its place
        if (!o.spec_given) {
            o.spec = drafter && o.batch <= 1 ? 3 : 0;
            if (opened && !drafter) basalt::log::info("basalt deepseek41", "no dspark/gguf section: plain decode");
        }
        if (o.spec >= 2 && o.spec_oracle.empty()) {
            const char* off = o.batch > 1 ? "--parallel slots draft nothing" : !drafter ? "the pack has no dspark/gguf section" : nullptr;
            if (off != nullptr) {
                basalt::log::warn("basalt deepseek41", "--spec %d off: %s", o.spec, off);
                o.spec = 0;
            }
        }
        if (!opened || !file.load_meta(err)) return std::nullopt;
        const core::Json* kv = file.meta().get("gguf_kv");
        const core::Json* native = kv ? kv->get("deepseek41.context_length") : nullptr;
        const ctxp::Plan p = ctxp::plan_context(o.max_context, native ? (int64_t) native->u64() : 0, UINT64_MAX, {});
        if (p.error.empty()) return std::nullopt;
        basalt::log::error("basalt deepseek41", "%s", p.error.c_str());
        return 2;
    }

    // The disk tier's records count positional cells; this model's caches keep compressed rows.
    model::TokenNames token_names() const override {
        return {{"<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"}, "<\xef\xbd\x9cUser\xef\xbd\x9c>",
                "<\xef\xbd\x9c" "deepseek_image\xef\xbd\x9c>"};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::deepseek41
