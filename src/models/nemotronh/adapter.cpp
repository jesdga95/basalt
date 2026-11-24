// Nemotron-H adapter: registers the architecture, validates and describes the file from its `nemotronh.*` metadata, and
// loads the pack into a NemotronModel.
#include "models/nemotronh/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/nemotronh/meta.hpp"
#include "models/nemotronh/model.hpp"

#include <algorithm>
#include <cstdlib>

namespace basalt::models::nemotronh {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<NemotronModel>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "nemotronh"}; }
    // Qwen's form, as the checkpoint's template writes it: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n
    // </function>\n</tool_call>\n, the newline after a call part of it (the grammar then ends there, and the turn may end)
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f = program::qwen_tool_form();
        f.call_close = "\n</tool_call>\n";
        return f;
    }
    PreTokenizer pretokenizer() const override { return {"nemotronh", pretok_nemotronh}; }
    program::GpuModes gpu_modes() const override { return {}; }   // one card: the experts stay on it and in RAM
    const char* model_name() const override { return "Nemotron-3.5-Super"; }
    int quality_window() const override { return 16; }

    // --kv: the attention caches are fp16; --suffix-draft: prompt lookup's least repeat (0: off); --spec: the rows of a
    // window the MTP head drafts for (0 or 1: none); --draft-log: the windows' log; --diag-tier-frozen:
    // the card keeps the experts it loaded (windows then bitwise one-row decode)
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") o.kv = args.next("--kv");
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(args.next("--suffix-draft")));
        else if (a == "--spec") {
            o.spec = std::atoi(args.next("--spec"));
            o.spec_given = true;
        } else if (a == "--draft-log") core::diagnostics().draft_log = args.next("--draft-log");
        else if (a == "--diag-tier-frozen") o.diag_tier_frozen = true;
        else return false;
        return true;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16") {
            basalt::log::error("basalt nemotronh", "--kv %s: the attention caches are fp16", o.kv.c_str());
            return 2;
        }
        // a pack with the MTP head drafts unless --spec says otherwise; the window policy takes the rows that pay
        core::ModelFile file;
        std::string err;
        const core::Json* kv = !o.model.empty() && file.open(o.model, err) && file.load_meta(err) ? file.meta().get("gguf_kv") : nullptr;
        const core::Json* mp = kv ? kv->get("nemotronh.mtp_pattern") : nullptr;
        const bool present = mp != nullptr && mp->is_string() && !mp->s.empty();
        if (!o.spec_given) o.spec = present ? kMtpSpec : 0;
        if (o.spec >= 2 && !present) {
            basalt::log::warn("basalt nemotronh", "--spec %d off: the pack has no MTP head", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }
    static constexpr int kMtpSpec = 4;   // the rows a drafted window may hold by default

    // <|im_end|> (and </s>) end a reply, <|im_start|> opens a turn, `assistant` names the model's own; no pictures yet
    model::TokenNames token_names() const override { return {{"</s>", "<|im_end|>"}, "<|im_start|>", "", "", "assistant"}; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::nemotronh
