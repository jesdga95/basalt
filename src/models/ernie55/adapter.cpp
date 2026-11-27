// src/models/ernie55/adapter.cpp - ERNIE-5.5-Flash ModelAdapter registration, flags, tool form, and token names.
#include "models/ernie55/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/model/route_flags.hpp"
#include "basalt/program/options.hpp"
#include "models/ernie55/context_policy.hpp"
#include "models/ernie55/dspark.hpp"
#include "models/ernie55/meta.hpp"
#include "models/ernie55/model.hpp"

#include <cstdlib>

namespace basalt::models::ernie55 {
namespace {

constexpr float kRoutePriorBare = 0.05f;

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "ernie55"; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<ErnieModel>(); }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "ernie55"}; }

    std::vector<std::string> shipped_profiles(const std::string& key) const override {
        if (key != "ernie55-43x512") return {};
        return {key + ".prof"};
    }

    // <ernie:tool_call>\n<invoke name="N">\n<parameter name="P">V</parameter>\n</invoke>\n</ernie:tool_call>
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<ernie:tool_call>\n";
        f.call_close = "\n</ernie:tool_call>";
        f.fn_open = "<invoke name=\"";
        f.fn_mid = "\">\n";
        f.fn_close = "</invoke>";
        f.param_open = "<parameter name=\"";
        f.param_mid_text = "\">";
        f.param_mid_value = "\">";
        f.param_close = "</parameter>\n";
        f.text_end = "</parameter>";
        f.calls_share_block = true;
        f.trigger = "<ernie:tool_call>";
        return f;
    }

    PreTokenizer pretokenizer() const override { return {"ernie55", pretok_ernie55}; }
    program::GpuModes gpu_modes() const override { return {kMaxSecondCards + 1, 0, 0}; }
    const char* model_name() const override { return "ERNIE-5.5-Flash"; }
    const char* community_kv() const override { return "fp8"; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        if (model::parse_route_flag(a, args, o.route, code)) return true;
        if (a == "--kv") {
            o.kv = args.next("--kv");
            return true;
        }
        if (a == "--spec") {
            o.spec = std::atoi(args.next("--spec"));
            o.spec_given = true;
            return true;
        }
        if (a == "--draft-log") {
            basalt::core::diagnostics().draft_log = args.next("--draft-log");
            return true;
        }
        if (a == "--diag-rows") {
            o.diag_rows = std::atoi(args.next("--diag-rows"));
            return true;
        }
        if (a == "--draft-experts") {
            o.draft_experts = std::atoi(args.next("--draft-experts"));
            draft_experts_given_ = true;
            return true;
        }
        return false;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        // The drafter's GPU cache of hot experts (slots per stage): off unless requested, leaving that VRAM for the
        // main model's expert tier and avoiding pageable host copies on the drafter's stream.
        if (!draft_experts_given_) o.draft_experts = 0;
        if (o.kv.empty()) o.kv = "fp8";
        ctxp::CacheFormat fmt = ctxp::CacheFormat::Fp8;
        if (!ctxp::cache_format(o.kv, fmt)) {
            basalt::log::error("basalt ernie55", "--kv %s: ERNIE-5.5 supports fp8 (default) or fp16", o.kv.c_str());
            return 2;
        }
        core::ModelFile file;
        std::string err;
        const bool opened = !o.model.empty() && file.open(o.model, err);
        const bool drafter = opened && file.find("dspark") != nullptr;
        if (!o.spec_given) {
            o.spec = drafter ? kDsparkBlock + 1 : 0;
            if (opened && !drafter) basalt::log::info("basalt ernie55", "no dspark section: plain decode");
        }
        if (o.spec >= 2 && o.spec_oracle.empty() && !drafter) {
            basalt::log::warn("basalt ernie55", "--spec %d off: the pack has no dspark section", o.spec);
            o.spec = 0;
        }
        int top_k = 0;
        if (o.route.tail_skip != 0 || o.route.prior != 0.0f) {
            const core::Json* kv = opened && file.load_meta(err) ? file.meta().get("gguf_kv") : nullptr;
            const core::Json* used = kv ? kv->get("ernie55.expert_used_count") : nullptr;
            top_k = used ? (int) used->num() : 0;
        }
        return model::finish_route_flags(o, top_k, kRoutePriorBare, "basalt ernie55");
    }

    model::TokenNames token_names() const override {
        return {{"<|ernie:im_end|>", "<|ERNIE_MASK|>"}, "<|ernie:im_start|>", "", "", "assistant"};
    }

private:
    mutable bool draft_experts_given_ = false;
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::ernie55
