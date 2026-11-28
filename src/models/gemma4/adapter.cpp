// Gemma 4 adapter: registers the architecture, validates and describes the file from its `gemma4.*` metadata, and loads
// the pack into a Gemma4Model.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/gemma4/kernels.hpp"
#include "basalt/program/options.hpp"
#include "models/gemma4/meta.hpp"
#include "models/gemma4/model.hpp"

#include <algorithm>
#include <cstdlib>

namespace basalt::models::gemma4 {
namespace {

// The checkpoint's tokenizer has no pre-tokenizer (its split on spaces runs after the normalizer has replaced them): a
// span between special tokens is one piece.
void pretok_gemma4(const char*, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (size > 0) ends.push_back(size);
}

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<Gemma4Model>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "gemma4"}; }
    // the 26B A4B's routing order (the experts the card keeps first); a dense pack routes nothing
    std::vector<std::string> shipped_profiles(const std::string& key) const override {
        if (key != "gemma4-30x128") return {};
        return {key + ".prof"};
    }
    // A call: <|tool_call>call:NAME{key:value,...}<tool_call|>, a string value between <|"|> marks.
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<|tool_call>";
        f.call_close = "<tool_call|>";
        f.fn_open = "call:";
        f.fn_mid = "{";
        f.fn_close = "}";
        f.param_mid_text = ":<|\"|>";
        f.param_mid_value = ":";
        f.text_end = "<|\"|>";
        f.trigger = "<|tool_call>";
        return f;
    }
    // A reply's thinking: <|channel>thought\n ... <channel|>, then the answer.
    program::grammar::ReplyForm reply_form() const override {
        program::grammar::ReplyForm f;
        f.think_open = "<|channel>";
        f.think_close = "<channel|>";
        return f;
    }
    PreTokenizer pretokenizer() const override {
        PreTokenizer p;
        p.name = "gemma4";
        p.split = pretok_gemma4;
        p.sentencepiece = true;
        return p;
    }
    // one card holds every weight; a second card has nothing to take yet
    program::GpuModes gpu_modes() const override { return {0, 0, 0}; }
    // each slot's window takes at least one row of a batch window; each slot keeps its own caches
    int max_slots() const override { return kRows; }
    // one architecture for every size: the pack's experts and layer count name it
    const char* model_name(const core::Json& kv) const override {
        if (kv.get("gemma4.expert_count") != nullptr) return "Gemma-4-26B-A4B";
        const core::Json* layers = kv.get("gemma4.block_count");
        return layers != nullptr && layers->i64() >= 60 ? "Gemma-4-31B" : "Gemma-4-12B";
    }
    // NVFP4 for a pack with routed experts or an NVFP4 MLP, else FP8
    const char* community_quant(const core::Json& meta) const override {
        const core::Json* kv = meta.get("gguf_kv");
        if (kv != nullptr && kv->get("gemma4.expert_count") != nullptr) return "NVFP4";
        const core::Json* tensors = meta.get("tensors");
        const core::Json* up = tensors != nullptr ? tensors->get("blk.0.ffn_up.weight") : nullptr;
        const core::Json* type = up != nullptr ? up->get("type") : nullptr;
        return type != nullptr && type->is_string() && type->s == "NVFP4_PLANES" ? "NVFP4" : "FP8";
    }

    // --kv: attention caches as fp16 (default) or fp8; --spec, --suffix-draft, --draft-log: the drafter's windows
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") {
            o.kv = args.next("--kv");
        } else if (a == "--spec") {
            o.spec = std::atoi(args.next("--spec"));
            o.spec_given = true;
        } else if (a == "--suffix-draft") {
            o.suffix_draft = std::atoi(args.next("--suffix-draft"));
        } else if (a == "--draft-log") {
            core::diagnostics().draft_log = args.next("--draft-log");
        } else {
            return false;
        }
        return true;
    }
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16" && o.kv != "fp8") {
            basalt::log::error("basalt gemma4", "--kv %s: the attention caches are fp16 (the default) or fp8", o.kv.c_str());
            return 2;
        }
        // the pack's assistant drafter offers its own draft count plus the anchor unless --spec caps it
        core::ModelFile file;
        std::string err;
        const core::Json* kv = !o.model.empty() && file.open(o.model, err) && file.load_meta(err) ? file.meta().get("gguf_kv") : nullptr;
        const core::Json* n = kv ? kv->get("gemma4.assistant.draft_tokens") : nullptr;
        if (!o.spec_given) o.spec = n ? std::min(kRows, (int) n->num() + 1) : 0;
        if (o.spec >= 2 && !n) {
            basalt::log::warn("basalt gemma4", "--spec %d off: the pack has no drafter", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }

    // <turn|> ends a reply (<eos> the text, <|tool_response> a tool call's turn), <|turn> opens a turn, runs of <|image|>
    // hold a picture's rows.
    model::TokenNames token_names() const override {
        return {{"<turn|>", "<eos>", "<|tool_response>"}, "<|turn>", "<|image|>", "<|video|>", "model"};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::gemma4
