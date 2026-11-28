// Inkling 2 adapter: registers the architecture, validates and describes the file from its `inkling2.*` metadata, and
// loads the pack into an InklingModel.
#include "models/inkling2/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/kernels/window_segs.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/meta.hpp"
#include "models/inkling2/model.hpp"

#include <cstdlib>

namespace basalt::models::inkling2 {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<InklingModel>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "inkling2"}; }
    // A call is its own block: <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>.
    // A block that opens with a name, not a content tag, starts a call.
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<|message_model|>";
        f.fn_mid = "<|content_invoke_tool_json|>{\"name\":{name},\"args\":";
        f.fn_close = "}";
        f.call_close = "<|end_message|>";
        f.json_args = true;
        f.trigger_pattern = "<\\|message_model\\|>[^<]";
        return f;
    }
    // A reply is a list of blocks, each opened by <|message_model|> and ended by <|end_message|>: thinking blocks
    // (<|content_thinking|>), then the answer (<|content_text|>, or text right after the header).
    program::grammar::ReplyForm reply_form() const override {
        program::grammar::ReplyForm f;
        f.header = "<|message_model|>";
        f.think_open = "<|content_thinking|>";
        f.think_close = "<|end_message|>";
        f.answer_open = "<|content_text|>";
        return f;
    }
    PreTokenizer pretokenizer() const override { return {"inkling2", pretok_inkling2, kIgnoreMerges}; }
    // peer: the second card holds and computes experts the main card cannot (model.hpp peer_held); split: it runs the later layers (card_of)
    int max_slots() const override { return basalt::kernels::kWindowSegsMax; }   // segments of one batch window
    program::GpuModes gpu_modes() const override { return {2, 2, 0}; }
    const char* model_name(const core::Json&) const override { return "Inkling-2-Mini"; }
    const char* community_quant(const core::Json&) const override { return "NVFP4"; }   // NVFP4 experts, Q8_0 dense

    // --kv: attention caches as fp16 (default), fp8 or scaled int8; --spec, --draft-log: drafter windows
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") {
            o.kv = args.next("--kv");
            return true;
        }
        return dspark_parse_flag(a, args, o);   // DSPARK HOOK
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16" && o.kv != "fp8" && o.kv != "int8") {
            basalt::log::error("basalt inkling2", "--kv %s: the attention caches are fp16 (the default), fp8 or int8", o.kv.c_str());
            return 2;
        }
        dspark_finish_flags(o);   // DSPARK HOOK
        return std::nullopt;
    }

    // <|content_model_end_sampling|> ends a reply, <|message_user|> opens the user's turn, runs of <|unused_200054|> hold a
    // picture's rows.
    model::TokenNames token_names() const override {
        return {{"<|content_model_end_sampling|>"}, "<|message_user|>", "<|unused_200054|>", "", ""};
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::inkling2
