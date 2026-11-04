// Flash-Next adapter. Decode, prefill and the drafter stay on the engine's own paths (core/verify, prefill,
// mtp); this adapter owns the architecture guard, the geometry and the side-table/tokenizer descriptions.
#include "models/qwen4exp/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "unicode_flags.hpp"
#include "basalt/program/options.hpp"

namespace basalt::models::qwen4exp {
namespace {

constexpr const char* kPrefix = "qwen4exp.";

// Keys the model must carry; a zero expectation checks presence only (pruned variants ship fewer experts).
struct Req {
    const char* key;
    uint64_t want;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "qwen4exp"; }

    std::string validate(const GgufFile& g) const override {
        const MetaValue* arch = g.get("general.architecture");
        if (!arch) return "missing general.architecture";
        if (arch->s != "qwen4exp") return "architecture is '" + arch->s + "', this adapter requires 'qwen4exp'";
        const core::ModelGeometry def;
        const Req reqs[] = {
            {"qwen4exp.block_count", (uint64_t) def.n_layers},
            {"qwen4exp.embedding_length", (uint64_t) def.n_embd},
            {"qwen4exp.expert_count", 0},
            {"qwen4exp.expert_used_count", 0},
            {"qwen4exp.attention.head_count", (uint64_t) def.n_head},
            {"qwen4exp.attention.head_count_kv", (uint64_t) def.n_head_kv},
        };
        for (const auto& r : reqs) {
            const MetaValue* v = g.get(r.key);
            if (!v) return std::string("missing ") + r.key;
            if (r.want && v->u != r.want)
                return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
        }
        return {};
    }

    model::ModelDesc describe(const GgufFile& g) const override {
        const core::ModelGeometry def;
        model::ModelDesc d;
        d.arch = "qwen4exp";
        d.n_layers = def.n_layers;
        d.n_embd = def.n_embd;
        d.n_expert = def.n_expert;
        d.n_expert_used = 10;
        d.n_ff_exp = def.n_ff;
        d.n_vocab = 248320;
        if (const MetaValue* v = g.get("qwen4exp.expert_count")) d.n_expert = (int64_t) v->u;
        if (const MetaValue* v = g.get("qwen4exp.expert_used_count")) d.n_expert_used = (int64_t) v->u;
        return d;
    }

    model::Result load(const model::LoadRequest&, std::unique_ptr<model::LoadedModel>&) const override {
        return {model::Status::NotImplemented, "Flash-Next runs its own start"};
    }

    std::optional<int> run(program::CommonOptions& o, std::vector<program::GpuCard>& cards) const override {
        return run_engine(static_cast<program::Options&>(o), cards);
    }

    int max_expert_cards() const override { return kernels::kDmMaxX; }

    std::vector<model::SideTableSpec> side_tables(const model::ModelDesc&) const override {
        return {{model::SideTableSpec::Kind::Ple, "per_layer_token_embd.weight", 0, 0}};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "qwen"}; }

    PreTokenizer pretokenizer() const override { return {"qwen35", program::tokenizer::pretok_qwen35}; }

    model::TokenNames token_names() const override {
        return {{"<|endoftext|>", "<|im_end|>"}, "<|im_start|>", "<|image_pad|>"};
    }

    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<tool_call>\n";
        f.call_close = "\n</tool_call>";
        f.fn_open = "<function=";
        f.fn_mid = ">\n";
        f.fn_close = "</function>";
        f.param_open = "<parameter=";
        f.param_mid_text = f.param_mid_value = ">\n";
        f.param_close = "\n</parameter>\n";
        f.text_end = "</parameter>";
        f.trigger = "<tool_call>";
        return f;
    }

    std::unique_ptr<program::CommonOptions> make_options() const override { return std::make_unique<program::Options>(); }
    bool parse_flag(const std::string& name, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        return qwen4exp::parse_flag(name, args, static_cast<program::Options&>(o), code);
    }
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        return qwen4exp::finish_flags(static_cast<program::Options&>(o));
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::qwen4exp
