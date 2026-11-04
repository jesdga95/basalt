// DeepSeek-V4.1-Flash adapter: registers the architecture, validates and describes the file from its
// `deepseek41.*` metadata, and loads the pack into a DsModel.
#include "models/deepseek41/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "models/deepseek41/model.hpp"

namespace basalt::models::deepseek41 {
namespace {

uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = g.get(std::string("deepseek41.") + suffix);
    return v ? v->u : 0;
}

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "deepseek41"; }

    std::string validate(const GgufFile& g) const override {
        const MetaValue* arch = g.get("general.architecture");
        if (!arch) return "missing general.architecture";
        if (arch->s != "deepseek41") return "architecture is '" + arch->s + "', this adapter requires 'deepseek41'";
        for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count",
                              "expert_feed_forward_length"})
            if (key_u(g, k) == 0) return std::string("missing or zero deepseek41.") + k;
        return {};
    }

    model::ModelDesc describe(const GgufFile& g) const override {
        model::ModelDesc d;
        d.arch = "deepseek41";
        d.n_layers = (int64_t) key_u(g, "block_count");
        d.n_embd = (int64_t) key_u(g, "embedding_length");
        d.n_expert = (int64_t) key_u(g, "expert_count");
        d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
        d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
        d.n_expert_shared = (int64_t) key_u(g, "expert_shared_count");
        d.context_length = (int64_t) key_u(g, "context_length");
        d.rope_from_model = true;
        if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
        return d;
    }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "deepseek41: no model path"};
        auto m = std::make_unique<DsModel>();
        std::string err;
        if (!m->load(req.paths.front(), err)) return {model::Status::Error, "deepseek41: " + err};
        if (!m->resolve_context(req.max_context, req.max_context_set, err)) return {model::Status::Error, err};
        m->set_diag_dump(req.diag_dump);
        m->set_diag_tier(req.diag_tier_ram, req.diag_tier_vram);
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
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::deepseek41
