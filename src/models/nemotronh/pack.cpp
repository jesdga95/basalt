// src/models/nemotronh/pack.cpp - reads the Nemotron-H pack's meta section.
#include "models/nemotronh/pack.hpp"

#include <sstream>

namespace basalt::models::nemotronh {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "nemotronh") {
        err = path + ": not a nemotronh pack";
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto k = [&](const char* s) { return kv->get(std::string("nemotronh.") + s); };
    auto u = [&](const char* s, double d = 0) { return (int) num_or(k(s), d); };
    auto f = [&](const char* s, double d = 0) { return (float) num_or(k(s), d); };
    Geometry& g = p.geo;
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.n_vocab = u("vocab_size");
    g.n_head = u("attention.head_count");
    g.n_kv = u("attention.head_count_kv");
    g.head_dim = u("attention.key_length");
    g.rms_eps = f("attention.layer_norm_rms_epsilon", 1e-5);
    g.ssm_conv = u("ssm.conv_kernel");
    g.ssm_state = u("ssm.state_size");
    g.ssm_heads = u("ssm.head_count");
    g.ssm_head_dim = u("ssm.head_dim");
    g.ssm_groups = u("ssm.group_count");
    g.ssm_inner = u("ssm.inner_size");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.n_ff_shared = u("expert_shared_feed_forward_length");
    g.n_latent = u("expert_latent_size");
    g.expert_scale = f("expert_weights_scale", 1.0);
    g.context_length = (int64_t) num_or(k("context_length"), 0);
    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    const Json* pat = k("layer_pattern");
    if (!pat || !pat->is_string() || (int) pat->s.size() != g.n_layers) {
        err = path + ": nemotronh.layer_pattern is missing or does not have a letter a layer";
        return false;
    }
    for (char c : pat->s) {
        if (c != 'M' && c != '*' && c != 'E') {
            err = path + ": a layer kind '" + std::string(1, c) + "' (M, * and E are run)";
            return false;
        }
        g.kinds.push_back((Kind) c);
    }
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_vocab <= 0 || g.n_head <= 0 || g.n_kv <= 0 || g.ssm_heads <= 0 ||
        g.n_expert <= 0 || g.n_used <= 0 || g.n_ff_exp <= 0 || g.n_ff_shared <= 0) {
        err = path + ": meta.gguf_kv is incomplete";
        return false;
    }
    if (g.n_latent == 0) g.n_latent = g.n_embd;   // experts over the layer's input itself
    if (g.ssm_heads * g.ssm_head_dim != g.ssm_inner) {
        err = path + ": ssm.head_count x ssm.head_dim is not ssm.inner_size";
        return false;
    }

    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& d : sh->items) r.shape.push_back(d.u64());
        r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
        const core::ModelFile::Section* sec = p.file.find("gguf");
        if (!sec || !t.get("offset")) {
            err = path + ": tensor " + name + " has no place in the gguf section";
            return false;
        }
        r.offset = sec->offset + t.get("offset")->u64();
        p.tensors.emplace(name, std::move(r));
    }

    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    int seen = 0;
    bool relu2 = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            relu2 = relu2 || line.find("relu2") != std::string::npos;
            continue;
        }
        std::istringstream ls(line);
        long long l, gu, d;
        unsigned long long off, by;
        if (!(ls >> l >> gu >> d >> off >> by) || l < 0 || l >= g.n_layers || g.kinds[(size_t) l] != Kind::Moe) {
            err = "experts.layout: bad line '" + line + "'";
            return false;
        }
        p.experts[(size_t) l] = {true, (int) gu, (int) d, off, by};
        ++seen;
    }
    if (!relu2) {
        err = path + ": experts.layout does not say its experts are squared-ReLU blobs";
        return false;
    }
    if (seen != g.count(Kind::Moe)) {
        err = "experts.layout lists " + std::to_string(seen) + " layers, the model has " + std::to_string(g.count(Kind::Moe)) +
              " with experts";
        return false;
    }
    return true;
}

}  // namespace basalt::models::nemotronh
