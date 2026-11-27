// src/models/gemma4/pack.cpp - reads the Gemma 4 pack's meta section.
#include "models/gemma4/pack.hpp"

#include <sstream>

namespace basalt::models::gemma4 {
namespace {

using core::Json;

std::vector<bool> bools(const Json* j) {
    std::vector<bool> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.is_number() ? x.num() != 0 : x.b);
    return v;
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "gemma4") {
        err = path + ": not a gemma4 pack";
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    std::string missing;
    auto k = [&](const std::string& s) {
        const Json* j = kv->get("gemma4." + s);
        if (!j && missing.empty()) missing = "gemma4." + s;
        return j;
    };
    auto num = [&](const std::string& s) {
        const Json* j = k(s);
        return j && j->is_number() ? j->num() : 0.0;
    };
    auto opt = [&](const std::string& s) { return kv->get("gemma4." + s); };
    Geometry& g = p.geo;
    g.n_layers = (int) num("block_count");
    g.n_embd = (int) num("embedding_length");
    g.n_ff = (int) num("feed_forward_length");
    g.n_head = (int) num("attention.head_count");
    g.n_kv_swa = (int) num("attention.head_count_kv");
    g.n_kv_full = (int) num("attention.head_count_kv_full");
    g.hd_swa = (int) num("attention.key_length");
    g.hd_full = (int) num("attention.key_length_full");
    g.window = (int) num("attention.sliding_window");
    if (const Json* j = k("attention.k_eq_v")) g.k_eq_v = j->b;
    g.swa = bools(k("attention.sliding_window_pattern"));
    g.rms_eps = (float) num("attention.layer_norm_rms_epsilon");
    g.rope_base = (float) num("rope.freq_base");
    g.rope_base_swa = (float) num("rope.freq_base_swa");
    g.partial_rotary = (float) num("rope.partial_rotary_factor");
    g.softcap = (float) num("final_logit_softcapping");
    g.embed_scale = (float) num("embedding_scale");
    g.image_token = (int) num("image_token_id");
    g.context_length = (int64_t) num("context_length");
    if (opt("expert_count")) {
        g.n_expert = (int) num("expert_count");
        g.n_used = (int) num("expert_used_count");
        g.n_ff_exp = (int) num("expert_feed_forward_length");
    }
    if (opt("assistant.block_count")) {
        DrafterGeometry& d = g.drafter;
        d.present = true;
        d.n_layers = (int) num("assistant.block_count");
        d.n_embd = (int) num("assistant.embedding_length");
        d.n_ff = (int) num("assistant.feed_forward_length");
        d.n_head = (int) num("assistant.attention.head_count");
        d.draft_tokens = (int) num("assistant.draft_tokens");
        d.swa = bools(k("assistant.attention.sliding_window_pattern"));
    }
    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (!missing.empty()) {
        err = path + ": meta.gguf_kv lacks " + missing;
        return false;
    }
    if (g.n_layers <= 0 || (int) g.swa.size() != g.n_layers || g.n_vocab <= 0 || g.n_embd % 128 || g.window <= 0 ||
        g.softcap <= 0.f || g.embed_scale <= 0.f) {
        err = path + ": meta.gguf_kv's gemma4 geometry is incomplete";
        return false;
    }
    for (int l = 0; l < g.n_layers; ++l) {
        const LayerShape s = g.shape(l);
        if ((s.hd != 256 && s.hd != 512) || s.n_kv < 1 || g.n_head % s.n_kv || g.n_head > kMaxHeads ||
            g.n_head / s.n_kv > kMaxGroup || (g.n_head / s.n_kv & (g.n_head / s.n_kv - 1))) {
            err = path + ": a Gemma 4 attention layout this engine does not run (heads of 256 or 512, at most " +
                  std::to_string(kMaxHeads) + " query heads, a power of two up to " + std::to_string(kMaxGroup) +
                  " per key head)";
            return false;
        }
    }
    if (g.moe() && (g.n_expert > 256 || g.n_used > 16 || g.n_ff_exp % 64)) {
        err = path + ": a Gemma 4 expert layout this engine does not run (up to 256 experts, 16 a token, widths of 64s)";
        return false;
    }
    if (g.drafter.present && ((int) g.drafter.swa.size() != g.drafter.n_layers || g.drafter.n_embd % 128)) {
        err = path + ": the assistant drafter's geometry is incomplete";
        return false;
    }

    p.vision = p.file.range("vision/mmproj");
    if (const Json* hf = meta.get("hf_config"); hf && hf->get("vision_config")) {
        const Json* n = hf->get("vision_config")->get("default_output_length");
        g.image_cells = n && n->is_number() ? (int) n->num() : 0;
    }
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = t.get("type_id") ? (int) t.get("type_id")->num() : 0;
        if (const Json* sh = t.get("shape"))
            for (const Json& dd : sh->items) r.shape.push_back(dd.u64());
        r.bytes = t.get("bytes") ? (uint64_t) t.get("bytes")->num() : 0;
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        r.offset = sec->offset + t.get("offset")->u64();
        p.tensors.emplace(name, std::move(r));
    }

    if (!g.moe()) return true;
    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    std::vector<bool> seen((size_t) g.n_layers, false);
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        long long l, gu, dt;
        unsigned long long off, by;
        if (!(ls >> l >> gu >> dt >> off >> by) || l < 0 || l >= g.n_layers || seen[(size_t) l]) {
            err = "experts.layout: bad line '" + line + "'";
            return false;
        }
        p.experts[(size_t) l] = {(int) gu, (int) dt, off, by};
        seen[(size_t) l] = true;
    }
    for (int l = 0; l < g.n_layers; ++l)
        if (!seen[(size_t) l]) {
            err = "experts.layout lists no experts for layer " + std::to_string(l);
            return false;
        }
    return true;
}

}  // namespace basalt::models::gemma4
