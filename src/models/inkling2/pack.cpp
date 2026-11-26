// src/models/inkling2/pack.cpp - reads the Inkling 2 pack's meta section.
#include "models/inkling2/pack.hpp"

#include <sstream>

namespace basalt::models::inkling2 {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

std::vector<bool> bools(const Json* j) {
    std::vector<bool> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.is_number() ? x.num() != 0 : x.b);
    return v;
}

std::vector<int> ints(const Json* j) {
    std::vector<int> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back((int) x.num());
    return v;
}

}  // namespace

std::vector<Visit> loop_visits(int layers, int start, int end, int passes) {
    std::vector<Visit> v;
    for (int l = 0; l < start; ++l) v.push_back({l, 0, l});
    for (int p = 0; p < passes; ++p)
        for (int l = start; l < end; ++l) v.push_back({l, p, p == 0 ? l : layers + (p - 1) * (end - start) + (l - start)});
    for (int l = end; l < layers; ++l) v.push_back({l, 0, l});
    return v;
}

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "inkling2") {
        err = path + ": not an inkling2 pack";
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto k = [&](const char* s) { return kv->get(std::string("inkling2.") + s); };
    auto u = [&](const char* s, double d = 0) { return (int) num_or(k(s), d); };
    auto f = [&](const char* s, double d = 0) { return (float) num_or(k(s), d); };
    Geometry& g = p.geo;
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.n_head = u("attention.head_count");
    g.n_head_kv = u("attention.head_count_kv");
    g.head_dim = u("attention.key_length");
    g.rms_eps = f("attention.layer_norm_rms_epsilon", 1e-6);
    g.window = u("attention.sliding_window");
    g.swa = bools(k("attention.sliding_window_pattern"));
    g.rel = u("attention.relative_length");
    g.extent = u("attention.relative_extent");
    g.qk_clip = f("attention.qk_clip");
    g.rel_clip = f("attention.relative_clip");
    g.log_floor = f("attention.log_scale_floor");
    g.log_alpha = f("attention.log_scale_alpha");
    g.conv = u("shortconv.l_cache");
    g.block = u("attn_res.block_size");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.expert_scale = f("expert_weights_scale", 1.0);
    g.gate_softcap = f("expert_gate_softcap");
    g.up_softcap = f("expert_up_softcap");
    g.logit_softcap = f("final_logit_softcapping");
    g.loop_start = u("loop.start");
    g.loop_end = u("loop.end");
    g.loop_passes = u("loop.count", 1);
    g.gate_rank = u("loop.gate_rank");
    g.image_token = u("image_token_id", -1);
    g.context_length = (int64_t) num_or(k("context_length"), 0);
    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_layers <= 0 || (int) g.swa.size() != g.n_layers || g.n_vocab <= 0 || g.block <= 0) {
        err = path + ": meta.gguf_kv's inkling2 geometry is incomplete";
        return false;
    }
    // the layout the kernels are built for (basalt/models/inkling2/kernels.hpp)
    if (g.n_embd != 2304 || g.n_head != 24 || g.n_head_kv != 8 || g.head_dim != 128 || g.rel != 16 || g.extent != 1024 ||
        g.window != g.extent || g.conv != 4 || g.n_expert > 512 || g.n_expert % 32 || g.n_used > 16 ||
        g.loop_start <= 0 || g.loop_start >= g.loop_end || g.loop_end > g.n_layers || g.loop_start % g.block ||
        g.loop_end % g.block || g.loop_passes < 1 || g.gate_rank <= 0) {
        err = path + ": an Inkling 2 layout this engine does not run (width 2304, 24 query and 8 key/value heads of 128, "
                     "relative states of 16 over 1,024 positions, the window 1,024, convolutions of 4, a loop on block "
                     "boundaries)";
        return false;
    }
    g.visits = loop_visits(g.n_layers, g.loop_start, g.loop_end, g.loop_passes);

    DsparkGeometry& d = p.dspark;
    if (const Json* dl = k("dspark.block_count"); dl && dl->is_number()) {
        d.present = true;
        d.n_layers = u("dspark.block_count");
        d.block = u("dspark.block_size");
        d.targets = ints(k("dspark.target_layers"));
        d.swa = bools(k("dspark.sliding_window_pattern"));
        d.window = u("dspark.sliding_window");
        d.n_head = u("dspark.attention.head_count");
        d.n_head_kv = u("dspark.attention.head_count_kv");
        d.head_dim = u("dspark.attention.key_length");
        d.n_ff = u("dspark.feed_forward_length");
        d.rope_base = f("dspark.rope.freq_base", 1e6);
        d.rms_eps = f("dspark.attention.layer_norm_rms_epsilon", 1e-6);
        d.markov_rank = u("dspark.markov_rank");
    }

    p.vision = p.file.range("vision/mmproj");
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& dd : sh->items) r.shape.push_back(dd.u64());
        r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
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

}  // namespace basalt::models::inkling2
