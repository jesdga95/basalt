// src/models/deepseek41/pack.cpp - reads the DeepSeek-V4.1 pack's meta section.
#include "models/deepseek41/pack.hpp"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace basalt::models::deepseek41 {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

const Json* member(const Json& o, const char* k) { return o.get(k); }

std::vector<double> numbers(const Json* j) {
    std::vector<double> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.num());
    return v;
}

std::vector<uint64_t> unsigneds(const Json* j) {
    std::vector<uint64_t> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.u64());
    return v;
}

// The meta parser keeps only the head of long arrays; the Engram token map (one entry per vocabulary id) is read
// from the raw text.
bool read_int_array(const std::string& text, const char* key, std::vector<int32_t>& out) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t p = text.find(needle);
    if (p == std::string::npos) return false;
    p = text.find('[', p);
    if (p == std::string::npos) return false;
    ++p;
    out.clear();
    const char* s = text.c_str();
    while (p < text.size()) {
        while (p < text.size() && (s[p] == ' ' || s[p] == ',' || s[p] == '\n')) ++p;
        if (p >= text.size() || s[p] == ']') break;
        char* end = nullptr;
        const long long v = std::strtoll(s + p, &end, 10);
        if (end == s + p) return false;
        out.push_back((int32_t) v);
        p = (size_t) (end - s);
    }
    return !out.empty();
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "deepseek41") {
        err = path + ": not a deepseek41 pack";
        return false;
    }
    const Json* gj = meta.get("geometry");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no geometry / tensors";
        return false;
    }
    Geometry& g = p.geo;
    auto u = [&](const char* k, double d = 0) { return (int) num_or(member(*gj, k), d); };
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.n_head = u("attention.head_count");
    g.head_dim = u("attention.key_length");
    g.rope_dim = u("rope.dimension_count");
    g.q_lora = u("attention.q_lora_rank");
    g.window = u("attention.sliding_window");
    g.o_groups = u("attention.output_group_count");
    g.o_lora = u("attention.output_lora_rank");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.n_shared = u("expert_shared_count");
    g.weight_scale = (float) num_or(member(*gj, "expert_weights_scale"), 1.0);
    g.rms_eps = (float) num_or(member(*gj, "attention.layer_norm_rms_epsilon"), 1e-20);
    g.hc_eps = (float) num_or(member(*gj, "hyper_connection.epsilon"), 1e-6);
    g.sinkhorn_iters = u("hyper_connection.sinkhorn_iterations", 20);
    g.idx_heads = u("attention.indexer.head_count");
    g.idx_dim = u("attention.indexer.key_length");
    g.idx_top_k = u("attention.indexer.top_k");
    g.rope_base = (float) num_or(member(*gj, "rope.freq_base"), 10000.0);
    g.comp_rope_base = (float) num_or(member(*gj, "attention.compress_rope_freq_base"), 160000.0);
    g.context_length = (int64_t) num_or(member(*gj, "context_length"), 0);
    g.yarn_factor = (float) num_or(member(*gj, "rope.scaling.factor"), 16.0);
    g.yarn_orig_ctx = (int) num_or(member(*gj, "rope.scaling.original_context_length"), 65536.0);
    if (g.yarn_factor < 1.f || g.yarn_orig_ctx < 1) {
        err = path + ": meta.geometry has an invalid rope scaling";
        return false;
    }
    for (double d : numbers(member(*gj, "swiglu_clamp_exp"))) g.clamp_exp.push_back((float) d);
    for (double d : numbers(member(*gj, "swiglu_clamp_shexp"))) g.clamp_shexp.push_back((float) d);
    for (double d : numbers(member(*gj, "attention.compress_ratios"))) g.compress_ratio.push_back((int) d);
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_expert <= 0) {
        err = path + ": meta.geometry is incomplete";
        return false;
    }
    g.compress_ratio.resize((size_t) g.n_layers, 0);
    g.clamp_exp.resize((size_t) g.n_layers, 10.0f);
    g.clamp_shexp.resize((size_t) g.n_layers, 10.0f);
    if (const Json* gk = meta.get("gguf_kv"))
        if (const Json* t = gk->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;

    // tensors; offsets are relative to the section a tensor names (the GGUF section by default)
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& d : sh->items) r.shape.push_back(d.u64());
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

    p.vision = p.file.range("vision/mmproj");
    p.dspark = p.file.range("dspark/gguf");

    // the experts section and its per-layer layout
    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    {
        std::istringstream in(std::string(text.begin(), text.end()));
        std::string line;
        p.experts.resize((size_t) g.n_layers);
        int seen = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            long long l, gu, d;
            unsigned long long off, by;
            if (!(ls >> l >> gu >> d >> off >> by) || l < 0 || l >= g.n_layers) {
                err = "experts.layout: bad line '" + line + "'";
                return false;
            }
            p.experts[(size_t) l] = {(int) gu, (int) d, off, by};
            ++seen;
        }
        if (seen != g.n_layers) {
            err = "experts.layout lists " + std::to_string(seen) + " layers, the model has " + std::to_string(g.n_layers);
            return false;
        }
    }

    // Engram
    if (const Json* ej = meta.get("engram"); ej && ej->is_object()) {
        EngramInfo& e = p.engram;
        const core::ModelFile::Section* sec = p.file.find(ej->get("section") ? ej->get("section")->s : "engram");
        if (!sec) {
            err = path + ": meta.engram names a missing section";
            return false;
        }
        e.present = true;
        e.file_offset = sec->offset;
        e.rows = ej->get("rows")->u64();
        e.row_bytes = (uint32_t) ej->get("row_bytes")->u64();
        e.row_values = (int) ej->get("row_values")->u64();
        for (double d : numbers(ej->get("ple.layers"))) e.layers.push_back((int) d);
        EngramHashParams& h = e.hash;
        h.n_gram = (int) num_or(ej->get("ple.ngram_size"), 4);
        h.n_head = (int) num_or(ej->get("ple.heads_per_ngram"), 8);
        h.pad_token_id = (int) num_or(ej->get("engram.pad_token_id"), 2);
        h.mult = unsigneds(ej->get("ple.layer_multipliers"));
        h.primes = unsigneds(ej->get("ple.head_vocab_sizes"));
        h.offsets = unsigneds(ej->get("ple.head_offsets"));
        const size_t ne = e.layers.size();
        if (h.mult.size() != ne * (size_t) h.n_gram || h.primes.size() != ne * (size_t) h.n_cols() ||
            h.offsets.size() != ne * (size_t) h.n_cols()) {
            err = path + ": the Engram hash arrays do not match its layer count";
            return false;
        }
        std::vector<uint8_t> raw;
        if (!p.file.read("meta", raw, err)) return false;
        if (!read_int_array(std::string(raw.begin(), raw.end()), "deepseek41.engram.token_map", h.token_map)) {
            err = path + ": the Engram token map is missing";
            return false;
        }
    }
    return true;
}

}  // namespace basalt::models::deepseek41
