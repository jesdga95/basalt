// src/models/hy4flash/pack.cpp - reads the Hy4-flash pack's meta section.
#include "models/hy4flash/pack.hpp"

#include <cstdlib>
#include <sstream>

namespace basalt::models::hy4flash {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

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

std::vector<bool> bools(const Json* j) {
    std::vector<bool> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.is_number() ? x.num() != 0 : x.b);
    return v;
}

// The meta parser keeps only the head of long arrays; the Engram token map (one entry per vocabulary id) is read from
// the raw text.
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
    if (!arch || !arch->is_string() || arch->s != "hy4flash") {
        err = path + ": not a hy4flash pack";
        return false;
    }
    const Json* gj = meta.get("geometry");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no geometry / tensors";
        return false;
    }
    Geometry& g = p.geo;
    auto u = [&](const char* k, double d = 0) { return (int) num_or(gj->get(k), d); };
    const int blocks = u("block_count"), nextn = u("nextn_predict_layers");
    g.n_layers = blocks - nextn;
    g.n_embd = u("embedding_length");
    g.n_ff = u("feed_forward_length");
    g.n_dense = u("leading_dense_block_count");
    g.q_lora = u("attention.q_lora_rank");
    g.kv_lora = u("attention.kv_lora_rank");
    g.qk_dim = u("attention.key_length_mla");
    g.v_dim = u("attention.value_length_mla");
    g.rope_dim = u("rope.dimension_count");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.n_shared = u("expert_shared_count");
    g.n_latent = u("expert_latent_length");
    g.expert_scale = (float) num_or(gj->get("expert_weights_scale"), 1.0);
    g.rms_eps = (float) num_or(gj->get("attention.layer_norm_rms_epsilon"), 1e-5);
    g.hc_eps = (float) num_or(gj->get("hyper_connection.epsilon"), 1e-6);
    g.hc_magnitude = (float) num_or(gj->get("hyper_connection.magnitude"), 2.0);
    g.hc_mult = u("hyper_connection.count", 4);
    g.window = u("attention.sliding_window", 512);
    g.rope_base = (float) num_or(gj->get("rope.freq_base"), 1e7);
    g.rope_base_swa = (float) num_or(gj->get("rope.freq_base_swa"), 1e4);
    g.idx_heads = u("attention.indexer.head_count");
    g.idx_dim = u("attention.indexer.key_length");
    g.idx_top_k = u("attention.indexer.top_k");
    g.idx_pool = u("attention.indexer.kpool");
    g.idx_select_tail = num_or(gj->get("attention.indexer.kpool_select_tail"), 1) != 0 ||
                        (gj->get("attention.indexer.kpool_select_tail") && gj->get("attention.indexer.kpool_select_tail")->b);
    g.context_length = (int64_t) num_or(gj->get("context_length"), 0);
    for (double d : numbers(gj->get("attention.head_count"))) g.heads.push_back((int) d);
    g.swa = bools(gj->get("attention.sliding_window_pattern"));
    g.indexed = bools(gj->get("attention.indexer.types"));
    for (double d : numbers(gj->get("swiglu_clamp_exp"))) g.clamp_exp.push_back((float) d);
    if (const Json* kv = meta.get("gguf_kv"))
        if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_vocab == 0) g.n_vocab = u("vocab_size");
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_embd % 32 != 0 || g.n_vocab <= 0 || g.n_expert <= 0 ||
        g.n_dense < 0 || g.n_dense >= g.n_layers || (g.n_dense > 0 && g.n_ff <= 0) || g.q_lora <= 0 ||
        g.n_used <= 0 || g.n_used > g.n_expert || g.window <= 0 ||
        (int) g.heads.size() < g.n_layers || (int) g.swa.size() < g.n_layers || (int) g.indexed.size() < g.n_layers) {
        err = path + ": meta.geometry is incomplete";
        return false;
    }
    g.heads.resize((size_t) g.n_layers);
    g.swa.resize((size_t) g.n_layers);
    g.indexed.resize((size_t) g.n_layers);
    g.clamp_exp.resize((size_t) g.n_layers, 10.0f);
    // the layouts this engine runs: four identity streams, one latent for both layer kinds, an indexer on every layer
    // that is not a window layer, experts that run in a latent
    if (g.hc_mult != 4 || g.qk_dim != 256 || g.kv_lora != 512 || g.rope_dim != 64 || g.v_dim != 256 ||
        g.qk_dim - g.rope_dim != 192 || g.n_latent <= 0 || g.n_ff_exp != g.n_latent || g.n_shared != 1 || g.idx_pool != 4 ||
        g.idx_dim != 128 || g.n_used > 16 || g.n_expert > 512) {
        err = path + ": a Hy4-flash layout this engine does not run (4 streams, MLA 192 + 64 over a 512 latent, a 1536 latent "
                     "for the experts, indexer keys of 128 pooled by 4)";
        return false;
    }
    bool any_indexed = false;
    for (int l = 0; l < g.n_layers; ++l) {
        if (g.heads[(size_t) l] <= 0 || g.heads[(size_t) l] % 8 != 0) {
            err = path + ": layer " + std::to_string(l) + " has an invalid attention head count";
            return false;
        }
        if (g.swa[(size_t) l] == g.indexed[(size_t) l]) {
            err = path + ": layer " + std::to_string(l) + " is neither a window layer nor an indexed one, or both";
            return false;
        }
        any_indexed = any_indexed || g.indexed[(size_t) l];
    }
    if (any_indexed &&
        (g.idx_heads <= 0 || g.idx_top_k <= 0 || g.idx_top_k % g.idx_pool != 0 ||
         g.idx_top_k / g.idx_pool - (g.idx_select_tail ? 1 : 0) < 1)) {
        err = path + ": meta.geometry has invalid indexer parameters";
        return false;
    }

    p.vision = p.file.range("vision/mmproj");
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
        const Json* off = t.get("offset");
        if (!off || !off->is_number() || off->neg) {
            err = path + ": tensor " + name + " has no offset";
            return false;
        }
        r.offset = sec->offset + off->u64();
        p.tensors.emplace(name, std::move(r));
    }

    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    {
        std::istringstream in(std::string(text.begin(), text.end()));
        std::string line;
        p.experts.assign((size_t) g.n_layers, ExpertLayer{});
        int seen = 0, version = 0;
        while (std::getline(in, line)) {
            if (line.rfind("# experts v", 0) == 0) version = std::atoi(line.c_str() + 11);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            long long l, gu, d, up = -1;
            unsigned long long off, by;
            if (!(ls >> l >> gu >> d >> off >> by) || l < g.n_dense || l >= g.n_layers || by == 0 ||
                p.experts[(size_t) l].present || (version >= 7 && !(ls >> up))) {
                err = "experts.layout: bad line '" + line + "'";
                return false;
            }
            if (version >= 7 && up != gu) {   // v7: the up rows' type; this model's kernels take one gate/up type
                err = "experts.layout: layer " + std::to_string(l) + "'s gate and up differ in type";
                return false;
            }
            p.experts[(size_t) l] = {true, (int) gu, (int) d, off, by};
            ++seen;
        }
        if (seen != g.n_layers - g.n_dense) {
            err = "experts.layout lists " + std::to_string(seen) + " layers, the model has " +
                  std::to_string(g.n_layers - g.n_dense) + " with experts";
            return false;
        }
    }

    if (const Json* ej = meta.get("engram"); ej && ej->is_object()) {
        EngramInfo& e = p.engram;
        const Json* sj = ej->get("section");
        const core::ModelFile::Section* sec = p.file.find(sj && sj->is_string() ? sj->s : "engram");
        const Json *rj = ej->get("rows"), *rbj = ej->get("row_bytes"), *rvj = ej->get("row_values");
        if (!sec || num_or(ej->get("type_id"), 0) != 43 || !rj || !rj->is_number() || rj->neg ||
            !rbj || !rbj->is_number() || rbj->neg || !rvj || !rvj->is_number() || rvj->neg) {
            err = path + ": meta.engram has no section, missing table geometry, or is not F8_E4M3 rows";
            return false;
        }
        e.present = true;
        e.file_offset = sec->offset;
        e.rows = rj->u64();
        const uint64_t rb = rbj->u64(), rv = rvj->u64();
        e.row_bytes = (uint32_t) rb;
        e.row_values = (int) rv;
        for (double d : numbers(ej->get("ple.layers"))) e.layers.push_back((int) d);
        e.n_gram = (int) num_or(ej->get("ple.ngram_size"), 3);
        e.n_head = (int) num_or(ej->get("ple.heads_per_ngram"), 8);
        e.pad_token_id = (int) num_or(ej->get("engram.pad_token_id"), 0);
        e.mult = unsigneds(ej->get("ple.layer_multipliers"));
        e.primes = unsigneds(ej->get("ple.head_vocab_sizes"));
        e.offsets = unsigneds(ej->get("ple.head_offsets"));
        const size_t ne = e.layers.size();
        if (e.rows == 0 || rv == 0 || rv > (uint64_t) g.n_embd || rb != rv ||
            e.n_gram < 2 || e.n_gram > 3 || e.n_head <= 0 || e.n_head > g.n_embd ||
            (int64_t) e.n_cols() * e.row_values > g.n_embd ||
            e.mult.size() != ne * (size_t) e.n_gram ||
            e.primes.size() != ne * (size_t) e.n_cols() || e.offsets.size() != e.primes.size()) {
            err = path + ": the Engram hash arrays or table geometry are invalid";
            return false;
        }
        for (int l : e.layers) {
            if (l < 0 || l >= g.n_layers) {
                err = path + ": meta.engram names an out-of-range layer " + std::to_string(l);
                return false;
            }
        }
        for (size_t i = 0; i < e.primes.size(); ++i) {
            if (e.primes[i] == 0 || e.offsets[i] >= e.rows || e.primes[i] > e.rows - e.offsets[i]) {
                err = path + ": meta.engram has a zero prime or out-of-range head offset";
                return false;
            }
        }
        std::vector<uint8_t> raw;
        if (!p.file.read("meta", raw, err)) return false;
        if (!read_int_array(std::string(raw.begin(), raw.end()), "hy_v4_flash.engram.token_map", e.token_map) ||
            e.pad_token_id < 0 || (size_t) e.pad_token_id >= e.token_map.size()) {
            err = path + ": the Engram token map is missing";
            return false;
        }
    }
    return true;
}

}  // namespace basalt::models::hy4flash
