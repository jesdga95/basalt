// src/models/deepseek41/pack.cpp - reads the DeepSeek-V4.1 pack's meta section.
#include "models/deepseek41/pack.hpp"

#include "models/deepseek41/engram_rows.hpp"

#include <climits>
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
    p.geo = {};
    p.tensors.clear();
    p.experts.clear();
    p.experts_range = {};
    p.engram = {};
    p.vision = {};
    p.dspark = {};
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
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_embd % 32 != 0 || g.n_expert <= 0 || g.n_used <= 0 || g.n_used > 8 ||
        g.n_used > g.n_expert || g.n_ff_exp <= 0 || g.n_ff_exp % 32 != 0 || g.n_shared < 0) {
        err = path + ": meta.geometry is incomplete or invalid";
        return false;
    }
    for (int r : g.compress_ratio) {
        if (r < 0) {
            err = path + ": meta.geometry has a negative compress ratio";
            return false;
        }
    }
    g.compress_ratio.resize((size_t) g.n_layers, 0);
    g.clamp_exp.resize((size_t) g.n_layers, 10.0f);
    g.clamp_shexp.resize((size_t) g.n_layers, 10.0f);
    if (const Json* gk = meta.get("gguf_kv"))
        if (const Json* t = gk->get("tokenizer.ggml.tokens"); t && t->is_array() && t->count <= (uint64_t) INT_MAX)
            g.n_vocab = (int) t->count;
    if (g.n_vocab <= 0) {
        err = path + ": meta has no vocabulary size";
        return false;
    }

    // tensors; offsets are relative to the section a tensor names (the GGUF section by default)
    for (const auto& [name, t] : tj->members) {
        if (!t.is_object()) {
            err = path + ": tensor " + name + " is not an object";
            return false;
        }
        const Json* off_j = t.get("offset");
        if (!off_j || off_j->type != Json::Type::Int || off_j->neg) {
            err = path + ": tensor " + name + " has no valid offset";
            return false;
        }
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"); sh && sh->is_array())
            for (const Json& d : sh->items) r.shape.push_back(d.u64());
        const Json* bytes_j = t.get("bytes");
        if (bytes_j && (bytes_j->type != Json::Type::Int || bytes_j->neg)) {
            err = path + ": tensor " + name + " has an invalid byte size";
            return false;
        }
        r.bytes = bytes_j ? bytes_j->u64() : 0;
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        const uint64_t rel_off = off_j->u64();
        if (rel_off > sec->bytes || r.bytes > sec->bytes - rel_off) {
            err = path + ": tensor " + name + " runs past the end of section " + section;
            return false;
        }
        r.offset = sec->offset + rel_off;
        p.tensors.emplace(name, std::move(r));
    }

    p.vision = p.file.range("vision/mmproj");
    p.dspark = p.file.range("dspark/gguf");

    // the experts section and its per-layer layout
    p.experts_range = p.file.range("experts");
    if (p.experts_range.empty()) {
        err = path + ": missing experts section";
        return false;
    }
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    {
        std::istringstream in(std::string(text.begin(), text.end()));
        std::string line;
        p.experts.resize((size_t) g.n_layers);
        std::vector<uint8_t> has((size_t) g.n_layers, 0);
        int seen = 0, version = 0;
        while (std::getline(in, line)) {
            if (line.rfind("# experts v", 0) == 0) version = std::atoi(line.c_str() + 11);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            long long l, gu, d, up = -1;
            unsigned long long off, by;
            if (!(ls >> l >> gu >> d >> off >> by) || l < 0 || l >= g.n_layers || by == 0 ||
                (uint64_t) off > p.experts_range.bytes ||
                (uint64_t) by > (p.experts_range.bytes - (uint64_t) off) / (uint64_t) g.n_expert ||
                (version >= 7 && !(ls >> up))) {
                err = "experts.layout: bad line '" + line + "'";
                return false;
            }
            if (version >= 7 && up != gu) {   // v7: the up rows' type; this model's kernels take one gate/up type
                err = "experts.layout: layer " + std::to_string(l) + "'s gate and up differ in type";
                return false;
            }
            if (has[(size_t) l]) {
                err = "experts.layout: duplicate layer " + std::to_string(l);
                return false;
            }
            has[(size_t) l] = 1;
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
        const Json* sec_j = ej->get("section");
        const std::string sec_name = sec_j && sec_j->is_string() ? sec_j->s : "engram";
        const core::ModelFile::Section* sec = p.file.find(sec_name);
        if (!sec) {
            err = path + ": meta.engram names a missing section";
            return false;
        }
        const Json *rows_j = ej->get("rows"), *rb_j = ej->get("row_bytes"), *rv_j = ej->get("row_values");
        if (!rows_j || rows_j->type != Json::Type::Int || rows_j->neg ||
            !rb_j || rb_j->type != Json::Type::Int || rb_j->neg ||
            !rv_j || rv_j->type != Json::Type::Int || rv_j->neg) {
            err = path + ": meta.engram is missing rows, row_bytes or row_values";
            return false;
        }
        e.file_offset = sec->offset;
        e.rows = rows_j->u64();
        const uint64_t row_bytes_u = rb_j->u64();
        const uint64_t row_values_u = rv_j->u64();
        e.type = (int) num_or(ej->get("type_id"), 20);
        if (row_values_u != 256) {   // every consumer dequantizes one row into 256 floats
            err = path + ": the Engram rows hold " + std::to_string(row_values_u) + " values; rows of 256 values are read";
            return false;
        }
        e.row_values = 256;
        const size_t want = engram_row_bytes(e.type, e.row_values);   // the whole row: blocks of 32 values, or of 256
        if (want == 0 || row_bytes_u != want) {
            err = path + ": the Engram rows are type " + std::to_string(e.type) + " of " + std::to_string(row_bytes_u) +
                  " bytes; IQ4_NL, Q5_K, Q6_K, MXFP4 and MXFP8 rows are read";
            return false;
        }
        e.row_bytes = (uint32_t) row_bytes_u;
        if (e.rows > sec->bytes / (uint64_t) e.row_bytes) {
            err = path + ": meta.engram rows exceed its section size";
            return false;
        }
        for (double d : numbers(ej->get("ple.layers"))) {
            const int l = (int) d;
            if (l < 0 || l >= g.n_layers) {
                err = path + ": meta.engram layer out of range";
                return false;
            }
            e.layers.push_back(l);
        }
        EngramHashParams& h = e.hash;
        h.n_gram = (int) num_or(ej->get("ple.ngram_size"), 4);
        h.n_head = (int) num_or(ej->get("ple.heads_per_ngram"), 8);
        h.pad_token_id = (int) num_or(ej->get("engram.pad_token_id"), 2);
        if (h.n_gram < 2 || h.n_gram > 9 || h.n_head <= 0 || h.n_head > INT_MAX / (h.n_gram - 1)) {
            err = path + ": meta.engram has invalid n-gram geometry";
            return false;
        }
        h.mult = unsigneds(ej->get("ple.layer_multipliers"));
        h.primes = unsigneds(ej->get("ple.head_vocab_sizes"));
        h.offsets = unsigneds(ej->get("ple.head_offsets"));
        const size_t ne = e.layers.size();
        if (h.mult.size() != ne * (size_t) h.n_gram || h.primes.size() != ne * (size_t) h.n_cols() ||
            h.offsets.size() != ne * (size_t) h.n_cols()) {
            err = path + ": the Engram hash arrays do not match its layer count";
            return false;
        }
        for (size_t i = 0; i < h.primes.size(); ++i) {
            if (h.primes[i] == 0) {
                err = path + ": meta.engram has a zero head vocab size";
                return false;
            }
            if (h.offsets[i] > e.rows || h.primes[i] > e.rows - h.offsets[i]) {
                err = path + ": meta.engram head range exceeds table rows";
                return false;
            }
        }
        std::vector<uint8_t> raw;
        if (!p.file.read("meta", raw, err)) return false;
        if (!read_int_array(std::string(raw.begin(), raw.end()), "deepseek41.engram.token_map", h.token_map)) {
            err = path + ": the Engram token map is missing";
            return false;
        }
        if (h.pad_token_id < 0 || (size_t) h.pad_token_id >= h.token_map.size()) {
            err = path + ": meta.engram pad_token_id is out of range";
            return false;
        }
        e.present = true;
    }
    return true;
}

}  // namespace basalt::models::deepseek41
