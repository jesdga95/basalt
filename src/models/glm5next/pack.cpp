// src/models/glm5next/pack.cpp - reads the glm5-next pack's meta section into a Pack.
#include "models/glm5next/pack.hpp"

#include "basalt/artifact/gguf_reader.hpp"   // ggml_type_name, for what a pack's experts are

#include <cstdlib>
#include <sstream>

namespace basalt::models::glm5next {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }
const Json* member(const Json& o, const char* k) { return o.get(k); }

// A per-block float array of the geometry (the SwiGLU clamps). Absent: an empty vector, which the model reads as "no
// clamp". A present array must hold exactly `n` numbers; the JSON reader keeps only the first core::Json::kJsonArrayKeep
// items of a longer one, so a longer array is refused here rather than read short.
bool clamp_array(const Json& geo, const char* key, int n, std::vector<float>& out, std::string& err) {
    const Json* j = member(geo, key);
    if (!j || j->type == Json::Type::Null) return true;
    if (!j->is_array() || j->count != (uint64_t) n || j->items.size() != j->count) {
        err = std::string("meta.geometry: ") + key + " wants exactly " + std::to_string(n) + " values";
        return false;
    }
    out.reserve((size_t) n);
    for (const Json& x : j->items) out.push_back((float) x.num());
    return true;
}

// The main or a drafter experts.layout: "layer gu down offset blob [up]" per line, a "# experts vN" header, dense lead
// layers absent. Layers not listed keep their default (gu_type/d_type 0).
//
// The layout's `blob` column is ONE expert's blob: a layer's region is all of its experts back to back, which is what
// the offsets' spacing says and what the header's `total` counts. That is also the number everything downstream
// indexes by - the tier's places, the card's grouped kernels, the copy of one expert into a slot - so it is stored as
// the file gives it, and a reader that wants a layer's whole region multiplies it by n_expert itself.
bool read_expert_layout(const core::ModelFile& file, const char* section, int n_layers, int n_expert,
                        std::vector<ExpertLayer>& out, std::string& err) {
    std::vector<uint8_t> text;
    if (!file.read(section, text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    int version = 0;
    while (std::getline(in, line)) {
        if (line.rfind("# experts v", 0) == 0) version = std::atoi(line.c_str() + 11);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        long long l, gu, d, up = -1;
        unsigned long long off, by;
        if (!(ls >> l >> gu >> d >> off >> by) || l < 0 || l >= n_layers || (version >= 7 && !(ls >> up))) {
            err = std::string(section) + ": bad line '" + line + "'";
            return false;
        }
        if (version >= 7 && up != gu) {   // the up rows' type; this model's kernels take one gate/up type
            err = std::string(section) + ": layer " + std::to_string(l) + "'s gate and up differ in type";
            return false;
        }
        // The tier computes an expert from its packed rows (the CPU pool's native formats, and the grouped kernels for
        // the ones the card homes), so a pack whose experts are fp32 is not something this port can run: refuse it
        // here, by layer, rather than feed the tier a type it has no dot product for.
        if (gu == 0 || d == 0) {
            err = std::string(section) + ": layer " + std::to_string(l) + "'s experts are " +
                  ggml_type_name((uint32_t) gu) + " (gate/up) and " + ggml_type_name((uint32_t) d) +
                  " (down); glm5next computes them in their packed formats";
            return false;
        }
        if (n_expert < 1) {
            err = std::string(section) + ": the layout has experts but the geometry has no expert count";
            return false;
        }
        if ((size_t) l >= out.size()) out.resize((size_t) l + 1);
        out[(size_t) l] = {(int) gu, (int) d, off, by};
    }
    return true;
}

/// The float tensors of a `dense` pool, from its `.index`: `<name> - <kind> <offset> <bytes> <offset> <stored> <ne0>
/// <ne1> - - <align> -...` (tools/convert/model_sections.py writes them; a row's two offsets are the same number).
/// A kind-0 row is a quantized tensor served from the `gguf` section, which the pack's tensor table already names, so
/// it is skipped here; what is left is the pool's own tensors, which the table does not name at all. That is why the
/// drafter's norms, router gate and index projection - and nothing else the model reads - can only be found here.
bool read_pool_index(const core::ModelFile& file, const char* section, const char* pool,
                     std::map<std::string, TensorRef>& out, std::string& err) {
    const core::ModelFile::Section* s = file.find(pool);
    if (!s) {
        err = std::string(pool) + ": the pack holds " + section + ", but no pool for it to point at";
        return false;
    }
    std::vector<uint8_t> text;
    if (!file.read(section, text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    int version = 0, declared = -1, rows = 0;
    unsigned long long pool_bytes = 0;
    while (std::getline(in, line)) {
        if (line.rfind("# dense index v", 0) == 0) {
            version = std::atoi(line.c_str() + 15);
            continue;
        }
        if (line.rfind("# align ", 0) == 0) {
            std::istringstream hs(line);
            std::string hash, kw;
            long long align = 0;
            if (!(hs >> hash >> kw >> align >> kw >> pool_bytes >> kw >> declared) || kw != "tensors") {
                err = std::string(section) + ": bad header '" + line + "'";
                return false;
            }
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string name;
        long long f1 = 0, kind = 0, again = 0, stored = 0, f9 = 0, f10 = 0, f11 = 0, ne0 = 0, ne1 = 0;
        unsigned long long off = 0, bytes = 0;
        if (!(ls >> name >> f1 >> kind >> off >> bytes >> again >> stored >> ne0 >> ne1 >> f9 >> f10 >> f11)) {
            err = std::string(section) + ": bad row '" + line + "'";
            return false;
        }
        ++rows;
        if (kind == 0) continue;
        // The pool's kinds are the converter's: 2 fp32, 4 bf16, 5 f16. Only the first is a pack this port runs - a
        // bf16 pool is what --compat-bf16 writes, and the model's own dense tensors would have arrived that way too.
        if (kind != 2) {
            err = std::string(section) + ": " + name + " is pool kind " + std::to_string(kind) +
                  "; glm5next reads the drafter's pool tensors as fp32 rows";
            return false;
        }
        if (ne0 <= 0 || off + bytes > s->bytes) {
            err = std::string(section) + ": " + name + " (" + std::to_string(off) + " +" + std::to_string(bytes) +
                  ") lies outside the " + std::string(pool) + " section's " + std::to_string(s->bytes) + " bytes";
            return false;
        }
        TensorRef r;
        r.name = name;
        r.type = 0;  // fp32
        r.shape.push_back((uint64_t) ne0);
        if (ne1 > 0) r.shape.push_back((uint64_t) ne1);
        r.offset = s->offset + (uint64_t) off;
        r.bytes = (uint64_t) bytes;
        out.emplace(name, std::move(r));
    }
    if (version != 3) {
        err = std::string(section) + ": index version " + std::to_string(version) + ", expected 3";
        return false;
    }
    if (declared != rows) {
        err = std::string(section) + ": the header names " + std::to_string(declared) +
              " tensors, the index holds " + std::to_string(rows);
        return false;
    }
    if (pool_bytes != s->bytes) {
        err = std::string(section) + ": the index counts " + std::to_string(pool_bytes) + " pool bytes, the " +
              std::string(pool) + " section holds " + std::to_string(s->bytes);
        return false;
    }
    return true;
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "glm5next") {
        err = path + ": not a glm5next pack";
        return false;
    }
    const Json* gj = meta.get("geometry");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no geometry / tensors";
        return false;
    }
    Geometry& g = p.geo;
    // every key a glm5-next pack must carry is required by name: one that the converter renamed or dropped is refused
    // here rather than read as 0, which would only show up later as a wrong size or a wrong answer
    std::vector<std::string> missing;
    auto at = [&](const char* k) {
        const Json* j = member(*gj, k);
        if (!j || !j->is_number()) {
            missing.emplace_back(k);
            return 0.;
        }
        return j->num();
    };
    auto u = [&](const char* k) { return (int) at(k); };
    auto f = [&](const char* k) { return (float) at(k); };
    g.n_layers = u("block_count");
    g.nextn = u("nextn_predict_layers");
    g.leading_dense = u("leading_dense_block_count");
    g.n_embd = u("embedding_length");
    g.n_vocab = u("vocab_size");
    g.n_head = u("attention.head_count");
    g.qk_dim = u("attention.key_length");
    g.v_dim = u("attention.value_length");
    g.q_lora = u("attention.q_lora_rank");
    g.kv_lora = u("attention.kv_lora_rank");
    g.kda_head_dim = u("kda.head_dim");
    g.conv = u("ssm.conv_kernel");
    g.gate_lower_bound = f("kda.gate_lower_bound");
    g.idx_heads = u("attention.indexer.head_count");
    g.idx_dim = u("attention.indexer.key_length");
    g.idx_top_k = u("attention.indexer.top_k");
    g.kpool = u("attention.indexer.kpool");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_shared = u("expert_shared_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.n_ff = u("feed_forward_length");
    g.expert_scale = f("expert_weights_scale");
    g.hc_count = u("hyper_connection.count");
    g.hc_eps = f("hyper_connection.epsilon");
    g.sinkhorn_iters = u("hyper_connection.sinkhorn_iterations");
    g.rms_eps = f("attention.layer_norm_rms_epsilon");
    g.context_length = (int64_t) at("context_length");
    if (const Json* kj = member(*gj, "attention.head_count_kv"); kj && kj->is_array())
        for (const Json& x : kj->items) g.kinds.push_back((int) x.u64());
    else
        missing.emplace_back("attention.head_count_kv");
    if (!missing.empty()) {
        err = path + ": meta.geometry lacks";
        for (const std::string& k : missing) err += " " + k + ",";
        err.pop_back();                               // no comma after the last name
        return false;
    }
    // The geometry beyond the required keys, read once the block count is known.
    //
    // The head widths of the sparse MLA come from their own keys when a pack separates them. The release keeps
    // attention.key_length / value_length at the compressed latent's length (512 both) and states the head widths
    // separately: its blk.N.attn_q_b projects q_lora_rank to head_count x 256 and its output projection takes
    // head_count x 256 in, so 256 (attention.*_length_mla) is what the weights want.
    if (const Json* j = member(*gj, "attention.key_length_mla"); j && j->is_number()) g.qk_dim = (int) j->num();
    if (const Json* j = member(*gj, "attention.value_length_mla"); j && j->is_number()) g.v_dim = (int) j->num();
    // The indexer's key LayerNorm has its own epsilon, apart from the RMS norms' (attention.layer_norm_rms_epsilon).
    if (const Json* j = member(*gj, "attention.layer_norm_epsilon"); j && j->is_number()) g.ln_eps = (float) j->num();
    // The shared experts' width, which a pack may set apart from the routed ones (Geometry::shared_ffn falls back).
    if (const Json* j = member(*gj, "expert_shared_feed_forward_length"); j && j->is_number())
        g.n_ff_shexp = (int) j->num();
    // A pack without the SwiGLU clamps keeps the unclamped gate.
    if (!clamp_array(*gj, "swiglu_clamp_exp", g.n_layers, g.clamp_exp, err) ||
        !clamp_array(*gj, "swiglu_clamp_shexp", g.n_layers, g.clamp_shexp, err))
        return false;
    if (const Json* gk = meta.get("gguf_kv"))
        if (const Json* t = gk->get("tokenizer.ggml.tokens"); t && (int) t->count != g.n_vocab) {
            err = path + ": geometry vocab_size " + std::to_string(g.n_vocab) +
                  " disagrees with the vocabulary's " + std::to_string(t->count) + " tokens";
            return false;
        }

    if (g.n_layers <= 0 || g.n_embd <= 0 || (int) g.kinds.size() != g.n_layers) {
        err = path + ": meta.geometry is incomplete (block_count / embedding_length / attention.head_count_kv)";
        return false;
    }
    // What this port's kernels are: the shared router's sigmoid gate with the chosen weights renormalised, and MLA heads
    // with no rotary part. A pack that states otherwise wants a path glm5next does not have, so it is refused rather than
    // run with the wrong gate or a rotary part dropped on the floor.
    if (const Json* j = member(*gj, "expert_gating_func"); j && j->is_number() && (int) j->num() != 2) {
        err = path + ": expert_gating_func is " + std::to_string((int) j->num()) +
              "; glm5next routes with the sigmoid gate (2)";
        return false;
    }
    if (const Json* j = member(*gj, "expert_weights_norm"); j && j->type == Json::Type::Bool && !j->b) {
        err = path + ": expert_weights_norm is false; glm5next renormalises the chosen experts' weights";
        return false;
    }
    if (const Json* j = member(*gj, "rope.dimension_count"); j && j->is_number() && (int) j->num() != 0) {
        err = path + ": rope.dimension_count is " + std::to_string((int) j->num()) +
              "; glm5next's MLA heads carry no rotary part";
        return false;
    }
    // The shared router picks its experts from the whole set at once. A pack that groups the experts and lets the gate
    // choose from only some groups wants a selection glm5next has not built; one group with all of it used is the no-op
    // the router already is, which is what every released pack states.
    for (const char* k : {"expert_group_count", "expert_group_used_count"}) {
        if (const Json* j = member(*gj, k); j && j->is_number() && (int) j->num() != 1) {
            err = path + std::string(": ") + k + " is " + std::to_string((int) j->num()) +
                  "; glm5next selects its experts across all of them at once";
            return false;
        }
    }
    if (const Json* j = member(*gj, "attention.indexer.index_share_mtp"); j && j->type == Json::Type::Bool)
        g.index_share_mtp = j->b;

    // tensors; offsets are relative to the section a tensor names (dense/quantized/experts per the converter)
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

    // routed experts (main), then the drafter's own section
    p.experts_range = p.file.range("experts");
    if (!read_expert_layout(p.file, "experts.layout", g.n_layers, g.n_expert, p.experts, err)) return false;
    p.mtp_experts_range = p.file.range("mtp/experts");
    p.mtp_dense_range = p.file.range("mtp/dense");
    // Half a drafter is a pack that was cut short: its block needs both its experts and its float pool.
    if (p.mtp_experts_range.empty() != p.mtp_dense_range.empty()) {
        err = path + ": the pack carries " +
              (p.mtp_dense_range.empty() ? std::string("mtp/experts but no mtp/dense")
                                         : std::string("mtp/dense but no mtp/experts"));
        return false;
    }
    if (!p.mtp_experts_range.empty()) {
        if (!read_expert_layout(p.file, "mtp/experts.layout", g.n_layers, g.n_expert, p.mtp_experts, err)) return false;
        if (!read_pool_index(p.file, "mtp/dense.index", "mtp/dense", p.mtp_tensors, err)) return false;
    }
    return true;
}

}  // namespace basalt::models::glm5next
