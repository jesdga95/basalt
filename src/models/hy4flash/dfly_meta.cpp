// src/models/hy4flash/dfly_meta.cpp - the host side of the DFly2 drafter that needs no device: the pack's record, the
// tensors it must hold, the draft loop, and the flags.
#include "models/hy4flash/dfly.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"

#include <cstdlib>

namespace basalt::models::hy4flash {

uint64_t DflySpec::elems() const {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

namespace {

using core::Json;

std::vector<int> ints(const Json* j) {
    std::vector<int> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back((int) x.num());
    return v;
}

}  // namespace

bool parse_dfly_meta(const Json& d, uint64_t section_offset, DflyPack& out, std::string& err) {
    out = DflyPack{};
    if (!d.is_object()) return true;
    auto num = [&](const char* k, double dflt) {
        const Json* j = d.get(k);
        return j && j->is_number() ? j->num() : dflt;
    };
    DflyDims& m = out.dims;
    m.layers = (int) num("layers", 0);
    m.mask_id = (int) num("mask_token_id", -1);
    m.topk = (int) num("target_index_topk", 2048);
    m.kv_layers = ints(d.get("target_kv_layers"));
    m.idx_layers = ints(d.get("target_index_layers"));
    if (num("context_local_window", 0) != 0) {
        err = "meta.dfly2: a local window of the context is not run";
        return false;
    }
    if (m.layers < 1 || m.layers > kDflyMaxLayers || (int) m.kv_layers.size() != m.layers || (int) m.idx_layers.size() != m.layers ||
        m.mask_id < 0) {
        err = "meta.dfly2: layers, target_kv_layers, target_index_layers or mask_token_id are missing or disagree";
        return false;
    }
    const Json* tj = d.get("tensors");
    if (!tj || !tj->is_object()) {
        err = "meta.dfly2 has no tensors";
        return false;
    }
    for (const auto& [name, t] : tj->members) {
        const Json* dt = t.get("dtype");
        if (!dt || dt->s != "BF16") {
            err = "meta.dfly2: " + name + " is not BF16";
            return false;
        }
        DflyTensor r;
        const Json *off = t.get("offset"), *by = t.get("bytes"), *sh = t.get("shape");
        if (!off || !by || !sh || !sh->is_array()) {
            err = "meta.dfly2: " + name + " has no offset, bytes or shape";
            return false;
        }
        r.offset = section_offset + off->u64();
        r.bytes = by->u64();
        for (const Json& x : sh->items) r.shape.push_back(x.u64());
        out.tensors.emplace(name, std::move(r));
    }
    out.present = true;
    return true;
}

std::vector<DflySpec> dfly_specs(const DflyDims& d) {
    const uint64_t E = (uint64_t) d.E, H = (uint64_t) d.heads, F = (uint64_t) d.ffn, Q = (uint64_t) d.q_lora, C = (uint64_t) d.hc_ffn;
    std::vector<DflySpec> v;
    auto add = [&](std::string n, std::vector<uint64_t> s, bool f32 = false) { v.push_back({std::move(n), std::move(s), f32, false}); };
    auto q8 = [&](std::string n, std::vector<uint64_t> s) { v.push_back({std::move(n), std::move(s), false, true}); };
    auto q4 = [&](std::string n, std::vector<uint64_t> s) { v.push_back({std::move(n), std::move(s), false, false, true}); };
    for (int i = 0; i < d.layers; ++i) {
        const std::string p = std::to_string(i) + ".";
        add(p + "input_layernorm.weight", {E}, true);
        add(p + "post_attention_layernorm.weight", {E}, true);
        q4(p + "mlp.gate_proj.weight", {F, E});
        q4(p + "mlp.up_proj.weight", {F, E});
        q4(p + "mlp.down_proj.weight", {E, F});
        q8(p + "self_attn.q_a_proj.weight", {Q, E});
        add(p + "self_attn.q_a_layernorm.weight", {Q}, true);
        q8(p + "self_attn.q_b_proj.weight", {H * 256, Q});
        q8(p + "self_attn.kv_a_proj_with_mqa.weight", {576, E});
        add(p + "self_attn.kv_a_layernorm.weight", {512}, true);
        add(p + "self_attn.kv_b_proj.weight", {H * 448, 512});
        q4(p + "self_attn.linear_gate.weight", {H * 256, E});
        q4(p + "self_attn.o_proj.weight", {E, H * 256});
        add(p + "self_attn.post_proj_norm.weight", {E}, true);
        add(p + "self_attn.learnable_sink_param", {H}, true);
    }
    add("final_norm.weight", {E}, true);
    q8("mtp_fusion.eh_proj.weight", {E, 2 * E});
    add("mtp_fusion.enorm.weight", {E}, true);
    add("mtp_fusion.hnorm.weight", {E}, true);
    add("mtp_fusion.gate", {E}, true);
    q8("hidden_correction.gate_proj.weight", {C, 2 * E});
    q8("hidden_correction.up_proj.weight", {C, 2 * E});
    q8("hidden_correction.down_proj.weight", {E, C});
    add("hidden_correction.hidden_norm.weight", {E}, true);
    add("hidden_correction.embed_norm.weight", {E}, true);
    add("confidence_head.proj.weight", {1, E}, true);
    add("confidence_head.proj.bias", {1}, true);
    return v;
}

bool dfly_check_shapes(const std::map<std::string, DflyTensor>& tensors, DflyDims& d, std::string& err) {
    auto shape = [&](const char* n) -> const std::vector<uint64_t>* {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second.shape;
    };
    const auto *fn = shape("final_norm.weight"), *gp = shape("0.mlp.gate_proj.weight"), *qa = shape("0.self_attn.q_a_proj.weight"),
               *lg = shape("0.self_attn.linear_gate.weight"), *hg = shape("hidden_correction.gate_proj.weight");
    if (!fn || fn->size() != 1 || !gp || gp->size() != 2 || !qa || qa->size() != 2 || !lg || lg->size() != 2 || !hg || hg->size() != 2) {
        err = "dfly: tensors final_norm, 0.mlp.gate_proj, 0.self_attn.q_a_proj, 0.self_attn.linear_gate or hidden_correction.gate_proj are missing";
        return false;
    }
    d.E = (int) (*fn)[0];
    d.ffn = (int) (*gp)[0];
    d.q_lora = (int) (*qa)[0];
    d.heads = (int) ((*lg)[0] / 256);
    d.hc_ffn = (int) (*hg)[0];
    for (const DflySpec& s : dfly_specs(d)) {
        auto it = tensors.find(s.name);
        if (it == tensors.end()) {
            err = "dfly: no tensor " + s.name;
            return false;
        }
        if (it->second.shape != s.shape || it->second.bytes != s.elems() * 2) {
            err = "dfly: tensor " + s.name + " has another shape than the drafter's";
            return false;
        }
    }
    return true;
}

bool dfly_draft_loop(int n, int32_t anchor, const std::function<bool(int, int32_t, int32_t&, float&, bool&)>& step, int32_t* ids,
                     float* conf, int& made) {
    int32_t prev = anchor;
    for (made = 0; made < n; ++made) {
        bool go = true;
        if (!step(made, prev, ids[made], conf[made], go)) return false;
        if (!go) break;
        prev = ids[made];
    }
    return true;
}

bool dfly_parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o) {
    if (a == "--spec") {
        o.spec = std::atoi(args.next("--spec"));
        o.spec_given = true;
    } else if (a == "--spec-oracle") o.spec_oracle = args.next("--spec-oracle");
    else if (a == "--diag-teacher") o.diag_teacher = true;
    else if (a == "--diag-tier-frozen") o.diag_tier_frozen = true;
    else if (a == "--draft-log") core::diagnostics().draft_log = args.next("--draft-log");
    else return false;
    return true;
}

void dfly_finish_flags(program::CommonOptions& o) {
    core::ModelFile file;
    std::string err;
    const bool opened = !o.model.empty() && file.open(o.model, err);
    const bool drafter = opened && file.find("dfly") != nullptr;
    // the pack's drafter offers a whole block (the window policy takes the rows that pay) unless --spec caps it; a token
    // file drafts in its place
    if (!o.spec_given) {
        o.spec = drafter ? kDflyBlock : 0;
        if (opened && !drafter) basalt::log::info("basalt hy4flash", "no dfly section: plain decode");
    }
    if (o.spec >= 2 && o.spec_oracle.empty() && !drafter) {
        basalt::log::warn("basalt hy4flash", "--spec %d off: the pack has no dfly section", o.spec);
        o.spec = 0;
    }
}

}  // namespace basalt::models::hy4flash
