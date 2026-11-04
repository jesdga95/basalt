// check_all over a synthetic weight table (host only): the canonical shapes pass; a wrong dimension, a wrong
// engine form, a missing tensor and a tensor of the wrong layer family each fail with a message naming it.
#include "basalt/models/qwen4exp/decode/layout.hpp"

#include <cstdio>
#include <cstdlib>

namespace basalt::core {

struct WeightTableFixture {
    static void put(WeightTable& t, const std::string& name, int64_t ne0, int64_t ne1, WeightKind kind) {
        WeightRef r;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.elements = ne0 * ne1;
        r.kind = kind;
        r.bytes = (uint64_t) r.elements * (kind == WeightKind::Bf16InF32 ? 2u : 4u);
        t.table_[name] = r;
    }
    static void erase(WeightTable& t, const std::string& name) { t.table_.erase(name); }
    static WeightRef* row(WeightTable& t, const std::string& name) { return &t.table_.at(name); }
};

}  // namespace basalt::core

using namespace basalt::core;

static void build(WeightTable& t, const ModelGeometry& g) {
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool qsa = is_qsa_layer(g, l);
        const std::string p = "blk." + std::to_string(l) + ".";
        auto m = [&](const char* s, int64_t a, int64_t b) { WeightTableFixture::put(t, p + s, a, b, WeightKind::Verbatim); };
        auto v = [&](const char* s, int64_t n, WeightKind k) { WeightTableFixture::put(t, p + s, n, 1, k); };
        m("hc_attn_down.weight", g.hc_dim(), g.hc_lr);
        m("hc_attn_up.weight", g.hc_lr, g.hc_dim());
        m("hc_attn_inject.weight", g.hc_dim(), g.hc);
        m("hc_ffn_down.weight", g.hc_dim(), g.hc_lr);
        m("hc_ffn_up.weight", g.hc_lr, g.hc_dim());
        m("hc_ffn_inject.weight", g.hc_dim(), g.hc);
        m("ffn_gate_inp.weight", g.n_embd, g.n_expert);
        m("ffn_gate_shexp.weight", g.n_embd, g.n_ff);
        m("ffn_up_shexp.weight", g.n_embd, g.n_ff);
        m("ffn_down_shexp.weight", g.n_ff, g.n_embd);
        v("hc_attn_norm.weight", g.hc_dim(), WeightKind::F32);
        v("hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32);
        v("ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32);
        if (qsa) {
            m("attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim);
            m("attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim);
            m("attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim);
            m("attn_output.weight", g.n_head * g.head_dim, g.n_embd);
            m("indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim);
            m("indexer.k_proj.weight", g.n_embd, g.idx_key_dim);
            v("attn_q_norm.weight", g.head_dim, WeightKind::F32);
            v("attn_k_norm.weight", g.head_dim, WeightKind::F32);
            v("indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32);
            v("indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32);
        } else {
            m("attn_qkv.weight", g.n_embd, g.ssm_conv_channels);
            m("attn_gate.weight", g.n_embd, g.ssm_value_dim);
            m("ssm_out.weight", g.ssm_value_dim, g.n_embd);
            m("ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels);
            m("ssm_alpha.weight", g.n_embd, g.ssm_v_heads);
            m("ssm_beta.weight", g.n_embd, g.ssm_v_heads);
            v("ssm_a", g.ssm_v_heads, WeightKind::F32);
            v("ssm_dt.bias", g.ssm_v_heads, WeightKind::F32);
            v("ssm_norm.weight", g.ssm_state_size, WeightKind::F32);
        }
    }
}

int main() {
    const ModelGeometry g;
    auto check = [](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "layout_check_test: %s\n", what); std::exit(1); }
    };
    std::string err;

    WeightTable t;
    build(t, g);
    check(check_all(t, g, err), "the canonical table was refused");

    // one wrong dimension in a QSA layer's projection
    WeightRef* r = WeightTableFixture::row(t, "blk.7.attn_k.weight");
    r->ne1 += 1;
    check(!check_all(t, g, err), "a wrong dimension passed");
    check(err.find("blk.7.attn_k.weight") != std::string::npos && err.find("ne1") != std::string::npos,
          "the message does not name the tensor and axis");
    r->ne1 -= 1;
    check(check_all(t, g, err), "the restored table was refused");

    // a 1-D tensor in the wrong engine form
    r = WeightTableFixture::row(t, "blk.0.ffn_gate_inp_shexp.weight");
    r->kind = WeightKind::F32;
    check(!check_all(t, g, err) && err.find("ffn_gate_inp_shexp") != std::string::npos, "a wrong form passed");
    r->kind = WeightKind::Bf16InF32;

    // a missing tensor
    WeightRef saved = *WeightTableFixture::row(t, "blk.20.ssm_a");
    WeightTableFixture::erase(t, "blk.20.ssm_a");
    check(!check_all(t, g, err) && err.find("blk.20.ssm_a") != std::string::npos, "a missing tensor passed");
    WeightTableFixture::put(t, "blk.20.ssm_a", saved.ne0, saved.ne1, saved.kind);
    check(check_all(t, g, err), "the restored table was refused");

    // a geometry that disagrees with the pack
    ModelGeometry other = g;
    other.n_ff = 512;
    check(!check_all(t, other, err), "a different geometry passed");
    std::puts("layout_check_test: passed");
}
