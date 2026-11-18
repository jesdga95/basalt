// src/models/hy4flash/dfly_test.cpp - CPU tests of the DFly2 drafter's host side: the pack record, the tensors it must hold
// (the real checkpoint's shapes), the source row, the draft loop, the flags, and the exactness of verification: whatever
// drafts a window holds, the stream equals plain decoding's, greedy or drawn, against a fake target.
#include "models/hy4flash/dfly.hpp"

#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::models::hy4flash;

static int g_fail = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "dfly_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                             \
        }                                                                         \
    } while (0)

namespace {

// tencent/Hy4-flash, model.dfly_layers.* of the last shard (safetensors order), layer 0 and the heads
struct Real {
    const char* name;
    std::vector<uint64_t> shape;
};
const std::vector<Real>& real_layer0() {
    static const std::vector<Real> v = {
        {"input_layernorm.weight", {3072}},
        {"mlp.down_proj.weight", {3072, 9216}},
        {"mlp.gate_proj.weight", {9216, 3072}},
        {"mlp.up_proj.weight", {9216, 3072}},
        {"post_attention_layernorm.weight", {3072}},
        {"self_attn.kv_a_layernorm.weight", {512}},
        {"self_attn.kv_a_proj_with_mqa.weight", {576, 3072}},
        {"self_attn.kv_b_proj.weight", {14336, 512}},
        {"self_attn.learnable_sink_param", {32}},
        {"self_attn.linear_gate.weight", {8192, 3072}},
        {"self_attn.o_proj.weight", {3072, 8192}},
        {"self_attn.post_proj_norm.weight", {3072}},
        {"self_attn.q_a_layernorm.weight", {1024}},
        {"self_attn.q_a_proj.weight", {1024, 3072}},
        {"self_attn.q_b_proj.weight", {8192, 1024}},
    };
    return v;
}
const std::vector<Real>& real_head() {
    static const std::vector<Real> v = {
        {"confidence_head.proj.bias", {1}},
        {"confidence_head.proj.weight", {1, 3072}},
        {"final_norm.weight", {3072}},
        {"hidden_correction.down_proj.weight", {3072, 3072}},
        {"hidden_correction.embed_norm.weight", {3072}},
        {"hidden_correction.gate_proj.weight", {3072, 6144}},
        {"hidden_correction.hidden_norm.weight", {3072}},
        {"hidden_correction.up_proj.weight", {3072, 6144}},
        {"mtp_fusion.eh_proj.weight", {3072, 6144}},
        {"mtp_fusion.enorm.weight", {3072}},
        {"mtp_fusion.gate", {3072}},
        {"mtp_fusion.hnorm.weight", {3072}},
    };
    return v;
}

std::map<std::string, DflyTensor> real_tensors(int layers) {
    std::map<std::string, DflyTensor> m;
    auto add = [&](const std::string& n, const std::vector<uint64_t>& s) {
        uint64_t e = 1;
        for (uint64_t d : s) e *= d;
        m[n] = {0, e * 2, s};
    };
    for (int i = 0; i < layers; ++i)
        for (const Real& r : real_layer0()) add(std::to_string(i) + "." + r.name, r.shape);
    for (const Real& r : real_head()) add(r.name, r.shape);
    return m;
}

std::string meta_text(const std::map<std::string, DflyTensor>& t, const std::string& extra = "") {
    std::string s = "{\"section\":\"dfly\",\"layers\":5,\"mask_token_id\":120009,\"target_kv_layers\":[25,29,33,37,41],"
                    "\"target_index_layers\":[25,29,33,37,41],\"target_index_topk\":2048,\"context_local_window\":0," + extra +
                    "\"tensors\":{";
    bool first = true;
    uint64_t off = 1000;
    for (const auto& [n, x] : t) {
        s += std::string(first ? "" : ",") + "\"" + n + "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t i = 0; i < x.shape.size(); ++i) s += (i ? "," : "") + std::to_string(x.shape[i]);
        s += "],\"offset\":" + std::to_string(off) + ",\"bytes\":" + std::to_string(x.bytes) + "}";
        off += x.bytes;
        first = false;
    }
    return s + "}}";
}

bool parse(const std::string& text, DflyPack& out, std::string& err, uint64_t sec = 1u << 21) {
    core::Json j;
    if (!core::parse_json(text.data(), text.size(), j, err)) return false;
    return parse_dfly_meta(j, sec, out, err);
}

void test_pack_record() {
    const auto tensors = real_tensors(5);
    DflyPack p;
    std::string err;
    CHECK(parse(meta_text(tensors), p, err, 4096));
    CHECK(p.present);
    CHECK(p.dims.layers == 5 && p.dims.mask_id == 120009 && p.dims.topk == 2048);
    CHECK(p.dims.kv_layers == (std::vector<int>{25, 29, 33, 37, 41}));
    CHECK(p.tensors.size() == 87);   // the checkpoint's count: 5 x 15 + 12
    CHECK(p.tensors.begin()->second.offset >= 4096 + 1000);   // offsets are the section's plus the record's
    CHECK(dfly_check_shapes(p.tensors, p.dims, err));
    CHECK(p.dims.E == 3072 && p.dims.q_lora == 1024 && p.dims.heads == 32 && p.dims.ffn == 9216 && p.dims.hc_ffn == 3072);
    // the drafter's bytes: 0.85B parameters in BF16 is the 1.69 GB the model card gives
    uint64_t bytes = 0;
    for (const DflySpec& s : dfly_specs(p.dims)) bytes += s.elems() * 2;
    CHECK(dfly_specs(p.dims).size() == 87);
    CHECK(bytes > 1'680'000'000ull && bytes < 1'700'000'000ull);

    // absent, a wrong shape, a missing tensor, a window of the context, a layer count that disagrees: refused or not present
    CHECK(parse("null", p, err) && !p.present);
    auto bad = tensors;
    bad["3.mlp.up_proj.weight"].shape = {9216, 3071};
    CHECK(parse(meta_text(bad), p, err) && !dfly_check_shapes(p.tensors, p.dims, err));
    bad = tensors;
    bad.erase("mtp_fusion.gate");
    CHECK(parse(meta_text(bad), p, err) && !dfly_check_shapes(p.tensors, p.dims, err));
    CHECK(!parse(meta_text(tensors).replace(meta_text(tensors).find("\"context_local_window\":0"), 24, "\"context_local_window\":64"), p, err));
    std::string s = meta_text(tensors);
    s.replace(s.find("\"layers\":5"), 10, "\"layers\":4");
    CHECK(!parse(s, p, err));
    s = meta_text(tensors);
    s.replace(s.find("\"dtype\":\"BF16\""), 14, "\"dtype\":\"F32\"");
    CHECK(!parse(s, p, err));
}

void test_source_row() {
    unsigned sel[kDflyTapRows] = {};
    const unsigned all = (1u << 5) - 1;
    // a window began at position 100 and kept rows 0..2 whole, row 3's hidden state only
    for (int r = 0; r < 3; ++r) sel[r] = all;
    sel[3] = all & ~(1u << 2);
    const unsigned hid = 0xFu;
    CHECK(dfly_source_row(100, hid, sel, 5, 101) == 0);   // the token at 101 was chosen by the row at 100
    CHECK(dfly_source_row(100, hid, sel, 5, 103) == 2);
    CHECK(dfly_source_row(100, hid, sel, 5, 104) == -1);  // row 3 lacks a selection
    CHECK(dfly_source_row(100, hid, sel, 5, 100) == -1);  // the row before the window is not kept
    CHECK(dfly_source_row(100, hid, sel, 5, 200) == -1);
    CHECK(dfly_source_row(100, 0x3u, sel, 5, 103) == -1);  // no hidden state for row 2
    CHECK(dfly_source_row(-1, hid, sel, 5, 1) == -1);      // invalidated
}

void test_draft_loop() {
    // draft j sees the token before it: the anchor, then each draft
    std::vector<int32_t> seen;
    int32_t ids[5];
    float conf[5];
    CHECK(dfly_draft_loop(
        5, 7,
        [&](int j, int32_t prev, int32_t& tok, float& c) {
            seen.push_back(prev);
            tok = 10 * (j + 1) + (int32_t) seen.size();
            c = 0.5f + 0.1f * (float) j;
            return true;
        },
        ids, conf));
    CHECK(seen == (std::vector<int32_t>{7, 11, 22, 33, 44}));
    CHECK(ids[4] == 55 && std::fabs(conf[3] - 0.8f) < 1e-6f);
    CHECK(!dfly_draft_loop(3, 0, [](int j, int32_t, int32_t&, float&) { return j < 1; }, ids, conf));
}

void test_flags() {
    program::CommonOptions o;
    char a0[] = "x", a1[] = "--spec", a2[] = "4", a3[] = "--spec-oracle", a4[] = "t.txt", a5[] = "--draft-log", a6[] = "d.log",
         a7[] = "--diag-teacher", a8[] = "--unknown";
    char* argv[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
    int i = 1;
    model::FlagArgs args(9, argv, i);
    CHECK(dfly_parse_flag("--spec", args, o) && o.spec == 4 && o.spec_given && i == 2);
    i = 3;
    CHECK(dfly_parse_flag("--spec-oracle", args, o) && o.spec_oracle == "t.txt");
    i = 5;
    CHECK(dfly_parse_flag("--draft-log", args, o));
    CHECK(dfly_parse_flag("--diag-teacher", args, o) && o.diag_teacher);
    CHECK(!dfly_parse_flag("--unknown", args, o));
    // no pack: a given --spec without an oracle is dropped, none stays 0
    program::CommonOptions p;
    p.spec = 6;
    p.spec_given = true;
    dfly_finish_flags(p);
    CHECK(p.spec == 0);
    program::CommonOptions q;
    dfly_finish_flags(q);
    CHECK(q.spec == 0);
    program::CommonOptions r;
    r.spec = 6;
    r.spec_given = true;
    r.spec_oracle = "t.txt";
    dfly_finish_flags(r);
    CHECK(r.spec == 6);
}

// ---- exact verification --------------------------------------------------------------------------------------------------

constexpr int kVocab = 23;

// The fake target: row i's logits are a function of the whole token history up to it, so a wrong draft in a window changes
// every later row, like a real model's. The pick is the argmax, or a draw keyed by the output position (a counter-based
// draw: the token at output index k is drawn from (seed, k) whatever the window it is made in).
struct FakeTarget {
    uint64_t seed;
    bool sampled;
    static uint64_t mix(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
        return x;
    }
    std::vector<double> logits(const std::vector<int32_t>& hist) const {
        uint64_t h = seed;
        for (size_t i = hist.size() > 3 ? hist.size() - 3 : 0; i < hist.size(); ++i) h = mix(h * 31 + (uint64_t) hist[i] + 1);
        std::vector<double> l(kVocab);
        for (int v = 0; v < kVocab; ++v) l[(size_t) v] = (double) (mix(h + (uint64_t) v * 977) % 10000) / 1000.0;
        return l;
    }
    int32_t pick(const std::vector<int32_t>& hist, int64_t out_index) const {
        const std::vector<double> l = logits(hist);
        if (!sampled) return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
        // Gumbel-max with noise from (seed, output index, token): the row's own draw
        int best = 0;
        double bv = -1e300;
        for (int v = 0; v < kVocab; ++v) {
            const double u = ((double) (mix(seed ^ mix((uint64_t) out_index * 131 + (uint64_t) v)) >> 11) + 0.5) / 9007199254740992.0;
            const double sc = l[(size_t) v] - std::log(-std::log(u));
            if (sc > bv) { bv = sc; best = v; }
        }
        return best;
    }
};

// Plain decoding: one row a step.
std::vector<int32_t> plain(const FakeTarget& t, std::vector<int32_t> prompt, int n) {
    std::vector<int32_t> out, hist = prompt;
    for (int k = 0; k < n; ++k) {
        const int32_t tok = t.pick(hist, k);
        out.push_back(tok);
        hist.push_back(tok);
    }
    return out;
}

// The window loop of the engine (src/program/generate/adapter.cpp): the window is the last pick and the drafts; row i's pick
// follows the window's first i + 1 tokens; the longest prefix of drafts the rows confirm is kept, with the row after it.
std::vector<int32_t> drafted(const FakeTarget& t, std::vector<int32_t> prompt, int n, int block, std::mt19937& rng, double quality,
                             long& offered, long& accepted) {
    std::vector<int32_t> out, hist = prompt;
    int32_t next = t.pick(hist, 0);
    out.push_back(next);
    while ((int) out.size() < n) {
        // the drafter: a draft is the true continuation with probability `quality`, else any token; each draft sees the one before
        const int nd = std::uniform_int_distribution<int>(0, block - 1)(rng);
        std::vector<float> c((size_t) block);
        std::vector<int32_t> truth_hist = hist;
        truth_hist.push_back(next);
        std::vector<int32_t> ids((size_t) block);
        dfly_draft_loop(
            nd, next,
            [&](int j, int32_t prev, int32_t& tok, float& chance) {
                (void) prev;
                const int64_t at = (int64_t) out.size() + j;
                const int32_t truth = t.pick(truth_hist, at);   // what plain decoding picks next
                truth_hist.push_back(truth);
                tok = std::bernoulli_distribution(quality)(rng) ? truth : std::uniform_int_distribution<int32_t>(0, kVocab - 1)(rng);
                chance = 0.5f;
                return true;
            },
            ids.data(), c.data());
        std::vector<int32_t> window = {next};
        window.insert(window.end(), ids.begin(), ids.begin() + nd);
        // row i: the pick after hist + window[0..i], drawn at the output index of the token it picks
        std::vector<int32_t> rows;
        std::vector<int32_t> h = hist;
        for (int i = 0; i <= nd; ++i) {
            h.push_back(window[(size_t) i]);
            rows.push_back(t.pick(h, (int64_t) out.size() + i));
        }
        int a = 0;
        while (a < nd && window[(size_t) a + 1] == rows[(size_t) a]) ++a;
        offered += nd;
        accepted += a;
        hist.push_back(next);
        for (int i = 0; i < a; ++i) hist.push_back(window[(size_t) i + 1]);
        for (int i = 0; i <= a && (int) out.size() < n; ++i) out.push_back(rows[(size_t) i]);
        next = rows[(size_t) a];
    }
    out.resize((size_t) n);
    return out;
}

void test_exact_verify() {
    long offered = 0, accepted = 0;
    for (int sampled = 0; sampled < 2; ++sampled) {
        for (uint64_t seed = 1; seed <= 40; ++seed) {
            const FakeTarget t{seed, sampled != 0};
            std::mt19937 rng((uint32_t) seed);
            std::vector<int32_t> prompt = {3, 1, 4, 1, 5};
            const std::vector<int32_t> want = plain(t, prompt, 60);
            for (double q : {0.0, 0.4, 0.9, 1.0}) {
                for (int block : {2, 4, 6}) {
                    const std::vector<int32_t> got = drafted(t, prompt, 60, block, rng, q, offered, accepted);
                    if (got != want) {
                        std::fprintf(stderr, "dfly_test: stream differs (sampled %d, seed %llu, quality %.1f, block %d)\n", sampled,
                                     (unsigned long long) seed, q, block);
                        ++g_fail;
                        return;
                    }
                }
            }
        }
    }
    CHECK(offered > 1000 && accepted > 100 && accepted < offered);   // the windows really held drafts, some right, some wrong
}

}  // namespace

int main() {
    test_pack_record();
    test_source_row();
    test_draft_loop();
    test_flags();
    test_exact_verify();
    if (g_fail) return 1;
    std::printf("dfly_test: ok\n");
    return 0;
}
