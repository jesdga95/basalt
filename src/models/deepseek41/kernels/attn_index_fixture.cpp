// Fixture check of the DeepSeek-V4.1 attention index side against per-layer tensors recorded from the llama.cpp
// support on the same weights: kv (window latent), comp_latent (compressor) and idx_top_k (indexer selection) for
// a Full, a first-level and a second-level layer across the 16 prefill and 16 decode positions of one prompt.
// The inputs and weights come from the exporter's directory (--dir).
#include "basalt/log.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

#include <algorithm>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int NT = 32;
std::string dir;

template <class T>
std::vector<T> load(const std::string& name) {
    std::vector<T> v;
    if (!read_vec(dir + "/" + name, v)) { basalt::log::info("basalt deepseek41", "cannot read %s/%s", dir.c_str(), name.c_str()); std::exit(2); }
    return v;
}

std::vector<float> table(bool compressed) {
    RopeParams p;
    if (compressed) p = RopeParams{160000.f, 1.f / 16.f, 1.f, 32.f, 1.f, 65536};
    std::vector<float> t(kRopePairs);
    rope_freq_table(p, t.data());
    return t;
}

float f16(uint16_t h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }

void rope_ref(std::vector<double>& v, double pos, const std::vector<float>& freq) {
    const size_t off = v.size() - kRopeDim;
    for (int p = 0; p < kRopePairs; ++p) {
        const double th = pos * (double) freq[p], c = std::cos(th), s = std::sin(th);
        const double x0 = v[off + 2 * p], x1 = v[off + 2 * p + 1];
        v[off + 2 * p] = x0 * c - x1 * s;
        v[off + 2 * p + 1] = x0 * s + x1 * c;
    }
}

std::vector<int> chunks() {
    std::vector<int> c = {4, 4, 4, 4};
    for (int i = 0; i < 16; ++i) c.push_back(i % 5 == 4 ? 2 : 1);
    int sum = 0;
    for (int x : c) sum += x;
    while (sum > NT) { c.pop_back(); sum = 0; for (int x : c) sum += x; }
    while (sum < NT) { c.push_back(1); ++sum; }
    return c;
}

template <class T>
Dev<T> up(const std::vector<T>& v) { return Dev<T>(v); }

void check_window(Verdict& v, int layer, bool compressed) {
    const auto freq = table(compressed);
    const auto raw = load<float>("kvraw_" + std::to_string(layer) + ".bin");
    const auto nw = load<float>("kvnorm_" + std::to_string(layer) + ".bin");
    const auto exp = load<float>("expkv_" + std::to_string(layer) + ".bin");
    Dev<float> draw(raw), dnw(nw), dfreq(freq), out((size_t) NT * kLatent);
    std::vector<int32_t> pos(NT);
    for (int i = 0; i < NT; ++i) pos[i] = i;
    Dev<int32_t> dpos(pos);
    Dev<uint16_t> ring((size_t) kWindowSlots * kLatent);
    int off = 0;
    for (int n : chunks()) {
        kv_window(draw.p + (size_t) off * kLatent, dnw.p, dpos.p + off, n, 1e-20f, dfreq.p, out.p + (size_t) off * kLatent, ring.p, nullptr);
        off += n;
    }
    check(cudaDeviceSynchronize(), "kv_window");
    // The recorded projection ran on 8-bit activations, so the unrotated dims carry that noise (about 1e-2); the
    // rotated dims must not add to it.
    const auto got = out.down();
    std::vector<float> g0, w0, g1, w1;
    for (int t = 0; t < NT; ++t)
        for (int c = 0; c < kLatent; ++c) {
            (c < kLatent - kRopeDim ? g0 : g1).push_back(got[(size_t) t * kLatent + c]);
            (c < kLatent - kRopeDim ? w0 : w1).push_back(exp[(size_t) t * kLatent + c]);
        }
    const Err e0 = compare(g0, w0), e1 = compare(g1, w1);
    std::printf("layer %2d kv (window latent): rel RMS %.2e unrotated dims, %.2e rotated dims\n", layer, e0.rms_rel, e1.rms_rel);
    v.expect(e0.rms_rel < 1.5e-2 && e1.rms_rel < 1.5e-2, "kv vs fixture");
}

struct Layer {
    int id, ratio;
    Dev<uint16_t> ckv, cgate, ik, qb, proj;
    Dev<float> cnorm, iknorm, freq, x, qr, skv, ssc;
    Dev<uint16_t> comp, cik;
};

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "--dir") dir = argv[i + 1];
    if (dir.empty()) { std::printf("SKIP: pass --dir EXPORT_DIR (fixture export)\n"); return 0; }
    Verdict v;
    check_window(v, 0, false);
    check_window(v, 2, true);
    check_window(v, 20, true);

    const auto freq = table(true);
    Dev<float> dfreq(freq);
    std::vector<int32_t> pos(NT);
    for (int i = 0; i < NT; ++i) pos[i] = i;
    Dev<int32_t> dpos(pos);
    Dev<char> cws(compress_workspace_bytes()), iws(indexer_workspace_bytes(64));
    const std::vector<float> zero(2 * kLatent, 0.f);
    struct L { int id, ratio; };
    // compressors of layers 2 (ratio 2) and 20 (ratio 1)
    Dev<uint16_t> ckv2(load<uint16_t>("ckv_2.bin")), cg2(load<uint16_t>("cgate_2.bin")), ik2(load<uint16_t>("ik_2.bin"));
    Dev<uint16_t> ckv20(load<uint16_t>("ckv_20.bin")), ik20(load<uint16_t>("ik_20.bin"));
    Dev<float> cn2(load<float>("cnorm_2.bin")), cn20(load<float>("cnorm_20.bin")), ikn2(load<float>("iknorm_2.bin")), ikn20(load<float>("iknorm_20.bin"));
    Dev<float> sk2(zero), ss2(zero), sk20(zero), ss20(zero);
    Dev<uint16_t> comp2((size_t) 64 * kLatent), cik2((size_t) 64 * kIdxDim), comp20((size_t) 64 * kLatent), cik20((size_t) 64 * kIdxDim);
    for (auto* d : {&comp2, &comp20}) d->up(std::vector<uint16_t>(d->n, 0));
    for (auto* d : {&cik2, &cik20}) d->up(std::vector<uint16_t>(d->n, 0));
    Dev<float> x2(load<float>("x_2.bin")), x20(load<float>("x_20.bin")), x24(load<float>("x_24.bin"));
    Dev<float> qr2(load<float>("qr_2.bin")), qr20(load<float>("qr_20.bin")), qr24(load<float>("qr_24.bin"));
    Dev<uint16_t> qb2(load<uint16_t>("qb_2.bin")), pj2(load<uint16_t>("proj_2.bin")), qb20(load<uint16_t>("qb_20.bin")),
        pj20(load<uint16_t>("proj_20.bin")), qb24(load<uint16_t>("qb_24.bin")), pj24(load<uint16_t>("proj_24.bin"));
    Dev<int32_t> cand(NT * kMaxCandBlocks), cn(NT), ids2(NT * kTopK), n2(NT), ids20(NT * kTopK), n20(NT), ids24(NT * kTopK), n24(NT);

    int off = 0;
    for (int n : chunks()) {
        CompressArgs a;
        a.rope_freq = dfreq.p;
        a.ratio = 2; a.w_kv = ckv2.p; a.w_gate = cg2.p; a.norm_w = cn2.p; a.w_idx_k = ik2.p; a.idx_k_norm_w = ikn2.p;
        a.state_kv = sk2.p; a.state_score = ss2.p; a.comp_kv = comp2.p; a.comp_ik = cik2.p;
        compress(a, x2.p + (size_t) off * 5120, dpos.p + off, n, cws.p, nullptr);
        CompressArgs b = a;
        b.ratio = 1; b.w_kv = ckv20.p; b.w_gate = nullptr; b.norm_w = cn20.p; b.w_idx_k = ik20.p; b.idx_k_norm_w = ikn20.p;
        b.state_kv = sk20.p; b.state_score = ss20.p; b.comp_kv = comp20.p; b.comp_ik = cik20.p;
        compress(b, x20.p + (size_t) off * 5120, dpos.p + off, n, cws.p, nullptr);

        const int last = off + n - 1;
        IndexerArgs ia;
        ia.rope_freq = dfreq.p;
        ia.ratio = 2; ia.n_comp = std::max(1, (last + 1) / 2); ia.mode = IndexMode::Full; ia.w_q = qb2.p; ia.w_proj = pj2.p;
        ia.comp_ik = cik2.p; ia.out_ids = ids2.p + (size_t) off * kTopK; ia.out_n = n2.p + off;
        indexer_select(ia, qr2.p + (size_t) off * 1280, x2.p + (size_t) off * 5120, dpos.p + off, n, iws.p, nullptr);
        IndexerArgs i1 = ia;
        i1.ratio = 1; i1.n_comp = last + 1; i1.mode = IndexMode::Level1; i1.w_q = qb20.p; i1.w_proj = pj20.p; i1.comp_ik = cik20.p;
        i1.cand_ids = cand.p + (size_t) off * kMaxCandBlocks; i1.cand_n = cn.p + off;
        i1.out_ids = ids20.p + (size_t) off * kTopK; i1.out_n = n20.p + off;
        indexer_select(i1, qr20.p + (size_t) off * 1280, x20.p + (size_t) off * 5120, dpos.p + off, n, iws.p, nullptr);
        IndexerArgs i2 = i1;
        i2.mode = IndexMode::Level2; i2.w_q = qb24.p; i2.w_proj = pj24.p;
        i2.out_ids = ids24.p + (size_t) off * kTopK; i2.out_n = n24.p + off;
        indexer_select(i2, qr24.p + (size_t) off * 1280, x24.p + (size_t) off * 5120, dpos.p + off, n, iws.p, nullptr);
        off += n;
    }
    check(cudaDeviceSynchronize(), "run");

    // compressor latents
    for (int layer : {2, 20}) {
        const int ratio = layer == 2 ? 2 : 1;
        const auto exp = load<float>("expcomp_" + std::to_string(layer) + ".bin");
        const auto got = (layer == 2 ? comp2 : comp20).down();
        std::vector<float> g, w;
        for (int p = 0; p < NT; ++p) {
            if ((p + 1) % ratio) continue;
            std::vector<double> r(exp.begin() + (size_t) p * kLatent, exp.begin() + (size_t) (p + 1) * kLatent);
            rope_ref(r, p + 1 - ratio, freq);
            for (int c = 0; c < kLatent; ++c) { g.push_back(f16(got[(size_t) (p / ratio) * kLatent + c])); w.push_back((float) r[c]); }
        }
        const Err e = compare(g, w);
        std::printf("layer %2d comp_latent (%zu rows, rotated): rel RMS %.2e\n", layer, g.size() / kLatent, e.rms_rel);
        v.expect(e.rms_rel < 3e-3, "comp_latent vs fixture");
    }
    // selections: fixture ids < visible rows (the rest of its 256 are padding) equal ours
    struct Sel { int layer, ratio; const Dev<int32_t>* ids; const Dev<int32_t>* n; };
    for (const Sel& s : {Sel{2, 2, &ids2, &n2}, Sel{20, 1, &ids20, &n20}, Sel{24, 1, &ids24, &n24}}) {
        const auto exp = load<int32_t>("exptopk_" + std::to_string(s.layer) + ".bin");
        const auto ids = s.ids->down(), cnt = s.n->down();
        int bad = 0, total = 0;
        for (int p = 0; p < NT; ++p) {
            const int nv = (p + 1) / s.ratio;
            std::vector<int> want;
            for (int k = 0; k < 256; ++k) if (exp[(size_t) p * 256 + k] >= 0 && exp[(size_t) p * 256 + k] < nv) want.push_back(exp[(size_t) p * 256 + k]);
            std::sort(want.begin(), want.end());
            want.erase(std::unique(want.begin(), want.end()), want.end());
            std::vector<int> got(ids.begin() + (size_t) p * kTopK, ids.begin() + (size_t) p * kTopK + cnt[p]);
            total += (int) want.size();
            if (want != got) ++bad;
        }
        std::printf("layer %2d idx_top_k: %d of %d tokens differ (%d ids compared)\n", s.layer, bad, NT, total);
        v.expect(bad == 0, "idx_top_k vs fixture");
    }
    std::printf(v.failures ? "FAILED (%d)\n" : "PASS\n", v.failures);
    return v.failures ? 1 : 0;
}
