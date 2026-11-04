// Fixture parity of the attention compute side on real weights: from a layer's hc_attn_pre, kv, comp_latent and
// idx_top_k fixtures (llama.cpp on the same weights) reproduce attn_derope and attn_out over the prefill and decode
// steps. Run by hand: attn_fixture_parity <pack> <fixture.bin> <index.txt> <layer> <ratio> <comp_src_layer> <steps>
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "test_util.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <tuple>

using namespace basalt::models::deepseek41;
using dstest::Dev;

namespace {

struct Weight { int type, n_in, n_out; long long off; };
struct Fx { long long bytes, off; };

std::vector<char> read_at(std::ifstream& f, long long off, size_t n) {
    std::vector<char> b(n);
    f.seekg(off);
    f.read(b.data(), (std::streamsize) n);
    if (!f) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: pack fixture.bin index.txt layer ratio comp_src steps\n"); return 2; }
    const std::string pack = argv[1], fixbin = argv[2], idxf = argv[3];
    const int layer = std::atoi(argv[4]), ratio = std::atoi(argv[5]), comp_src = std::atoi(argv[6]);
    const int n_steps = std::atoi(argv[7]);
    const int shift = argc > 8 ? std::atoi(argv[8]) : 0;
    const int chunk = argc > 9 ? std::atoi(argv[9]) : 8;

    std::map<std::string, Weight> W;
    std::map<std::tuple<int, std::string, int>, Fx> F;
    int n_prompt = 0;
    {
        std::ifstream in(idxf);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            char k; ss >> k;
            if (k == 'W') { std::string n; Weight w; ss >> n >> w.type >> w.n_in >> w.n_out >> w.off; W[n] = w; }
            else if (k == 'N') ss >> n_prompt;
            else if (k == 'F') { int st, ly; std::string n; Fx x; ss >> st >> n >> ly >> x.bytes >> x.off; F[{st, n, ly}] = x; }
        }
    }
    std::ifstream pk(pack, std::ios::binary), fb(fixbin, std::ios::binary);
    auto wbytes = [&](const char* n) {
        const Weight& w = W.at(n);
        return read_at(pk, w.off, basalt::kernels::native_mmvq_weight_bytes(w.type, w.n_in, w.n_out));
    };
    auto wfloat = [&](const char* n) {
        const Weight& w = W.at(n);
        auto b = read_at(pk, w.off, (size_t) w.n_in * 4);
        std::vector<float> v(w.n_in);
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    };
    auto fx = [&](int step, const char* name, int ly) {
        auto it = F.find({step, name, ly});
        if (it == F.end()) { std::fprintf(stderr, "missing fixture %d %s %d\n", step, name, ly); std::exit(1); }
        auto b = read_at(fb, it->second.off, (size_t) it->second.bytes);
        std::vector<float> v(b.size() / 4);
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    };

    auto bq_a = wbytes("attn_q_a"), bq_b = wbytes("attn_q_b"), bo_a = wbytes("attn_output_a"), bo_b = wbytes("attn_output_b");
    Dev<char> wq_a(bq_a.size()), wq_b(bq_b.size()), wo_a(bo_a.size()), wo_b(bo_b.size());
    cudaMemcpy(wq_a.p, bq_a.data(), bq_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(wq_b.p, bq_b.data(), bq_b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(wo_a.p, bo_a.data(), bo_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(wo_b.p, bo_b.data(), bo_b.size(), cudaMemcpyHostToDevice);
    const auto qn = wfloat("attn_q_a_norm"), sk = wfloat("attn_sinks"), an = wfloat("attn_norm");
    Dev<float> dqn(qn), dsk(sk);

    QPathWeights qw{wq_a.p, W["attn_q_a"].type, dqn.p, wq_b.p, W["attn_q_b"].type, 9.99999968e-21f};
    AttnOutWeights ow{wo_a.p, W["attn_output_a"].type, wo_b.p, W["attn_output_b"].type};
    const RopeTable rope = ratio ? rope_table_compressed() : rope_table_window();
    const RopeTable crope = rope_table_compressed();

    const int cap = 512;
    std::vector<float> kvw((size_t) cap * 512);
    std::vector<float> comp;   // compressed rows, rope applied
    cudaStream_t s; cudaStreamCreate(&s);
    Dev<float> dwin((size_t) cap * 512), dcomp((size_t) 512 * 512), dq(8 * 32768), dqr(8 * 1280), dattn(8 * 32768),
        dout(8 * 5120), doa(attn_out_scratch_floats(8)), dx(8 * 5120), dws(sparse_mqa_workspace_floats(8, 256) + 64);
    Dev<char> dxq(std::max(attn_out_scratch_bytes(8), (size_t) 8 * 5120 / 32 * 36)), dxq2(q_path_scratch_bytes(8));
    Dev<int32_t> dpos(8), dstart(8), dn(8), dlim(8), dsel(8 * 256);
    Dev<float> dref(8 * 32768);

    int n_pos = 0;
    int n_comp_rows = 0;
    double worst_derope = 0, worst_out = 0, worst_out_iso = 0;
    for (int step = 0; step <= n_steps; ++step) {
        const int T = step == 0 ? n_prompt : 1;
        auto kv = fx(step, "kv", layer);
        std::memcpy(&kvw[(size_t) n_pos * 512], kv.data(), kv.size() * 4);
        if (ratio) {
            auto cl = fx(step, "comp_latent", comp_src);
            // a decode step dumps its latent every step; the row exists only when a block completes
            const int nr = (step > 0 && (n_pos + 1) % ratio != 0) ? 0 : (int) (cl.size() / 512);
            std::vector<int32_t> cp(nr);
            for (int i = 0; i < nr; ++i) cp[i] = (n_comp_rows + i) * ratio;
            if (nr) {
            Dev<int32_t> dcp(cp);
            std::vector<float> rows(cl.begin(), cl.begin() + (size_t) nr * 512);
            Dev<float> dr(rows);
            rope_rows(dr.p, nr, 512, 1, dcp.p, crope, false, s);
            cudaStreamSynchronize(s);
            auto rr = dr.down();
            comp.insert(comp.end(), rr.begin(), rr.end());
            n_comp_rows += nr;
            cudaMemcpy(dcomp.p, comp.data(), comp.size() * 4, cudaMemcpyHostToDevice);
            }
        }
        cudaMemcpy(dwin.p, kvw.data(), (size_t) (n_pos + T) * 512 * 4, cudaMemcpyHostToDevice);
        auto x = fx(step, "hc_attn_pre", layer);
        auto top = ratio ? fx(step, "idx_top_k", comp_src) : std::vector<float>();
        const int n_sel = ratio ? 256 : 0;
        auto ref_d = fx(step, "attn_derope", layer), ref_o = fx(step, "attn_out", layer);
        std::vector<float> got_d, got_o, got_oi;
        for (int c0 = 0; c0 < T; c0 += chunk) {
            const int n = std::min(chunk, T - c0);
            std::vector<float> xn((size_t) n * 5120);
            for (int t = 0; t < n; ++t) {
                const float* r = &x[(size_t) (c0 + t) * 5120];
                double ss = 0;
                for (int i = 0; i < 5120; ++i) ss += (double) r[i] * r[i];
                const float sc = 1.0f / std::sqrt((float) (ss / 5120) + 9.99999968e-21f);
                for (int i = 0; i < 5120; ++i) xn[(size_t) t * 5120 + i] = r[i] * sc * an[i];
            }
            dx.up(xn);   // note: sized for 8 tokens; n <= 8
            std::vector<int32_t> pos(n), st(n), wn(n), lim(n), sel((size_t) n * std::max(n_sel, 1));
            for (int t = 0; t < n; ++t) {
                const int p = n_pos + c0 + t + (step ? shift : 0);
                pos[t] = p; st[t] = std::max(0, p - 127); wn[t] = p - st[t] + 1; lim[t] = ratio ? (p + 1) / ratio : 0;
                for (int i = 0; i < n_sel; ++i) {
                    int32_t id;
                    std::memcpy(&id, &top[(size_t) (c0 + t) * 256 + i], 4);
                    sel[(size_t) t * n_sel + i] = id;
                }
            }
            cudaMemcpy(dpos.p, pos.data(), n * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dstart.p, st.data(), n * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dn.p, wn.data(), n * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dlim.p, lim.data(), n * 4, cudaMemcpyHostToDevice);
            if (n_sel) cudaMemcpy(dsel.p, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice);
            basalt::kernels::native_quantize_q8_1(dx.p, dxq.p, 5120, n, s);
            q_path(dxq.p, qw, n, dpos.p, rope, dxq2.p, dqr.p, dq.p, s);
            MqaArgs a;
            a.n_tokens = n; a.q = dq.p; a.win = dwin.p; a.win_cap = cap; a.win_start = dstart.p; a.win_n = dn.p;
            a.comp = dcomp.p; a.comp_stride = 512; a.sel = n_sel ? dsel.p : nullptr; a.n_sel = n_sel;
            a.comp_limit = dlim.p; a.sinks = dsk.p; a.pos = dpos.p; a.rope = rope; a.out = dattn.p;
            sparse_mqa(a, dws.p, s);
            attn_out(dattn.p, ow, n, dxq.p, doa.p, dout.p, s);
            dstest::check(cudaStreamSynchronize(s), "run");
            auto d = dattn.down(); auto o = dout.down();
            got_d.insert(got_d.end(), d.begin(), d.begin() + (size_t) n * 32768);
            got_o.insert(got_o.end(), o.begin(), o.begin() + (size_t) n * 5120);
            // attn_out alone, from the fixture's attn_derope
            dref.up(std::vector<float>(ref_d.begin() + (size_t) c0 * 32768, ref_d.begin() + (size_t) (c0 + n) * 32768));
            for (int t = 0; t < n; ++t) attn_out(dref.p + (size_t) t * 32768, ow, 1, dxq.p, doa.p, dout.p + (size_t) t * 5120, s);
            cudaStreamSynchronize(s);
            auto oi = dout.down();
            got_oi.insert(got_oi.end(), oi.begin(), oi.begin() + (size_t) n * 5120);
        }
        const auto ed = dstest::compare(got_d, ref_d), eo = dstest::compare(got_o, ref_o), ei = dstest::compare(got_oi, ref_o);
        std::printf("step %2d T=%2d  derope rms_rel %.2e  out rms_rel %.2e  out(from fixture derope) rms_rel %.2e\n", step, T,
                    ed.rms_rel, eo.rms_rel, ei.rms_rel);
        if (shift || chunk != 8) {
            for (int t = 0; t < T; ++t) {
                std::vector<float> a1(got_d.begin() + (size_t) t * 32768, got_d.begin() + (size_t) (t + 1) * 32768),
                    b1(ref_d.begin() + (size_t) t * 32768, ref_d.begin() + (size_t) (t + 1) * 32768);
                std::vector<float> a2(got_oi.begin() + (size_t) t * 5120, got_oi.begin() + (size_t) (t + 1) * 5120),
                    b2(ref_o.begin() + (size_t) t * 5120, ref_o.begin() + (size_t) (t + 1) * 5120);
                std::printf("  tok %2d derope %.2e out_iso %.2e  ref_o[0]=%g got_oi[0]=%g d: %g/%g %g/%g %g/%g\n", t,
                            dstest::compare(a1, b1).rms_rel, dstest::compare(a2, b2).rms_rel, b2[0], a2[0], a1[0], b1[0], a1[1000], b1[1000], a1[20000], b1[20000]);
            }
        }
        worst_derope = std::max(worst_derope, ed.rms_rel);
        worst_out = std::max(worst_out, eo.rms_rel);
        worst_out_iso = std::max(worst_out_iso, ei.rms_rel);
        n_pos += T;
    }
    std::printf("layer %d ratio %d worst rms_rel: derope %.2e out %.2e out-from-fixture %.2e\n", layer, ratio,
                worst_derope, worst_out, worst_out_iso);
    return 0;
}
