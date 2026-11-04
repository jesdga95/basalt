// Parity test of the DeepSeek-V4.1 router: random inputs at the real geometry against a double-precision host
// reference, independence from the window width and position, tie order, a microbench, and (with --fixture DIR)
// real layers against the per-layer fixtures.
#include "basalt/models/deepseek41/kernels.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int NE = 384, ND = 5120, K = 6;

struct Ref {
    std::vector<double> logits, biased;
    std::vector<int> ids;
    std::vector<double> w;
};

Ref reference(const float* x, const std::vector<float>& gate, const std::vector<float>& bias, float scale) {
    Ref r;
    r.logits.resize(NE);
    r.biased.resize(NE);
    std::vector<double> s(NE);
    for (int e = 0; e < NE; ++e) {
        double acc = 0;
        for (int d = 0; d < ND; ++d) acc += (double) gate[(size_t) e * ND + d] * (double) x[d];
        r.logits[e] = acc;
        s[e] = std::sqrt(std::log1p(std::exp(acc)));
        r.biased[e] = s[e] + (double) bias[e];
    }
    std::vector<int> idx(NE);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return r.biased[a] > r.biased[b]; });
    double sum = 0;
    for (int k = 0; k < K; ++k) { r.ids.push_back(idx[k]); sum += s[idx[k]]; }
    for (int k = 0; k < K; ++k) r.w.push_back(s[idx[k]] / (sum + 1e-20) * scale);
    return r;
}

struct Out {
    std::vector<float> logits;
    std::vector<int32_t> ids;
    std::vector<float> w;
};

Out run(const Dev<float>& gate, const Dev<float>& bias, const std::vector<float>& x, int T, const RouterArgs& a) {
    Dev<float> dx(x), dl((size_t) T * NE), dw((size_t) T * K);
    Dev<int32_t> di((size_t) T * K);
    router(dx.p, gate.p, bias.p, T, a, dl.p, di.p, dw.p, nullptr);
    return {dl.down(), di.down(), dw.down()};
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
    Verdict v;
    RouterArgs a;

    // ---- random inputs vs the double reference
    auto gate_h = randn((size_t) NE * ND, 1, 0.02f);
    auto bias_h = randn(NE, 2, 0.05f);
    // two identical experts: equal logits and equal bias must pick the lower index first
    std::memcpy(&gate_h[(size_t) 201 * ND], &gate_h[(size_t) 77 * ND], ND * sizeof(float));
    bias_h[201] = bias_h[77];
    Dev<float> gate(gate_h), bias(bias_h);

    const int Tmax = 11;
    auto x_h = randn((size_t) Tmax * ND, 3);
    Out all = run(gate, bias, x_h, Tmax, a);
    double worst_logit = 0, worst_w = 0, rms_num = 0, rms_den = 0;
    int near_ties = 0, id_mismatch = 0;
    for (int t = 0; t < Tmax; ++t) {
        Ref r = reference(&x_h[(size_t) t * ND], gate_h, bias_h, a.weight_scale);
        for (int e = 0; e < NE; ++e) {
            const double d = std::fabs(all.logits[(size_t) t * NE + e] - r.logits[e]);
            worst_logit = std::max(worst_logit, d);
            rms_num += d * d;
            rms_den += r.logits[e] * r.logits[e];
        }
        bool same = true;
        for (int k = 0; k < K; ++k) same = same && all.ids[(size_t) t * K + k] == r.ids[k];
        if (same) {
            for (int k = 0; k < K; ++k) worst_w = std::max(worst_w, std::fabs(all.w[(size_t) t * K + k] - r.w[k]) / r.w[k]);
        } else {
            // only a near-tie of the biased scores may reorder or swap experts
            ++id_mismatch;
            std::vector<double> gap;
            for (int k = 0; k < K; ++k) {
                const int g = all.ids[(size_t) t * K + k];
                if (std::find(r.ids.begin(), r.ids.end(), g) == r.ids.end()) gap.push_back(std::fabs(r.biased[g] - r.biased[r.ids[K - 1]]));
            }
            bool tie = true;
            for (double g : gap) tie = tie && g < 1e-5;
            if (tie) ++near_ties;
        }
    }
    std::printf("random T=%d: logits max abs %.3g rel rms %.3g; weights max rel %.3g; id differences %d (near-ties %d)\n",
                Tmax, worst_logit, std::sqrt(rms_num / rms_den), worst_w, id_mismatch, near_ties);
    v.expect(worst_logit < 1e-4, "logits abs error");
    v.expect(worst_w < 1e-4, "weights rel error");
    v.expect(id_mismatch == near_ties, "ids differ beyond near-ties");

    // ---- window width and position independence: bitwise
    bool bitwise = true;
    for (int T : {1, 2, 3, 4}) {
        for (int start : {0, 3, 7}) {
            if (start + T > Tmax) continue;
            std::vector<float> xs(x_h.begin() + (size_t) start * ND, x_h.begin() + (size_t)(start + T) * ND);
            Out o = run(gate, bias, xs, T, a);
            for (int t = 0; t < T; ++t) {
                bitwise = bitwise && !std::memcmp(&o.logits[(size_t) t * NE], &all.logits[(size_t)(start + t) * NE], NE * 4);
                bitwise = bitwise && !std::memcmp(&o.ids[(size_t) t * K], &all.ids[(size_t)(start + t) * K], K * 4);
                bitwise = bitwise && !std::memcmp(&o.w[(size_t) t * K], &all.w[(size_t)(start + t) * K], K * 4);
            }
        }
    }
    std::printf("window/position independence: %s\n", bitwise ? "bitwise identical" : "DIFFERS");
    v.expect(bitwise, "output depends on window width or position");

    // ---- tie order: experts 77 and 201 are identical; a token that selects either selects 77 first
    {
        std::vector<float> xs(x_h.begin(), x_h.begin() + ND);
        std::vector<float> tie_bias(NE, -10.f);
        tie_bias[201] = tie_bias[77] = 5.f;
        Dev<float> tb(tie_bias);
        Out o = run(gate, tb, xs, 1, a);
        std::printf("tie: top ids %d %d\n", o.ids[0], o.ids[1]);
        v.expect(o.ids[0] == 77 && o.ids[1] == 201, "tie did not resolve to the lower index first");
    }

    // ---- microbench
    for (int T : {1, 4}) {
        Dev<float> dx(std::vector<float>(x_h.begin(), x_h.begin() + (size_t) T * ND)), dl((size_t) T * NE), dw((size_t) T * K);
        Dev<int32_t> di((size_t) T * K);
        const double us = bench_stream_us([&](void* st) { router(dx.p, gate.p, bias.p, T, a, dl.p, di.p, dw.p, st); });
        char name[64];
        std::snprintf(name, sizeof name, "router T=%d", T);
        bench_line(name, us, (double) NE * ND * 4 + (double) T * ND * 4);
    }

    // ---- cold-gate microbench: the gate cycles through copies far larger than L2, as the layers do
    {
        constexpr int kCopies = 16;
        std::vector<float*> gates(kCopies);
        for (auto& g : gates) {
            check(cudaMalloc(&g, (size_t) NE * ND * 4), "cudaMalloc");
            check(cudaMemcpy(g, gate.p, (size_t) NE * ND * 4, cudaMemcpyDeviceToDevice), "copy");
        }
        for (int T : {1, 4}) {
            Dev<float> dx(std::vector<float>(x_h.begin(), x_h.begin() + (size_t) T * ND)), dl((size_t) T * NE), dw((size_t) T * K);
            Dev<int32_t> di((size_t) T * K);
            int i = 0;
            const double us = bench_stream_us([&](void* st) { router(dx.p, gates[i++ % kCopies], bias.p, T, a, dl.p, di.p, dw.p, st); });
            char name[64];
            std::snprintf(name, sizeof name, "router cold gate T=%d", T);
            bench_line(name, us, (double) NE * ND * 4 + (double) T * ND * 4);
        }
        for (auto g : gates) cudaFree(g);
    }

    // ---- fixtures: ffn input = rms_norm(hc_ffn_pre) * ffn_norm.weight, router weights from the pack
    if (!fixture.empty()) {
        for (int L : {1, 2, 13, 14}) {
            std::vector<float> g, b, nw;
            if (!read_vec(fixture + "/w.blk." + std::to_string(L) + ".ffn_gate_inp.weight.bin", g) ||
                !read_vec(fixture + "/w.blk." + std::to_string(L) + ".exp_probs_b.bias.bin", b) ||
                !read_vec(fixture + "/w.blk." + std::to_string(L) + ".ffn_norm.weight.bin", nw)) {
                std::printf("fixture weights for layer %d missing\n", L);
                v.failures++;
                continue;
            }
            Dev<float> dg(g), db(b);
            for (const char* p : {"p1", "p2", "p3"}) {
                for (int step : {0, 1}) {
                    const std::string base = fixture + "/" + p + ".s" + std::to_string(step) + ".";
                    const std::string sfx = "." + std::to_string(L) + ".bin";
                    std::vector<float> in, flog, fw;
                    std::vector<int32_t> fid;
                    if (!read_vec(base + "hc_ffn_pre" + sfx, in) || !read_vec(base + "ffn_moe_logits" + sfx, flog) ||
                        !read_vec(base + "ffn_moe_topk" + sfx, fid) || !read_vec(base + "ffn_moe_weights_scaled" + sfx, fw)) {
                        std::printf("fixture %s step %d layer %d missing\n", p, step, L);
                        v.failures++;
                        continue;
                    }
                    const int T = (int) (in.size() / ND);
                    for (int t = 0; t < T; ++t) {
                        double ss = 0;
                        for (int d = 0; d < ND; ++d) ss += (double) in[(size_t) t * ND + d] * in[(size_t) t * ND + d];
                        const double rs = 1.0 / std::sqrt(ss / ND + 1e-20);
                        for (int d = 0; d < ND; ++d) in[(size_t) t * ND + d] = (float) (in[(size_t) t * ND + d] * rs * nw[d]);
                    }
                    Out o = run(dg, db, in, T, a);
                    Err le = compare(o.logits, flog, 1e-2);
                    int id_diff = 0;
                    double wr = 0;
                    for (int t = 0; t < T; ++t) {
                        // compared as (id, weight) sets: the fixture's order inside the top-k is not specified
                        std::vector<std::pair<int, float>> mine, ref;
                        for (int k = 0; k < K; ++k) {
                            mine.push_back({o.ids[(size_t) t * K + k], o.w[(size_t) t * K + k]});
                            ref.push_back({fid[(size_t) t * K + k], fw[(size_t) t * K + k]});
                        }
                        std::sort(mine.begin(), mine.end());
                        std::sort(ref.begin(), ref.end());
                        bool same = true;
                        for (int k = 0; k < K; ++k) same = same && mine[k].first == ref[k].first;
                        if (!same) {
                            ++id_diff;
                            continue;
                        }
                        for (int k = 0; k < K; ++k) wr = std::max(wr, (double) std::fabs(mine[k].second - ref[k].second) / ref[k].second);
                    }
                    std::printf("fixture %s s%d L%-2d T=%2d: logits rel rms %.3g max abs %.3g; ids differ on %d/%d tokens; weights max rel %.3g\n",
                                p, step, L, T, le.rms_rel, le.max_abs, id_diff, T, wr);
                    v.expect(le.rms_rel < (T > 1 ? 1e-3 : 1e-5), "fixture logits rel rms");
                    v.expect(id_diff * 20 <= T + 19, "fixture ids differ on too many tokens");
                    v.expect(wr < 1e-3, "fixture weights");
                }
            }
        }
    }
    std::printf(v.failures ? "router_parity: FAILED\n" : "router_parity: ok\n");
    return v.failures ? 1 : 0;
}
