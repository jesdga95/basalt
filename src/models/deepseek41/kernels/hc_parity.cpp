// Parity test of the DeepSeek-V4.1 hyper-connection kernels (mixes with Sinkhorn, pre collapse, post expansion):
// random inputs at the real geometry against a double-precision host reference, independence from the window
// width and position, the shared RMS norm at this model's epsilon, a microbench, and (with --fixture DIR) real
// layers chained through the single-pass shift against the per-layer fixtures.
#include "basalt/kernels/elementwise.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cstring>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int D = 5120, FLAT = 4 * D;

// coef[24] for one token in double.
std::vector<double> ref_coef(const float* x, const std::vector<float>& fn, const std::vector<float>& scale,
                             const std::vector<float>& base, const HcArgs& a) {
    double ss = 0;
    for (int i = 0; i < FLAT; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / FLAT + a.rms_eps);
    double mix[24];
    for (int m = 0; m < 24; ++m) {
        double acc = 0;
        for (int i = 0; i < FLAT; ++i) acc += (double) fn[(size_t) m * FLAT + i] * x[i];
        mix[m] = acc * rs;
    }
    std::vector<double> c(24);
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    for (int i = 0; i < 4; ++i) {
        c[i] = sig(mix[i] * scale[0] + base[i]) + a.hc_eps;
        c[4 + i] = 2.0 * sig(mix[4 + i] * scale[1] + base[4 + i]);
    }
    double m4[4][4];   // [src][dst]
    for (int s = 0; s < 4; ++s) {
        double mx = -1e300, sum = 0;
        for (int d = 0; d < 4; ++d) { m4[s][d] = mix[8 + s * 4 + d] * scale[2] + base[8 + s * 4 + d]; mx = std::max(mx, m4[s][d]); }
        for (int d = 0; d < 4; ++d) { m4[s][d] = std::exp(m4[s][d] - mx); sum += m4[s][d]; }
        for (int d = 0; d < 4; ++d) m4[s][d] = m4[s][d] / sum + a.hc_eps;
    }
    auto cols = [&] {
        for (int d = 0; d < 4; ++d) {
            double sum = 0;
            for (int s = 0; s < 4; ++s) sum += m4[s][d];
            for (int s = 0; s < 4; ++s) m4[s][d] /= sum + a.hc_eps;
        }
    };
    auto rows = [&] {
        for (int s = 0; s < 4; ++s) {
            double sum = 0;
            for (int d = 0; d < 4; ++d) sum += m4[s][d];
            for (int d = 0; d < 4; ++d) m4[s][d] /= sum + a.hc_eps;
        }
    };
    cols();
    for (int i = 1; i < a.sinkhorn_iters; ++i) { rows(); cols(); }
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < 4; ++d) c[8 + s * 4 + d] = m4[s][d];
    return c;
}

struct Weights {
    std::vector<float> fn, scale, base;
};

struct DevW {
    Dev<float> fn, scale, base;
    explicit DevW(const Weights& w) : fn(w.fn), scale(w.scale), base(w.base) {}
};

std::vector<float> run_coef(const DevW& w, const std::vector<float>& x, int T, const HcArgs& a) {
    Dev<float> dx(x), ws(hc_mixes_workspace_floats(T)), coef((size_t) T * kHcCoef);
    hc_mixes(dx.p, w.fn.p, w.scale.p, w.base.p, T, a, ws.p, coef.p, nullptr);
    return coef.down();
}

std::vector<float> run_pre(const std::vector<float>& x, const std::vector<float>& pre, int stride, int T) {
    Dev<float> dx(x), dp(pre), out((size_t) T * D);
    hc_pre(dx.p, dp.p, stride, T, D, out.p, nullptr);
    return out.down();
}

std::vector<float> run_post(const std::vector<float>& f, const std::vector<float>& x, const std::vector<float>& coef,
                            int T) {
    Dev<float> df(f), dx(x), dc(coef), out((size_t) T * FLAT);
    hc_post(df.p, dx.p, dc.p, kHcCoef, T, D, out.p, nullptr);
    return out.down();
}

std::vector<double> ref_pre(const std::vector<float>& x, const std::vector<double>& pre, int t) {
    std::vector<double> o(D);
    for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int s = 0; s < 4; ++s) acc += pre[s] * (double) x[(size_t) t * FLAT + (size_t) s * D + d];
        o[d] = acc;
    }
    return o;
}

std::vector<double> ref_post(const std::vector<float>& f, const std::vector<float>& x, const std::vector<double>& c, int t) {
    std::vector<double> o(FLAT);
    for (int dst = 0; dst < 4; ++dst)
        for (int d = 0; d < D; ++d) {
            double acc = c[4 + dst] * (double) f[(size_t) t * D + d];
            for (int s = 0; s < 4; ++s) acc += c[8 + s * 4 + dst] * (double) x[(size_t) t * FLAT + (size_t) s * D + d];
            o[(size_t) dst * D + d] = acc;
        }
    return o;
}

bool load_weights(const std::string& dir, int L, const char* kind, Weights& w) {
    const std::string b = dir + "/w.blk." + std::to_string(L) + ".hc_" + kind + "_";
    return read_vec(b + "fn.weight.bin", w.fn) && read_vec(b + "scale.weight.bin", w.scale) &&
           read_vec(b + "base.weight.bin", w.base);
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
    Verdict v;
    HcArgs a;

    // ---- random inputs
    Weights w;
    w.fn = randn((size_t) 24 * FLAT, 11, 0.01f);
    w.scale = {0.4f, 0.3f, 0.5f};
    w.base = randn(24, 12, 0.5f);
    DevW dw(w);
    const int Tmax = 9;
    auto x_h = randn((size_t) Tmax * FLAT, 13, 1.5f);
    auto f_h = randn((size_t) Tmax * D, 14);
    std::vector<float> coef = run_coef(dw, x_h, Tmax, a);
    double ce = 0, pre_e = 0, post_e = 0, pre_n = 0, post_n = 0, col_dev = 0;
    for (int t = 0; t < Tmax; ++t) {
        auto rc = ref_coef(&x_h[(size_t) t * FLAT], w.fn, w.scale, w.base, a);
        for (int i = 0; i < 24; ++i) ce = std::max(ce, std::fabs(coef[(size_t) t * 24 + i] - rc[i]));
        for (int d = 0; d < 4; ++d) {
            double s = 0;
            for (int sr = 0; sr < 4; ++sr) s += coef[(size_t) t * 24 + 8 + sr * 4 + d];
            col_dev = std::max(col_dev, std::fabs(s - 1.0));
        }
    }
    auto pre_dev = run_pre(x_h, coef, kHcCoef, Tmax);
    auto post_dev = run_post(f_h, x_h, coef, Tmax);
    for (int t = 0; t < Tmax; ++t) {
        std::vector<double> cd(coef.begin() + (size_t) t * 24, coef.begin() + (size_t)(t + 1) * 24);
        auto rp = ref_pre(x_h, cd, t);
        auto ro = ref_post(f_h, x_h, cd, t);
        for (int d = 0; d < D; ++d) { pre_e = std::max(pre_e, std::fabs(pre_dev[(size_t) t * D + d] - rp[d])); pre_n = std::max(pre_n, std::fabs(rp[d])); }
        for (int i = 0; i < FLAT; ++i) { post_e = std::max(post_e, std::fabs(post_dev[(size_t) t * FLAT + i] - ro[i])); post_n = std::max(post_n, std::fabs(ro[i])); }
    }
    std::printf("random T=%d: coef max abs %.3g (column sums within %.3g of 1); hc_pre max abs %.3g (max |v| %.2f); "
                "hc_post max abs %.3g (max |v| %.2f)\n", Tmax, ce, col_dev, pre_e, pre_n, post_e, post_n);
    v.expect(ce < 2e-5, "coef error");
    v.expect(pre_e < 1e-4 * pre_n, "hc_pre error");
    v.expect(post_e < 1e-4 * post_n, "hc_post error");

    // ---- window width and position independence (bitwise)
    bool bitwise = true;
    for (int T : {1, 2, 3, 4})
        for (int start : {0, 2, 5}) {
            if (start + T > Tmax) continue;
            std::vector<float> xs(x_h.begin() + (size_t) start * FLAT, x_h.begin() + (size_t)(start + T) * FLAT);
            std::vector<float> fs(f_h.begin() + (size_t) start * D, f_h.begin() + (size_t)(start + T) * D);
            auto c = run_coef(dw, xs, T, a);
            bitwise = bitwise && !std::memcmp(c.data(), &coef[(size_t) start * 24], (size_t) T * 24 * 4);
            auto p = run_pre(xs, c, kHcCoef, T);
            bitwise = bitwise && !std::memcmp(p.data(), &pre_dev[(size_t) start * D], (size_t) T * D * 4);
            auto o = run_post(fs, xs, c, T);
            bitwise = bitwise && !std::memcmp(o.data(), &post_dev[(size_t) start * FLAT], (size_t) T * FLAT * 4);
        }
    std::printf("window/position independence: %s\n", bitwise ? "bitwise identical" : "DIFFERS");
    v.expect(bitwise, "output depends on window width or position");

    // ---- the shared RMS norm at this model's epsilon (a tiny eps must not change the result)
    {
        auto g = randn(D, 15, 1.f, 1.f);
        auto xr = randn((size_t) 3 * D, 16, 2.f);
        Dev<float> dx(xr), dg(g);
        basalt::kernels::rms_norm_weighted(dx.p, dg.p, 3, D, 1e-20f, nullptr);
        auto got = dx.down();
        std::vector<double> ref(3 * (size_t) D);
        for (int r = 0; r < 3; ++r) {
            double ss = 0;
            for (int d = 0; d < D; ++d) ss += (double) xr[(size_t) r * D + d] * xr[(size_t) r * D + d];
            const double rs = 1.0 / std::sqrt(ss / D + 1e-20);
            for (int d = 0; d < D; ++d) ref[(size_t) r * D + d] = xr[(size_t) r * D + d] * rs * g[d];
        }
        Err e = compare(got, ref, 1e-2);
        std::printf("rms_norm_weighted eps 1e-20: max abs %.3g rel rms %.3g\n", e.max_abs, e.rms_rel);
        v.expect(e.rms_rel < 1e-6, "rms norm at tiny eps");
    }

    // ---- microbench
    for (int T : {1, 4}) {
        std::vector<float> xs(x_h.begin(), x_h.begin() + (size_t) T * FLAT), fs(f_h.begin(), f_h.begin() + (size_t) T * D);
        Dev<float> dx(xs), df(fs), ws(hc_mixes_workspace_floats(T)), dc((size_t) T * 24), o1((size_t) T * D), o2((size_t) T * FLAT);
        char name[64];
        std::snprintf(name, sizeof name, "hc_mixes T=%d", T);
        bench_line(name, bench_stream_us([&](void* st) { hc_mixes(dx.p, dw.fn.p, dw.scale.p, dw.base.p, T, a, ws.p, dc.p, st); }),
                   24.0 * FLAT * 4 + (double) T * FLAT * 4);
        std::snprintf(name, sizeof name, "hc_pre T=%d", T);
        bench_line(name, bench_stream_us([&](void* st) { hc_pre(dx.p, dc.p, kHcCoef, T, D, o1.p, st); }), (double) T * (FLAT + D) * 4);
        std::snprintf(name, sizeof name, "hc_post T=%d", T);
        bench_line(name, bench_stream_us([&](void* st) { hc_post(df.p, dx.p, dc.p, kHcCoef, T, D, o2.p, st); }), (double) T * (D + 2 * FLAT) * 4);
    }

    // ---- fixtures: layer L chained through the single-pass shift (attention collapses with the previous
    // sublayer's pre; the FFN with the attention sublayer's pre of its own layer)
    if (!fixture.empty()) {
        for (int L : {2, 14}) {
            Weights wa, wf, wfp;
            if (!load_weights(fixture, L, "attn", wa) || !load_weights(fixture, L, "ffn", wf) ||
                !load_weights(fixture, L - 1, "ffn", wfp)) {
                std::printf("fixture weights for layer %d missing\n", L);
                v.failures++;
                continue;
            }
            DevW da(wa), df(wf), dfp(wfp);
            for (const char* p : {"p1", "p2", "p3"})
                for (int step : {0, 1}) {
                    const std::string base = fixture + "/" + p + ".s" + std::to_string(step) + ".";
                    auto fx = [&](const char* name, int layer, std::vector<float>& out) {
                        return read_vec(base + name + "." + std::to_string(layer) + ".bin", out);
                    };
                    std::vector<float> x_in, prev_post, attn_pre, attn_out, attn_post, ffn_pre, moe, shexp, l_out;
                    bool ok = fx(L == 14 ? "engram_out" : "l_out", L == 14 ? 14 : L - 1, x_in) &&
                              fx("hc_attn_post", L - 1, prev_post) && fx("hc_attn_pre", L, attn_pre) &&
                              fx("attn_out", L, attn_out) && fx("hc_attn_post", L, attn_post) &&
                              fx("hc_ffn_pre", L, ffn_pre) && fx("ffn_moe_out", L, moe) && fx("ffn_shexp", L, shexp) &&
                              fx("l_out", L, l_out);
                    if (!ok) { std::printf("fixture %s step %d layer %d missing\n", p, step, L); v.failures++; continue; }
                    const int T = (int) (x_in.size() / FLAT);
                    auto c_attn = run_coef(da, x_in, T, a);
                    auto c_prev = run_coef(dfp, prev_post, T, a);
                    auto got_attn_pre = run_pre(x_in, c_prev, kHcCoef, T);
                    auto got_attn_post = run_post(attn_out, x_in, c_attn, T);
                    auto got_ffn_pre = run_pre(attn_post, c_attn, kHcCoef, T);
                    auto c_ffn = run_coef(df, attn_post, T, a);
                    std::vector<float> ffn_out(moe.size());
                    for (size_t i = 0; i < ffn_out.size(); ++i) ffn_out[i] = moe[i] + shexp[i];
                    auto got_l_out = run_post(ffn_out, attn_post, c_ffn, T);
                    Err e1 = compare(got_attn_pre, attn_pre), e2 = compare(got_attn_post, attn_post),
                        e3 = compare(got_ffn_pre, ffn_pre), e4 = compare(got_l_out, l_out);
                    std::printf("fixture %s s%d L%-2d T=%2d: rel rms attn_pre %.2g, attn_post %.2g, ffn_pre %.2g, l_out %.2g "
                                "(max abs %.2g %.2g %.2g %.2g)\n", p, step, L, T, e1.rms_rel, e2.rms_rel, e3.rms_rel, e4.rms_rel,
                                e1.max_abs, e2.max_abs, e3.max_abs, e4.max_abs);
                    for (double r : {e1.rms_rel, e2.rms_rel, e3.rms_rel, e4.rms_rel}) v.expect(r < 1e-3, "fixture rel rms");
                }
        }
    }
    std::printf(v.failures ? "hc_parity: FAILED\n" : "hc_parity: ok\n");
    return v.failures ? 1 : 0;
}
