// Parity test of the DeepSeek-V4.1 row norms: rms_norm_rows and hc_pre_rms_norm at the real width against a
// double-precision host reference, hc_pre_rms_norm against hc_pre (bitwise for the collapsed row), independence
// from the window width and position, and a microbench.
#include "basalt/models/deepseek41/kernels.hpp"
#include "test_util.hpp"

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

constexpr int ND = 5120;
constexpr float kEps = 1e-6f;

std::vector<double> ref_norm(const float* x, const std::vector<float>& w) {
    double ss = 0;
    for (int d = 0; d < ND; ++d) ss += (double) x[d] * x[d];
    const double rs = 1.0 / std::sqrt(ss / ND + (double) kEps);
    std::vector<double> o(ND);
    for (int d = 0; d < ND; ++d) o[d] = (double) x[d] * rs * w[d];
    return o;
}

}  // namespace

int main() {
    Verdict v;
    const int Tmax = 9;
    auto x_h = randn((size_t) Tmax * ND, 1, 2.f, 0.3f);
    auto w_h = randn(ND, 2, 0.5f, 1.f);
    auto s_h = randn((size_t) Tmax * kHcMult * ND, 3);
    auto pre_h = randn((size_t) Tmax * kHcCoef, 4, 0.5f, 0.5f);
    Dev<float> x(x_h), w(w_h), s(s_h), pre(pre_h);

    // ---- rms_norm_rows vs the double reference
    Dev<float> out((size_t) Tmax * ND);
    rms_norm_rows(x.p, w.p, Tmax, ND, kEps, out.p, nullptr);
    auto got = out.down();
    double worst = 0;
    for (int t = 0; t < Tmax; ++t) {
        auto r = ref_norm(&x_h[(size_t) t * ND], w_h);
        Err e = compare(std::vector<float>(got.begin() + (size_t) t * ND, got.begin() + (size_t)(t + 1) * ND), r, 1e-3);
        worst = std::fmax(worst, e.max_rel);
    }
    std::printf("rms_norm_rows T=%d: max rel %.3g\n", Tmax, worst);
    v.expect(worst < 1e-5, "rms_norm_rows error");

    // ---- hc_pre_rms_norm: collapsed row bitwise equal to hc_pre, normed row vs the double reference
    Dev<float> cur((size_t) Tmax * ND), nrm((size_t) Tmax * ND), cur_ref((size_t) Tmax * ND);
    hc_pre_rms_norm(s.p, pre.p, kHcCoef, w.p, Tmax, ND, kEps, cur.p, nrm.p, nullptr);
    hc_pre(s.p, pre.p, kHcCoef, Tmax, ND, cur_ref.p, nullptr);
    auto cur_h = cur.down(), cur_r = cur_ref.down(), nrm_h = nrm.down();
    const bool same = !std::memcmp(cur_h.data(), cur_r.data(), cur_h.size() * 4);
    std::printf("hc_pre_rms_norm: collapsed row %s hc_pre\n", same ? "bitwise equal to" : "DIFFERS from");
    v.expect(same, "collapsed row differs from hc_pre");
    worst = 0;
    for (int t = 0; t < Tmax; ++t) {
        auto r = ref_norm(&cur_r[(size_t) t * ND], w_h);
        Err e = compare(std::vector<float>(nrm_h.begin() + (size_t) t * ND, nrm_h.begin() + (size_t)(t + 1) * ND), r, 1e-3);
        worst = std::fmax(worst, e.max_rel);
    }
    std::printf("hc_pre_rms_norm normed row: max rel %.3g\n", worst);
    v.expect(worst < 1e-5, "hc_pre_rms_norm error");

    // ---- window width and position independence: bitwise
    bool bitwise = true;
    for (int T : {1, 2, 4}) {
        for (int start : {0, 3, 5}) {
            Dev<float> o1((size_t) T * ND), o2((size_t) T * ND), c2((size_t) T * ND);
            rms_norm_rows(x.p + (size_t) start * ND, w.p, T, ND, kEps, o1.p, nullptr);
            hc_pre_rms_norm(s.p + (size_t) start * kHcMult * ND, pre.p + (size_t) start * kHcCoef, kHcCoef, w.p, T, ND, kEps,
                            c2.p, o2.p, nullptr);
            auto a = o1.down(), b = o2.down();
            bitwise = bitwise && !std::memcmp(a.data(), got.data() + (size_t) start * ND, a.size() * 4);
            bitwise = bitwise && !std::memcmp(b.data(), nrm_h.data() + (size_t) start * ND, b.size() * 4);
        }
    }
    std::printf("window/position independence: %s\n", bitwise ? "bitwise identical" : "DIFFERS");
    v.expect(bitwise, "output depends on window width or position");

    // ---- microbench (the rows are L2-resident across calls)
    for (int T : {1, 4}) {
        char name[64];
        double us = bench_stream_us([&](void* st) { rms_norm_rows(x.p, w.p, T, ND, kEps, out.p, st); });
        std::snprintf(name, sizeof name, "rms_norm_rows T=%d", T);
        bench_line(name, us, (double) T * ND * 8 + ND * 4);
        us = bench_stream_us([&](void* st) { hc_pre_rms_norm(s.p, pre.p, kHcCoef, w.p, T, ND, kEps, cur.p, nrm.p, st); });
        std::snprintf(name, sizeof name, "hc_pre_rms_norm T=%d", T);
        bench_line(name, us, (double) T * ND * 4 * (kHcMult + 2) + ND * 4);
    }
    std::printf(v.failures ? "norm_parity: FAILED\n" : "norm_parity: ok\n");
    return v.failures ? 1 : 0;
}
