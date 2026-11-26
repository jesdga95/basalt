// V4's 4-stream mHC at the model's 4096-wide geometry, against a double reference.
#include "basalt/kernels/mhc.hpp"
#include "parity_util.hpp"

#include <cstring>
#include <limits>

using namespace ds4test;

namespace {

constexpr int D = 4096;
constexpr int FLAT = 4 * D;

std::vector<double> coefficients(const float* x, const std::vector<float>& fn, const std::vector<float>& scale,
                                 const std::vector<float>& base, const basalt::kernels::HcArgs& args) {
    double square = 0.0;
    for (int i = 0; i < FLAT; ++i) square += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(square / FLAT + args.rms_eps);
    double mix[basalt::kernels::kHcCoef]{};
    for (int row = 0; row < basalt::kernels::kHcCoef; ++row) {
        double dot = 0.0;
        for (int i = 0; i < FLAT; ++i) dot += (double) fn[(size_t) row * FLAT + i] * x[i];
        mix[row] = dot * rs;
    }
    std::vector<double> c(basalt::kernels::kHcCoef);
    const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    for (int i = 0; i < 4; ++i) {
        c[(size_t) i] = sigmoid(mix[i] * scale[0] + base[(size_t) i]) + args.hc_eps;
        c[(size_t) 4 + i] = 2.0 * sigmoid(mix[4 + i] * scale[1] + base[4 + i]);
    }
    double m[4][4]{};
    for (int src = 0; src < 4; ++src) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int dst = 0; dst < 4; ++dst) {
            m[src][dst] = mix[8 + src * 4 + dst] * scale[2] + base[8 + src * 4 + dst];
            maximum = std::max(maximum, m[src][dst]);
        }
        double sum = 0.0;
        for (int dst = 0; dst < 4; ++dst) sum += m[src][dst] = std::exp(m[src][dst] - maximum);
        for (int dst = 0; dst < 4; ++dst) m[src][dst] = m[src][dst] / sum + args.hc_eps;
    }
    const auto columns = [&] {
        for (int dst = 0; dst < 4; ++dst) {
            double sum = 0.0;
            for (int src = 0; src < 4; ++src) sum += m[src][dst];
            for (int src = 0; src < 4; ++src) m[src][dst] /= sum + args.hc_eps;
        }
    };
    const auto rows = [&] {
        for (int src = 0; src < 4; ++src) {
            double sum = 0.0;
            for (int dst = 0; dst < 4; ++dst) sum += m[src][dst];
            for (int dst = 0; dst < 4; ++dst) m[src][dst] /= sum + args.hc_eps;
        }
    };
    columns();
    for (int i = 1; i < args.sinkhorn_iters; ++i) { rows(); columns(); }
    for (int src = 0; src < 4; ++src)
        for (int dst = 0; dst < 4; ++dst) c[(size_t) 8 + src * 4 + dst] = m[src][dst];
    return c;
}

}  // namespace

int main() {
    using namespace basalt::kernels;
    constexpr int T = 5;
    HcArgs args;
    args.n_embd = D;
    args.rms_eps = 1e-20f;
    args.hc_eps = 1e-6f;
    args.sinkhorn_iters = 20;

    const std::vector<float> x = randn((size_t) T * FLAT, 0x4d484334, 0.8f);
    const std::vector<float> fn = randn((size_t) kHcCoef * FLAT, 0x4d484335, 0.002f);
    const std::vector<float> scale = {0.7f, 0.9f, 0.6f};
    const std::vector<float> base = randn(kHcCoef, 0x4d484336, 0.25f);
    Dev<float> dx(x), df(fn), ds(scale), db(base), ws(hc_mixes_workspace_floats(T)), dc((size_t) T * kHcCoef);
    hc_mixes(dx.p, df.p, ds.p, db.p, T, args, ws.p, dc.p, nullptr);
    const std::vector<float> got = dc.down();

    std::vector<double> ref((size_t) T * kHcCoef);
    double worst_coef = 0.0;
    for (int t = 0; t < T; ++t) {
        const auto row = coefficients(x.data() + (size_t) t * FLAT, fn, scale, base, args);
        for (int c = 0; c < kHcCoef; ++c) {
            ref[(size_t) t * kHcCoef + c] = row[(size_t) c];
            worst_coef = std::max(worst_coef, std::fabs(got[(size_t) t * kHcCoef + c] - row[(size_t) c]));
        }
    }

    Dev<float> pre_out((size_t) T * D), post_out((size_t) T * FLAT), df_out((size_t) T * D);
    const std::vector<float> branch = randn((size_t) T * D, 0x4d484337, 0.4f);
    df_out.up(branch);
    hc_pre(dx.p, dc.p, kHcCoef, T, D, pre_out.p, nullptr);
    hc_post(df_out.p, dx.p, dc.p, kHcCoef, T, D, post_out.p, nullptr);
    const auto pre = pre_out.down();
    const auto post = post_out.down();
    double worst_pre = 0.0, worst_post = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
            double v = 0.0;
            for (int src = 0; src < 4; ++src)
                v += ref[(size_t) t * kHcCoef + src] * x[(size_t) t * FLAT + (size_t) src * D + d];
            worst_pre = std::max(worst_pre, std::fabs(pre[(size_t) t * D + d] - v));
        }
        for (int dst = 0; dst < 4; ++dst) {
            for (int d = 0; d < D; ++d) {
                double v = ref[(size_t) t * kHcCoef + 4 + dst] * branch[(size_t) t * D + d];
                for (int src = 0; src < 4; ++src)
                    v += ref[(size_t) t * kHcCoef + 8 + src * 4 + dst] * x[(size_t) t * FLAT + (size_t) src * D + d];
                worst_post = std::max(worst_post, std::fabs(post[(size_t) t * FLAT + (size_t) dst * D + d] - v));
            }
        }
    }

    bool independent = true;
    for (int t = 0; t < T; ++t) {
        Dev<float> sx(std::vector<float>(x.begin() + (size_t) t * FLAT, x.begin() + (size_t) (t + 1) * FLAT));
        Dev<float> sws(hc_mixes_workspace_floats(1)), sc(kHcCoef);
        hc_mixes(sx.p, df.p, ds.p, db.p, 1, args, sws.p, sc.p, nullptr);
        const std::vector<float> single = sc.down();
        independent = independent && std::memcmp(single.data(), got.data() + (size_t) t * kHcCoef,
                                                  kHcCoef * sizeof(float)) == 0;
    }

    std::printf("DeepSeek4 mHC D=%d T=%d: coefficients max abs %.3g, pre max abs %.3g, post max abs %.3g; "
                "single-row independence %s\n", D, T, worst_coef, worst_pre, worst_post,
                independent ? "bitwise" : "DIFFERS");
    return (worst_coef <= 3e-5 && worst_pre <= 1e-4 && worst_post <= 1e-4 && independent) ? 0 : 1;
}
