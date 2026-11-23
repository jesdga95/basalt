// tests/kernels/ssd_parity.cpp - the Mamba-2 mixer kernels (ssd.hpp) at Nemotron-H Super's geometry: 128 heads of 64,
// d_state 128, 8 groups, a convolution of 4 taps.
//
//     build/ssd_parity [--bench]
//
// A run of N tokens three ways: (a) a double reference of the whole mixer (convolution, recurrence, gated group norm)
// token by token; (b) one prompt run of all N rows (ssd_conv, ssd_scan with state_out in place, ssd_conv_advance,
// ssd_gated_norm); (c) decode windows of 1 to 16 rows, each accepting a random number of its rows and committing
// them (ssd_commit, ssd_conv_commit), the next window starting after the last accepted row.  (b) and (c) must give
// every token's output bitwise, and the same state and history after the N tokens; both against (a) to float
// rounding.  Then the microbench: a decode window of 1 and 8 rows, its commit, and a prompt chunk.
#include "basalt/kernels/ssd.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = basalt::kernels;

namespace {
constexpr int NH = 128, HD = 64, DS = 128, NG = 8, KC = 4;
constexpr int N = 48;                        // tokens
constexpr float kEps = 1e-5f;

struct Model {
    K::SsdGeometry g{NH, HD, DS, NG, KC};
    int DI = g.d_inner(), C = g.conv_dim(), W = g.d_inner() + g.conv_dim() + NH;   // a projection row: [z | xBC | dt]
    std::vector<float> conv_w, conv_b, dt_bias, A, D, norm_w;
    std::vector<float> state0, hist0;        // the state and history before the run
    std::vector<float> proj;                 // [N][W]
};

Model make_model() {
    Model m;
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    m.conv_w.resize((size_t) m.C * KC);
    for (auto& v : m.conv_w) v = 0.5f * nd(rng);
    m.conv_b.resize((size_t) m.C);
    for (auto& v : m.conv_b) v = 0.1f * nd(rng);
    m.dt_bias.resize(NH);
    m.A.resize(NH);
    m.D.resize(NH);
    for (int h = 0; h < NH; ++h) {
        m.dt_bias[(size_t) h] = -4.f + 3.f * ud(rng);
        m.A[(size_t) h] = (float) -std::exp(2.7 * ud(rng));
        m.D[(size_t) h] = 0.5f + ud(rng);
    }
    m.norm_w.resize((size_t) m.DI);
    for (auto& v : m.norm_w) v = 0.5f + ud(rng);
    m.state0.resize(m.g.state_floats());
    for (auto& v : m.state0) v = 0.1f * nd(rng);
    m.hist0.resize(m.g.hist_floats());
    for (auto& v : m.hist0) v = nd(rng);
    m.proj.resize((size_t) N * m.W);
    for (int t = 0; t < N; ++t)
        for (int i = 0; i < m.W; ++i) {
            float v = nd(rng);
            if (i >= m.DI + m.C) v = -1.f + nd(rng);   // dt_raw
            if (t == 7) v *= 4.f;                      // one loud token
            m.proj[(size_t) t * m.W + i] = v;
        }
    return m;
}

// (a) the double reference, token by token
std::vector<double> reference(const Model& m) {
    std::vector<double> S(m.state0.begin(), m.state0.end()), hist(m.hist0.begin(), m.hist0.end());
    std::vector<double> out((size_t) N * m.DI), xbc((size_t) m.C), y((size_t) m.DI);
    const int hpg = NH / NG, G = m.DI / NG;
    for (int t = 0; t < N; ++t) {
        const float* row = m.proj.data() + (size_t) t * m.W;
        for (int c = 0; c < m.C; ++c) {
            double acc = 0;
            for (int k = 0; k < KC; ++k) {
                const double v = k < KC - 1 ? hist[(size_t) c * (KC - 1) + k] : (double) row[m.DI + c];
                acc += (double) m.conv_w[(size_t) c * KC + k] * v;
            }
            acc += m.conv_b[(size_t) c];
            xbc[(size_t) c] = acc / (1.0 + std::exp(-acc));
            for (int k = 0; k < KC - 2; ++k) hist[(size_t) c * (KC - 1) + k] = hist[(size_t) c * (KC - 1) + k + 1];
            hist[(size_t) c * (KC - 1) + KC - 2] = row[m.DI + c];
        }
        for (int h = 0; h < NH; ++h) {
            const int g = h / hpg;
            const double r = (double) row[m.DI + m.C + h] + m.dt_bias[(size_t) h];
            const double dt = r > 20 ? r : std::log1p(std::exp(r));
            const double dA = std::exp(dt * m.A[(size_t) h]);
            for (int p = 0; p < HD; ++p) {
                const double x = xbc[(size_t) h * HD + p];
                double acc = 0;
                for (int n = 0; n < DS; ++n) {
                    double& s = S[((size_t) h * HD + p) * DS + n];
                    s = s * dA + dt * x * xbc[(size_t) m.DI + g * DS + n];
                    acc += s * xbc[(size_t) m.DI + NG * DS + g * DS + n];
                }
                y[(size_t) h * HD + p] = acc + m.D[(size_t) h] * x;
            }
        }
        for (int grp = 0; grp < NG; ++grp) {
            double ss = 0;
            std::vector<double> v((size_t) G);
            for (int c = 0; c < G; ++c) {
                const double z = row[grp * G + c];
                v[(size_t) c] = y[(size_t) grp * G + c] * (z / (1.0 + std::exp(-z)));
                ss += v[(size_t) c] * v[(size_t) c];
            }
            const double rs = 1.0 / std::sqrt(ss / G + kEps);
            for (int c = 0; c < G; ++c) out[(size_t) t * m.DI + grp * G + c] = v[(size_t) c] * rs * m.norm_w[(size_t) grp * G + c];
        }
    }
    return out;
}

struct Dev {
    float *proj, *conv_w, *conv_b, *dt_bias, *A, *D, *norm_w, *state, *hist, *xbc, *y, *dt, *dA, *out;
    int32_t* keep;
};

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    Model m = make_model();
    if (std::string e = K::ssd_check(m.g); !e.empty()) {
        std::printf("ssd_check: %s  FAIL\n", e.c_str());
        return 1;
    }
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    Dev d{};
    auto up = [&](float*& p, const std::vector<float>& v) {
        cudaMalloc((void**) &p, v.size() * 4);
        cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    };
    up(d.proj, m.proj);
    up(d.conv_w, m.conv_w);
    up(d.conv_b, m.conv_b);
    up(d.dt_bias, m.dt_bias);
    up(d.A, m.A);
    up(d.D, m.D);
    up(d.norm_w, m.norm_w);
    up(d.state, m.state0);
    up(d.hist, m.hist0);
    const int maxT = 1024;
    cudaMalloc((void**) &d.xbc, (size_t) maxT * m.C * 4);
    cudaMalloc((void**) &d.y, (size_t) maxT * m.DI * 4);
    cudaMalloc((void**) &d.dt, (size_t) maxT * NH * 4);
    cudaMalloc((void**) &d.dA, (size_t) maxT * NH * 4);
    cudaMalloc((void**) &d.out, (size_t) maxT * m.DI * 4);
    cudaMalloc((void**) &d.keep, 4);
    const int64_t W = m.W;
    // a run of T rows from projection row r0, no commit: the outputs in d.out
    auto window = [&](int r0, int T, bool pass_state) {
        const float* pr = d.proj + (size_t) r0 * W;
        K::ssd_conv(m.g, pr + m.DI, W, d.hist, d.conv_w, d.conv_b, T, d.xbc, s);
        K::ssd_scan(m.g, d.state, d.xbc, pr + m.DI + m.C, W, d.dt_bias, d.A, d.D, T, d.y, d.dt, d.dA,
                    pass_state ? d.state : nullptr, s);
        K::ssd_gated_norm(m.g, d.y, pr, W, d.norm_w, kEps, T, d.out, nullptr, s);
    };
    // (b) the prompt run
    std::vector<float> prompt_out((size_t) N * m.DI), prompt_state(m.g.state_floats()), prompt_hist(m.g.hist_floats());
    window(0, N, true);
    K::ssd_conv_advance(m.g, d.hist, d.proj + m.DI, W, N, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(prompt_out.data(), d.out, prompt_out.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(prompt_state.data(), d.state, prompt_state.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(prompt_hist.data(), d.hist, prompt_hist.size() * 4, cudaMemcpyDeviceToHost);
    // (c) decode windows with partial acceptance
    cudaMemcpy(d.state, m.state0.data(), m.state0.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.hist, m.hist0.data(), m.hist0.size() * 4, cudaMemcpyHostToDevice);
    std::vector<float> dec_out((size_t) N * m.DI), dec_state(m.g.state_floats()), dec_hist(m.g.hist_floats());
    std::mt19937 rng(3);
    const int widths[] = {1, 3, 8, 16, 5, 2, 16, 7, 1, 12};
    int pos = 0, wi = 0, windows = 0;
    while (pos < N) {
        const int T = std::min(widths[wi++ % 10], N - pos);
        window(pos, T, false);
        const int keep = 1 + (int) (rng() % (unsigned) T);
        cudaMemcpyAsync(d.keep, &keep, 4, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(dec_out.data() + (size_t) pos * m.DI, d.out, (size_t) keep * m.DI * 4, cudaMemcpyDeviceToHost);
        K::ssd_commit(m.g, d.state, d.xbc, d.dt, d.dA, d.keep, s);
        K::ssd_conv_commit(m.g, d.hist, d.proj + (size_t) pos * W + m.DI, W, d.keep, s);
        pos += keep;
        ++windows;
    }
    cudaStreamSynchronize(s);
    cudaMemcpy(dec_state.data(), d.state, dec_state.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dec_hist.data(), d.hist, dec_hist.size() * 4, cudaMemcpyDeviceToHost);
    const cudaError_t ce = cudaGetLastError();
    const bool same_out = std::memcmp(prompt_out.data(), dec_out.data(), prompt_out.size() * 4) == 0;
    const bool same_state = std::memcmp(prompt_state.data(), dec_state.data(), prompt_state.size() * 4) == 0;
    const bool same_hist = std::memcmp(prompt_hist.data(), dec_hist.data(), prompt_hist.size() * 4) == 0;
    // (a) against the double reference: each token's relative L1 error and the largest absolute one
    const std::vector<double> ref = reference(m);
    double worst_rel = 0, worst_abs = 0, mean_rel = 0;
    for (int t = 0; t < N; ++t) {
        double num = 0, den = 0;
        for (int i = 0; i < m.DI; ++i) {
            const double e = std::fabs((double) prompt_out[(size_t) t * m.DI + i] - ref[(size_t) t * m.DI + i]);
            num += e;
            den += std::fabs(ref[(size_t) t * m.DI + i]);
            worst_abs = std::max(worst_abs, e);
        }
        worst_rel = std::max(worst_rel, num / den);
        mean_rel += num / den / N;
    }
    const bool ok = ce == cudaSuccess && same_out && same_state && same_hist && worst_rel < 1e-5 && std::isfinite(worst_rel);
    std::printf("ssd %d tokens: prompt run vs %d decode windows (1-16 rows, random acceptance): outputs %s, state %s, "
                "history %s; vs double: rel %.2e worst / %.2e mean, max abs %.2e  %s\n",
                N, windows, same_out ? "bitwise" : "DIFFER", same_state ? "bitwise" : "DIFFER", same_hist ? "bitwise" : "DIFFER",
                worst_rel, mean_rel, worst_abs, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    if (bench) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        auto time_us = [&](int R, auto&& fn) {
            for (int i = 0; i < 5; ++i) fn();
            cudaEventRecord(a, s);
            for (int i = 0; i < R; ++i) fn();
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return 1000.0 * ms / R;
        };
        // 40 layers' states in turn, as decode reads them: more than any L2 holds
        constexpr int kLayers = 40;
        float* states = nullptr;
        cudaMalloc((void**) &states, (size_t) kLayers * m.g.state_floats() * 4);
        cudaMemset(states, 0, (size_t) kLayers * m.g.state_floats() * 4);
        int li = 0;
        auto st = [&] { li = (li + 1) % kLayers; return states + (size_t) li * m.g.state_floats(); };
        const double state_mb = m.g.state_floats() * 4 / 1e6;
        for (int T : {1, 8}) {
            const double conv = time_us(200, [&] { K::ssd_conv(m.g, d.proj + m.DI, W, d.hist, d.conv_w, d.conv_b, T, d.xbc, s); });
            const double scan = time_us(200, [&] {
                K::ssd_scan(m.g, st(), d.xbc, d.proj + m.DI + m.C, W, d.dt_bias, d.A, d.D, T, d.y, d.dt, d.dA, nullptr, s);
            });
            const double scan_out = time_us(200, [&] {
                float* x = st();
                K::ssd_scan(m.g, x, d.xbc, d.proj + m.DI + m.C, W, d.dt_bias, d.A, d.D, T, d.y, d.dt, d.dA, x, s);
            });
            const double norm = time_us(200, [&] { K::ssd_gated_norm(m.g, d.y, d.proj, W, d.norm_w, kEps, T, d.out, nullptr, s); });
            const int keep = T;
            cudaMemcpy(d.keep, &keep, 4, cudaMemcpyHostToDevice);
            const double commit = time_us(200, [&] { K::ssd_commit(m.g, st(), d.xbc, d.dt, d.dA, d.keep, s); });
            std::printf("bench window T=%d (40 layers' states in turn): conv %.1f us, scan %.1f us (%.0f GB/s of state read), "
                        "scan passing the state %.1f us (%.0f GB/s read + write), norm %.1f us, commit %.1f us (%.0f GB/s)\n",
                        T, conv, scan, state_mb * 1e3 / scan, scan_out, 2 * state_mb * 1e3 / scan_out, norm, commit,
                        2 * state_mb * 1e3 / commit);
        }
        cudaFree(states);
        std::vector<float> big((size_t) maxT * W);
        std::mt19937 r2(5);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : big) v = nd(r2);
        float* dbig = nullptr;
        cudaMalloc((void**) &dbig, big.size() * 4);
        cudaMemcpy(dbig, big.data(), big.size() * 4, cudaMemcpyHostToDevice);
        const double chunk = time_us(5, [&] {
            K::ssd_conv(m.g, dbig + m.DI, W, d.hist, d.conv_w, d.conv_b, maxT, d.xbc, s);
            K::ssd_scan(m.g, d.state, d.xbc, dbig + m.DI + m.C, W, d.dt_bias, d.A, d.D, maxT, d.y, nullptr, nullptr, nullptr, s);
            K::ssd_gated_norm(m.g, d.y, dbig, W, d.norm_w, kEps, maxT, d.out, nullptr, s);
        });
        std::printf("bench prompt chunk of %d rows: %.0f us, %.2f us per row\n", maxT, chunk, chunk / maxT);
        cudaFree(dbig);
    }
    for (void* p : {(void*) d.proj, (void*) d.conv_w, (void*) d.conv_b, (void*) d.dt_bias, (void*) d.A, (void*) d.D,
                    (void*) d.norm_w, (void*) d.state, (void*) d.hist, (void*) d.xbc, (void*) d.y, (void*) d.dt, (void*) d.dA,
                    (void*) d.out, (void*) d.keep})
        cudaFree(p);
    cudaStreamDestroy(s);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
