// src/models/inkling2/kernels/kernels_parity.cpp - the Inkling 2 kernels against plain CPU references (double accumulation)
// at the model's geometry: the block-residual mixes, the loop controllers' norm and blend, the attention preparation
// (short convolutions, clips, cache writes), the attention with its relative bias and sink over a window ring and a full
// cache, the post-sublayer convolutions, the experts with the softcapped SwiGLU and their weighted sum, and the softcap.
// Every kernel is checked for a row's bits not depending on the rows beside it, and timed. Exit 77 without a GPU.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace basalt::models::inkling2;
namespace bk = basalt::kernels;

namespace {

constexpr int E = kWidth;
std::mt19937 rng(23);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.f, float mean = 0.f) {
    std::normal_distribution<float> d(mean, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T) + 16);
    cudaMemset(p, 0, n * sizeof(T) + 16);
    return p;
}

void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-34s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void same_bits(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    std::printf("%-34s %s\n", what, ok ? "bitwise" : "FAIL");
    failures += !ok;
}
// microseconds a call, averaged over `n` calls after one warm-up
double time_us(const std::function<void()>& f, int n = 50) {
    f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}
double h2d(uint16_t h) {
    __half_raw r;
    r.x = h;
    return (double) __half2float(__half(r));
}
uint16_t f2h(float f) { return __half_raw(__float2half(f)).x; }
double tanhd(double x) { return std::tanh(x); }
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---- block residuals ---------------------------------------------------------------------------------------------------

void test_mix() {
    for (int n_q : {1, 3, 4}) {
        const int R = 5, n = n_q == 4 ? 2 : 5;
        std::vector<std::vector<float>> st(n);
        std::vector<float*> ds(n);
        for (int i = 0; i < n; ++i) {
            st[i] = randn((size_t) R * E, 0.5f + 0.3f * i);
            ds[i] = dev(st[i]);
        }
        const std::vector<float> q = randn((size_t) n_q * E, 0.05f);
        float* dq = dev(q);
        float* out = zeros<float>((size_t) n_q * R * E);
        MixStates ms;
        for (int i = 0; i < n; ++i) ms.s[ms.n++] = ds[i];
        attnres_mix(ms, dq, n_q, R, 1e-6f, out, (int64_t) R * E, nullptr);
        std::vector<float> got = host(out, (size_t) n_q * R * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r) {
            std::vector<double> inv(n);
            for (int i = 0; i < n; ++i) {
                double ss = 0;
                for (int c = 0; c < E; ++c) ss += (double) st[i][(size_t) r * E + c] * st[i][(size_t) r * E + c];
                inv[i] = 1.0 / std::sqrt(ss / E + 1e-6);
            }
            for (int j = 0; j < n_q; ++j) {
                std::vector<double> sc(n);
                double m = -1e300;
                for (int i = 0; i < n; ++i) {
                    double d = 0;
                    for (int c = 0; c < E; ++c) d += (double) q[(size_t) j * E + c] * st[i][(size_t) r * E + c];
                    sc[i] = d * inv[i] / std::sqrt((double) E);
                    m = std::max(m, sc[i]);
                }
                double den = 0;
                for (double& s : sc) den += (s = std::exp(s - m));
                for (int c = 0; c < E; ++c) {
                    double acc = 0;
                    for (int i = 0; i < n; ++i) acc += sc[i] / den * st[i][(size_t) r * E + c] * inv[i];
                    ref[((size_t) j * R + r) * E + c] = acc;
                }
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "attnres_mix (%d states, %d queries)", n, n_q);
        report(name, got, ref, 2e-6);
        MixStates one;   // row 3 alone
        for (int i = 0; i < n; ++i) one.s[one.n++] = ds[i] + 3 * E;
        float* out1 = zeros<float>((size_t) n_q * E);
        attnres_mix(one, dq, n_q, 1, 1e-6f, out1, E, nullptr);
        std::vector<float> g1 = host(out1, (size_t) n_q * E);
        bool same = true;
        for (int j = 0; j < n_q; ++j) same = same && !std::memcmp(&g1[(size_t) j * E], &got[((size_t) j * R + 3) * E], E * 4);
        std::printf("%-34s %s\n", "  a row alone", same ? "bitwise" : "FAIL");
        failures += !same;
        if (n_q == 1)
            std::printf("  %.1f us a call (1 row, %d states)\n", time_us([&] { attnres_mix(one, dq, 1, 1, 1e-6f, out1, E, nullptr); }), n);
    }
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

void test_ctrl() {
    const int R = 3;
    const std::vector<float> a = randn((size_t) R * E, 0.7f), b = randn((size_t) R * E, 1.3f), gf = randn((size_t) R * 2 * E, 0.8f);
    float *da = dev(a), *db = dev(b), *dgf = dev(gf), *pair = zeros<float>((size_t) R * 2 * E), *out = zeros<float>((size_t) R * E);
    ctrl_pair_norm(da, db, R, pair, nullptr);
    ctrl_blend(da, db, dgf, R, out, nullptr);
    std::vector<float> gp = host(pair, (size_t) R * 2 * E), go = host(out, (size_t) R * E);
    std::vector<double> rp(gp.size()), ro(go.size());
    for (int r = 0; r < R; ++r) {
        double sa = 0, sb = 0;
        for (int c = 0; c < E; ++c) {
            sa += (double) a[(size_t) r * E + c] * a[(size_t) r * E + c];
            sb += (double) b[(size_t) r * E + c] * b[(size_t) r * E + c];
        }
        const double ia = 1 / std::sqrt(sa / E + 1e-6), ib = 1 / std::sqrt(sb / E + 1e-6);
        for (int c = 0; c < E; ++c) {
            rp[(size_t) r * 2 * E + c] = a[(size_t) r * E + c] * ia;
            rp[(size_t) r * 2 * E + E + c] = b[(size_t) r * E + c] * ib;
            const double g = tanhd(gf[(size_t) r * 2 * E + c]), f = tanhd(gf[(size_t) r * 2 * E + E + c]);
            ro[(size_t) r * E + c] = (1 - f) * a[(size_t) r * E + c] + g * b[(size_t) r * E + c];
        }
    }
    report("ctrl_pair_norm", gp, rp, 2e-6);
    report("ctrl_blend", go, ro, 2e-6);
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

// The CPU reference of one row's convolution input at position p - j (window rows, else the ring, else 0).
double conv_input(const std::vector<float>& y, int ld, const std::vector<float>& ring, int width, const std::vector<int>& pos, int r, int j,
                  int c) {
    if (r - j >= 0) return y[(size_t) (r - j) * ld + c];
    const int p = pos[r] - j;
    if (p < 0) return 0.0;
    return ring[(size_t) (p % kConvRing) * width + c];
}

void test_prepare() {
    for (bool full : {false, true}) {
        const int R = 3, p0 = full ? 9000 : 2;   // full: past the log scale's floor; a window row: near the start (zero history)
        const int64_t cap = full ? 9100 : kExtent + 8;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        const std::vector<float> yq = randn((size_t) R * kProjQ, 1.5f), yk = randn((size_t) R * kProjKV, 2.0f), yv = randn((size_t) R * kProjKV, 0.6f);
        const std::vector<float> kc = randn(1024 * kConv, 0.3f), vc = randn(1024 * kConv, 0.3f);
        const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rs = randn(kHeads * kRel, 0.2f, 1.f);
        const std::vector<float> kr = randn((size_t) kConvRing * 1024, 2.0f), vr = randn((size_t) kConvRing * 1024, 0.6f);
        float *dyq = dev(yq), *dyk = dev(yk), *dyv = dev(yv), *dkc = dev(kc), *dvc = dev(vc), *dqs = dev(qs), *drs = dev(rs);
        float *dkr = dev(kr), *dvr = dev(vr);
        int* dpos = dev(pos);
        uint16_t *kcache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim), *vcache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim);
        float *q = zeros<float>((size_t) R * kHeads * kHeadDim), *rr = zeros<float>((size_t) R * kHeads * kRel), *sink = zeros<float>((size_t) R * kHeads);
        AttnPrep a;
        a.yq = dyq; a.yk = dyk; a.yv = dyv; a.pos = dpos; a.full = full; a.k_conv = dkc; a.v_conv = dvc; a.q_scale = dqs; a.r_scale = drs;
        a.k_ring = dkr; a.v_ring = dvr; a.k_cache = kcache; a.v_cache = vcache; a.cap = cap; a.q = q; a.r = rr; a.sink = sink;
        attn_prepare(a, R, nullptr);
        cudaDeviceSynchronize();
        // keys and values
        std::vector<float> gk, gv;
        std::vector<double> rk, rv;
        const std::vector<uint16_t> hk = host(kcache, (size_t) kKvHeads * cap * kHeadDim), hv = host(vcache, (size_t) kKvHeads * cap * kHeadDim);
        for (int r = 0; r < R; ++r)
            for (int which = 0; which < 2; ++which) {
                const std::vector<float>& y = which ? yv : yk;
                const std::vector<float>& taps = which ? vc : kc;
                const std::vector<float>& ring = which ? vr : kr;
                for (int h = 0; h < kKvHeads; ++h) {
                    std::vector<double> val(kHeadDim);
                    double ss = 0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const int c = h * kHeadDim + d;
                        double acc = 0;
                        for (int j = 0; j < kConv; ++j) {
                            double t = taps[(size_t) c * kConv + (kConv - 1 - j)];
                            if (full) t += y[(size_t) r * kProjKV + kOffTaps + h * kConv + j];
                            acc += t * conv_input(y, kProjKV, ring, 1024, pos, r, j, c);
                        }
                        val[d] = (acc + y[(size_t) r * kProjKV + c]) / std::sqrt(full ? 9.0 : 5.0);
                        ss += val[d] * val[d];
                    }
                    const double f = which == 0 ? std::min(22.627417 / (std::sqrt(ss) + 1e-6), 1.0) : 1.0 / std::sqrt(ss / kHeadDim + 1e-6);
                    for (int d = 0; d < kHeadDim; ++d) {
                        const size_t at = ((size_t) h * cap + pos[r] % cap) * kHeadDim + d;
                        (which ? gv : gk).push_back((float) h2d((which ? hv : hk)[at]));
                        (which ? rv : rk).push_back(val[d] * f);
                    }
                }
            }
        report(full ? "attn_prepare keys (full, fp16)" : "attn_prepare keys (window, fp16)", gk, rk, 1e-3);
        report(full ? "attn_prepare values (full, fp16)" : "attn_prepare values (window, fp16)", gv, rv, 1e-3);
        // queries, relative states, sinks
        const std::vector<float> gq = host(q, (size_t) R * kHeads * kHeadDim), gr = host(rr, (size_t) R * kHeads * kRel), gs = host(sink, (size_t) R * kHeads);
        std::vector<double> rq(gq.size()), rr2(gr.size()), rsk(gs.size());
        for (int r = 0; r < R; ++r) {
            const double tau = full ? 1.0 + 0.1 * std::log(std::max((pos[r] + 1.0) / 8192.0, 1.0)) : 1.0;
            for (int h = 0; h < kHeads; ++h) {
                double ss = 0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const double v = (double) yq[(size_t) r * kProjQ + h * kHeadDim + d] * qs[h * kHeadDim + d];
                    ss += v * v;
                }
                const double f = std::min(22.627417 / (std::sqrt(ss) + 1e-6), 1.0) * tau;
                for (int d = 0; d < kHeadDim; ++d)
                    rq[((size_t) r * kHeads + h) * kHeadDim + d] = (double) yq[(size_t) r * kProjQ + h * kHeadDim + d] * qs[h * kHeadDim + d] * f;
                double s2 = 0;
                for (int i = 0; i < kRel; ++i) {
                    const double v = (double) yq[(size_t) r * kProjQ + kOffR + h * kRel + i] * rs[h * kRel + i];
                    s2 += v * v;
                }
                const double fr = std::min(8.0 / (std::sqrt(s2) + 1e-6), 1.0) * tau;
                for (int i = 0; i < kRel; ++i)
                    rr2[((size_t) r * kHeads + h) * kRel + i] = (double) yq[(size_t) r * kProjQ + kOffR + h * kRel + i] * rs[h * kRel + i] * fr;
                rsk[(size_t) r * kHeads + h] = yq[(size_t) r * kProjQ + kOffSink + h];
            }
        }
        report("attn_prepare queries", gq, rq, 2e-6);
        report("attn_prepare relative states", gr, rr2, 2e-6);
        report("attn_prepare sinks", gs, rsk, 0);
        // the rings took the window's inputs
        const std::vector<float> nkr = host(dkr, (size_t) kConvRing * 1024);
        bool ring_ok = true;
        for (int r = 0; r < R; ++r) ring_ok = ring_ok && !std::memcmp(&nkr[(size_t) (pos[r] % kConvRing) * 1024], &yk[(size_t) r * kProjKV], 1024 * 4);
        std::printf("%-34s %s\n", "  the key ring", ring_ok ? "ok" : "FAIL");
        failures += !ring_ok;
        if (!full)
            std::printf("  %.1f us a call (1 row)\n", time_us([&] { attn_prepare(a, 1, nullptr); }));
    }
}

// ---- attention ---------------------------------------------------------------------------------------------------------

void test_attention() {
    for (bool window : {true, false}) {
        const int R = 4;
        const int p0 = window ? 2900 : 2995;
        const int64_t cap = window ? kExtent + 8 : 3000;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        std::vector<uint16_t> kh((size_t) kKvHeads * cap * kHeadDim), vh(kh.size());
        for (auto& x : kh) x = f2h(randn(1, 0.6f)[0]);
        for (auto& x : vh) x = f2h(randn(1, 1.0f)[0]);
        const std::vector<float> q = randn((size_t) R * kHeads * kHeadDim, 0.8f), rs = randn((size_t) R * kHeads * kRel, 1.0f);
        const std::vector<float> sink = randn((size_t) R * kHeads, 2.0f), table = randn((size_t) kRel * kExtent, 0.3f);
        uint16_t *dk = dev(kh), *dv = dev(vh);
        float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table), *out = zeros<float>((size_t) R * kHeads * kHeadDim);
        int* dpos = dev(pos);
        AttnArgs a;
        a.q = dq; a.r = drs; a.sink = dsink; a.rel_table = dt; a.pos = dpos; a.k_cache = dk; a.v_cache = dv; a.cap = cap;
        a.window = window ? kExtent : 0;
        const int64_t max_keys = window ? kExtent : p0 + R;
        void* scr = zeros<uint8_t>(attention_decode_scratch(R, max_keys));
        attention_decode(a, R, max_keys, scr, out, nullptr);
        const std::vector<float> got = host(out, (size_t) R * kHeads * kHeadDim);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r)
            for (int h = 0; h < kHeads; ++h) {
                const int g = h / 3;
                const int lo = window ? std::max(0, pos[r] - kExtent + 1) : 0;
                std::vector<double> sc;
                double m = sink[(size_t) r * kHeads + h];
                for (int k = lo; k <= pos[r]; ++k) {
                    double d = 0;
                    for (int i = 0; i < kHeadDim; ++i)
                        d += (double) q[((size_t) r * kHeads + h) * kHeadDim + i] * h2d(kh[((size_t) g * cap + k % cap) * kHeadDim + i]);
                    d *= a.scale;
                    const int dist = pos[r] - k;
                    if (dist < kExtent)
                        for (int i = 0; i < kRel; ++i) d += (double) rs[((size_t) r * kHeads + h) * kRel + i] * table[(size_t) i * kExtent + dist];
                    sc.push_back(d);
                    m = std::max(m, d);
                }
                double den = std::exp(sink[(size_t) r * kHeads + h] - m);
                for (double& s : sc) den += (s = std::exp(s - m));
                for (int i = 0; i < kHeadDim; ++i) {
                    double acc = 0;
                    for (int k = lo; k <= pos[r]; ++k) acc += sc[(size_t) (k - lo)] * h2d(vh[((size_t) g * cap + k % cap) * kHeadDim + i]);
                    ref[((size_t) r * kHeads + h) * kHeadDim + i] = acc / den;
                }
            }
        report(window ? "attention (window ring)" : "attention (full, 3k keys)", got, ref, 2e-3);
        // row 2 alone
        AttnArgs one = a;
        one.q = dq + 2 * kHeads * kHeadDim; one.r = drs + 2 * kHeads * kRel; one.sink = dsink + 2 * kHeads; one.pos = dpos + 2;
        float* out1 = zeros<float>((size_t) kHeads * kHeadDim);
        attention_decode(one, 1, max_keys, scr, out1, nullptr);
        const std::vector<float> g1 = host(out1, (size_t) kHeads * kHeadDim);
        same_bits("  a row alone", g1.data(), &got[(size_t) 2 * kHeads * kHeadDim], g1.size() * 4);
        const double us = time_us([&] { attention_decode(one, 1, max_keys, scr, out1, nullptr); });
        const double keys = window ? kExtent : p0 + 3;
        std::printf("  %.1f us a call (1 row, %.0f keys: %.0f GB/s of K and V)\n", us, keys, keys * kKvHeads * kHeadDim * 2 * 2 / us / 1e3);
    }
}

// ---- post-sublayer convolutions ----------------------------------------------------------------------------------------

void test_post_conv() {
    const int R = 4;
    std::vector<int> pos = {40, 41, 42, 43};
    const std::vector<float> x = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f), ring = randn((size_t) kConvRing * E, 0.4f),
                             add = randn((size_t) R * E, 1.0f);
    float *dx = dev(x), *dt = dev(taps), *dring = dev(ring), *dadd = dev(add), *out = zeros<float>((size_t) R * E);
    int* dpos = dev(pos);
    post_conv(dx, dpos, dt, dring, dadd, R, out, nullptr);
    const std::vector<float> got = host(out, (size_t) R * E);
    std::vector<double> ref(got.size());
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < E; ++c) {
            double acc = 0;
            for (int j = 0; j < kConv; ++j) acc += (double) taps[(size_t) c * kConv + (kConv - 1 - j)] * conv_input(x, E, ring, E, pos, r, j, c);
            ref[(size_t) r * E + c] = (acc + x[(size_t) r * E + c]) / std::sqrt(5.0) + add[(size_t) r * E + c];
        }
    report("post_conv", got, ref, 2e-6);
}

// ---- experts -----------------------------------------------------------------------------------------------------------

void test_experts() {
    const int H = E, FF = 1152, NE = 6, R = 3, K = 4;
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    bk::NativeExpertLayout L = bk::native_expert_layout(40, 40, H, FF);
    L.gate_softcap = 4.f;
    L.up_softcap = 25.f;
    std::vector<uint8_t> blobs((size_t) NE * L.bytes);
    std::vector<std::vector<float>> wg(NE), wu(NE), wd(NE);   // the dequantised weights
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * L.bytes;
        auto put = [&](std::vector<float>& deq, int rows, int k, size_t row_bytes, uint8_t* dst, float s) {
            const std::vector<float> w = randn((size_t) rows * k, s);
            deq.resize((size_t) rows * k);
            for (int r = 0; r < rows; ++r) {
                tr->from_float_ref(&w[(size_t) r * k], dst + (size_t) r * row_bytes, k);
                tr->to_float(dst + (size_t) r * row_bytes, &deq[(size_t) r * k], k);
            }
        };
        put(wg[e], FF, H, L.gu_row, b, 0.05f);
        put(wu[e], FF, H, L.up_row, b + L.up_off, 0.05f);
        put(wd[e], H, FF, L.d_row, b + L.down_off, 0.05f);
        const float tail[4] = {1.f, 1.f, 1.f, 0.f};
        std::memcpy(b + L.bytes - 16, tail, 16);
    }
    std::vector<int32_t> ids = {0, 2, 5, 1, 2, 3, 4, 0, 5, 1, 0, 3};   // [R][K]
    const std::vector<float> w = randn((size_t) R * K, 0.5f, 1.f), x = randn((size_t) R * H, 1.0f);
    uint8_t* dblobs = dev(blobs);
    int32_t *dids = dev(ids), *start = zeros<int32_t>(R * K + 2), *ng = zeros<int32_t>(4), *dst = zeros<int32_t>(R * K), *tok = zeros<int32_t>(R * K);
    unsigned long long* ptr = zeros<unsigned long long>(R * K);
    float *dw = dev(w), *dx = dev(x), *rows = zeros<float>((size_t) R * K * H), *y = zeros<float>((size_t) R * H);
    void* xq = zeros<uint8_t>(bk::native_q8_1_bytes(H, R));
    void* scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(R * K, FF));
    cudaStream_t st;
    cudaStreamCreate(&st);
    moe_plan(dids, R, K, NE, dblobs, L.bytes, ptr, start, ng, dst, tok, st);
    bk::native_quantize_q8_1(dx, xq, H, R, st);
    bk::native_expert_grouped(L, ptr, start, ng, dst, tok, NE, R * K, xq, scr, rows, st);
    moe_combine(rows, dw, R, K, y, st);
    cudaStreamSynchronize(st);
    const std::vector<float> got = host(y, (size_t) R * H);
    std::vector<double> ref(got.size(), 0.0);
    for (int r = 0; r < R; ++r)
        for (int s = 0; s < K; ++s) {
            const int e = ids[(size_t) r * K + s];
            std::vector<double> h(FF);
            for (int f = 0; f < FF; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < H; ++c) {
                    g += (double) wg[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                    u += (double) wu[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                }
                h[f] = 4.0 * tanhd(g / 4.0) * sig(g) * 25.0 * tanhd(u / 25.0);
            }
            for (int c = 0; c < H; ++c) {
                double o = 0;
                for (int f = 0; f < FF; ++f) o += (double) wd[e][(size_t) c * FF + f] * h[f];
                ref[(size_t) r * H + c] += (double) w[(size_t) r * K + s] * o;
            }
        }
    report("experts (NVFP4, softcapped SwiGLU)", got, ref, 2e-2);   // q8_1 activations
    // the plan of row 1 alone gives that row's bits
    moe_plan(dids + K, 1, K, NE, dblobs, L.bytes, ptr, start, ng, dst, tok, st);
    bk::native_quantize_q8_1(dx + H, xq, H, 1, st);
    bk::native_expert_grouped(L, ptr, start, ng, dst, tok, K, K, xq, scr, rows, st);
    float* y1 = zeros<float>(H);
    moe_combine(rows, dw + K, 1, K, y1, st);
    cudaStreamSynchronize(st);
    const std::vector<float> g1 = host(y1, H);
    same_bits("  a row alone", g1.data(), &got[H], (size_t) H * 4);
}

void test_softcap() {
    std::vector<float> x = randn(3 * 1000, 40.f);
    float* dx = dev(x);
    softcap(dx, 3, 1000, 1000, 30.f, nullptr);
    const std::vector<float> got = host(dx, x.size());
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = 30.0 * tanhd(x[i] / 30.0);
    report("softcap", got, ref, 2e-6);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    test_mix();
    test_ctrl();
    test_prepare();
    test_attention();
    test_post_conv();
    test_experts();
    test_softcap();
    std::printf("inkling2_kernels_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
