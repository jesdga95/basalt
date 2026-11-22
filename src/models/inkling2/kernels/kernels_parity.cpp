// src/models/inkling2/kernels/kernels_parity.cpp - the Inkling 2 kernels against plain CPU references (double accumulation)
// at the model's geometry: the block-residual mixes, the loop controllers' norm and blend, the attention preparation
// (short convolutions, clips, cache writes), the attention with its relative bias and sink over a window ring and a full
// cache, the post-sublayer convolutions, the experts with the softcapped SwiGLU and their weighted sum, and the softcap;
// and the decode window's fused kernels against the references and the kernels they replace, their q8_1 blocks against a
// CPU rounding of their own fp32 rows. Every kernel is checked for a row's bits not depending on the rows beside it, and
// timed. Exit 77 without a GPU.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

// The block-residual mix of row r of `st` (each [rows][E]) scored by the query q[0..E).
std::vector<double> mix_ref(const std::vector<const std::vector<float>*>& st, const float* q, int r) {
    const size_t n = st.size();
    std::vector<double> inv(n), sc(n);
    double m = -1e300;
    for (size_t i = 0; i < n; ++i) {
        double ss = 0, d = 0;
        for (int c = 0; c < E; ++c) {
            const double x = (*st[i])[(size_t) r * E + c];
            ss += x * x;
            d += (double) q[c] * x;
        }
        inv[i] = 1.0 / std::sqrt(ss / E + 1e-6);
        sc[i] = d * inv[i] / std::sqrt((double) E);
        m = std::max(m, sc[i]);
    }
    double den = 0;
    for (double& v : sc) den += (v = std::exp(v - m));
    std::vector<double> out(E, 0.0);
    for (int c = 0; c < E; ++c)
        for (size_t i = 0; i < n; ++i) out[c] += sc[i] / den * (*st[i])[(size_t) r * E + c] * inv[i];
    return out;
}
std::vector<double> norm_ref(const std::vector<double>& x, const std::vector<float>& w, double eps = 1e-6) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / (double) x.size() + eps);
    std::vector<double> y(x.size());
    for (size_t c = 0; c < x.size(); ++c) y[c] = x[c] * inv * w[c];
    return y;
}

// The q8_1 blocks of `rows` rows of `width` against a CPU rounding of the kernel's own fp32 rows `x`: the int8 values and
// the scale exactly, the sum to its fp16 rounding.
void check_q8(const char* what, const void* dq8, const std::vector<float>& x, int rows, int width) {
    const size_t nb = (size_t) rows * width / 32;
    std::vector<uint8_t> q = host(static_cast<const uint8_t*>(dq8), nb * 36);
    bool ok = true;
    for (size_t b = 0; b < nb && ok; ++b) {
        const float* v = &x[b * 32];
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(v[i]));
            sum += v[i];
        }
        const float d = amax / 127.0f;
        uint16_t dh, sh;
        std::memcpy(&dh, &q[b * 36], 2);
        std::memcpy(&sh, &q[b * 36 + 2], 2);
        ok = dh == f2h(d) && std::fabs(h2d(sh) - (double) sum) <= 1e-3 * std::max(1.0, std::fabs((double) sum));
        for (int i = 0; i < 32 && ok; ++i) {
            const float t = amax == 0.f ? 0.f : std::max(-127.f, std::min(127.f, std::round(v[i] / d)));
            ok = (int8_t) q[b * 36 + 4 + i] == (int8_t) t;
        }
    }
    std::printf("%-34s %s\n", what, ok ? "ok" : "FAIL");
    failures += !ok;
}

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
                        const size_t at = ((size_t) (pos[r] % cap) * kKvHeads + h) * kHeadDim + d;
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
    for (int mode = 0; mode < 3; ++mode) {   // a window ring, a full cache, a full cache of E4M3 codes
        const bool window = mode == 0, fp8 = mode == 2;
        const int R = 4;
        const int p0 = window ? 2900 : 2995;
        const int64_t cap = window ? kExtent + 8 : 3000;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        std::vector<double> kd((size_t) kKvHeads * cap * kHeadDim), vd(kd.size());   // the cached values as the kernel reads them
        std::vector<uint16_t> kh(kd.size()), vh(kd.size());
        std::vector<uint8_t> k8(kd.size()), v8(kd.size());
        for (size_t i = 0; i < kd.size(); ++i) {
            const float kf = randn(1, 0.6f)[0], vf = randn(1, 1.0f)[0];
            if (fp8) {
                const __nv_fp8_e4m3 a(kf), b(vf);
                k8[i] = a.__x;
                v8[i] = b.__x;
                kd[i] = (double) (float) a;
                vd[i] = (double) (float) b;
            } else {
                kh[i] = f2h(kf);
                vh[i] = f2h(vf);
                kd[i] = h2d(kh[i]);
                vd[i] = h2d(vh[i]);
            }
        }
        const std::vector<float> q = randn((size_t) R * kHeads * kHeadDim, 0.8f), rs = randn((size_t) R * kHeads * kRel, 1.0f);
        const std::vector<float> sink = randn((size_t) R * kHeads, 2.0f), table = randn((size_t) kRel * kExtent, 0.3f);
        const std::vector<float> gate = randn((size_t) R * kProjQ, 1.5f);
        void* dk = fp8 ? (void*) dev(k8) : (void*) dev(kh);
        void* dv = fp8 ? (void*) dev(v8) : (void*) dev(vh);
        float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table), *out = zeros<float>((size_t) R * kHeads * kHeadDim);
        float* dgate = dev(gate);
        int* dpos = dev(pos);
        AttnArgs a;
        a.q = dq; a.r = drs; a.sink = dsink; a.rel_table = dt; a.pos = dpos; a.k_cache = dk; a.v_cache = dv; a.cap = cap; a.fp8 = fp8;
        a.window = window ? kExtent : 0;
        const int64_t max_keys = window ? kExtent : p0 + R;
        void* scr = zeros<uint8_t>(attention_decode_scratch(R, 1 << 20));
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
                    for (int i = 0; i < kHeadDim; ++i) d += (double) q[((size_t) r * kHeads + h) * kHeadDim + i] * kd[((size_t) (k % cap) * kKvHeads + g) * kHeadDim + i];
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
                    for (int k = lo; k <= pos[r]; ++k) acc += sc[(size_t) (k - lo)] * vd[((size_t) (k % cap) * kKvHeads + g) * kHeadDim + i];
                    ref[((size_t) r * kHeads + h) * kHeadDim + i] = acc / den;
                }
            }
        report(window ? "attention (window ring)" : fp8 ? "attention (full, 3k keys, E4M3)" : "attention (full, 3k keys)", got, ref, 2e-3);
        // row 2 alone, and under a launch bound of a million keys
        AttnArgs one = a;
        one.q = dq + 2 * kHeads * kHeadDim; one.r = drs + 2 * kHeads * kRel; one.sink = dsink + 2 * kHeads; one.pos = dpos + 2;
        float* out1 = zeros<float>((size_t) kHeads * kHeadDim);
        attention_decode(one, 1, max_keys, scr, out1, nullptr);
        std::vector<float> g1 = host(out1, (size_t) kHeads * kHeadDim);
        same_bits("  a row alone", g1.data(), &got[(size_t) 2 * kHeads * kHeadDim], g1.size() * 4);
        attention_decode(one, 1, 1 << 20, scr, out1, nullptr);
        g1 = host(out1, (size_t) kHeads * kHeadDim);
        same_bits("  any launch bound", g1.data(), &got[(size_t) 2 * kHeads * kHeadDim], g1.size() * 4);
        // gated, and as q8_1 blocks
        AttnArgs gq = a;
        gq.gate = dgate + kOffGate;
        gq.gate_ld = kProjQ;
        gq.q8 = zeros<uint8_t>((size_t) R * kHeads * kHeadDim / 32 * 36);
        float* outg = zeros<float>((size_t) R * kHeads * kHeadDim);
        attention_decode(gq, R, max_keys, scr, outg, nullptr);
        const std::vector<float> gg = host(outg, (size_t) R * kHeads * kHeadDim);
        std::vector<float> want(gg.size());
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < kHeads * kHeadDim; ++c) {
                const size_t i = (size_t) r * kHeads * kHeadDim + c;
                want[i] = got[i] * (1.f / (1.f + std::exp(-gate[(size_t) r * kProjQ + kOffGate + c])));
            }
        std::vector<double> wd(want.begin(), want.end());
        report("  gated", gg, wd, 1e-6);
        check_q8("  gated, its q8_1 blocks", gq.q8, gg, R, kHeads * kHeadDim);
        const double us = time_us([&] { attention_decode(one, 1, max_keys, scr, out1, nullptr); });
        const double keys = window ? kExtent : p0 + 3;
        std::printf("  %.1f us a call (1 row, %.0f keys: %.0f GB/s of K and V)\n", us, keys,
                    keys * kKvHeads * kHeadDim * (fp8 ? 1 : 2) * 2 / us / 1e3);
        if (mode == 1) {   // a short context: the latency a decode token sees early on
            std::vector<int> p1 = {40};
            int* dp1 = dev(p1);
            AttnArgs s1 = one;
            s1.pos = dp1;
            std::printf("  %.1f us a call (1 row, 41 keys)\n", time_us([&] { attention_decode(s1, 1, 1024, scr, out1, nullptr); }));
        }
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
    std::vector<unsigned long long> table(NE);
    for (int e = 0; e < NE; ++e) table[(size_t) e] = (unsigned long long) (dblobs + (size_t) e * L.bytes);
    unsigned long long* dtable = dev(table);
    int32_t *dids = dev(ids), *start = zeros<int32_t>(R * K + 2), *ng = zeros<int32_t>(4), *dst = zeros<int32_t>(R * K), *tok = zeros<int32_t>(R * K);
    unsigned long long* ptr = zeros<unsigned long long>(R * K);
    float *dw = dev(w), *dx = dev(x), *rows = zeros<float>((size_t) R * K * H), *y = zeros<float>((size_t) R * H);
    void* xq = zeros<uint8_t>(bk::native_q8_1_bytes(H, R));
    void* scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(R * K, FF));
    cudaStream_t st;
    cudaStreamCreate(&st);
    moe_plan(dids, R, K, NE, dtable, ptr, start, ng, dst, tok, st);
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
    moe_plan(dids + K, 1, K, NE, dtable, ptr, start, ng, dst, tok, st);
    bk::native_quantize_q8_1(dx + H, xq, H, 1, st);
    bk::native_expert_grouped(L, ptr, start, ng, dst, tok, K, K, xq, scr, rows, st);
    float* y1 = zeros<float>(H);
    moe_combine(rows, dw + K, 1, K, y1, st);
    cudaStreamSynchronize(st);
    const std::vector<float> g1 = host(y1, H);
    same_bits("  a row alone", g1.data(), &got[H], (size_t) H * 4);
    // an expert off the card: its pairs are left out of the plan, the others planned as before
    std::vector<unsigned long long> off = table;
    off[2] = 0;
    unsigned long long* doff = dev(off);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st);
    cudaStreamSynchronize(st);
    const int n_g = host(ng, 1)[0];
    const std::vector<int32_t> hs = host(start, (size_t) R * K + 2), hd = host(dst, (size_t) R * K);
    const std::vector<unsigned long long> hp = host(ptr, (size_t) R * K);
    int want = 0;
    for (int32_t e : ids) want += e != 2;
    bool plan_ok = n_g == NE - 1 && hs[(size_t) n_g] == want;
    for (int i = 0; plan_ok && i < want; ++i) plan_ok = ids[(size_t) hd[(size_t) i]] != 2;
    for (int gi = 0; plan_ok && gi < n_g; ++gi) plan_ok = hp[(size_t) gi] != 0 && hp[(size_t) gi] != table[2];
    std::printf("%-34s %s\n", "  an expert off the card", plan_ok ? "left out" : "FAIL");
    failures += !plan_ok;
    // the console's tally: two plans add each expert's pairs on the card to its entry, the expert off it stays at zero
    uint32_t* tally = zeros<uint32_t>(NE);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st, tally);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st, tally);
    cudaStreamSynchronize(st);
    const std::vector<uint32_t> ht = host(tally, NE);
    bool tally_ok = true;
    for (int e = 0; e < NE; ++e) {
        uint32_t n = 0;
        for (int32_t i : ids) n += i == e && e != 2 ? 2u : 0u;
        tally_ok = tally_ok && ht[(size_t) e] == n;
    }
    std::printf("%-34s %s\n", "  the experts' tally", tally_ok ? "counted" : "FAIL");
    failures += !tally_ok;
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

// ---- a decode window's fused glue ---------------------------------------------------------------------------------------

void test_mix_norm() {
    const int R = 5;
    for (int n_out : {1, 3}) {
        const int n = 4;
        std::vector<std::vector<float>> st(n);
        std::vector<const std::vector<float>*> sp;
        MixStates ms;
        for (int i = 0; i < n; ++i) {
            st[i] = randn((size_t) R * E, 0.5f + 0.4f * i);
            ms.s[ms.n++] = dev(st[i]);
            sp.push_back(&st[i]);
        }
        const std::vector<float> q = randn((size_t) n_out * E, 0.05f);
        std::vector<std::vector<float>> w(3);
        MixNorm a;
        a.st = ms;
        a.queries = dev(q);
        for (int j = 0; j < n_out; ++j) {
            w[j] = randn(E, 0.1f, 1.f);
            a.norm[j] = dev(w[j]);
            a.q8[j] = zeros<uint8_t>((size_t) R * E / 32 * 36);
        }
        a.norm_out = zeros<float>((size_t) R * E);
        mix_norm(a, n_out, R, nullptr);
        const std::vector<float> got = host(a.norm_out, (size_t) R * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r) {
            const std::vector<double> y = norm_ref(mix_ref(sp, q.data(), r), w[0]);
            std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
        }
        char name[64];
        std::snprintf(name, sizeof name, "mix_norm (%d outputs): output 0", n_out);
        report(name, got, ref, 4e-6);
        check_q8("  its q8_1 blocks", a.q8[0], got, R, E);
        if (n_out == 3) {   // output 2 against the reference, through its q8_1 blocks
            const std::vector<uint8_t> q2 = host(static_cast<uint8_t*>(a.q8[2]), (size_t) R * E / 32 * 36);
            std::vector<float> deq((size_t) R * E);
            std::vector<double> r2(deq.size());
            for (int r = 0; r < R; ++r) {
                const std::vector<double> y = norm_ref(mix_ref(sp, q.data() + 2 * E, r), w[2]);
                std::copy(y.begin(), y.end(), r2.begin() + (size_t) r * E);
            }
            for (size_t b = 0; b < deq.size() / 32; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &q2[b * 36], 2);
                for (int i = 0; i < 32; ++i) deq[b * 32 + i] = (float) (h2d(dh) * (int8_t) q2[b * 36 + 4 + i]);
            }
            report("  output 2 through q8_1", deq, r2, 1.5e-2);
        }
        // the mixes as they are: the same bits as attnres_mix
        MixNorm m2 = a;
        m2.mix_out = zeros<float>((size_t) n_out * R * E);
        m2.out_stride = (int64_t) R * E;
        mix_norm(m2, n_out, R, nullptr);
        float* old = zeros<float>((size_t) n_out * R * E);
        attnres_mix(ms, a.queries, n_out, R, 1e-6f, old, (int64_t) R * E, nullptr);
        {
            const std::vector<float> mg = host(m2.mix_out, (size_t) n_out * R * E), mo = host(old, (size_t) n_out * R * E);
            report("  the mixes against attnres_mix", mg, std::vector<double>(mo.begin(), mo.end()), 1e-6);
        }
        // row 3 alone
        MixNorm one = a;
        for (int i = 0; i < n; ++i) one.st.s[i] = ms.s[i] + 3 * E;
        float* nrm1 = zeros<float>(E);
        void* q81 = zeros<uint8_t>(E / 32 * 36);
        one.norm_out = nrm1;
        one.q8[0] = q81;
        mix_norm(one, n_out, 1, nullptr);
        same_bits("  a row alone", host(nrm1, E).data(), &got[(size_t) 3 * E], (size_t) E * 4);
        same_bits("  a row alone (q8_1)", host(static_cast<uint8_t*>(q81), E / 32 * 36).data(),
                  host(static_cast<uint8_t*>(a.q8[0]) + (size_t) 3 * E / 32 * 36, E / 32 * 36).data(), E / 32 * 36);
        std::printf("  %.1f us a call (1 row, %d states, %d outputs)\n", time_us([&] { mix_norm(one, n_out, 1, nullptr); }), n, n_out);
    }
    {   // no query: the last state normed
        const int R2 = 2;
        const std::vector<float> x = randn((size_t) R2 * E, 1.3f), w = randn(E, 0.1f, 1.f);
        MixNorm a;
        a.st.s[a.st.n++] = dev(x);
        a.norm[0] = dev(w);
        a.q8[0] = zeros<uint8_t>((size_t) R2 * E / 32 * 36);
        a.norm_out = zeros<float>((size_t) R2 * E);
        mix_norm(a, 1, R2, nullptr);
        const std::vector<float> got = host(a.norm_out, (size_t) R2 * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R2; ++r) {
            const std::vector<double> y = norm_ref(std::vector<double>(x.begin() + (size_t) r * E, x.begin() + (size_t) (r + 1) * E), w);
            std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
        }
        report("mix_norm (no query)", got, ref, 2e-6);
    }
}

void test_controllers() {
    const int R = 8, S = 3, rank = 64;
    std::vector<std::vector<float>> dest(S), msg(S), nw(S);
    std::vector<std::vector<uint16_t>> down(S), up(S);
    CtrlArgs a;
    a.rank = rank;
    a.scratch = zeros<float>(ctrl_scratch_floats(rank));
    auto bf16 = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    };
    auto bf2d = [](uint16_t h) {
        const uint32_t u = (uint32_t) h << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return (double) f;
    };
    for (int s = 0; s < S; ++s) {
        dest[s] = randn((size_t) R * E, 0.8f + 0.2f * s);
        msg[s] = randn((size_t) R * E, 1.2f);
        nw[s] = randn(E, 0.1f, 1.f);
        const std::vector<float> d = randn((size_t) rank * 2 * E, 0.02f), u = randn((size_t) 2 * E * rank, 0.3f);
        for (float f : d) down[s].push_back(bf16(f));
        for (float f : u) up[s].push_back(bf16(f));
        a.dest[s] = dev(dest[s]);
        a.msg[s] = dev(msg[s]);
        a.down[s] = dev(down[s]);
        a.up[s] = dev(up[s]);
        a.norm[s] = dev(nw[s]);
        a.q8[s] = zeros<uint8_t>((size_t) R * E / 32 * 36);
    }
    a.norm_out = zeros<float>((size_t) R * E);
    controllers(a, S, R, nullptr);
    const std::vector<float> got = host(a.norm_out, (size_t) R * E);
    auto ref_of = [&](int s, int r) {
        double sa = 0, sb = 0;
        for (int c = 0; c < E; ++c) {
            sa += (double) dest[s][(size_t) r * E + c] * dest[s][(size_t) r * E + c];
            sb += (double) msg[s][(size_t) r * E + c] * msg[s][(size_t) r * E + c];
        }
        const double ia = 1 / std::sqrt(sa / E + 1e-6), ib = 1 / std::sqrt(sb / E + 1e-6);
        std::vector<double> hid(rank, 0.0), y(E);
        for (int o = 0; o < rank; ++o)
            for (int c = 0; c < E; ++c)
                hid[o] += bf2d(down[s][(size_t) o * 2 * E + c]) * dest[s][(size_t) r * E + c] * ia +
                          bf2d(down[s][(size_t) o * 2 * E + E + c]) * msg[s][(size_t) r * E + c] * ib;
        for (int c = 0; c < E; ++c) {
            double g = 0, f = 0;
            for (int k = 0; k < rank; ++k) {
                g += bf2d(up[s][(size_t) c * rank + k]) * hid[k];
                f += bf2d(up[s][(size_t) (E + c) * rank + k]) * hid[k];
            }
            y[c] = (1 - std::tanh(f)) * dest[s][(size_t) r * E + c] + std::tanh(g) * msg[s][(size_t) r * E + c];
        }
        return norm_ref(y, nw[s]);
    };
    std::vector<double> ref(got.size());
    for (int r = 0; r < R; ++r) {
        const std::vector<double> y = ref_of(0, r);
        std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
    }
    report("controllers: stream 0", got, ref, 2e-5);
    check_q8("  its q8_1 blocks", a.q8[0], got, R, E);
    {   // stream 2 through its q8_1 blocks
        const std::vector<uint8_t> q2 = host(static_cast<uint8_t*>(a.q8[2]), (size_t) R * E / 32 * 36);
        std::vector<float> deq((size_t) R * E);
        std::vector<double> r2(deq.size());
        for (int r = 0; r < R; ++r) {
            const std::vector<double> y = ref_of(2, r);
            std::copy(y.begin(), y.end(), r2.begin() + (size_t) r * E);
        }
        for (size_t b = 0; b < deq.size() / 32; ++b) {
            uint16_t dh;
            std::memcpy(&dh, &q2[b * 36], 2);
            for (int i = 0; i < 32; ++i) deq[b * 32 + i] = (float) (h2d(dh) * (int8_t) q2[b * 36 + 4 + i]);
        }
        report("  stream 2 through q8_1", deq, r2, 1.5e-2);
    }
    CtrlArgs one = a;   // row 1 alone
    for (int s = 0; s < S; ++s) {
        one.dest[s] = a.dest[s] + E;
        one.msg[s] = a.msg[s] + E;
        one.q8[s] = zeros<uint8_t>(E / 32 * 36);
    }
    one.norm_out = zeros<float>(E);
    controllers(one, S, 1, nullptr);
    same_bits("  a row alone", host(one.norm_out, E).data(), &got[E], (size_t) E * 4);
    CtrlArgs three = a;   // the first three rows (the 3-4-row launch) against the 8-row one
    for (int s = 0; s < S; ++s) three.q8[s] = zeros<uint8_t>((size_t) 3 * E / 32 * 36);
    three.norm_out = zeros<float>((size_t) 3 * E);
    controllers(three, S, 3, nullptr);
    same_bits("  three rows", host(three.norm_out, (size_t) 3 * E).data(), got.data(), (size_t) 3 * E * 4);
    std::printf("  %.1f us a call (1 row, 3 streams)\n", time_us([&] { controllers(one, S, 1, nullptr); }));
    std::printf("  %.1f us a call (%d rows, 3 streams)\n", time_us([&] { controllers(a, S, R, nullptr); }), R);
}

void test_decode_prepare() {
    for (bool full : {false, true}) {
        const int R = 4, p0 = full ? 9000 : 1;
        const int64_t cap = full ? 9100 : kExtent + 8;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        // the projections as one [rows][kProj] array (the stride of a single product over all rows)
        const std::vector<float> y = randn((size_t) R * kProj, 1.5f);
        const std::vector<float> kc = randn(1024 * kConv, 0.3f), vc = randn(1024 * kConv, 0.3f);
        const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rs = randn(kHeads * kRel, 0.2f, 1.f);
        const std::vector<float> kr = randn((size_t) kConvRing * 1024, 2.0f), vr = randn((size_t) kConvRing * 1024, 0.6f);
        float* dy = dev(y);
        int* dpos = dev(pos);
        AttnPrep a;
        a.yq = dy; a.yk = dy + kProjQ; a.yv = dy + kProjQ + kProjKV; a.ldq = a.ldk = a.ldv = kProj;
        a.pos = dpos; a.full = full; a.k_conv = dev(kc); a.v_conv = dev(vc); a.q_scale = dev(qs); a.r_scale = dev(rs);
        a.cap = cap;
        AttnPrep b = a;
        for (AttnPrep* x : {&a, &b}) {
            x->k_ring = dev(kr); x->v_ring = dev(vr);
            x->k_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim); x->v_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim);
            x->q = zeros<float>((size_t) R * kHeads * kHeadDim); x->r = zeros<float>((size_t) R * kHeads * kRel); x->sink = zeros<float>((size_t) R * kHeads);
        }
        decode_prepare(a, R, nullptr);
        attn_prepare(b, R, nullptr);
        cudaDeviceSynchronize();
        const size_t kvb = (size_t) kKvHeads * cap * kHeadDim * 2;
        const char* tag = full ? "decode_prepare (full)" : "decode_prepare (window)";
        bool same = !std::memcmp(host((uint8_t*) a.k_cache, kvb).data(), host((uint8_t*) b.k_cache, kvb).data(), kvb) &&
                    !std::memcmp(host((uint8_t*) a.v_cache, kvb).data(), host((uint8_t*) b.v_cache, kvb).data(), kvb) &&
                    host(a.q, (size_t) R * kHeads * kHeadDim) == host(b.q, (size_t) R * kHeads * kHeadDim) &&
                    host(a.r, (size_t) R * kHeads * kRel) == host(b.r, (size_t) R * kHeads * kRel) &&
                    host(a.sink, (size_t) R * kHeads) == host(b.sink, (size_t) R * kHeads) &&
                    host(a.k_ring, (size_t) kConvRing * 1024) == host(b.k_ring, (size_t) kConvRing * 1024) &&
                    host(a.v_ring, (size_t) kConvRing * 1024) == host(b.v_ring, (size_t) kConvRing * 1024);
        std::printf("%-34s %s\n", tag, same ? "bitwise = attn_prepare" : "FAIL");
        failures += !same;
        if (!full) std::printf("  %.1f us a call (1 row)\n", time_us([&] { decode_prepare(a, 1, nullptr); }));
    }
}

// attention_window (the fused kernel where a row's keys split in at most 8 slices) against decode_prepare then
// attention_decode: the q8_1 output, the caches and the rings bitwise, for windows of 1, 3 and 8 rows over a window ring
// (also past its wrap) and a full cache, in fp16 and E4M3.
void test_attention_window() {
    struct Case { bool window; int p0, R; int64_t cap; };
    const Case cases[] = {{true, 0, 1, kExtent + 8}, {true, 300, 3, kExtent + 8}, {true, 2900, 8, kExtent + 8},
                          {false, 700, 8, 4096}, {false, 120, 2, 4096}, {false, 2000, 2, 4096},
                          {true, 0, 2, kExtent + 8}, {true, 127, 2, kExtent + 8}, {false, 1, 2, 4096}};   // 2 rows: one slice, a slice each, the ring before position 0
    for (bool fp8 : {false, true})
        for (const Case& k : cases) {
            const int R = k.R;
            std::vector<int> pos(R);
            for (int r = 0; r < R; ++r) pos[r] = k.p0 + r;
            const size_t n_kv = (size_t) kKvHeads * k.cap * kHeadDim, elem = fp8 ? 1 : 2;
            std::vector<uint8_t> kc(n_kv * elem), vc(n_kv * elem);
            for (size_t i = 0; i < n_kv; ++i) {
                const float kf = randn(1, 0.6f)[0], vf = randn(1, 1.0f)[0];
                if (fp8) {
                    kc[i] = __nv_fp8_e4m3(kf).__x;
                    vc[i] = __nv_fp8_e4m3(vf).__x;
                } else {
                    const uint16_t a = f2h(kf), b = f2h(vf);
                    std::memcpy(&kc[2 * i], &a, 2);
                    std::memcpy(&vc[2 * i], &b, 2);
                }
            }
            const std::vector<float> y = randn((size_t) R * kProj, 1.5f);
            const std::vector<float> tk = randn(1024 * kConv, 0.3f), tv = randn(1024 * kConv, 0.3f);
            const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rsc = randn(kHeads * kRel, 0.2f, 1.f);
            const std::vector<float> kr = randn((size_t) kConvRing * 1024, 2.0f), vr = randn((size_t) kConvRing * 1024, 0.6f);
            const std::vector<float> table = randn((size_t) kRel * kExtent, 0.3f);
            float* dy = dev(y);
            int* dpos = dev(pos);
            AttnPrep pa[2];
            AttnArgs aa[2];
            for (int i = 0; i < 2; ++i) {
                AttnPrep& p = pa[i];
                p.yq = dy; p.yk = dy + kProjQ; p.yv = dy + kProjQ + kProjKV; p.ldq = p.ldk = p.ldv = kProj;
                p.pos = dpos; p.full = !k.window; p.k_conv = dev(tk); p.v_conv = dev(tv); p.q_scale = dev(qs); p.r_scale = dev(rsc);
                p.k_ring = dev(kr); p.v_ring = dev(vr); p.k_cache = dev(kc); p.v_cache = dev(vc); p.cap = k.cap; p.fp8 = fp8;
                p.q = zeros<float>((size_t) R * kHeads * kHeadDim); p.r = zeros<float>((size_t) R * kHeads * kRel);
                p.sink = zeros<float>((size_t) R * kHeads);
                AttnArgs& a = aa[i];
                a.q = p.q; a.r = p.r; a.sink = p.sink; a.rel_table = dev(table); a.pos = dpos; a.k_cache = p.k_cache; a.v_cache = p.v_cache;
                a.cap = k.cap; a.fp8 = fp8; a.window = k.window ? kExtent : 0; a.gate = dy + kOffGate; a.gate_ld = kProj;
                a.q8 = zeros<uint8_t>((size_t) R * kHeads * kHeadDim / 32 * 36);
            }
            int64_t max_keys = 1024;
            while (max_keys < k.p0 + R) max_keys <<= 1;
            if (k.window) max_keys = kExtent;
            void* scr = zeros<uint8_t>(attention_decode_scratch(R, max_keys));
            decode_prepare(pa[0], R, nullptr);
            attention_decode(aa[0], R, max_keys, scr, nullptr, nullptr);
            attention_window(pa[1], aa[1], R, max_keys, scr, nullptr);
            cudaDeviceSynchronize();
            const size_t q8b = (size_t) R * kHeads * kHeadDim / 32 * 36, kvb = n_kv * elem, ringb = (size_t) kConvRing * 1024;
            const bool same = host((uint8_t*) aa[0].q8, q8b) == host((uint8_t*) aa[1].q8, q8b) &&
                              host((uint8_t*) pa[0].k_cache, kvb) == host((uint8_t*) pa[1].k_cache, kvb) &&
                              host((uint8_t*) pa[0].v_cache, kvb) == host((uint8_t*) pa[1].v_cache, kvb) &&
                              host(pa[0].k_ring, ringb) == host(pa[1].k_ring, ringb) && host(pa[0].v_ring, ringb) == host(pa[1].v_ring, ringb);
            char tag[96];
            std::snprintf(tag, sizeof tag, "attention_window (%s%s, %d rows at %d)", k.window ? "window" : "full", fp8 ? ", E4M3" : "", R, k.p0);
            std::printf("%-46s %s\n", tag, same ? "bitwise = prepare + attention" : "FAIL");
            failures += !same;
            if (!fp8 && k.window && R == 1) {
                std::printf("  %.1f us a call (1 row, 1 key, fused)  %.1f us (prepare + attention)\n",
                            time_us([&] { attention_window(pa[1], aa[1], 1, max_keys, scr, nullptr); }),
                            time_us([&] { decode_prepare(pa[0], 1, nullptr); attention_decode(aa[0], 1, max_keys, scr, nullptr, nullptr); }));
            }
            if (!fp8 && k.window && R == 3) {
                std::printf("  %.1f us a call (1 row, 301 keys, fused)  %.1f us (prepare + attention)\n",
                            time_us([&] { attention_window(pa[1], aa[1], 1, max_keys, scr, nullptr); }),
                            time_us([&] { decode_prepare(pa[0], 1, nullptr); attention_decode(aa[0], 1, max_keys, scr, nullptr, nullptr); }));
            }
        }
}

void test_post_attn() {
    for (bool boundary : {false, true}) {
        for (bool mix_only : {false, true}) {
            const int R = 4, nb = 2;
            std::vector<int> pos = {70, 71, 72, 73};
            const std::vector<float> o = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f),
                                     ring = randn((size_t) kConvRing * E, 0.4f), hid = randn((size_t) R * E, 1.0f), q = randn(E, 0.05f),
                                     w = randn(E, 0.1f, 1.f);
            std::vector<std::vector<float>> blocks(nb);
            PostAttn a;
            for (int i = 0; i < nb; ++i) {
                blocks[i] = randn((size_t) R * E, 0.7f + 0.5f * i);
                a.blocks[a.n_blocks++] = dev(blocks[i]);
            }
            a.o = dev(o); a.pos = dev(pos); a.taps = dev(taps); a.ring = dev(ring); a.hid = dev(hid); a.boundary = boundary;
            a.resid = zeros<float>((size_t) R * E); a.query = dev(q); a.norm = dev(w);
            a.q8 = zeros<uint8_t>((size_t) R * E / 32 * 36); a.norm_out = zeros<float>((size_t) R * E);
            if (mix_only) a.mix_out = zeros<float>((size_t) R * E);
            post_attn(a, R, nullptr);
            const std::vector<float> res = host(a.resid, (size_t) R * E);
            std::vector<double> rres(res.size());
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < E; ++c) {
                    double acc = 0;
                    for (int j = 0; j < kConv; ++j) acc += (double) taps[(size_t) c * kConv + (kConv - 1 - j)] * conv_input(o, E, ring, E, pos, r, j, c);
                    rres[(size_t) r * E + c] = (acc + o[(size_t) r * E + c]) / std::sqrt(5.0) + (boundary ? 0.0 : hid[(size_t) r * E + c]);
                }
            char name[80];
            std::snprintf(name, sizeof name, "post_attn%s%s: residual", boundary ? " (boundary)" : "", mix_only ? " (mix)" : "");
            report(name, res, rres, 2e-6);
            std::vector<const std::vector<float>*> sp;
            for (auto& b : blocks) sp.push_back(&b);
            if (boundary) sp.push_back(&hid);
            sp.push_back(&res);
            std::vector<double> ref((size_t) R * E);
            for (int r = 0; r < R; ++r) {
                std::vector<double> y = mix_ref(sp, q.data(), r);
                if (!mix_only) y = norm_ref(y, w);
                std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
            }
            const std::vector<float> got = host(mix_only ? a.mix_out : a.norm_out, (size_t) R * E);
            report(mix_only ? "  the MLP's mix" : "  the MLP's normed input", got, ref, 4e-6);
            if (!mix_only) check_q8("  its q8_1 blocks", a.q8, got, R, E);
            const std::vector<float> nr = host(a.ring, (size_t) kConvRing * E);
            bool ring_ok = true;
            for (int r = 0; r < R; ++r) ring_ok = ring_ok && !std::memcmp(&nr[(size_t) (pos[r] % kConvRing) * E], &o[(size_t) r * E], E * 4);
            std::printf("%-34s %s\n", "  the ring", ring_ok ? "ok" : "FAIL");
            failures += !ring_ok;
            if (!boundary && !mix_only) {   // row 2 alone (its window rows before it read from a ring holding them)
                std::vector<float> ring2 = ring;
                for (int r = 0; r < 2; ++r) std::memcpy(&ring2[(size_t) (pos[r] % kConvRing) * E], &o[(size_t) r * E], E * 4);
                PostAttn one = a;
                one.o = a.o + 2 * E; one.pos = a.pos + 2; one.ring = dev(ring2); one.hid = a.hid + 2 * E;
                for (int i = 0; i < nb; ++i) one.blocks[i] = a.blocks[i] + 2 * E;
                one.resid = zeros<float>(E); one.norm_out = zeros<float>(E); one.q8 = zeros<uint8_t>(E / 32 * 36);
                post_attn(one, 1, nullptr);
                same_bits("  a row alone", host(one.norm_out, E).data(), &got[(size_t) 2 * E], (size_t) E * 4);
                std::printf("  %.1f us a call (1 row, %d blocks)\n", time_us([&] { post_attn(one, 1, nullptr); }), nb);
            }
        }
    }
}

// The guessed experts' prefetch over a stand-in blob table runs and reads only.
void test_expert_prefetch() {
    const int NE = 128;
    const size_t blob = 65536;
    uint8_t* blobs = zeros<uint8_t>((size_t) NE * blob);
    std::vector<unsigned long long> tab(NE);
    for (int e = 0; e < NE; ++e) tab[e] = (unsigned long long) (blobs + (size_t) e * blob);
    const std::vector<int32_t> ids = {3, 17, 64, 127, 0, 90, 41, 8};
    unsigned long long* dtab = dev(tab);
    int32_t* dids = dev(ids);
    uint32_t* sink = zeros<uint32_t>(1);
    expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr);
    const bool ok = cudaDeviceSynchronize() == cudaSuccess;
    std::printf("%-34s %s\n", "expert_prefetch: runs", ok ? "ok" : "FAIL");
    failures += !ok;
    std::printf("  %.1f us a call (8 blobs of 64 KiB)\n", time_us([&] { expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr); }));
}

void test_moe_finish() {
    const int R = 4, K = 8;
    std::vector<int> pos = {130, 131, 132, 133};
    const std::vector<float> rows = randn((size_t) R * K * E, 0.3f), w = randn((size_t) R * K, 0.3f, 1.f), taps = randn((size_t) E * kConv, 0.3f),
                             ring = randn((size_t) kConvRing * E, 0.4f), resid = randn((size_t) R * E, 1.0f);
    MoeFinish a;
    a.rows = dev(rows); a.w = dev(w); a.k = K; a.pos = dev(pos); a.taps = dev(taps); a.ring = dev(ring); a.resid = dev(resid);
    a.out = zeros<float>((size_t) R * E);
    moe_finish(a, R, nullptr);
    // the kernels it replaces: moe_combine, then post_conv with the residual
    float *y = zeros<float>((size_t) R * E), *out2 = zeros<float>((size_t) R * E), *ring2 = dev(ring);
    moe_combine(a.rows, a.w, R, K, y, nullptr);
    post_conv(y, a.pos, a.taps, ring2, a.resid, R, out2, nullptr);
    const std::vector<float> got = host(a.out, (size_t) R * E);
    same_bits("moe_finish = moe_combine + post_conv", got.data(), host(out2, (size_t) R * E).data(), (size_t) R * E * 4);
    same_bits("  the ring", host(a.ring, (size_t) kConvRing * E).data(), host(ring2, (size_t) kConvRing * E).data(), (size_t) kConvRing * E * 4);
    MoeFinish t = a;   // with the drafter's tap: the same rows in both outputs
    t.ring = dev(ring);
    t.out = zeros<float>((size_t) R * E);
    t.tap = zeros<float>((size_t) R * E);
    moe_finish(t, R, nullptr);
    same_bits("  with a tap: the output", host(t.out, (size_t) R * E).data(), got.data(), (size_t) R * E * 4);
    same_bits("  with a tap: the tap", host(t.tap, (size_t) R * E).data(), got.data(), (size_t) R * E * 4);
    MoeFinish one = a;
    one.rows = a.rows + (size_t) 3 * K * E; one.w = a.w + 3 * K; one.pos = a.pos + 3; one.resid = a.resid + 3 * E; one.out = zeros<float>(E);
    std::printf("  %.1f us a call (1 row)\n", time_us([&] { moe_finish(one, 1, nullptr); }));
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
    test_mix_norm();
    test_controllers();
    test_decode_prepare();
    test_attention_window();
    test_post_attn();
    test_expert_prefetch();
    test_moe_finish();
    std::printf("inkling2_kernels_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
