// src/models/hy4flash/kernels/kernels_parity.cpp - the Hy4-flash decode kernels against plain CPU references (double
// accumulation) at the model's geometry: the identity hyper-connections, the per-head Q8_0 products, the MLA
// preparation and attention with a sink, the router and the Engram apply. Exit 77 without a GPU.
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::hy4flash;
namespace bk = basalt::kernels;

namespace {

std::mt19937 rng(11);
std::vector<float> randn(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
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
float* devf(size_t n) {
    float* p = nullptr;
    cudaMalloc(&p, n * 4 + 16);
    return p;
}
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int failures = 0;
void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-30s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}

uint16_t f2h(float f) {   // fp16, round to nearest even (normal range)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000, e = (u >> 23) & 0xff, m = u & 0x7fffff;
    if (e < 113) return (uint16_t) sign;
    uint32_t h = ((e - 112) << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) ++h;
    return (uint16_t) (sign | h);
}
float h2f(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp((double) m, -24);
    return (float) ((h & 0x8000) ? -v : v);
}

// Q8_0 rows [rows][k]: the bytes and the dequantised values.
struct Q8 {
    std::vector<uint8_t> bytes;
    std::vector<float> vals;
};
Q8 make_q8(int rows, int k) {
    Q8 q;
    q.vals.resize((size_t) rows * k);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < k / 32; ++b) {
            const uint16_t d = f2h(std::fabs(randn(1, 0.02f)[0]) + 1e-3f);
            q.bytes.push_back((uint8_t) (d & 255));
            q.bytes.push_back((uint8_t) (d >> 8));
            for (int i = 0; i < 32; ++i) {
                const int8_t v = (int8_t) std::clamp((int) std::lround(randn(1, 40.f)[0]), -127, 127);
                q.bytes.push_back((uint8_t) v);
                q.vals[(size_t) r * k + b * 32 + i] = h2f(d) * (float) v;
            }
        }
    return q;
}

void test_hc() {
    const int E = 3072, T = 3;
    for (int n_mix : {8, 4}) {
        std::vector<float> x = randn((size_t) T * 4 * E), fn = randn((size_t) n_mix * 4 * E, 0.02f), base = randn(n_mix, 0.5f),
                           scale = {0.7f, 0.3f};
        bk::HcPlainArgs a;
        const int stride = bk::hc_plain_coef_stride(n_mix);
        float *dx = dev(x), *dfn = dev(fn), *dbase = dev(base), *dscale = dev(scale);
        float* ws = devf(bk::hc_plain_workspace_floats(T, n_mix));
        float* coef = devf((size_t) T * stride);
        bk::hc_mixes_plain(dx, dfn, dscale, dbase, T, n_mix, a, ws, coef, nullptr);
        std::vector<float> got = host(coef, (size_t) T * stride);
        std::vector<double> ref((size_t) T * stride, 0.0);
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int i = 0; i < 4 * E; ++i) ss += (double) x[(size_t) t * 4 * E + i] * x[(size_t) t * 4 * E + i];
            const double rs = 1.0 / std::sqrt(ss / (4 * E) + a.rms_eps);
            for (int m = 0; m < n_mix; ++m) {
                double d = 0;
                for (int i = 0; i < 4 * E; ++i) d += (double) x[(size_t) t * 4 * E + i] * fn[(size_t) m * 4 * E + i];
                d *= rs;
                double* o = &ref[(size_t) t * stride];
                if (m < 4) o[m] = sig(d * scale[0] + base[m]) + a.hc_eps;
                else if (n_mix == 8) o[m] = a.magnitude * sig(d * scale[1] + base[m]) + a.hc_eps;
            }
            if (n_mix == 8)
                for (int e = 0; e < 16; ++e) ref[(size_t) t * stride + 8 + e] = (e / 4 == e % 4) ? 1.0 : 0.0;
        }
        report(n_mix == 8 ? "hc_mixes_plain (8 mixes)" : "hc_mixes_plain (head)", got, ref, 1e-5);
        // T invariance: token 1 alone
        float* coef1 = devf(stride);
        bk::hc_mixes_plain(dx + 4 * E, dfn, dscale, dbase, 1, n_mix, a, ws, coef1, nullptr);
        std::vector<float> one = host(coef1, stride);
        bool same = std::memcmp(one.data(), got.data() + stride, stride * 4) == 0;
        std::printf("%-30s %s\n", "  T invariance", same ? "bitwise" : "FAIL");
        failures += !same;
        if (n_mix == 8) {   // post: out[i] = x[i] + post[i] f, through hc_post
            std::vector<float> f = randn((size_t) T * E);
            float *df = dev(f), *dout = devf((size_t) T * 4 * E);
            bk::hc_post(df, dx, coef, stride, T, E, dout, nullptr);
            std::vector<float> o = host(dout, (size_t) T * 4 * E);
            std::vector<double> r2(o.size());
            for (int t = 0; t < T; ++t)
                for (int s = 0; s < 4; ++s)
                    for (int d = 0; d < E; ++d)
                        r2[((size_t) t * 4 + s) * E + d] = (double) x[((size_t) t * 4 + s) * E + d] + ref[(size_t) t * stride + 4 + s] * f[(size_t) t * E + d];
            report("hc_post (identity comb)", o, r2, 1e-6);
        }
    }
}

void test_gemv() {
    for (auto [heads, rows, k] : {std::tuple<int, int, int>{32, 512, 192}, {64, 256, 512}}) {
        Q8 w = make_q8(heads * rows, k);
        const int xs = k == 192 ? 256 : k;   // the query's head stride vs a packed latent
        std::vector<float> x = randn((size_t) heads * xs);
        uint8_t* dw = dev(w.bytes);
        float *dx = dev(x), *dy = devf((size_t) heads * rows);
        q8_0_heads_gemv(dw, dx, xs, heads, rows, k, dy, nullptr);
        std::vector<float> got = host(dy, (size_t) heads * rows);
        std::vector<double> ref((size_t) heads * rows);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < rows; ++r) {
                double d = 0;
                for (int i = 0; i < k; ++i) d += (double) w.vals[((size_t) h * rows + r) * k + i] * x[(size_t) h * xs + i];
                ref[(size_t) h * rows + r] = d;
            }
        char name[64];
        std::snprintf(name, sizeof name, "q8_0_heads_gemv %dx%dx%d", heads, rows, k);
        report(name, got, ref, 1e-5);
    }
}

void test_mla() {
    for (auto [heads, base_f, window] : {std::tuple<int, float, int>{32, 1e7f, 0}, {64, 1e4f, 512}}) {
        MlaArgs a;
        a.heads = heads;
        a.rope_base = base_f;
        const int W = a.rank + a.rope, Q = a.nope + a.rope, cells = 700, pos = cells - 1;
        std::vector<float> q = randn((size_t) heads * Q), qabs = randn((size_t) heads * a.rank, 0.3f), kvraw = randn(W),
                           norm = randn(a.rank, 0.2f), cache = randn((size_t) cells * W, 0.5f), sinks = randn(heads, 1.f);
        for (auto& v : norm) v += 1.f;
        float *dq = dev(q), *dqa = dev(qabs), *dkv = dev(kvraw), *dn = dev(norm), *dc = dev(cache), *ds = dev(sinks);
        float *dqcur = devf((size_t) heads * W), *dlat = devf((size_t) heads * a.rank);
        mla_prepare(dq, dqa, dkv, dn, a, pos, dqcur, dc + (size_t) pos * W, nullptr);
        std::vector<float> qcur = host(dqcur, (size_t) heads * W), row = host(dc + (size_t) pos * W, W);
        // references
        std::vector<double> rq((size_t) heads * W), rrow(W);
        auto rot = [&](const float* p, double* o, int h_unused) {
            (void) h_unused;
            for (int i = 0; i < a.rope / 2; ++i) {
                const double th = pos * std::pow((double) base_f, -2.0 * i / a.rope);
                o[2 * i] = p[2 * i] * std::cos(th) - p[2 * i + 1] * std::sin(th);
                o[2 * i + 1] = p[2 * i] * std::sin(th) + p[2 * i + 1] * std::cos(th);
            }
        };
        for (int h = 0; h < heads; ++h) {
            for (int c = 0; c < a.rank; ++c) rq[(size_t) h * W + c] = qabs[(size_t) h * a.rank + c];
            rot(&q[(size_t) h * Q + a.nope], &rq[(size_t) h * W + a.rank], h);
        }
        double ss = 0;
        for (int c = 0; c < a.rank; ++c) ss += (double) kvraw[c] * kvraw[c];
        const double rs = 1.0 / std::sqrt(ss / a.rank + a.rms_eps);
        for (int c = 0; c < a.rank; ++c) rrow[c] = kvraw[c] * rs * norm[c];
        rot(&kvraw[a.rank], &rrow[a.rank], 0);
        report("mla_prepare query", qcur, rq, 2e-5);
        report("mla_prepare cache row", row, rrow, 2e-5);
        // attention over the window, over the prepared (device) rows
        cache = host(dc, (size_t) cells * W);
        const int first = window ? std::max(0, pos + 1 - window) : 0, n = pos + 1 - first;
        mla_attend(dqcur, dc, first, n, ds, a, dlat, nullptr);
        std::vector<float> got = host(dlat, (size_t) heads * a.rank);
        std::vector<double> ref((size_t) heads * a.rank);
        for (int h = 0; h < heads; ++h) {
            std::vector<double> s(n);
            double m = sinks[h];
            for (int j = 0; j < n; ++j) {
                double d = 0;
                for (int c = 0; c < W; ++c) d += (double) qcur[(size_t) h * W + c] * cache[(size_t) (first + j) * W + c];
                s[j] = d * a.scale;
                m = std::max(m, s[j]);
            }
            double den = std::exp(sinks[h] - m);
            for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - m); den += s[j]; }
            for (int c = 0; c < a.rank; ++c) {
                double o = 0;
                for (int j = 0; j < n; ++j) o += s[j] * cache[(size_t) (first + j) * W + c];
                ref[(size_t) h * a.rank + c] = o / den;
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "mla_attend %d heads, %d cells", heads, n);
        report(name, got, ref, 1e-4);
    }
}

void test_router() {
    const int NE = 512, E = 3072, K = 16;
    std::vector<float> x = randn(E), gate = randn((size_t) NE * E, 0.05f), bias = randn(NE, 0.3f);
    float *dx = dev(x), *dg = dev(gate), *db = dev(bias), *dl = devf(NE), *dw = devf(K);
    int32_t* di = nullptr;
    cudaMalloc(&di, K * 4);
    bk::RouterArgs ra;
    ra.n_expert = NE;
    ra.n_embd = E;
    ra.k = K;
    ra.weight_scale = 3.998f;
    ra.gate = bk::RouterGate::Sigmoid;
    bk::router(dx, dg, db, 1, ra, dl, di, dw, nullptr);
    std::vector<int32_t> ids = host(di, K);
    std::vector<float> w = host(dw, K), logits = host(dl, NE);
    std::vector<double> s(NE), sel(NE), rlog(NE);
    for (int e = 0; e < NE; ++e) {
        double d = 0;
        for (int i = 0; i < E; ++i) d += (double) gate[(size_t) e * E + i] * x[i];
        rlog[e] = d;
        s[e] = sig(d);
        sel[e] = s[e] + bias[e];
    }
    std::vector<int> order(NE);
    for (int i = 0; i < NE; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    double sum = 0;
    for (int r = 0; r < K; ++r) sum += s[order[r]];
    bool same = true;
    std::vector<double> rw(K);
    for (int r = 0; r < K; ++r) {
        same = same && ids[r] == order[r];
        rw[r] = s[order[r]] / sum * 3.998;
    }
    report("router logits", logits, rlog, 1e-5);
    report("router weights", w, rw, 1e-5);
    std::printf("%-30s %s\n", "router ids", same ? "equal" : "FAIL");
    failures += !same;
}

void test_engram() {
    const int n = 3072;
    EngramArgs a;
    std::vector<float> x = randn((size_t) 4 * n), kv = randn((size_t) 5 * n), kvb = randn((size_t) 5 * n, 0.1f), gw = randn((size_t) 4 * n, 0.3f),
                       gamma = randn(n, 0.5f), hs = randn((size_t) 4 * n, 0.2f);
    for (float valid : {1.f, 0.f}) {
        float *dx = dev(x), *dkv = dev(kv), *dkb = dev(kvb), *dgw = dev(gw), *dgm = dev(gamma), *dhs = dev(hs);
        engram_apply(dkv, dkb, dgw, dgm, dhs, valid, a, dx, nullptr);
        std::vector<float> got = host(dx, (size_t) 4 * n);
        std::vector<double> ref((size_t) 4 * n);
        auto rms = [&](const std::vector<double>& v) {
            double ss = 0;
            for (double e : v) ss += e * e;
            return 1.0 / std::sqrt(ss / v.size() + a.eps);
        };
        std::vector<double> val(n);
        for (int j = 0; j < n; ++j) val[j] = (double) kv[(size_t) 4 * n + j] + kvb[(size_t) 4 * n + j];
        const double rv = rms(val);
        for (int i = 0; i < 4; ++i) {
            std::vector<double> xi(n), key(n);
            for (int j = 0; j < n; ++j) {
                xi[j] = x[(size_t) i * n + j];
                key[j] = (double) kv[(size_t) i * n + j] + kvb[(size_t) i * n + j];
            }
            const double rx = rms(xi), rk = rms(key);
            double d = 0;
            for (int j = 0; j < n; ++j) d += xi[j] * rx * key[j] * rk * gw[(size_t) i * n + j];
            d /= std::sqrt((double) n);
            const double mag = std::sqrt(std::max(std::fabs(d), 1e-6));
            const double g = sig(d < 0 ? -mag : mag);
            for (int j = 0; j < n; ++j)
                ref[(size_t) i * n + j] = xi[j] + valid * val[j] * rv * gamma[j] * (g + hs[(size_t) i * n + j]);
        }
        report(valid > 0 ? "engram_apply" : "engram_apply (masked)", got, ref, 2e-6);
    }
}

void test_indexer() {
    IndexerArgs a;
    const int D = a.dim, P = a.pool;
    // pooling
    std::vector<float> vals = randn((size_t) P * D), gates = randn((size_t) P * D), ape = randn((size_t) P * D, 0.3f), nw = randn(D, 0.2f);
    for (auto& v : nw) v += 1.f;
    const int first = 1236;
    float *dv = dev(vals), *dg = dev(gates), *da = dev(ape), *dn = dev(nw), *dp = devf(D);
    idx_pool(dv, dg, da, dn, a, first, dp, nullptr);
    std::vector<float> got = host(dp, D);
    std::vector<double> ref(D), row(D);
    for (int d = 0; d < D; ++d) {
        double mx = -1e30, s = 0, acc = 0;
        for (int j = 0; j < P; ++j) mx = std::max(mx, (double) gates[(size_t) j * D + d] + ape[(size_t) j * D + d]);
        for (int j = 0; j < P; ++j) s += std::exp(gates[(size_t) j * D + d] + ape[(size_t) j * D + d] - mx);
        for (int j = 0; j < P; ++j) acc += std::exp(gates[(size_t) j * D + d] + ape[(size_t) j * D + d] - mx) / s * vals[(size_t) j * D + d];
        row[d] = acc;
    }
    double ss = 0;
    for (double v : row) ss += v * v;
    for (int d = 0; d < D; ++d) row[d] = row[d] / std::sqrt(ss / D + a.eps) * nw[d];
    ref = row;
    for (int i = 0; i < a.rope / 2; ++i) {
        const double th = first * std::pow((double) a.rope_base, -2.0 * i / a.rope);
        const double x0 = row[D - a.rope + 2 * i], x1 = row[D - a.rope + 2 * i + 1];
        ref[D - a.rope + 2 * i] = x0 * std::cos(th) - x1 * std::sin(th);
        ref[D - a.rope + 2 * i + 1] = x0 * std::sin(th) + x1 * std::cos(th);
    }
    report("idx_pool", got, ref, 2e-5);
    // scores and selection over many pools, with ties
    const int NP = 5000, K = 511;
    std::vector<float> q = randn((size_t) a.heads * D), w = randn(a.heads, 0.05f), pooled = randn((size_t) NP * D);
    float *dq = dev(q), *dw = dev(w), *dpool = dev(pooled), *dsc = devf(NP);
    idx_scores(dq, dw, dpool, NP, a, dsc, nullptr);
    std::vector<float> sc = host(dsc, NP);
    std::vector<double> rsc(NP);
    for (int p = 0; p < NP; ++p) {
        double t = 0;
        for (int h = 0; h < a.heads; ++h) {
            double d = 0;
            for (int i = 0; i < D; ++i) d += (double) q[(size_t) h * D + i] * pooled[(size_t) p * D + i];
            t += w[h] * std::max(d, 0.0);
        }
        rsc[p] = t;
    }
    report("idx_scores", sc, rsc, 1e-5);
    for (int ties = 0; ties < 2; ++ties) {
        std::vector<float> s2 = sc;
        if (ties)   // many equal scores around the cut
            for (int p = 0; p < NP; ++p) s2[p] = std::floor(s2[p] * 4.f) / 4.f;
        float* ds2 = dev(s2);
        const int tail0 = NP * P, tail_n = 3;
        int32_t* dc = nullptr;
        cudaMalloc(&dc, (size_t) (K * P + tail_n) * 4);
        idx_select(ds2, NP, K, P, tail0, tail_n, dc, nullptr);
        std::vector<int32_t> cells = host(dc, (size_t) K * P + tail_n);
        std::vector<int> order(NP);
        for (int i = 0; i < NP; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return s2[x] > s2[y]; });
        std::vector<int> pick(order.begin(), order.begin() + K);
        std::sort(pick.begin(), pick.end());
        bool ok = true;
        for (int i = 0; i < K && ok; ++i)
            for (int j = 0; j < P; ++j) ok = ok && cells[(size_t) i * P + j] == pick[i] * P + j;
        for (int j = 0; j < tail_n; ++j) ok = ok && cells[(size_t) K * P + j] == tail0 + j;
        std::printf("%-30s %s\n", ties ? "idx_select (ties)" : "idx_select", ok ? "equal" : "FAIL");
        failures += !ok;
    }
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void test_mla_bf16() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, Q = a.nope + a.rope, cells = 900, pos = cells - 1, n = cells;
    std::vector<float> q = randn((size_t) a.heads * Q), qabs = randn((size_t) a.heads * a.rank, 0.3f), kvraw = randn(W),
                       norm = randn(a.rank, 0.2f), sinks = randn(a.heads);
    for (auto& v : norm) v += 1.f;
    std::vector<float> rows = randn((size_t) cells * W, 0.5f);
    std::vector<uint16_t> bf((size_t) cells * W);
    for (size_t i = 0; i < bf.size(); ++i) bf[i] = to_bf16(rows[i]);
    float *dq = dev(q), *dqa = dev(qabs), *dkv = dev(kvraw), *dn = dev(norm), *ds = dev(sinks);
    uint16_t* dc = dev(bf);
    float *dqcur = devf((size_t) a.heads * W), *dlat = devf((size_t) a.heads * a.rank);
    mla_prepare(dq, dqa, dkv, dn, a, pos, dqcur, dc + (size_t) pos * W, nullptr);
    std::vector<uint16_t> row = host(dc + (size_t) pos * W, W);
    // the row against fp32 preparation, within BF16's rounding
    float* dr32 = devf(W);
    mla_prepare(dq, dqa, dkv, dn, a, pos, dqcur, dr32, nullptr);
    std::vector<float> r32 = host(dr32, W);
    bool exact = true;
    for (int c = 0; c < W; ++c) exact = exact && row[c] == to_bf16(r32[c]);
    std::printf("%-30s %s\n", "mla_prepare bf16 row", exact ? "bitwise = rn(fp32 row)" : "FAIL");
    failures += !exact;
    bf = host(dc, (size_t) cells * W);
    std::vector<float> qcur = host(dqcur, (size_t) a.heads * W);
    mla_attend(dqcur, dc, 0, n, ds, a, dlat, nullptr);
    std::vector<float> got = host(dlat, (size_t) a.heads * a.rank);
    std::vector<double> ref((size_t) a.heads * a.rank);
    for (int h = 0; h < a.heads; ++h) {
        std::vector<double> sc(n);
        double m = sinks[h];
        for (int j = 0; j < n; ++j) {
            double d = 0;
            for (int c = 0; c < W; ++c) d += (double) qcur[(size_t) h * W + c] * from_bf16(bf[(size_t) j * W + c]);
            sc[j] = d * a.scale;
            m = std::max(m, sc[j]);
        }
        double den = std::exp(sinks[h] - m);
        for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - m); den += sc[j]; }
        for (int c = 0; c < a.rank; ++c) {
            double o = 0;
            for (int j = 0; j < n; ++j) o += sc[j] * from_bf16(bf[(size_t) j * W + c]);
            ref[(size_t) h * a.rank + c] = o / den;
        }
    }
    report("mla_attend (bf16 cache)", got, ref, 1e-4);
}

void test_attend_cells() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, cells_n = 3000, n = 2052;
    std::vector<float> qcur = randn((size_t) a.heads * W, 0.3f), cache = randn((size_t) cells_n * W, 0.5f), sinks = randn(a.heads);
    std::vector<int32_t> cells(n);
    for (int j = 0; j < n; ++j) cells[j] = (j * 7 + 3) % cells_n;   // distinct for n <= cells_n (7 and 3000 are coprime)
    std::sort(cells.begin(), cells.end());
    float *dq = dev(qcur), *dc = dev(cache), *ds = dev(sinks), *dl = devf((size_t) a.heads * a.rank);
    int32_t* dcells = dev(cells);
    mla_attend(dq, dc, 0, n, ds, a, dl, nullptr, dcells);
    std::vector<float> got = host(dl, (size_t) a.heads * a.rank);
    std::vector<double> ref((size_t) a.heads * a.rank);
    for (int h = 0; h < a.heads; ++h) {
        std::vector<double> s(n);
        double m = sinks[h];
        for (int j = 0; j < n; ++j) {
            double d = 0;
            for (int c = 0; c < W; ++c) d += (double) qcur[(size_t) h * W + c] * cache[(size_t) cells[j] * W + c];
            s[j] = d * a.scale;
            m = std::max(m, s[j]);
        }
        double den = std::exp(sinks[h] - m);
        for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - m); den += s[j]; }
        for (int c = 0; c < a.rank; ++c) {
            double o = 0;
            for (int j = 0; j < n; ++j) o += s[j] * cache[(size_t) cells[j] * W + c];
            ref[(size_t) h * a.rank + c] = o / den;
        }
    }
    report("mla_attend (cell list)", got, ref, 1e-4);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    test_hc();
    test_gemv();
    test_mla();
    test_router();
    test_engram();
    test_indexer();
    test_attend_cells();
    test_mla_bf16();
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
