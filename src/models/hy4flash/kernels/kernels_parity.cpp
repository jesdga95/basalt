// src/models/hy4flash/kernels/kernels_parity.cpp - the Hy4-flash decode kernels against plain CPU references (double
// accumulation) at the model's geometry: the identity hyper-connections, the per-head Q8_0 products, the MLA
// preparation and attention with a sink, the router and the Engram apply. Exit 77 without a GPU.
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/prefill/moe_mmq.hpp"
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

void test_gemv() {
    const int T = 3;
    for (auto [heads, rows, k] : {std::tuple<int, int, int>{32, 512, 192}, {64, 256, 512}}) {
        Q8 w = make_q8(heads * rows, k);
        const int xs = k == 192 ? 256 : k;   // the query's head stride vs a packed latent
        const int tok = heads * xs;
        std::vector<float> x = randn((size_t) T * tok);
        uint8_t* dw = dev(w.bytes);
        float *dx = dev(x), *dy = devf((size_t) T * heads * rows);
        q8_0_heads_gemv(dw, dx, xs, tok, heads, rows, k, T, dy, nullptr);
        std::vector<float> got = host(dy, (size_t) T * heads * rows);
        // the reference rounds the activation to q8_1 per 32 values as the kernel does (d = amax / 127, half away from zero) and
        // keeps the integer sums exact; the exact-activation product is compared too
        std::vector<double> ref((size_t) T * heads * rows), exact((size_t) T * heads * rows);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < heads; ++h)
                for (int r = 0; r < rows; ++r) {
                    const uint8_t* row = &w.bytes[((size_t) h * rows + r) * (k / 32) * 34];
                    double acc = 0, ex = 0;
                    for (int b = 0; b < k / 32; ++b) {
                        const uint8_t* blk = row + (size_t) b * 34;
                        const float dw = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
                        const float* xv = &x[(size_t) t * tok + (size_t) h * xs + b * 32];
                        float amax = 0;
                        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xv[i]));
                        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
                        long isum = 0;
                        for (int i = 0; i < 32; ++i) {
                            isum += (long) (int8_t) blk[2 + i] * (long) std::lround(xv[i] * id);
                            ex += (double) dw * (int8_t) blk[2 + i] * xv[i];
                        }
                        acc += (double) (dw * d) * (double) isum;
                    }
                    ref[((size_t) t * heads + h) * rows + r] = acc;
                    exact[((size_t) t * heads + h) * rows + r] = ex;
                }
        char name[64];
        std::snprintf(name, sizeof name, "q8_0_heads_gemv %dx%dx%d", heads, rows, k);
        report(name, got, ref, 2e-5);
        report("  against exact activations", got, exact, 5e-2);   // the q8_1 rounding
        float* d1 = devf((size_t) heads * rows);
        q8_0_heads_gemv(dw, dx + (size_t) 1 * tok, xs, tok, heads, rows, k, 1, d1, nullptr);
        const bool same = std::memcmp(host(d1, (size_t) heads * rows).data(), got.data() + (size_t) heads * rows, (size_t) heads * rows * 4) == 0;
        std::printf("%-30s %s\n", "  row independence", same ? "bitwise" : "FAIL");
        failures += !same;
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

// a reference for the attention of one row over `cells` of a cache (given as floats)
std::vector<double> ref_attend(const std::vector<float>& qcur, const std::vector<float>& cache, const std::vector<int>& cells, int h,
                               const MlaArgs& a, float sink) {
    const int W = a.rank + a.rope, n = (int) cells.size();
    std::vector<double> s(n);
    double m = sink;
    for (int j = 0; j < n; ++j) {
        double d = 0;
        for (int c = 0; c < W; ++c) d += (double) qcur[(size_t) h * W + c] * cache[(size_t) cells[j] * W + c];
        s[j] = d * a.scale;
        m = std::max(m, s[j]);
    }
    double den = std::exp(sink - m);
    for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - m); den += s[j]; }
    std::vector<double> o(a.rank);
    for (int c = 0; c < a.rank; ++c) {
        double acc = 0;
        for (int j = 0; j < n; ++j) acc += s[j] * cache[(size_t) cells[j] * W + c];
        o[c] = acc / den;
    }
    return o;
}

void test_mla() {
    // 1: three rows at pos0, a window ring; 2: the rows over explicit cell lists; both with fp32 and BF16 caches
    for (auto [heads, base_f, ring] : {std::tuple<int, float, int>{32, 1e7f, 0}, {64, 1e4f, 700}}) {
        MlaArgs a;
        a.heads = heads;
        a.rope_base = base_f;
        const int W = a.rank + a.rope, Q = a.nope + a.rope, T = 12, pos0 = 650, cap = ring ? ring : 1000;
        std::vector<float> q = randn((size_t) T * heads * Q), qabs = randn((size_t) T * heads * a.rank, 0.3f), kvraw = randn((size_t) T * W),
                           norm = randn(a.rank, 0.2f), sinks = randn(heads, 1.f), cache0 = randn((size_t) cap * W, 0.5f);
        for (auto& v : norm) v += 1.f;
        for (int bf = 0; bf < 2; ++bf) {
            float *dq = dev(q), *dqa = dev(qabs), *dkv = dev(kvraw), *dn = dev(norm), *ds = dev(sinks);
            float *dqcur = devf((size_t) T * heads * W), *dlat = devf((size_t) T * heads * a.rank);
            std::vector<float> cache = cache0;
            std::vector<uint16_t> cbf(cache.size());
            for (size_t i = 0; i < cache.size(); ++i) cbf[i] = to_bf16(cache[i]);
            float* dc32 = dev(cache);
            uint16_t* dc16 = dev(cbf);
            if (bf) mla_prepare(dq, dqa, dkv, dn, a, pos0, T, ring, dqcur, dc16, nullptr);
            else mla_prepare(dq, dqa, dkv, dn, a, pos0, T, ring, dqcur, dc32, nullptr);
            std::vector<float> qcur = host(dqcur, (size_t) T * heads * W);
            std::vector<float> rows32 = bf ? std::vector<float>() : host(dc32, cache.size());
            if (bf) {
                std::vector<uint16_t> r = host(dc16, cbf.size());
                rows32.resize(r.size());
                for (size_t i = 0; i < r.size(); ++i) rows32[i] = from_bf16(r[i]);
            }
            // references: query rows and cache rows
            std::vector<double> rq((size_t) T * heads * W);
            bool rows_ok = true;
            for (int t = 0; t < T; ++t) {
                const int pos = pos0 + t;
                auto rot = [&](const float* p, double* o) {
                    for (int i = 0; i < a.rope / 2; ++i) {
                        const double th = pos * std::pow((double) base_f, -2.0 * i / a.rope);
                        o[2 * i] = p[2 * i] * std::cos(th) - p[2 * i + 1] * std::sin(th);
                        o[2 * i + 1] = p[2 * i] * std::sin(th) + p[2 * i + 1] * std::cos(th);
                    }
                };
                for (int h = 0; h < heads; ++h) {
                    for (int c = 0; c < a.rank; ++c) rq[((size_t) t * heads + h) * W + c] = qabs[((size_t) t * heads + h) * a.rank + c];
                    rot(&q[((size_t) t * heads + h) * Q + a.nope], &rq[((size_t) t * heads + h) * W + a.rank]);
                }
                double ss = 0;
                for (int c = 0; c < a.rank; ++c) ss += (double) kvraw[(size_t) t * W + c] * kvraw[(size_t) t * W + c];
                const double rs = 1.0 / std::sqrt(ss / a.rank + a.rms_eps);
                std::vector<double> row(W);
                for (int c = 0; c < a.rank; ++c) row[c] = kvraw[(size_t) t * W + c] * rs * norm[c];
                rot(&kvraw[(size_t) t * W + a.rank], &row[a.rank]);
                const size_t cell = ring ? pos % ring : pos;
                for (int c = 0; c < W; ++c) {
                    const double got = rows32[cell * W + c];
                    rows_ok = rows_ok && std::fabs(got - row[c]) <= (bf ? 8e-3 * (std::fabs(row[c]) + 1e-2) : 6e-5 + 3e-5 * std::fabs(row[c]));   // the rotation's angle is fp32
                }
            }
            std::vector<float> qcur_ref(rq.begin(), rq.end());
            report(bf ? "mla_prepare query (bf16 run)" : "mla_prepare query", qcur, rq, 3e-5);
            std::printf("%-30s %s\n", bf ? "mla_prepare rows (bf16)" : "mla_prepare rows", rows_ok ? "ok" : "FAIL");
            failures += !rows_ok;
            // attention: row t over the last (up to) 512 positions, or over a given list
            std::vector<int32_t> firsts(T), counts(T), cells_list;
            const int stride = 1100;
            cells_list.assign((size_t) T * stride, 0);
            for (int use_cells = 0; use_cells < 2; ++use_cells) {
                std::vector<std::vector<int>> lists(T);
                for (int t = 0; t < T; ++t) {
                    const int pos = pos0 + t;
                    if (!use_cells) {
                        const int first = ring ? std::max(0, pos + 1 - 512) : 0;
                        firsts[t] = first;
                        counts[t] = pos + 1 - first;
                        for (int j = 0; j < counts[t]; ++j) lists[t].push_back(ring ? (first + j) % ring : first + j);
                    } else {   // every 3rd cell of the cap: a selection
                        firsts[t] = 0;
                        for (int c = t % 3; c < cap && (int) lists[t].size() < 900 + t; c += 3) lists[t].push_back(c);
                        counts[t] = (int) lists[t].size();
                        for (int j = 0; j < counts[t]; ++j) cells_list[(size_t) t * stride + j] = lists[t][j];
                    }
                }
                int32_t *df = dev(firsts), *dcnt = dev(counts), *dcl = dev(cells_list);
                RowSpans rs;
                rs.firsts = df;
                rs.counts = dcnt;
                rs.cells = use_cells ? dcl : nullptr;
                rs.cells_stride = stride;
                if (bf) mla_attend(dqcur, dc16, rs, T, ds, a, use_cells ? 0 : ring, dlat, nullptr);
                else mla_attend(dqcur, dc32, rs, T, ds, a, use_cells ? 0 : ring, dlat, nullptr);
                std::vector<float> got = host(dlat, (size_t) T * heads * a.rank);
                std::vector<double> ref;
                for (int t = 0; t < T; ++t)
                    for (int h = 0; h < heads; ++h) {
                        std::vector<float> qrow(qcur.begin() + (size_t) t * heads * W, qcur.begin() + (size_t) (t + 1) * heads * W);
                        std::vector<double> o = ref_attend(qrow, rows32, lists[t], h, a, sinks[h]);
                        ref.insert(ref.end(), o.begin(), o.end());
                    }
                char name[96];
                std::snprintf(name, sizeof name, "mla_attend %d heads%s%s", heads, use_cells ? ", cell lists" : ", spans", bf ? ", bf16" : "");
                report(name, got, ref, 1e-4);
                if (!use_cells && !ring) {   // row independence: row 1 alone
                    float* d1 = devf((size_t) 9 * heads * a.rank);
                    RowSpans r1;
                    r1.firsts = df + 1;
                    r1.counts = dcnt + 1;
                    // a BF16 chunk (more than 8 rows) takes the tensor-core kernel: row 1 as the first of 9 rows
                    if (bf) mla_attend(dqcur + (size_t) heads * W, dc16, r1, 9, ds, a, 0, d1, nullptr);
                    else mla_attend(dqcur + (size_t) heads * W, dc32, r1, 1, ds, a, 0, d1, nullptr);
                    const bool same = std::memcmp(host(d1, (size_t) heads * a.rank).data(), got.data() + (size_t) heads * a.rank, (size_t) heads * a.rank * 4) == 0;
                    std::printf("%-30s %s\n", "  row independence", same ? "bitwise" : "FAIL");
                    failures += !same;
                }
            }
            // iota
            int32_t* dcl = nullptr;
            cudaMalloc(&dcl, (size_t) T * 64 * 4);
            cudaMemset(dcl, 0xff, (size_t) T * 64 * 4);
            std::vector<int32_t> fill = {10, 0, 33};
            int32_t* dfill = dev(fill);
            iota_cells(dcl, 64, dfill, T, nullptr);
            std::vector<int32_t> ic = host(dcl, (size_t) T * 64);
            bool ok = true;
            for (int j = 0; j < 10; ++j) ok = ok && ic[j] == j;
            ok = ok && ic[64] == -1;
            for (int j = 0; j < 33; ++j) ok = ok && ic[2 * 64 + j] == j;
            if (!bf) { std::printf("%-30s %s\n", "iota_cells", ok ? "equal" : "FAIL"); failures += !ok; }
        }
    }
}

void test_engram() {
    const int n = 3072, T = 3;
    EngramArgs a;
    std::vector<float> x = randn((size_t) T * 4 * n), kv = randn((size_t) T * 5 * n), kvb = randn((size_t) 5 * n, 0.1f), gw = randn((size_t) 4 * n, 0.3f),
                       gamma = randn(n, 0.5f), hs = randn((size_t) 4 * n, 0.2f);
    std::vector<float> valid = {1.f, 0.f, 1.f};
    float *dx = dev(x), *dkv = dev(kv), *dkb = dev(kvb), *dgw = dev(gw), *dgm = dev(gamma), *dhs = dev(hs), *dv = dev(valid);
    engram_apply(dkv, dkb, dgw, dgm, dhs, dv, T, a, dx, nullptr);
    std::vector<float> got = host(dx, (size_t) T * 4 * n);
    std::vector<double> ref((size_t) T * 4 * n);
    auto rms = [&](const std::vector<double>& v, double eps) {
        double ss = 0;
        for (double e : v) ss += e * e;
        return 1.0 / std::sqrt(ss / v.size() + eps);
    };
    for (int t = 0; t < T; ++t) {
        std::vector<double> val(n);
        for (int j = 0; j < n; ++j) val[j] = (double) kv[((size_t) t * 5 + 4) * n + j] + kvb[(size_t) 4 * n + j];
        const double rv = rms(val, a.eps);
        for (int i = 0; i < 4; ++i) {
            std::vector<double> xi(n), key(n);
            for (int j = 0; j < n; ++j) {
                xi[j] = x[((size_t) t * 4 + i) * n + j];
                key[j] = (double) kv[((size_t) t * 5 + i) * n + j] + kvb[(size_t) i * n + j];
            }
            const double rx = rms(xi, a.gate_eps), rk = rms(key, a.gate_eps);
            double d = 0;
            for (int j = 0; j < n; ++j) d += xi[j] * rx * key[j] * rk * gw[(size_t) i * n + j];
            d /= std::sqrt((double) n);
            const double mag = std::sqrt(std::max(std::fabs(d), 1e-6));
            const double g = sig(d < 0 ? -mag : mag);
            for (int j = 0; j < n; ++j)
                ref[((size_t) t * 4 + i) * n + j] = xi[j] + valid[t] * val[j] * rv * gamma[j] * (g + hs[(size_t) i * n + j]);
        }
    }
    report("engram_apply (rows, one masked)", got, ref, 2e-6);
}

void test_indexer() {
    IndexerArgs a;
    const int D = a.dim, P = a.pool, RING = 16;
    // pooling of 3 groups out of a ring (the groups start at 1228, a multiple of 4)
    std::vector<float> vals = randn((size_t) RING * D), gates = randn((size_t) RING * D), ape = randn((size_t) P * D, 0.3f), nw = randn(D, 0.2f);
    for (auto& v : nw) v += 1.f;
    const int first = 1228, G = 3, NP = 5000;
    float *dv = dev(vals), *dg = dev(gates), *da = dev(ape), *dn = dev(nw);
    uint16_t* dp = nullptr;
    cudaMalloc(&dp, (size_t) (first / P + G) * D * 2);
    idx_pool(dv, dg, da, dn, a, RING, first, G, dp, nullptr);
    std::vector<uint16_t> gh = host(dp + (size_t) (first / P) * D, (size_t) G * D);
    std::vector<float> got(gh.size());
    for (size_t i = 0; i < gh.size(); ++i) got[i] = h2f(gh[i]);
    std::vector<double> ref((size_t) G * D);
    for (int g = 0; g < G; ++g) {
        const int p0 = first + g * P;
        std::vector<double> row(D);
        for (int d = 0; d < D; ++d) {
            double mx = -1e30, s = 0, acc = 0;
            for (int j = 0; j < P; ++j) mx = std::max(mx, (double) gates[(size_t) ((p0 + j) % RING) * D + d] + ape[(size_t) j * D + d]);
            for (int j = 0; j < P; ++j) s += std::exp(gates[(size_t) ((p0 + j) % RING) * D + d] + ape[(size_t) j * D + d] - mx);
            for (int j = 0; j < P; ++j)
                acc += std::exp(gates[(size_t) ((p0 + j) % RING) * D + d] + ape[(size_t) j * D + d] - mx) / s * vals[(size_t) ((p0 + j) % RING) * D + d];
            row[d] = acc;
        }
        double ss = 0;
        for (double v : row) ss += v * v;
        for (int d = 0; d < D; ++d) row[d] = row[d] / std::sqrt(ss / D + a.eps) * nw[d];
        for (int d = 0; d < D; ++d) ref[(size_t) g * D + d] = row[d];
        for (int i = 0; i < a.rope / 2; ++i) {
            const double th = p0 * std::pow((double) a.rope_base, -2.0 * i / a.rope);
            const double x0 = row[D - a.rope + 2 * i], x1 = row[D - a.rope + 2 * i + 1];
            ref[(size_t) g * D + D - a.rope + 2 * i] = x0 * std::cos(th) - x1 * std::sin(th);
            ref[(size_t) g * D + D - a.rope + 2 * i + 1] = x0 * std::sin(th) + x1 * std::cos(th);
        }
    }
    report("idx_pool (3 groups, ring, fp16)", got, ref, 2e-3);   // the keys are rounded to fp16
    // the query rotation of 2 rows
    std::vector<float> qq = randn((size_t) 2 * a.heads * D);
    float* dqq = dev(qq);
    idx_rope_heads(dqq, a, 700, 2, nullptr);
    std::vector<float> qg = host(dqq, qq.size());
    std::vector<double> qr(qq.begin(), qq.end());
    for (int t = 0; t < 2; ++t)
        for (int h = 0; h < a.heads; ++h)
            for (int i = 0; i < a.rope / 2; ++i) {
                const double th = (700 + t) * std::pow((double) a.rope_base, -2.0 * i / a.rope);
                const size_t o = ((size_t) t * a.heads + h) * D + D - a.rope + 2 * i;
                qr[o] = qq[o] * std::cos(th) - qq[o + 1] * std::sin(th);
                qr[o + 1] = qq[o] * std::sin(th) + qq[o + 1] * std::cos(th);
            }
    report("idx_rope_heads (2 rows)", qg, qr, 2e-5);
    // scores and selection over many pools, with ties
    const int K = 511;
    std::vector<float> q = randn((size_t) a.heads * D), w = randn(a.heads, 0.05f), pooled = randn((size_t) NP * D);
    std::vector<uint16_t> ph(pooled.size());
    for (size_t i = 0; i < pooled.size(); ++i) {
        ph[i] = f2h(pooled[i]);
        pooled[i] = h2f(ph[i]);   // the reference reads what the kernel reads
    }
    float *dq = dev(q), *dw = dev(w), *dsc = devf(NP);
    uint16_t* dpool = dev(ph);
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
    {   // the tensor-core scores of 3 rows (different pool counts) against the same reference
        const int rows = 3, pc[3] = {NP, NP / 2 + 3, 100};
        std::vector<float> q3, w3;
        std::vector<int32_t> meta3;
        for (int r = 0; r < rows; ++r) {
            q3.insert(q3.end(), q.begin(), q.end());
            w3.insert(w3.end(), w.begin(), w.end());
            meta3.insert(meta3.end(), {pc[r], pc[r] * a.pool, 1});
        }
        float *dq3 = dev(q3), *dw3 = dev(w3), *dsc3 = devf((size_t) rows * NP);
        int32_t* dm3 = dev(meta3);
        idx_scores_rows(dq3, dw3, dpool, dm3, NP, rows, a, dsc3, NP, nullptr);
        std::vector<float> s3 = host(dsc3, (size_t) rows * NP);
        std::vector<float> g3;
        std::vector<double> r3;
        for (int r = 0; r < rows; ++r)
            for (int p = 0; p < pc[r]; ++p) {
                g3.push_back(s3[(size_t) r * NP + p]);
                r3.push_back(rsc[p]);
            }
        report("idx_scores_rows (tensor cores)", g3, r3, 1e-5);
    }
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


void test_fp8() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, Q = a.nope + a.rope, T = 3, pos0 = 100, cap = 400;
    std::vector<float> q = randn((size_t) T * a.heads * Q), qabs = randn((size_t) T * a.heads * a.rank, 0.3f), kvraw = randn((size_t) T * W, 2.f),
                       norm = randn(a.rank, 0.5f), sinks = randn(a.heads);
    for (auto& v : norm) v += 1.f;
    float *dq = dev(q), *dqa = dev(qabs), *dkv = dev(kvraw), *dn = dev(norm), *ds = dev(sinks);
    float *dqcur = devf((size_t) T * a.heads * W), *dlat = devf((size_t) T * a.heads * a.rank);
    uint8_t* dc = nullptr;
    cudaMalloc(&dc, (size_t) cap * kFp8RowBytes);
    cudaMemset(dc, 0, (size_t) cap * kFp8RowBytes);
    mla_prepare(dq, dqa, dkv, dn, a, pos0, T, 0, dqcur, dc, nullptr);
    std::vector<uint8_t> rows = host(dc, (size_t) cap * kFp8RowBytes);
    auto e4m3 = [](uint8_t b) {
        const int e = (b >> 3) & 15, m = b & 7;
        const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp((double) m, -9);
        return (b & 0x80) ? -v : v;
    };
    // decoded rows (what the attention reads) and a check of the rows against the fp32 preparation
    std::vector<float> dec((size_t) cap * W, 0.f);
    bool ok = true;
    for (int t = 0; t < T; ++t) {
        const uint8_t* row = &rows[(size_t) (pos0 + t) * kFp8RowBytes];
        double ss = 0;
        for (int c = 0; c < a.rank; ++c) ss += (double) kvraw[(size_t) t * W + c] * kvraw[(size_t) t * W + c];
        const double rs = 1.0 / std::sqrt(ss / a.rank + a.rms_eps);
        const float* scales = reinterpret_cast<const float*>(row + 512);
        for (int g = 0; g < 4; ++g) {
            double amax = 0;
            for (int c = g * 128; c < g * 128 + 128; ++c) amax = std::max(amax, std::fabs(kvraw[(size_t) t * W + c] * rs * norm[c]));
            ok = ok && std::fabs(scales[g] - amax / 448.0) <= 1e-6 * amax / 448.0 + 1e-12;
            for (int c = g * 128; c < g * 128 + 128; ++c) {
                const double ref = kvraw[(size_t) t * W + c] * rs * norm[c], got = e4m3(row[c]) * scales[g];
                dec[(size_t) (pos0 + t) * W + c] = (float) got;
                ok = ok && std::fabs(got - ref) <= 0.0625 * amax + 1e-6;   // E4M3: 3 mantissa bits
            }
        }
        for (int c = 0; c < a.rope; ++c) {
            uint16_t b;
            std::memcpy(&b, row + 528 + 2 * c, 2);
            dec[(size_t) (pos0 + t) * W + a.rank + c] = from_bf16(b);
        }
    }
    std::printf("%-30s %s\n", "mla_prepare fp8 rows", ok ? "ok" : "FAIL");
    failures += !ok;
    // attention over the decoded rows of the three written positions and some earlier (zero) ones
    std::vector<int32_t> firsts = {pos0, pos0, pos0}, counts = {1, 2, 3};
    int32_t *df = dev(firsts), *dcnt = dev(counts);
    RowSpans rs;
    rs.firsts = df;
    rs.counts = dcnt;
    mla_attend(dqcur, dc, rs, T, ds, a, 0, dlat, nullptr);
    std::vector<float> got = host(dlat, (size_t) T * a.heads * a.rank), qcur = host(dqcur, (size_t) T * a.heads * W);
    std::vector<double> ref;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < a.heads; ++h) {
            std::vector<float> qrow(qcur.begin() + (size_t) t * a.heads * W, qcur.begin() + (size_t) (t + 1) * a.heads * W);
            std::vector<int> cl;
            for (int j = 0; j <= t; ++j) cl.push_back(pos0 + j);
            std::vector<double> o = ref_attend(qrow, dec, cl, h, a, sinks[h]);
            ref.insert(ref.end(), o.begin(), o.end());
        }
    report("mla_attend (fp8 cache)", got, ref, 1e-4);
}

void test_mmq_dense() {
    namespace mmq = basalt::prefill::mmq;
    const int R = 64, rows = 1536, K = 3072;
    if (!mmq::built() || !mmq::fits(8, rows)) {
        std::printf("%-30s skipped (no MMQ for Q8_0)\n", "mmq dense Q8_0");
        return;
    }
    Q8 w = make_q8(rows, K);
    std::vector<float> x = randn((size_t) R * K);
    uint8_t* dw = dev(w.bytes);
    float *dx = dev(x), *dy = devf((size_t) R * rows);
    void* xq = nullptr;
    cudaMalloc(&xq, mmq::q8_bytes(R, K));
    int32_t *bounds = nullptr, *ident = nullptr;
    cudaMalloc(&bounds, 8);
    cudaMalloc(&ident, R * 4);
    mmq::iota(ident, R, nullptr);
    const int32_t b[2] = {0, R};
    cudaMemcpy(bounds, b, 8, cudaMemcpyHostToDevice);
    mmq::load(8, rows);
    mmq::quantize(dx, nullptr, xq, 8, K, K, R, nullptr);
    mmq::Product p;
    p.w = dw;
    p.type = 8;
    p.w_rows = rows;
    p.w_cols = K;
    p.expert_bytes = mmq::matrix_bytes(8, rows, K);
    p.n = 1;
    p.xq = xq;
    p.bounds = bounds;
    p.ids = ident;
    p.total_rows = R;
    p.max_rows = R;
    p.dst = dy;
    p.ld_dst = rows;
    mmq::Context ctx;
    ctx.run(p, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> got = host(dy, (size_t) R * rows);
    std::vector<double> ref((size_t) R * rows);
    for (int t = 0; t < R; ++t)
        for (int r = 0; r < rows; ++r) {
            double d = 0;
            for (int i = 0; i < K; ++i) d += (double) w.vals[(size_t) r * K + i] * x[(size_t) t * K + i];
            ref[(size_t) t * rows + r] = d;
        }
    report("mmq dense Q8_0 64x1536x3072", got, ref, 2e-2);   // the activations are rounded to q8_1
}


// the expert matrices of the K-quant formats through MMQ, as the prompt path runs them: n experts at a stride, rows given by bounds
void test_mmq_expert(int type, const char* label) {
    namespace mmq = basalt::prefill::mmq;
    const int rows = 1536, K = 1536, NE = 3, R0 = 5, R1 = 3, R2 = 9;
    if (!mmq::built() || !mmq::fits(type, rows)) {
        std::printf("%-30s skipped\n", label);
        return;
    }
    const size_t row_b = ggml_row_size((ggml_type) type, K), mat = row_b * rows, unit = ggml_type_size((ggml_type) type) * 16, stride = (mat + 4096 + unit - 1) / unit * unit;   // whole blocks, as MMQ steps
    std::vector<uint8_t> blobs(stride * NE, 0);
    std::vector<std::vector<float>> deq(NE, std::vector<float>((size_t) rows * K));
    for (int e = 0; e < NE; ++e) {
        std::vector<float> w((size_t) rows * K);
        for (auto& v : w) v = randn(1, 0.05f)[0];
        ggml_quantize_chunk((ggml_type) type, w.data(), blobs.data() + stride * e, 0, rows, K);
        for (int r = 0; r < rows; ++r)
            ggml_get_type_traits((ggml_type) type)->to_float(blobs.data() + stride * e + (size_t) r * row_b, deq[e].data() + (size_t) r * K, K);
    }
    const int R = R0 + R1 + R2;
    std::vector<float> x = randn((size_t) R * K);
    std::vector<int32_t> bounds = {0, R0, R0 + R1, R}, tok(R);
    for (int i = 0; i < R; ++i) tok[i] = i;
    uint8_t* dw = dev(blobs);
    float *dx = dev(x), *dy = devf((size_t) R * rows);
    int32_t *db = dev(bounds), *dt = dev(tok), *ident = nullptr;
    void* xq = nullptr;
    cudaMalloc(&xq, mmq::q8_bytes(R, K));
    cudaMalloc(&ident, R * 4);
    mmq::iota(ident, R, nullptr);
    mmq::load(type, rows);
    mmq::quantize(dx, dt, xq, type, K, K, R, nullptr);
    mmq::Product p;
    p.w = dw;
    p.type = type;
    p.w_rows = rows;
    p.w_cols = K;
    p.expert_bytes = stride;
    p.n = NE;
    p.xq = xq;
    p.bounds = db;
    p.ids = ident;
    p.total_rows = R;
    p.max_rows = std::max(R0, std::max(R1, R2));
    p.dst = dy;
    p.ld_dst = rows;
    mmq::Context ctx;
    ctx.run(p, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> got = host(dy, (size_t) R * rows);
    std::vector<double> ref((size_t) R * rows);
    for (int e = 0; e < NE; ++e)
        for (int t = bounds[e]; t < bounds[e + 1]; ++t)
            for (int r = 0; r < rows; ++r) {
                double d = 0;
                for (int i = 0; i < K; ++i) d += (double) deq[e][(size_t) r * K + i] * x[(size_t) t * K + i];
                ref[(size_t) t * rows + r] = d;
            }
    for (int t = 0; t < R; ++t) {
        int bad = 0;
        for (int r = 0; r < rows; ++r) bad += !(std::fabs(got[(size_t) t * rows + r] - ref[(size_t) t * rows + r]) <= 0.05 * 3 + 0.03 * std::fabs(ref[(size_t) t * rows + r]));
        std::printf("  row %d (expert %d): %d of %d outputs off, first got %g want %g\n", t, t < bounds[1] ? 0 : t < bounds[2] ? 1 : 2, bad, rows, got[(size_t) t * rows], ref[(size_t) t * rows]);
    }
    report(label, got, ref, 3e-2);
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
    test_fp8();
    test_mmq_dense();
    test_mmq_expert(12, "mmq experts Q4_K");
    test_mmq_expert(14, "mmq experts Q6_K");
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
