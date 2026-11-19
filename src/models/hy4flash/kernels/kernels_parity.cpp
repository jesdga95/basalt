// src/models/hy4flash/kernels/kernels_parity.cpp - the Hy4-flash decode kernels against plain CPU references (double
// accumulation) at the model's geometry: the identity hyper-connections, the per-head Q8_0 products, the MLA
// preparation and attention with a sink, the router and the Engram apply. Exit 77 without a GPU.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_fp16.h>
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
uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
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
            // the MoE's end fused: add_vectors, rms_norm_rows and hc_post in one launch, bit for bit
            std::vector<float> ra = randn((size_t) T * E), rb = randn((size_t) T * E, 0.3f), w = randn(E, 0.2f);
            for (float& v : w) v += 1.0f;
            float *da = dev(ra), *db = dev(rb), *dw = dev(w), *dsum = devf((size_t) T * E), *dn = devf((size_t) T * E);
            float *d3 = devf((size_t) T * 4 * E), *d1 = devf((size_t) T * 4 * E);
            cudaStream_t st;
            cudaStreamCreate(&st);
            add_vectors(da, db, dsum, T * E, st);
            bk::rms_norm_rows(dsum, dw, dn, T, E, 1e-6f, st);
            bk::hc_post(dn, dx, coef, stride, T, E, d3, st);
            add_norm_hc_post(da, db, dw, 1e-6f, dx, coef, stride, T, E, d1, st);
            cudaStreamSynchronize(st);
            std::vector<float> h3 = host(d3, (size_t) T * 4 * E), h1 = host(d1, (size_t) T * 4 * E);
            const bool fused_same = std::memcmp(h3.data(), h1.data(), h3.size() * 4) == 0;
            std::printf("%-30s %s\n", "add_norm_hc_post", fused_same ? "bitwise" : "FAIL");
            failures += !fused_same;
            // and without the add (the attention's end): rms_norm_rows and hc_post
            bk::rms_norm_rows(da, dw, dn, T, E, 1e-6f, st);
            bk::hc_post(dn, dx, coef, stride, T, E, d3, st);
            add_norm_hc_post(da, nullptr, dw, 1e-6f, dx, coef, stride, T, E, d1, st);
            cudaStreamSynchronize(st);
            h3 = host(d3, (size_t) T * 4 * E);
            h1 = host(d1, (size_t) T * 4 * E);
            const bool norm_same = std::memcmp(h3.data(), h1.data(), h3.size() * 4) == 0;
            std::printf("%-30s %s\n", "add_norm_hc_post (no add)", norm_same ? "bitwise" : "FAIL");
            failures += !norm_same;
            cudaStreamDestroy(st);
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

// The kernel's own arithmetic on the CPU: the q8_1 rounding of each 32 values, exact integer block sums, the blocks added in
// order with fmaf. Equal bits are expected.
std::vector<float> gemv_emulated(const Q8& w, const std::vector<float>& x, int xs, int tok, int heads, int rows, int k, int T) {
    std::vector<float> y((size_t) T * heads * rows);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < rows; ++r) {
                const uint8_t* row = &w.bytes[((size_t) h * rows + r) * (k / 32) * 34];
                float acc = 0.f;
                for (int b = 0; b < k / 32; ++b) {
                    const uint8_t* blk = row + (size_t) b * 34;
                    const float* xv = &x[(size_t) t * tok + (size_t) h * xs + b * 32];
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xv[i]));
                    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
                    int isum = 0;
                    for (int i = 0; i < 32; ++i) isum += (int) (int8_t) blk[2 + i] * (int) std::round(xv[i] * id);
                    acc = std::fmaf(h2f((uint16_t) (blk[0] | (blk[1] << 8))) * d, (float) isum, acc);
                }
                y[((size_t) t * heads + h) * rows + r] = acc;
            }
    return y;
}

void test_gemv() {
    const int T = 9;   // two launches of up to 8 rows
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
        const std::vector<float> emu = gemv_emulated(w, x, xs, tok, heads, rows, k, T);
        const bool bit = std::memcmp(emu.data(), got.data(), got.size() * 4) == 0;
        std::printf("%-30s %s\n", "  against the emulation", bit ? "bitwise" : "FAIL");
        failures += !bit;
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
uint16_t to_f16(float f) { return __half_as_ushort(__float2half_rn(f)); }
float from_f16(uint16_t b) { return __half2float(__ushort_as_half(b)); }

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
            for (size_t i = 0; i < cache.size(); ++i) cbf[i] = to_f16(cache[i]);
            float* dc32 = dev(cache);
            uint16_t* dc16 = dev(cbf);
            if (bf) mla_prepare(dq, dqa, dkv, dn, a, pos0, T, ring, dqcur, dc16, nullptr);
            else mla_prepare(dq, dqa, dkv, dn, a, pos0, T, ring, dqcur, dc32, nullptr);
            std::vector<float> qcur = host(dqcur, (size_t) T * heads * W);
            std::vector<float> rows32 = bf ? std::vector<float>() : host(dc32, cache.size());
            if (bf) {
                std::vector<uint16_t> r = host(dc16, cbf.size());
                rows32.resize(r.size());
                for (size_t i = 0; i < r.size(); ++i) rows32[i] = from_f16(r[i]);
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
                    rows_ok = rows_ok && std::fabs(got - row[c]) <= (bf ? 6e-5 + 1e-3 * std::fabs(row[c]) : 6e-5 + 3e-5 * std::fabs(row[c]));   // the rotation's angle is fp32
                }
            }
            std::vector<float> qcur_ref(rq.begin(), rq.end());
            report(bf ? "mla_prepare query (fp16 run)" : "mla_prepare query", qcur, rq, 3e-5);
            std::printf("%-30s %s\n", bf ? "mla_prepare rows (fp16)" : "mla_prepare rows", rows_ok ? "ok" : "FAIL");
            failures += !rows_ok;
            // attention: row t over the last (up to) 512 positions, or over a given list
            std::vector<int32_t> firsts(T), counts(T), cells_list;
            const int stride = 1100;
            cells_list.assign((size_t) T * stride, 0);
            const int max_cells = 1100;
            void* work = nullptr;
            cudaMalloc(&work, mla_attend_work_bytes(heads, max_cells, T));
            cudaMemset(work, 0, mla_attend_work_bytes(heads, max_cells, T));
            std::vector<int32_t> fill(T, 0);
            for (int use_cells = 0; use_cells < 3; ++use_cells) {   // spans, cell lists, lists with some rows filled (spans from 0)
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
                        fill[t] = 0;
                        if (use_cells == 2 && t % 4 == 1) {   // a filled row: cells 0..pos, its list ignored
                            counts[t] = fill[t] = pos + 1;
                            for (int j = 0; j < counts[t]; ++j) lists[t].push_back(j);
                            for (int j = 0; j < stride; ++j) cells_list[(size_t) t * stride + j] = cap - 1;
                            continue;
                        }
                        for (int c = t % 3; c < cap && (int) lists[t].size() < 900 + t; c += 3) lists[t].push_back(c);
                        counts[t] = (int) lists[t].size();
                        for (int j = 0; j < counts[t]; ++j) cells_list[(size_t) t * stride + j] = lists[t][j];
                    }
                }
                int32_t *df = dev(firsts), *dcnt = dev(counts), *dcl = dev(cells_list), *dfill = dev(fill);
                RowSpans rs;
                rs.firsts = df;
                rs.counts = dcnt;
                rs.cells = use_cells ? dcl : nullptr;
                rs.cells_stride = stride;
                rs.fill = use_cells == 2 ? dfill : nullptr;
                const int rg = use_cells ? 0 : ring;
                std::vector<double> ref;
                for (int t = 0; t < T; ++t)
                    for (int h = 0; h < heads; ++h) {
                        std::vector<float> qrow(qcur.begin() + (size_t) t * heads * W, qcur.begin() + (size_t) (t + 1) * heads * W);
                        std::vector<double> o = ref_attend(qrow, rows32, lists[t], h, a, sinks[h]);
                        ref.insert(ref.end(), o.begin(), o.end());
                    }
                const char* mode = use_cells == 0 ? ", spans" : use_cells == 1 ? ", cell lists" : ", lists + filled";
                // all T rows: a BF16 chunk (more than 8 rows) takes the tensor-core kernel (fp16 query and probabilities), fp32 two
                // launches of the chunked kernel; then a window of the first 6 rows
                if (bf) mla_attend(dqcur, dc16, rs, T, ds, a, rg, max_cells, work, dlat, nullptr);
                else mla_attend(dqcur, dc32, rs, T, ds, a, rg, max_cells, work, dlat, nullptr);
                std::vector<float> got = host(dlat, (size_t) T * heads * a.rank);
                char name[96];
                std::snprintf(name, sizeof name, "mla_attend %d heads%s%s", heads, mode, bf ? ", fp16 chunk" : "");
                report(name, got, ref, bf ? 1e-3 : 1e-4);
                if (bf) std::printf("%-30s %016llx\n", "  output hash", (unsigned long long) fnv(got));
                const int TW = 6;
                if (bf) mla_attend(dqcur, dc16, rs, TW, ds, a, rg, max_cells, work, dlat, nullptr);
                else mla_attend(dqcur, dc32, rs, TW, ds, a, rg, max_cells, work, dlat, nullptr);
                std::vector<float> win = host(dlat, (size_t) TW * heads * a.rank);
                std::snprintf(name, sizeof name, "  window of %d%s", TW, bf ? " (fp16)" : "");
                report(name, win, std::vector<double>(ref.begin(), ref.begin() + win.size()), 1e-5);
                if (!bf) {   // the chunked kernel's rows in the window are the T-row call's
                    const bool same = std::memcmp(win.data(), got.data(), win.size() * 4) == 0;
                    std::printf("%-30s %s\n", "  window width", same ? "bitwise" : "FAIL");
                    failures += !same;
                }
                {   // row independence: row 1 alone against row 1 of the window
                    float* d1 = devf((size_t) heads * a.rank);
                    RowSpans r1 = rs;
                    r1.firsts = df + 1;
                    r1.counts = dcnt + 1;
                    r1.cells = use_cells ? dcl + stride : nullptr;
                    r1.fill = use_cells == 2 ? dfill + 1 : nullptr;
                    if (bf) mla_attend(dqcur + (size_t) heads * W, dc16, r1, 1, ds, a, rg, max_cells, work, d1, nullptr);
                    else mla_attend(dqcur + (size_t) heads * W, dc32, r1, 1, ds, a, rg, max_cells, work, d1, nullptr);
                    const bool same = std::memcmp(host(d1, (size_t) heads * a.rank).data(), win.data() + (size_t) heads * a.rank, (size_t) heads * a.rank * 4) == 0;
                    std::printf("%-30s %s\n", "  row independence", same ? "bitwise" : "FAIL");
                    failures += !same;
                    cudaFree(d1);
                }
            }
            cudaFree(work);
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


// The selection at long range: 262,144 pools (a 1M context), random scores and runs of ties around the cut, by one row and by
// rows of different pool counts, against a stable sort.
// Microbench of a prompt's indexer batch at long range: 64 rows against 131,072 pools (the middle of a 1M prompt), the
// tensor-core scores and the selection.
void bench_indexer_batch() {
    IndexerArgs a;
    const int rows = 64, NP = 131072, K = 511, P = 4, cs = K * P + 8;
    std::vector<float> q = randn((size_t) rows * a.heads * a.dim, 0.1f), w = randn((size_t) rows * a.heads);
    std::vector<float> kf = randn((size_t) NP * a.dim);
    std::vector<uint16_t> keys((size_t) NP * a.dim);
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = f2h(kf[i]);
    std::vector<int32_t> meta;
    for (int r = 0; r < rows; ++r) meta.insert(meta.end(), {NP - r, (NP - r) * P, 1});
    float *dq = dev(q), *dw = dev(w), *ds = devf((size_t) rows * NP);
    uint16_t* dk = dev(keys);
    int32_t *dm = dev(meta), *dc = nullptr;
    cudaMalloc(&dc, (size_t) rows * cs * 4);
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    idx_scores_rows(dq, dw, dk, dm, NP, rows, a, ds, NP, st);
    idx_select_rows(ds, NP, dm, rows, K, P, dc, cs, st);
    cudaEventRecord(e0, st);
    for (int it = 0; it < 10; ++it) idx_scores_rows(dq, dw, dk, dm, NP, rows, a, ds, NP, st);
    cudaEventRecord(e1, st);
    for (int it = 0; it < 10; ++it) idx_select_rows(ds, NP, dm, rows, K, P, dc, cs, st);
    cudaEventRecord(e2, st);
    cudaEventSynchronize(e2);
    float m1 = 0.f, m2 = 0.f;
    cudaEventElapsedTime(&m1, e0, e1);
    cudaEventElapsedTime(&m2, e1, e2);
    const double flop = 2.0 * 2.0 * rows * a.heads * (double) NP * a.dim;   // q in two fp16 halves
    std::printf("%-30s scores %.0f us (%.1f TFLOP/s), select %.0f us, per 64 rows x 131072 pools\n", "indexer batch microbench",
                m1 * 100.f, flop / (m1 / 10.0 * 1e-3) / 1e12, m2 * 100.f);
    cudaFree(dq);
    cudaFree(dw);
    cudaFree(ds);
    cudaFree(dk);
    cudaFree(dm);
    cudaFree(dc);
    cudaStreamDestroy(st);
}

void test_select_long() {
    const int NP = 262144, K = 511, P = 4, tail_n = 2;
    std::vector<int> counts = {NP, NP - 77, 3000, 513};
    for (int ties = 0; ties < 3; ++ties) {
        std::vector<float> s = randn((size_t) NP);
        if (ties == 1)
            for (auto& v : s) v = std::floor(v * 64.f) / 64.f;   // many equal scores around the cut
        if (ties == 2)
            for (int p = 0; p < NP; ++p) s[(size_t) p] = (p % 7 == 0) ? 1.f : -std::fabs(s[(size_t) p]);   // more ties than K at the top
        const int rows = (int) counts.size(), cs = K * P + 8;
        std::vector<float> sr((size_t) rows * NP);
        std::vector<int32_t> meta;
        for (int r = 0; r < rows; ++r) {
            std::copy(s.begin(), s.begin() + counts[(size_t) r], sr.begin() + (size_t) r * NP);
            meta.insert(meta.end(), {counts[(size_t) r], counts[(size_t) r] * P, tail_n});
        }
        float* ds = dev(sr);
        int32_t *dm = dev(meta), *dc = nullptr;
        cudaMalloc(&dc, (size_t) rows * cs * 4);
        idx_select_rows(ds, NP, dm, rows, K, P, dc, cs, nullptr);
        std::vector<int32_t> got = host(dc, (size_t) rows * cs);
        int32_t* d1 = nullptr;
        cudaMalloc(&d1, (size_t) cs * 4);
        idx_select(ds, NP, K, P, NP * P, tail_n, d1, nullptr);
        std::vector<int32_t> one = host(d1, (size_t) cs);
        if (ties == 0) {   // microbench: one row of 262,144 pools, and a decode window of 8 rows
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            float ms1 = 0.f, ms8 = 0.f;
            std::vector<int32_t> m8;
            for (int r = 0; r < 8; ++r) m8.insert(m8.end(), {NP, NP * P, tail_n});
            int32_t* dm8 = dev(m8);
            float* ds8 = devf((size_t) 8 * NP);
            for (int r = 0; r < 8; ++r) cudaMemcpy(ds8 + (size_t) r * NP, ds, (size_t) NP * 4, cudaMemcpyDeviceToDevice);
            int32_t* dc8 = nullptr;
            cudaMalloc(&dc8, (size_t) 8 * cs * 4);
            idx_select(ds, NP, K, P, NP * P, tail_n, d1, nullptr);
            cudaEventRecord(e0);
            for (int it = 0; it < 20; ++it) idx_select_rows(ds8, NP, dm8, 1, K, P, dc8, cs, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms1, e0, e1);
            cudaEventRecord(e0);
            for (int it = 0; it < 20; ++it) idx_select_rows(ds8, NP, dm8, 8, K, P, dc8, cs, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms8, e0, e1);
            std::printf("%-30s %.1f us per row alone, %.1f us per window of 8 rows\n", "  microbench", ms1 * 1000.f / 20.f, ms8 * 1000.f / 20.f);
            cudaFree(dm8); cudaFree(ds8); cudaFree(dc8);
        }
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            const int n = counts[(size_t) r];
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[(size_t) i] = i;
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return s[(size_t) x] > s[(size_t) y]; });
            std::vector<int> pick(order.begin(), order.begin() + K);
            std::sort(pick.begin(), pick.end());
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < P; ++j) {
                    ok = ok && got[(size_t) r * cs + i * P + j] == pick[(size_t) i] * P + j;
                    if (r == 0) ok = ok && one[(size_t) i * P + j] == pick[(size_t) i] * P + j;
                }
            for (int j = 0; j < tail_n; ++j) ok = ok && got[(size_t) r * cs + K * P + j] == n * P + j;
        }
        std::printf("%-30s %s\n", ties == 0 ? "idx_select 262144 pools" : ties == 1 ? "  ties near the cut" : "  more ties than k", ok ? "equal" : "FAIL");
        failures += !ok;
        cudaFree(ds); cudaFree(dm); cudaFree(dc); cudaFree(d1);
    }
}

void test_fp8() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, Q = a.nope + a.rope, T = 3, pos0 = 100, cap = 400;
    std::vector<float> q = randn((size_t) T * a.heads * Q), qabs = randn((size_t) T * a.heads * a.rank, 0.3f), kvraw = randn((size_t) T * W, 2.f),
                       norm = randn(a.rank, 0.5f), sinks = randn(a.heads);
    for (auto& v : norm) v += 1.f;
    for (int c = 0; c < 128; ++c) norm[(size_t) c] *= 1e-5f;   // a group under the scale's floor
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
    auto e4m3_rn = [&](double x) {   // nearest code, ties to the even code, saturating at 448
        uint8_t best = 0;
        double bd = 1e300;
        for (int b = 0; b < 127; ++b) {
            const double d = std::fabs(std::fabs(x) - e4m3((uint8_t) b));
            if (d < bd || (d == bd && (b & 1) == 0)) { bd = d; best = (uint8_t) b; }
        }
        return (uint8_t) (best | (std::signbit(x) ? 0x80 : 0));
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
            // vLLM's rule over the BF16 latent: scale 2^ceil(log2(max(amax / 448, 1e-4))), codes E4M3(v / scale) to nearest even
            std::vector<double> v(128);
            double amax = 0;
            for (int c = 0; c < 128; ++c) {
                const int k = g * 128 + c;
                v[(size_t) c] = from_bf16(to_bf16((float) (kvraw[(size_t) t * W + k] * rs * norm[k])));
                amax = std::max(amax, std::fabs(v[(size_t) c]));
            }
            int e;
            const double fr = std::frexp(std::max(amax / 448.0, 1e-4), &e);
            const double scale = std::ldexp(1.0, fr == 0.5 ? e - 1 : e);
            ok = ok && scales[g] == (float) scale;
            for (int c = 0; c < 128; ++c) {
                const int k = g * 128 + c;
                ok = ok && row[k] == e4m3_rn(v[(size_t) c] / scale);
                const double got = e4m3(row[k]) * scales[g];
                dec[(size_t) (pos0 + t) * W + k] = (float) got;
                ok = ok && from_bf16(to_bf16((float) got)) == (float) got;   // decodes to a BF16 value
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
    void* work = nullptr;
    cudaMalloc(&work, mla_attend_work_bytes(a.heads, cap, T));
    cudaMemset(work, 0, mla_attend_work_bytes(a.heads, cap, T));
    mla_attend(dqcur, dc, rs, T, ds, a, 0, cap, work, dlat, nullptr);
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

// mla_query (the k_b product, the query rows and the cache rows of a window in one launch) against q8_0_heads_gemv then
// mla_prepare: the same bits, for each cache type, both head counts, 1 and 6 rows.
void test_query() {
    for (auto [heads, base_f, ring] : {std::tuple<int, float, int>{32, 1e7f, 0}, {64, 1e4f, 700}}) {
        MlaArgs a;
        a.heads = heads;
        a.rope_base = base_f;
        const int W = a.rank + a.rope, Q = a.nope + a.rope, pos0 = 650, cap = 1000;
        Q8 kb = make_q8(heads * a.rank, a.nope);
        uint8_t* dkb = dev(kb.bytes);
        for (int T : {1, 6}) {
            std::vector<float> q = randn((size_t) T * heads * Q), kvraw = randn((size_t) T * W), norm = randn(a.rank, 0.2f);
            for (auto& v : norm) v += 1.f;
            std::vector<int32_t> pos(T);
            for (int t = 0; t < T; ++t) pos[t] = pos0 + 3 * t;   // positions from device memory, not consecutive
            float *dq = dev(q), *dkv = dev(kvraw), *dn = dev(norm), *dqa = devf((size_t) T * heads * a.rank);
            float *qc1 = devf((size_t) T * heads * W), *qc2 = devf((size_t) T * heads * W);
            int32_t* dpos = dev(pos);
            for (int ty = 0; ty < 3; ++ty) {
                const size_t rb = ty == 0 ? (size_t) W * 4 : ty == 1 ? (size_t) W * 2 : (size_t) kFp8RowBytes;
                uint8_t *c1 = nullptr, *c2 = nullptr;
                cudaMalloc(&c1, cap * rb);
                cudaMalloc(&c2, cap * rb);
                cudaMemset(c1, 0, cap * rb);
                cudaMemset(c2, 0, cap * rb);
                q8_0_heads_gemv(dkb, dq, Q, heads * Q, heads, a.rank, a.nope, T, dqa, nullptr);
                for (int t = 0; t < T; ++t) {   // mla_prepare takes consecutive positions: a row at a time
                    const size_t qo = (size_t) t * heads;
                    if (ty == 0) mla_prepare(dq + qo * Q, dqa + qo * a.rank, dkv + (size_t) t * W, dn, a, pos[t], 1, ring, qc1 + qo * W, (float*) c1, nullptr);
                    if (ty == 1) mla_prepare(dq + qo * Q, dqa + qo * a.rank, dkv + (size_t) t * W, dn, a, pos[t], 1, ring, qc1 + qo * W, (uint16_t*) c1, nullptr);
                    if (ty == 2) mla_prepare(dq + qo * Q, dqa + qo * a.rank, dkv + (size_t) t * W, dn, a, pos[t], 1, ring, qc1 + qo * W, c1, nullptr);
                }
                if (ty == 0) mla_query(dkb, dq, dkv, dn, a, dpos, T, ring, qc2, (float*) c2, nullptr);
                if (ty == 1) mla_query(dkb, dq, dkv, dn, a, dpos, T, ring, qc2, (uint16_t*) c2, nullptr);
                if (ty == 2) mla_query(dkb, dq, dkv, dn, a, dpos, T, ring, qc2, c2, nullptr);
                const bool sq = host(qc1, (size_t) T * heads * W) == host(qc2, (size_t) T * heads * W);
                const bool sc = host(c1, cap * rb) == host(c2, cap * rb);
                char name[96];
                std::snprintf(name, sizeof name, "mla_query %d heads, %d rows, %s", heads, T, ty == 0 ? "fp32" : ty == 1 ? "fp16" : "fp8");
                std::printf("%-30s query %s, rows %s\n", name, sq ? "bitwise" : "FAIL", sc ? "bitwise" : "FAIL");
                failures += !sq + !sc;
                cudaFree(c1);
                cudaFree(c2);
            }
        }
    }
}

// The sparse layer at a long context: 2,048 selected cells (and the group's tail) of a 20,000-cell fp16 cache, rows of 6 with
// differing counts (one of a single cell), and the row's independence of the window.
void test_mla_long() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, T = 6, cap = 20000, stride = 2056;
    std::vector<float> qcur = randn((size_t) T * a.heads * W, 0.3f), sinks = randn(a.heads), cache = randn((size_t) cap * W, 0.5f);
    std::vector<uint16_t> cbf(cache.size());
    for (size_t i = 0; i < cache.size(); ++i) {
        cbf[i] = to_f16(cache[i]);
        cache[i] = from_f16(cbf[i]);
    }
    std::vector<int32_t> firsts(T, 0), counts(T), cells((size_t) T * stride);
    std::vector<std::vector<int>> lists(T);
    std::mt19937 r(5);
    for (int t = 0; t < T; ++t) {
        const int n = t == 3 ? 1 : 2044 + t % 4 + 1;
        std::vector<int> all(cap - 8);
        for (int i = 0; i < cap - 8; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), r);
        std::sort(all.begin(), all.begin() + n);
        for (int j = 0; j < n; ++j) {
            lists[t].push_back(all[j]);
            cells[(size_t) t * stride + j] = all[j];
        }
        counts[t] = n;
    }
    float *dq = dev(qcur), *ds = dev(sinks), *dlat = devf((size_t) T * a.heads * a.rank), *d1 = devf((size_t) a.heads * a.rank);
    uint16_t* dc = dev(cbf);
    int32_t *df = dev(firsts), *dcnt = dev(counts), *dcl = dev(cells);
    void* work = nullptr;
    cudaMalloc(&work, mla_attend_work_bytes(a.heads, stride, T));
    cudaMemset(work, 0, mla_attend_work_bytes(a.heads, stride, T));
    RowSpans rs;
    rs.firsts = df;
    rs.counts = dcnt;
    rs.cells = dcl;
    rs.cells_stride = stride;
    mla_attend(dq, dc, rs, T, ds, a, 0, stride, work, dlat, nullptr);
    std::vector<float> got = host(dlat, (size_t) T * a.heads * a.rank);
    std::vector<double> ref;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < a.heads; ++h) {
            std::vector<float> qrow(qcur.begin() + (size_t) t * a.heads * W, qcur.begin() + (size_t) (t + 1) * a.heads * W);
            std::vector<double> o = ref_attend(qrow, cache, lists[t], h, a, sinks[h]);
            ref.insert(ref.end(), o.begin(), o.end());
        }
    report("mla_attend 2,048 selected", got, ref, 1e-5);
    RowSpans r1 = rs;
    r1.firsts = df + 4;
    r1.counts = dcnt + 4;
    r1.cells = dcl + (size_t) 4 * stride;
    mla_attend(dq + (size_t) 4 * a.heads * W, dc, r1, 1, ds, a, 0, stride, work, d1, nullptr);
    const bool same = std::memcmp(host(d1, (size_t) a.heads * a.rank).data(), got.data() + (size_t) 4 * a.heads * a.rank, (size_t) a.heads * a.rank * 4) == 0;
    std::printf("%-30s %s\n", "  row independence", same ? "bitwise" : "FAIL");
    failures += !same;
    cudaFree(work);
}

// Microbench of a window's attention kernels at the model's geometry (the weights rotate through more than the L2 holds):
// us per call and GB/s of the bytes each must read.
void bench_mla() {
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const size_t pool = (size_t) 256 << 20;
    uint8_t* wp = nullptr;
    cudaMalloc(&wp, pool);
    cudaMemset(wp, 0x11, pool);
    auto timed = [&](int reps, auto&& fn) {
        fn(0);
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) fn(i);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return 1000.0 * ms / reps;
    };
    for (int heads : {32, 64}) {
        MlaArgs a;
        a.heads = heads;
        const int W = a.rank + a.rope, Q = a.nope + a.rope;
        const size_t kb = (size_t) heads * a.rank * (a.nope / 32) * 34, vb = (size_t) heads * 256 * (a.rank / 32) * 34;
        const int nkb = (int) (pool / kb), nvb = (int) (pool / vb);
        float *dq = dev(randn((size_t) 8 * heads * Q)), *dkv = dev(randn((size_t) 8 * W)), *dn = dev(randn(a.rank)),
              *qc = devf((size_t) 8 * heads * W), *lat = dev(randn((size_t) 8 * heads * a.rank)), *vo = devf((size_t) 8 * heads * 256);
        std::vector<int32_t> pos(8);
        for (int t = 0; t < 8; ++t) pos[t] = 100 + t;
        int32_t* dpos = dev(pos);
        uint16_t* cache = nullptr;
        cudaMalloc(&cache, (size_t) 4096 * W * 2);
        for (int T : {1, 6}) {
            const double uq = timed(200, [&](int i) { mla_query(wp + (size_t) (i % nkb) * kb, dq, dkv, dn, a, dpos, T, 0, qc, cache, st); });
            const double uv = timed(200, [&](int i) { q8_0_heads_gemv(wp + (size_t) (i % nvb) * vb, lat, a.rank, heads * a.rank, heads, 256, a.rank, T, vo, st); });
            std::printf("bench: %d heads, %d rows: mla_query %.1f us (%.0f GB/s of k_b), v_b %.1f us (%.0f GB/s)\n", heads, T, uq, kb / uq / 1e3, uv,
                        vb / uv / 1e3);
        }
        cudaFree(cache);
    }
    // attention: SWA (64 heads, 512 cells) and sparse (32 heads, 2,048 cells; and 100 at a short context), BF16 rows
    for (auto [heads, n, ring] : {std::tuple<int, int, int>{64, 512, 520}, {32, 2048, 0}, {32, 100, 0}}) {
        MlaArgs a;
        a.heads = heads;
        const int W = a.rank + a.rope, cap = 20000, stride = 2056;
        const int ncopies = (int) (pool / ((size_t) cap * W * 2));
        float *qc = dev(randn((size_t) 8 * heads * W, 0.3f)), *sk = dev(randn(heads)), *lat = devf((size_t) 8 * heads * a.rank);
        std::vector<int32_t> firsts(8), counts(8, n), cl((size_t) 8 * stride);
        for (int t = 0; t < 8; ++t) {
            firsts[t] = ring ? 7 : 0;
            for (int j = 0; j < stride; ++j) cl[(size_t) t * stride + j] = (j * 9 + t) % cap;
        }
        int32_t *df = dev(firsts), *dc = dev(counts), *dl = dev(cl);
        const int maxc = ring ? 512 : stride;
        void* work = nullptr;
        cudaMalloc(&work, mla_attend_work_bytes(heads, maxc, 8));
        cudaMemset(work, 0, mla_attend_work_bytes(heads, maxc, 8));
        RowSpans rs;
        rs.firsts = df;
        rs.counts = dc;
        if (!ring) {
            rs.cells = dl;
            rs.cells_stride = stride;
        }
        for (int T : {1, 6}) {
            const double u = timed(200, [&](int i) {
                mla_attend(qc, (const uint16_t*) (wp + (size_t) (i % ncopies) * cap * W * 2), rs, T, sk, a, ring, maxc, work, lat, st);
            });
            const double bytes = (double) n * W * 2;   // a row's cells, read once for all heads (the rows of a window overlap)
            std::printf("bench: attend %d heads, %d cells, %d rows: %.1f us (%.0f GB/s of the cells)\n", heads, n, T, u, bytes / u / 1e3);
        }
        cudaFree(work);
    }
    cudaFree(wp);
    cudaStreamDestroy(st);
}

// A chunk of rows (the tensor-core kernel) over FP8 rows: against the plain reference over the decoded rows, against the window
// kernel row by row, and a row's output alone among 9 rows.
void test_fp8_chunk() {
    MlaArgs a;
    a.heads = 32;
    const int W = a.rank + a.rope, Q = a.nope + a.rope, N = 1200, T = 12, stride = 1300;
    std::vector<float> q = randn((size_t) N * a.heads * Q), qabs = randn((size_t) N * a.heads * a.rank, 0.3f), kvraw = randn((size_t) N * W, 2.f),
                       norm = randn(a.rank, 0.5f), sinks = randn(a.heads);
    for (auto& v : norm) v += 1.f;
    float *dq = dev(q), *dqa = dev(qabs), *dkv = dev(kvraw), *dn = dev(norm), *ds = dev(sinks);
    float *dqcur = devf((size_t) N * a.heads * W), *dlat = devf((size_t) T * a.heads * a.rank), *d1 = devf((size_t) 9 * a.heads * a.rank);
    void* work = nullptr;
    cudaMalloc(&work, mla_attend_work_bytes(a.heads, N, T));
    cudaMemset(work, 0, mla_attend_work_bytes(a.heads, N, T));
    uint8_t* dc = nullptr;
    cudaMalloc(&dc, (size_t) N * kFp8RowBytes);
    mla_prepare(dq, dqa, dkv, dn, a, 0, N, 0, dqcur, dc, nullptr);
    std::vector<uint8_t> rows = host(dc, (size_t) N * kFp8RowBytes);
    std::vector<float> dec((size_t) N * W), qcur = host(dqcur, (size_t) T * a.heads * W);
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = &rows[(size_t) n * kFp8RowBytes];
        float sc[4];
        std::memcpy(sc, row + 512, sizeof sc);
        for (int c = 0; c < a.rank; ++c) {
            const uint8_t b = row[c];
            const int e = (b >> 3) & 15, m = b & 7;
            const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp((double) m, -9);
            dec[(size_t) n * W + c] = (float) (((b & 0x80) ? -v : v) * sc[c >> 7]);
        }
        for (int c = 0; c < a.rope; ++c) {
            uint16_t b;
            std::memcpy(&b, row + 528 + 2 * c, 2);
            dec[(size_t) n * W + a.rank + c] = from_bf16(b);
        }
    }
    for (int use_cells = 0; use_cells < 2; ++use_cells) {
        std::vector<int32_t> firsts(T), counts(T), cl((size_t) T * stride, 0);
        std::vector<std::vector<int>> lists(T);
        for (int t = 0; t < T; ++t) {
            firsts[t] = 0;
            if (use_cells)
                for (int c = t % 2; c < N && (int) lists[t].size() < 600 + 37 * t; c += 2) lists[t].push_back(c);
            else
                for (int c = 0; c < 1000 + 13 * t; ++c) lists[t].push_back(c);
            counts[t] = (int) lists[t].size();
            for (int j = 0; j < counts[t]; ++j) cl[(size_t) t * stride + j] = lists[t][j];
        }
        int32_t *df = dev(firsts), *dcnt = dev(counts), *dcl = dev(cl);
        RowSpans rs;
        rs.firsts = df;
        rs.counts = dcnt;
        rs.cells = use_cells ? dcl : nullptr;
        rs.cells_stride = stride;
        mla_attend(dqcur, dc, rs, T, ds, a, 0, N, work, dlat, nullptr);
        const std::vector<float> got = host(dlat, (size_t) T * a.heads * a.rank);
        std::vector<double> ref, win;
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < a.heads; ++h) {
                std::vector<float> qrow(qcur.begin() + (size_t) t * a.heads * W, qcur.begin() + (size_t) (t + 1) * a.heads * W);
                std::vector<double> o = ref_attend(qrow, dec, lists[t], h, a, sinks[h]);
                ref.insert(ref.end(), o.begin(), o.end());
            }
            RowSpans r1;   // the window kernel: the row alone
            r1.firsts = df + t;
            r1.counts = dcnt + t;
            r1.cells = use_cells ? dcl + (size_t) t * stride : nullptr;
            r1.cells_stride = stride;
            mla_attend(dqcur + (size_t) t * a.heads * W, dc, r1, 1, ds, a, 0, N, work, d1, nullptr);
            const std::vector<float> w1 = host(d1, (size_t) a.heads * a.rank);
            win.insert(win.end(), w1.begin(), w1.end());
        }
        report(use_cells ? "mla_attend fp8 chunk, cells" : "mla_attend fp8 chunk, spans", got, ref, 1e-3);
        report("  against the window kernel", got, win, 1e-3);
        if (!use_cells) {
            RowSpans r9;
            r9.firsts = df + 1;
            r9.counts = dcnt + 1;
            mla_attend(dqcur + (size_t) a.heads * W, dc, r9, 9, ds, a, 0, N, work, d1, nullptr);
            const bool same = std::memcmp(host(d1, (size_t) a.heads * a.rank).data(), got.data() + (size_t) a.heads * a.rank, (size_t) a.heads * a.rank * 4) == 0;
            std::printf("%-30s %s\n", "  row independence", same ? "bitwise" : "FAIL");
            failures += !same;
        }
        cudaFree(df); cudaFree(dcnt); cudaFree(dcl);
    }
    cudaFree(dq); cudaFree(dqa); cudaFree(dkv); cudaFree(dn); cudaFree(ds); cudaFree(dqcur); cudaFree(dlat); cudaFree(d1); cudaFree(dc); cudaFree(work);
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

// An NVFP4 expert (blob [gate | up | down] and its FP32 tail {s_gate, s_up, s_down, 0}) through the prompt path's chain - MMQ
// gate/up, the clamped SwiGLU with the blob's scales, MMQ down - against the decode path's grouped kernel on the same rows. The
// MMQ products leave the tail out, so a prompt path that forgets it is off by the scales.
void test_prompt_vs_decode_nvfp4() {
    namespace mmq = basalt::prefill::mmq;
    constexpr int kNvfp4 = 40;
    const int L = 1536, F = 1536, NE = 3, R0 = 5, R1 = 3, R2 = 9, R = R0 + R1 + R2;
    const float limit = 10.f;
    const char* label = "prompt vs decode NVFP4 expert";
    if (!mmq::built() || !mmq::fits(kNvfp4, 2 * F) || !mmq::fits(kNvfp4, L) || !bk::native_expert_supported(kNvfp4, kNvfp4, L, F)) {
        std::printf("%-30s skipped\n", label);
        return;
    }
    const bk::NativeExpertLayout lay = bk::native_expert_layout(kNvfp4, kNvfp4, L, F, limit);
    const size_t unit = 144, stride = (lay.bytes + 4096 + unit - 1) / unit * unit;   // whole 36-byte blocks and 16 bytes, as MMQ steps
    std::vector<uint8_t> blobs(stride * NE, 0);
    const float scales[NE][3] = {{3.f, 2.f, 0.5f}, {0.25f, 4.f, 2.f}, {1.5f, 0.75f, 0.125f}};
    // random NVFP4 blocks (64 values: four UE4M3 sub-block scales near 1/16 to 1/8, then 32 bytes of E2M1 codes), the tails set
    std::uniform_int_distribution<int> sd(0x18, 0x20), code(0, 255);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blobs.data() + stride * e;
        for (size_t o = 0; o + 36 <= lay.bytes - 16; o += 36) {
            for (int i = 0; i < 4; ++i) b[o + i] = (uint8_t) sd(rng);
            for (int i = 4; i < 36; ++i) b[o + i] = (uint8_t) code(rng);
        }
        const float tail[4] = {scales[e][0], scales[e][1], scales[e][2], 0.f};
        std::memcpy(b + lay.bytes - 16, tail, 16);
    }
    std::vector<float> x = randn((size_t) R * L);
    std::vector<int32_t> bounds = {0, R0, R0 + R1, R}, tok(R);
    for (int i = 0; i < R; ++i) tok[i] = i;
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);   // the native kernels take an explicit stream
    uint8_t* dw = dev(blobs);
    float* dx = dev(x);
    int32_t *db = dev(bounds), *dt = dev(tok), *ident = nullptr;
    cudaMalloc(&ident, R * 4);
    mmq::iota(ident, R, st);
    // the decode path: one group per expert
    std::vector<unsigned long long> gp(NE);
    for (int e = 0; e < NE; ++e) gp[e] = (unsigned long long) (dw + stride * e);
    std::vector<int32_t> ng = {NE, R, 0, 0};
    unsigned long long* dgp = dev(gp);
    int32_t* dng = dev(ng);
    void *x8 = nullptr, *scr = nullptr;
    cudaMalloc(&x8, bk::native_q8_1_bytes(L, 1) * R);
    cudaMalloc(&scr, bk::native_expert_scratch_bytes(R, F));
    float* dec = devf((size_t) R * L);
    const size_t row_q8 = bk::native_q8_1_bytes(L, 1);
    for (int c0 = 0; c0 < R; c0 += 8)   // the decode quantizer takes up to 8 rows a launch
        bk::native_quantize_q8_1(dx + (size_t) c0 * L, (uint8_t*) x8 + row_q8 * c0, L, std::min(8, R - c0), st);
    bk::native_expert_grouped_clamped(lay, limit, dgp, db, dng, dt, dt, NE, R, x8, scr, dec, st);
    // the prompt path's chain
    void *xq1 = nullptr, *xq2 = nullptr;
    cudaMalloc(&xq1, mmq::q8_bytes(R, L));
    cudaMalloc(&xq2, mmq::q8_bytes(R, F));
    float *gu = devf((size_t) R * 2 * F), *h = devf((size_t) R * F), *out = devf((size_t) R * L);
    mmq::load(kNvfp4, 2 * F);
    mmq::load(kNvfp4, L);
    mmq::Context ctx;
    mmq::quantize(dx, dt, xq1, kNvfp4, L, L, R, st);
    mmq::Product p;
    p.w = dw; p.type = kNvfp4; p.w_rows = 2 * F; p.w_cols = L; p.expert_bytes = stride; p.n = NE; p.xq = xq1; p.bounds = db;
    p.ids = ident; p.total_rows = R; p.max_rows = R2; p.dst = gu; p.ld_dst = 2 * F;
    ctx.run(p, st);
    const ExpertTails tails{dw, stride, lay.bytes - 16, db, NE};
    const auto chain = [&](const ExpertTails* t) {
        swiglu_split_clamped(gu, h, R, F, limit, st, t);
        mmq::quantize(h, nullptr, xq2, kNvfp4, F, F, R, st);
        mmq::Product d = p;
        d.w = dw + lay.down_off; d.w_rows = L; d.w_cols = F; d.xq = xq2; d.dst = out; d.ld_dst = L;
        ctx.run(d, st);
        cudaStreamSynchronize(st);
        return host(out, (size_t) R * L);
    };
    cudaStreamSynchronize(st);
    const std::vector<float> want = host(dec, (size_t) R * L);
    const std::vector<double> ref(want.begin(), want.end());
    const std::vector<float> got = chain(&tails);
    report(label, got, ref, 3e-2);
    // the same chain without the scales must not pass: the test sees a prompt path that forgets them
    const std::vector<float> bare = chain(nullptr);
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < bare.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(bare[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const bool caught = max_abs / std::max(max_ref, 1e-30) > 3e-2;
    std::printf("%-30s max rel %.3e without the tails  %s\n", label, max_abs / std::max(max_ref, 1e-30), caught ? "ok (caught)" : "FAIL (not caught)");
    failures += !caught;
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
    test_select_long();
    bench_indexer_batch();
    test_fp8();
    test_query();
    test_mla_long();
    test_fp8_chunk();
    test_mmq_dense();
    test_mmq_expert(12, "mmq experts Q4_K");
    test_mmq_expert(14, "mmq experts Q6_K");
    test_prompt_vs_decode_nvfp4();
    bench_mla();
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
