// src/models/gemma4/kernels/kernels_parity.cpp - the Gemma 4 kernels against CPU references in double: the attention's
// preparation and decode attention (a window layer's ring past its wrap, a full layer whose keys are its values), the glue;
// a batch window of two conversations bitwise equal to each alone; and a microbench line of the decode attention.
#include "basalt/models/gemma4/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::gemma4;

namespace {

int failures = 0;

void expect(bool ok, const char* what, double err) {
    std::printf("%-58s %s (max err %.3g)\n", what, ok ? "ok" : "FAIL", err);
    if (!ok) ++failures;
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

std::vector<float> randv(size_t n, std::mt19937& rng, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

double h2d(float x) { return (double) __half2float(__float2half_rn(x)); }

// The reference of one layer: the rows' q, k, v (projection outputs), the cache's earlier positions filled the same way
// from `past` rows, then attention. Returns out [rows][n_head][hd].
struct Ref {
    LayerShape sh;
    std::vector<float> qn, kn;
    std::vector<double> finv;
    double eps = 1e-6;
    // normed and rotated head row `x` at `pos` (w null: unweighted, rotate false: as is)
    std::vector<double> prep(const float* x, const float* w, bool rotate, int pos) const {
        const int hd = sh.hd;
        double ss = 0;
        for (int i = 0; i < hd; ++i) ss += (double) x[i] * x[i];
        const double inv_r = 1.0 / std::sqrt(ss / hd + eps);
        std::vector<double> y(hd);
        for (int i = 0; i < hd; ++i) y[i] = x[i] * inv_r * (w ? w[i] : 1.0);
        if (rotate) {
            std::vector<double> z = y;
            for (int i = 0; i < sh.rot; ++i) {
                const double ang = pos * finv[i];
                const double c = std::cos(ang), s = std::sin(ang);
                z[i] = y[i] * c - y[i + hd / 2] * s;
                z[i + hd / 2] = y[i + hd / 2] * c + y[i] * s;
            }
            y = z;
        }
        return y;
    }
};

// One layer's test: `n_pos` positions of keys written through attn_prep in windows of `rows`, the last window's
// attention checked against the reference. Returns the GPU outputs of the last window (for the batch check).
struct Case {
    const char* name;
    LayerShape sh;
    int n_pos, rows;
};

std::vector<float> run_case(const Case& cs, std::mt19937& rng, bool check, std::vector<float>* qkv_out = nullptr,
                            const std::vector<float>* qkv_in = nullptr) {
    const LayerShape& sh = cs.sh;
    const int hd = sh.hd, nq = sh.n_head, nk = sh.n_kv, P = cs.n_pos, R = cs.rows;
    const int qd = nq * hd, kd = nk * hd;
    const int64_t cap = sh.window > 0 ? sh.window + kRows : P;
    Ref ref;
    ref.sh = sh;
    ref.qn = randv(hd, rng, 0.3f);
    for (float& x : ref.qn) x += 1.f;
    ref.kn = randv(hd, rng, 0.3f);
    for (float& x : ref.kn) x += 1.f;
    const double base = sh.window > 0 ? 1e4 : 1e6;
    ref.finv.assign(hd / 2, 0.0);
    for (int i = 0; i < sh.rot; ++i) ref.finv[i] = std::pow(base, -2.0 * i / hd);
    std::vector<float> lo(rope_lo_floats(hd)), hi(rope_hi_floats(hd, P + kRows));
    rope_tables(hd, sh.rot, base, P + kRows, lo.data(), hi.data());
    // every position's projections (q, k, v), positions 0..P-1
    std::vector<float> q, k, v;
    if (qkv_in) {
        q.assign(qkv_in->begin(), qkv_in->begin() + (size_t) P * qd);
        k.assign(qkv_in->begin() + (size_t) P * qd, qkv_in->begin() + (size_t) P * (qd + kd));
        v.assign(qkv_in->begin() + (size_t) P * (qd + kd), qkv_in->end());
    } else {
        q = randv((size_t) P * qd, rng, 2.f), k = randv((size_t) P * kd, rng, 2.f), v = randv((size_t) P * kd, rng, 2.f);
    }
    if (qkv_out) {
        *qkv_out = q;
        qkv_out->insert(qkv_out->end(), k.begin(), k.end());
        qkv_out->insert(qkv_out->end(), v.begin(), v.end());
    }
    float *dq = dev(q), *dk = dev(k), *dv = dev(v), *dqn = dev(ref.qn), *dkn = dev(ref.kn), *dlo = dev(lo), *dhi = dev(hi);
    void *ck = nullptr, *cv = nullptr;
    cudaMalloc(&ck, (size_t) cap * kd * 2);
    cudaMalloc(&cv, (size_t) cap * kd * 2);
    float *qp = nullptr, *out = nullptr, *scr = nullptr;
    cudaMalloc(&qp, (size_t) R * qd * 4);
    cudaMalloc(&out, (size_t) R * qd * 4);
    cudaMalloc(&scr, attn_scratch_floats(R, nq, hd) * 4);
    std::vector<int32_t> pos(R);
    int32_t* dpos = nullptr;
    cudaMalloc(&dpos, R * 4);
    for (int p0 = 0; p0 < P; p0 += R) {
        const int n = std::min(R, P - p0);
        for (int i = 0; i < n; ++i) pos[i] = p0 + i;
        cudaMemcpy(dpos, pos.data(), n * 4, cudaMemcpyHostToDevice);
        Prep pr;
        pr.q = dq + (size_t) p0 * qd, pr.k = dk + (size_t) p0 * kd, pr.v = dv + (size_t) p0 * kd;
        pr.ldq = qd, pr.ldk = kd, pr.ldv = kd, pr.pos = dpos, pr.q_norm = dqn, pr.k_norm = dkn, pr.rope = {dlo, dhi}, pr.sh = sh;
        pr.q_out = qp;
        pr.kv.c[0] = {ck, cv, cap};
        attn_prep(pr, n, nullptr);
        Attn a;
        a.q = qp, a.pos = dpos, a.k_norm = dkn, a.rope = {dlo, dhi}, a.sh = sh, a.kv = pr.kv;
        a.max_keys = sh.window > 0 ? sh.window : P, a.scratch = scr, a.out = out;
        attn_decode(a, n, nullptr);
    }
    cudaDeviceSynchronize();
    const int last0 = (P - 1) / R * R, nl = P - last0;
    std::vector<float> got = host(out, (size_t) nl * qd);
    if (check) {
        // reference: every position's prepared key and value (rounded to fp16 as the cache holds them)
        std::vector<std::vector<double>> K((size_t) P * nk), V((size_t) P * nk);
        for (int p = 0; p < P; ++p)
            for (int h = 0; h < nk; ++h) {
                const float* kr = k.data() + (size_t) p * kd + (size_t) h * hd;
                const float* vr = (sh.one ? k : v).data() + (size_t) p * kd + (size_t) h * hd;
                std::vector<double> vv = ref.prep(vr, nullptr, false, p);
                for (double& x : vv) x = h2d((float) x);
                std::vector<double> kk;
                if (sh.one) {   // K = rope(V_fp16 * k_norm)
                    std::vector<float> t(hd);
                    for (int i = 0; i < hd; ++i) t[i] = (float) vv[i] * ref.kn[i];
                    std::vector<float> ones(hd, 1.f);
                    double ss = 0;
                    for (float x : t) ss += (double) x * x;
                    kk.assign(t.begin(), t.end());
                    std::vector<double> z = kk;
                    for (int i = 0; i < sh.rot; ++i) {
                        const double ang = p * ref.finv[i];
                        const double c = std::cos(ang), s = std::sin(ang);
                        z[i] = kk[i] * c - kk[i + hd / 2] * s;
                        z[i + hd / 2] = kk[i + hd / 2] * c + kk[i] * s;
                    }
                    kk = z;
                } else {
                    kk = ref.prep(kr, ref.kn.data(), true, p);
                    for (double& x : kk) x = h2d((float) x);
                }
                K[(size_t) p * nk + h] = kk;
                V[(size_t) p * nk + h] = vv;
            }
        double maxerr = 0;
        for (int t = 0; t < nl; ++t) {
            const int p = last0 + t;
            const int lo = sh.window > 0 ? std::max(0, p - sh.window + 1) : 0;
            for (int hq = 0; hq < nq; ++hq) {
                const int h = hq / (nq / nk);
                std::vector<double> qq = ref.prep(q.data() + (size_t) p * qd + (size_t) hq * hd, ref.qn.data(), true, p);
                std::vector<double> s;
                double M = -1e300;
                for (int j = lo; j <= p; ++j) {
                    double d = 0;
                    for (int i = 0; i < hd; ++i) d += qq[i] * K[(size_t) j * nk + h][i];
                    s.push_back(d);
                    M = std::max(M, d);
                }
                double L = 0;
                for (double& x : s) L += (x = std::exp(x - M));
                for (int i = 0; i < hd; ++i) {
                    double o = 0;
                    for (int j = lo; j <= p; ++j) o += s[(size_t) (j - lo)] * V[(size_t) j * nk + h][i];
                    maxerr = std::max(maxerr, std::fabs(o / L - got[((size_t) t * nq + hq) * hd + i]));
                }
            }
        }
        expect(maxerr < 5e-3, cs.name, maxerr);   // keys rounded to fp16 in the cache
    }
    for (void* p : {(void*) dq, (void*) dk, (void*) dv, (void*) dqn, (void*) dkn, (void*) dlo, (void*) dhi, ck, cv, (void*) qp, (void*) out,
                    (void*) scr, (void*) dpos})
        cudaFree(p);
    return got;
}

void glue_checks(std::mt19937& rng) {
    const int R = 3, E = 2816, F = 2112, FP = 2176, K = 8;
    const double eps = 1e-6;
    auto x = randv((size_t) R * E, rng), y = randv((size_t) R * E, rng, 3.f), w = randv(E, rng, 0.2f);
    float *dx = dev(x), *dy = dev(y), *dw = dev(w);
    add_norm(dx, dy, dw, (float) eps, R, E, nullptr);
    std::vector<float> got = host(dx, x.size());
    double err = 0;
    for (int r = 0; r < R; ++r) {
        double ss = 0;
        for (int i = 0; i < E; ++i) ss += (double) y[r * E + i] * y[r * E + i];
        const double inv = 1 / std::sqrt(ss / E + eps);
        for (int i = 0; i < E; ++i) err = std::max(err, std::fabs(x[r * E + i] + y[r * E + i] * inv * w[i] - got[r * E + i]));
    }
    expect(err < 1e-4, "add_norm", err);
    auto g = randv((size_t) R * FP, rng, 2.f), u = randv((size_t) R * FP, rng);
    float *dg = dev(g), *du = dev(u), *dh = nullptr;
    cudaMalloc(&dh, (size_t) R * FP * 4);
    gelu_mul(dg, du, R, F, FP, FP, dh, nullptr);
    got = host(dh, (size_t) R * FP);
    err = 0;
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < FP; ++i) {
            const double a = g[r * FP + i];
            const double want = i < F ? 0.5 * a * (1 + std::tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))) * u[r * FP + i] : 0;
            err = std::max(err, std::fabs(want - got[r * FP + i]));
        }
    expect(err < 1e-5, "gelu_mul (with zero padding)", err);
    // ffn_end with experts
    auto mlp = randv((size_t) R * E, rng), ex = randv((size_t) R * K * E, rng), wt = randv((size_t) R * K, rng, 0.3f);
    auto n1 = randv(E, rng, 0.2f), n2 = randv(E, rng, 0.2f), np = randv(E, rng, 0.2f);
    std::vector<float> sc{0.75f};
    auto x2 = randv((size_t) R * E, rng);
    float *dx2 = dev(x2), *dm = dev(mlp), *de = dev(ex), *dwt = dev(wt), *dn1 = dev(n1), *dn2 = dev(n2), *dnp = dev(np), *dsc = dev(sc);
    FfnEnd fe;
    fe.x = dx2, fe.mlp = dm, fe.experts = de, fe.weights = dwt, fe.k = K, fe.n1 = dn1, fe.n2 = dn2, fe.n_post = dnp, fe.scalar = dsc;
    ffn_end(fe, R, E, nullptr);
    got = host(dx2, x2.size());
    err = 0;
    auto rms = [&](const std::vector<double>& a) {
        double ss = 0;
        for (double v : a) ss += v * v;
        return 1 / std::sqrt(ss / a.size() + eps);
    };
    for (int r = 0; r < R; ++r) {
        std::vector<double> m(mlp.begin() + r * E, mlp.begin() + (r + 1) * E), s(E, 0.0), a(E);
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < E; ++i) s[i] += (double) wt[r * K + k] * ex[((size_t) r * K + k) * E + i];
        const double i1 = rms(m), i2 = rms(s);
        for (int i = 0; i < E; ++i) a[i] = m[i] * i1 * n1[i] + s[i] * i2 * n2[i];
        const double ip = rms(a);
        for (int i = 0; i < E; ++i) err = std::max(err, std::fabs((x2[r * E + i] + a[i] * ip * np[i]) * 0.75 - got[r * E + i]));
    }
    expect(err < 1e-4, "ffn_end (experts and MLP)", err);
    auto lg = randv(1000, rng, 40.f);
    float* dl = dev(lg);
    softcap(dl, 1000, 30.f, nullptr);
    got = host(dl, 1000);
    err = 0;
    for (int i = 0; i < 1000; ++i) err = std::max(err, std::fabs(30.0 * std::tanh(lg[i] / 30.0) - got[i]));
    expect(err < 1e-4, "softcap", err);
    // embed rows of a BF16 table
    std::vector<uint16_t> tab((size_t) 10 * E);
    for (size_t i = 0; i < tab.size(); ++i) tab[i] = (uint16_t) (0x3c00 + (i * 7919) % 2000);
    std::vector<int32_t> tok{3, 7, 0};
    uint16_t* dt = dev(tab);
    int32_t* dtok = dev(tok);
    float* de2 = nullptr;
    cudaMalloc(&de2, (size_t) 3 * E * 4);
    embed_rows(dt, dtok, 3, E, 53.f, de2, nullptr);
    got = host(de2, (size_t) 3 * E);
    err = 0;
    for (int r = 0; r < 3; ++r)
        for (int i = 0; i < E; ++i) {
            uint32_t b = (uint32_t) tab[(size_t) tok[r] * E + i] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            err = std::max(err, std::fabs((double) f * 53.0 - got[r * E + i]));
        }
    expect(err == 0, "embed_rows", err);
}

// Two conversations of one layer in one call against each alone: bitwise.
void batch_check(std::mt19937& rng, const LayerShape& sh, const char* name) {
    const int hd = sh.hd, nq = sh.n_head, nk = sh.n_kv, qd = nq * hd, kd = nk * hd;
    const int P[2] = {sh.window > 0 ? sh.window + 37 : 300, 77};
    const int R[2] = {2, 3};
    const int64_t cap[2] = {sh.window > 0 ? sh.window + kRows : 1024, sh.window > 0 ? sh.window + kRows : 1024};
    auto qn = randv(hd, rng), kn = randv(hd, rng);
    std::vector<float> lo(rope_lo_floats(hd)), hi(rope_hi_floats(hd, 1024));
    rope_tables(hd, sh.rot, 1e4, 1024, lo.data(), hi.data());
    float *dqn = dev(qn), *dkn = dev(kn);
    const RopeTab rt{dev(lo), dev(hi)};
    void* c[2][2];
    for (int s = 0; s < 2; ++s)
        for (int j = 0; j < 2; ++j) {
            cudaMalloc(&c[s][j], (size_t) cap[s] * kd * 2);
            auto fill = randv((size_t) cap[s] * kd, rng);
            std::vector<__half> hf(fill.size());
            for (size_t i = 0; i < fill.size(); ++i) hf[i] = __float2half_rn(fill[i]);
            cudaMemcpy(c[s][j], hf.data(), hf.size() * 2, cudaMemcpyHostToDevice);
        }
    auto q = randv((size_t) 5 * qd, rng, 2.f), k = randv((size_t) 5 * kd, rng, 2.f), v = randv((size_t) 5 * kd, rng, 2.f);
    float *dq = dev(q), *dk = dev(k), *dv = dev(v), *qp, *out, *scr;
    cudaMalloc(&qp, (size_t) 5 * qd * 4);
    cudaMalloc(&out, (size_t) 5 * qd * 4);
    cudaMalloc(&scr, attn_scratch_floats(5, nq, hd) * 4);
    int32_t* dpos;
    cudaMalloc(&dpos, 5 * 4);
    auto run = [&](int row0, int n, const std::vector<int32_t>& pos, const KvSegs& kv) {
        cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
        Prep pr;
        pr.q = dq + (size_t) row0 * qd, pr.k = dk + (size_t) row0 * kd, pr.v = dv + (size_t) row0 * kd;
        pr.ldq = qd, pr.ldk = kd, pr.ldv = kd, pr.pos = dpos, pr.q_norm = dqn, pr.k_norm = dkn, pr.rope = rt, pr.sh = sh;
        pr.q_out = qp, pr.kv = kv;
        attn_prep(pr, n, nullptr);
        Attn a;
        a.q = qp, a.pos = dpos, a.k_norm = dkn, a.rope = rt, a.sh = sh, a.kv = kv, a.max_keys = 1024, a.scratch = scr, a.out = out;
        attn_decode(a, n, nullptr);
        return host(out, (size_t) n * qd);
    };
    std::vector<float> solo;
    for (int s = 0; s < 2; ++s) {
        KvSegs kv;
        kv.c[0] = {c[s][0], c[s][1], cap[s]};
        std::vector<int32_t> pos;
        for (int i = 0; i < R[s]; ++i) pos.push_back(P[s] + i);
        auto o = run(s == 0 ? 0 : R[0], R[s], pos, kv);
        solo.insert(solo.end(), o.begin(), o.end());
    }
    KvSegs kv;
    kv.at.n = 2;
    kv.at.row0[0] = 0, kv.at.row0[1] = R[0], kv.at.row0[2] = R[0] + R[1];
    for (int s = 0; s < 2; ++s) kv.c[s] = {c[s][0], c[s][1], cap[s]};
    std::vector<int32_t> pos;
    for (int s = 0; s < 2; ++s)
        for (int i = 0; i < R[s]; ++i) pos.push_back(P[s] + i);
    auto both = run(0, R[0] + R[1], pos, kv);
    expect(std::memcmp(both.data(), solo.data(), both.size() * 4) == 0, name, 0.0);
}

void bench(std::mt19937& rng) {
    // a full layer's decode attention at 4,096 keys, one row (the microbench line)
    LayerShape sh{16, 2, 512, 64, 0, true};
    const int P = 4096, hd = sh.hd, kd = sh.n_kv * hd, qd = sh.n_head * hd;
    std::vector<__half> cache((size_t) P * kd);
    for (auto& x : cache) x = __float2half_rn(std::normal_distribution<float>()(rng));
    void* ck = nullptr;
    cudaMalloc(&ck, cache.size() * 2);
    cudaMemcpy(ck, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice);
    auto q = randv(qd, rng), kn = randv(hd, rng);
    std::vector<float> lo(rope_lo_floats(hd)), hi(rope_hi_floats(hd, P));
    rope_tables(hd, 64, 1e6, P, lo.data(), hi.data());
    std::vector<int32_t> pos{P - 1};
    float *dq = dev(q), *dkn = dev(kn), *out, *scr;
    int32_t* dpos = dev(pos);
    cudaMalloc(&out, (size_t) qd * 4);
    cudaMalloc(&scr, attn_scratch_floats(1, sh.n_head, hd) * 4);
    Attn a;
    a.q = dq, a.pos = dpos, a.k_norm = dkn, a.rope = {dev(lo), dev(hi)}, a.sh = sh, a.max_keys = P, a.scratch = scr, a.out = out;
    a.kv.c[0] = {ck, nullptr, P};
    for (int i = 0; i < 20; ++i) attn_decode(a, 1, nullptr);
    cudaDeviceSynchronize();
    const int iters = 200;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) attn_decode(a, 1, nullptr);
    cudaDeviceSynchronize();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("bench attn_decode full layer (16x512, 2 kv heads, one cache) 4096 keys, 1 row: %.1f us, %.0f GB/s\n", us,
                (double) cache.size() * 2 / us / 1e3);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device\n");
        return 77;
    }
    std::mt19937 rng(7);
    run_case({"window layer (256, 8 kv, window 64), ring wrapped, 5 rows", {16, 8, 256, 128, 64, false}, 203, 5}, rng, true);
    run_case({"window layer, 1 row", {16, 8, 256, 128, 64, false}, 150, 1}, rng, true);
    run_case({"full layer (512, 2 kv, rot 64, keys = values), 4 rows", {16, 2, 512, 64, 0, true}, 300, 4}, rng, true);
    run_case({"full layer, chunked keys (9,000 positions), 1 row", {16, 2, 512, 64, 0, true}, 9000, 16}, rng, true);
    batch_check(rng, {16, 8, 256, 128, 64, false}, "batch of two conversations = each alone (window layer)");
    batch_check(rng, {16, 2, 512, 64, 0, true}, "batch of two conversations = each alone (full layer)");
    glue_checks(rng);
    bench(rng);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
