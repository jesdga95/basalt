// tests/kernels/dec_gdn_parity.cpp - the verify window's GDN block around dec_gdn_rec against the per-stage verify
// kernels, bitwise.
//
//     build/dec_gdn_parity [--bench | --bench-only]
//
// The reference is the verify window's GDN sequence: native_quantize_q8_1 + native_mmvq (qkv, z),
// gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, native_quantize_q8_1 + native_mmvq (out).  The path under
// test is the native in/out projections around dec_gdn_rec (mode 1) or dec_gdn_rec_split + dec_gdn_rec_norm with z on
// a side stream forked after qkv (mode 2) or after the q8_1 (mode 3).  Every output both paths write (qkv, z, h,
// gate, beta, the q8_1 of y, the out projection) must match bit for bit at the model's geometry, for T = 1..8 and
// every projection type the model uses in every position, and each column must equal the same column of an
// eight-token window; the state must be left untouched.  Weights are random bytes with normal fp16 scales
// (mmvq_multi_parity's rule), activations and state random normal.
//
// Segmented windows (part of the default run): conv_l2, the verify recurrence, dec_gdn_rec and the commit over a
// device table of segments (1, 2, 5 or 8 segments, mixed tokens per segment, windows past kDecMaxT rows) against the
// current kernels called once per segment (bitwise) and a CPU reference (tolerance).
//
// --bench: the reference and the three modes timed as CUDA graphs over enough layers' distinct weights that the L2
// holds none of them, per stage and whole, in microseconds per layer and GB/s over the bytes each must move.
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {

constexpr int N = 2560, HK = 16, HV = 48, S = 128, C = 2 * HK * S + HV * S, ZV = HV * S;
constexpr float EPS = 1e-6f;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

struct Fmt {
    int type;
    const char* name;
    int block_bytes;
    int scale_at[2];
};
const Fmt FMTS[] = {
    {12, "Q4_K", 144, {0, 2}}, {13, "Q5_K", 176, {0, 2}}, {14, "Q6_K", 210, {208, -1}},
    {21, "IQ3_S", 110, {0, -1}}, {23, "IQ4_XS", 136, {0, -1}},
};
const Fmt& fmt(int t) {
    for (const Fmt& f : FMTS)
        if (f.type == t) return f;
    std::printf("no format %d\n", t);
    std::exit(1);
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

size_t mat_bytes(int t, int n_in, int n_out) { return (size_t) n_out * (n_in / 256) * fmt(t).block_bytes; }

void* rand_mat(int t, int n_in, int n_out, std::mt19937& rng) {
    const Fmt& f = fmt(t);
    const size_t nb = (size_t) n_out * (n_in / 256), bytes = nb * f.block_bytes;
    std::vector<uint8_t> w(bytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (size_t k = 0; k < nb; ++k)
        for (int at : f.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * f.block_bytes + at], &h, 2);
            }
    void* d = nullptr;
    CK(cudaMalloc(&d, bytes + 64));
    CK(cudaMemcpy(d, w.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());   // a pageable copy may return before its DMA lands; the test streams do not wait for it
    return d;
}

template<typename V>
V* upload(const std::vector<V>& h) {
    V* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return d;
}
std::vector<float> randn(size_t n, float sd, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One GDN layer's weights and constants.
struct Layer {
    int tq, tz, to;
    void *wqkv, *wz, *wo;
    float *conv_w, *dt, *ssm_a, *gamma, *state;
    uint16_t *wa, *wb;
    size_t bytes = 0;   // the bytes a window must read: weights + state
};
Layer make_layer(int tq, int tz, int to, std::mt19937& rng) {
    Layer L{};
    L.tq = tq; L.tz = tz; L.to = to;
    L.wqkv = rand_mat(tq, N, C, rng);
    L.wz = rand_mat(tz, N, ZV, rng);
    L.wo = rand_mat(to, ZV, N, rng);
    L.conv_w = upload(randn((size_t) C * 4, 0.5f, rng));
    std::vector<float> dt = randn(HV, 1.f, rng), sa(HV), gm(S);
    for (auto& v : sa) v = -std::exp(randn(1, 1.f, rng)[0]);
    for (auto& v : gm) v = 1.f + randn(1, 0.1f, rng)[0];
    L.dt = upload(dt); L.ssm_a = upload(sa); L.gamma = upload(gm);
    std::vector<float> st = randn((size_t) S * HV * S + (size_t) C * 3, 0.1f, rng);
    L.state = upload(st);
    std::vector<uint16_t> a((size_t) HV * N), b((size_t) HV * N);
    std::normal_distribution<float> nd(0.f, 0.03f);
    for (auto& v : a) v = bf16(nd(rng));
    for (auto& v : b) v = bf16(nd(rng));
    L.wa = upload(a); L.wb = upload(b);
    L.bytes = mat_bytes(tq, N, C) + mat_bytes(tz, N, ZV) + mat_bytes(to, ZV, N) + 2 * a.size() * 2 + (size_t) C * 16 +
              st.size() * 4;
    return L;
}

// The window's activations / outputs (one set per path).
struct Bufs {
    float *qkv, *z, *h, *gate, *beta, *y, *bo;
    void* xq;
    void* yq;
};
Bufs make_bufs() {
    Bufs b{};
    CK(cudaMalloc(&b.qkv, sizeof(float) * kDecMaxT * C));
    CK(cudaMalloc(&b.z, sizeof(float) * kDecMaxT * ZV));
    CK(cudaMalloc(&b.h, sizeof(float) * kDecMaxT * C));
    CK(cudaMalloc(&b.gate, sizeof(float) * kDecMaxT * HV));
    CK(cudaMalloc(&b.beta, sizeof(float) * kDecMaxT * HV));
    CK(cudaMalloc(&b.y, sizeof(float) * kDecMaxT * ZV));
    CK(cudaMalloc(&b.bo, sizeof(float) * kDecMaxT * N));
    CK(cudaMalloc(&b.xq, native_q8_1_bytes(ZV, kDecMaxT)));
    CK(cudaMalloc(&b.yq, native_q8_1_bytes(ZV, kDecMaxT)));
    return b;
}
void clear(Bufs& b, cudaStream_t s) {
    CK(cudaMemsetAsync(b.qkv, 0xff, sizeof(float) * kDecMaxT * C, s));
    CK(cudaMemsetAsync(b.z, 0xff, sizeof(float) * kDecMaxT * ZV, s));
    CK(cudaMemsetAsync(b.h, 0xff, sizeof(float) * kDecMaxT * C, s));
    CK(cudaMemsetAsync(b.gate, 0xff, sizeof(float) * kDecMaxT * HV, s));
    CK(cudaMemsetAsync(b.beta, 0xff, sizeof(float) * kDecMaxT * HV, s));
    CK(cudaMemsetAsync(b.bo, 0xff, sizeof(float) * kDecMaxT * N, s));
    CK(cudaMemsetAsync(b.yq, 0xff, native_q8_1_bytes(ZV, kDecMaxT), s));
}

// stage masks for the bench: 1 in-projections, 2 conv/ab/recurrence, 4 out projection
void run_ref(const Layer& L, const float* x, Bufs& b, int T, cudaStream_t s, int stages = 7, unsigned long long* = nullptr) {
    const float* conv = L.state + (size_t) S * HV * S;
    if (stages & 1) {
        native_quantize_q8_1(x, b.xq, N, T, s);
        native_mmvq(L.tq, L.wqkv, b.xq, b.qkv, N, C, T, s);
    }
    if (stages & 2) {
        gdn_conv_l2_multi(conv, b.qkv, L.conv_w, b.h, C, 2 * HK, EPS, T, s, 0);
        gdn_ab_multi(x, L.wa, L.wb, L.dt, L.ssm_a, b.gate, b.beta, N, HV, T, s);
    }
    if (stages & 1) native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, s);
    if (stages & 2) gdn_step_norm_multi(L.state, b.h, C, b.gate, b.beta, b.z, L.gamma, EPS, b.y, HK, HV, T, nullptr, s, 0);
    if (stages & 4) {
        native_quantize_q8_1(b.y, b.yq, ZV, T, s);
        native_mmvq(L.to, L.wo, b.yq, b.bo, ZV, N, T, s);
    }
}

// native in-projections, dec_gdn_rec (1) or dec_gdn_rec_split + dec_gdn_rec_norm with z on a
// side stream forked after qkv (2) or after the q8_1 (3), native out projection on the q8_1 (stage masks as
// run_ref's; a partial mask runs modes 2/3 on one stream)
cudaStream_t g_side = nullptr;
cudaEvent_t g_fork = nullptr, g_join = nullptr;
int g_gdnf = 1;
void run_gdnf(const Layer& L, const float* x, Bufs& b, int T, cudaStream_t s, int stages = 7,
              unsigned long long* = nullptr) {
    const bool fork = g_gdnf >= 2 && stages == 7;
    auto z_side = [&] {
        CK(cudaEventRecord(g_fork, s));
        CK(cudaStreamWaitEvent(g_side, g_fork, 0));
        PdlScope pz(true, true);
        native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, g_side);
        CK(cudaEventRecord(g_join, g_side));
    };
    DecRecArgs a;
    a.state = L.state; a.qkv = b.qkv; a.conv_w = L.conv_w; a.x = x; a.w_alpha = L.wa; a.w_beta = L.wb;
    a.dt = L.dt; a.ssm_a = L.ssm_a; a.z = b.z; a.gamma = L.gamma; a.h = b.h; a.gate = b.gate; a.beta = b.beta;
    a.y = nullptr; a.yq = b.yq; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS;
    a.early_trigger = true;
    a.o = b.y;
    if (stages & 1) {
        native_quantize_q8_1(x, b.xq, N, T, s);
        if (fork && g_gdnf == 3) z_side();
        native_mmvq(L.tq, L.wqkv, b.xq, b.qkv, N, C, T, s);
        if (!fork) native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, s);
    }
    if (stages & 2) {
        if (g_gdnf == 1) {
            dec_gdn_rec(a, T, s);
        } else {
            if (fork && g_gdnf == 2) z_side();
            dec_gdn_rec_split(a, T, s);
            if (fork) CK(cudaStreamWaitEvent(s, g_join, 0));
            PdlScope pn(true, fork);
            dec_gdn_rec_norm(a, T, s);
        }
    }
    if (stages & 4) native_mmvq(L.to, L.wo, b.yq, b.bo, ZV, N, T, s);
}

template<typename V>
std::vector<V> down(const V* d, size_t n) {
    std::vector<V> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(V), cudaMemcpyDeviceToHost));
    return h;
}

// bit differences between two float arrays; prints the first
long long cmp_f(const char* what, const float* a, const float* b, size_t n, double& max_abs) {
    std::vector<float> ha = down(a, n), hb = down(b, n);
    long long d = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&ha[i], &hb[i], 4) != 0) {
            if (d == 0) std::printf("    %s[%zu]: ref %.9g new %.9g\n", what, i, ha[i], hb[i]);
            ++d;
            const double e = std::fabs((double) ha[i] - hb[i]);
            if (e > max_abs) max_abs = e;
        }
        if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) { std::printf("    %s[%zu] not finite\n", what, i); ++d; }
    }
    return d;
}
long long cmp_b(const char* what, const void* a, const void* b, size_t n) {
    std::vector<uint8_t> ha = down((const uint8_t*) a, n), hb = down((const uint8_t*) b, n);
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += ha[i] != hb[i];
    if (d) std::printf("    %s: %lld bytes differ\n", what, d);
    return d;
}

bool parity_gdnf(cudaStream_t s) {
  bool ok = true;
  for (g_gdnf = 1; g_gdnf <= 3; ++g_gdnf) {
    std::mt19937 rng(43);
    const int combos[5][3] = {{14, 23, 14}, {23, 21, 13}, {12, 14, 21}, {13, 12, 23}, {14, 13, 12}};
    Bufs r = make_bufs(), n = make_bufs(), w8 = make_bufs();
    for (const auto& cb : combos) {
        Layer L = make_layer(cb[0], cb[1], cb[2], rng);
        const std::vector<float> st0 = down(L.state, (size_t) S * HV * S + (size_t) C * 3);
        float* x = upload(randn((size_t) kDecMaxT * N, 1.f, rng));
        clear(w8, s);
        run_gdnf(L, x, w8, kDecMaxT, s);
        for (int T = 1; T <= kDecMaxT; ++T) {
            clear(r, s); clear(n, s);
            run_ref(L, x, r, T, s);
            run_gdnf(L, x, n, T, s);
            CK(cudaStreamSynchronize(s));
            double ma = 0;
            long long d = 0, inv = 0;
            d += cmp_f("qkv", r.qkv, n.qkv, (size_t) T * C, ma);
            d += cmp_f("z", r.z, n.z, (size_t) T * ZV, ma);
            d += cmp_f("gate", r.gate, n.gate, (size_t) T * HV, ma);
            d += cmp_f("beta", r.beta, n.beta, (size_t) T * HV, ma);
            d += cmp_f("h", r.h, n.h, (size_t) T * C, ma);
            d += cmp_b("yq", r.yq, n.yq, native_q8_1_bytes(ZV, T));
            d += cmp_f("out", r.bo, n.bo, (size_t) T * N, ma);
            inv += cmp_f("h@8", w8.h, n.h, (size_t) T * C, ma);
            inv += cmp_b("yq@8", w8.yq, n.yq, native_q8_1_bytes(ZV, T));
            inv += cmp_f("out@8", w8.bo, n.bo, (size_t) T * N, ma);
            const std::vector<float> st1 = down(L.state, st0.size());
            const bool st_ok = std::memcmp(st0.data(), st1.data(), st0.size() * 4) == 0;
            std::printf("gdnf=%d qkv %-6s z %-6s out %-6s T=%d: vs ref %s (max abs %.3g), columns vs T=8 %s%s\n",
                        g_gdnf, fmt(cb[0]).name, fmt(cb[1]).name, fmt(cb[2]).name, T, d == 0 ? "bitwise" : "DIFF", ma,
                        inv == 0 ? "bitwise" : "DIFF", st_ok ? "" : " STATE WRITTEN");
            ok = ok && d == 0 && inv == 0 && st_ok;
        }
        CK(cudaFree(x));
    }
  }
    return ok;
}

// ---- segmented windows
// CPU reference of one segment: conv + L2 (h) and the recurrence from `st0` over n tokens (y)
struct CpuSeg {
    std::vector<float> h, y;
};
CpuSeg cpu_seg(const std::vector<float>& st0, const std::vector<float>& qkv, const std::vector<float>& z,
               const std::vector<float>& cw, const std::vector<float>& gamma, const std::vector<float>& gate,
               const std::vector<float>& beta, int n) {
    CpuSeg r;
    r.h.assign((size_t) n * C, 0.f);
    r.y.assign((size_t) n * ZV, 0.f);
    const float* hist = st0.data() + (size_t) S * HV * S;
    for (int t = 0; t < n; ++t) {
        for (int c = 0; c < C; ++c) {
            double win[4];
            for (int j = 0; j < 4; ++j) {
                const int src = t + j;
                win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
            }
            double sum = 0;
            for (int j = 0; j < 4; ++j) sum += win[j] * cw[(size_t) c * 4 + j];
            r.h[(size_t) t * C + c] = (float) (sum / (1.0 + std::exp(-sum)));
        }
        for (int hd = 0; hd < 2 * HK; ++hd) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += (double) r.h[(size_t) t * C + hd * S + i] * r.h[(size_t) t * C + hd * S + i];
            const double sc = 1.0 / std::sqrt(ss + EPS);
            for (int i = 0; i < S; ++i) r.h[(size_t) t * C + hd * S + i] = (float) (r.h[(size_t) t * C + hd * S + i] * sc);
        }
    }
    std::vector<double> sd(st0.begin(), st0.begin() + (size_t) S * HV * S);
    const int qk = S * HK;
    for (int t = 0; t < n; ++t)
        for (int hd = 0; hd < HV; ++hd) {
            const int qh = hd % HK;
            const float* ht = &r.h[(size_t) t * C];
            const double g = std::exp((double) gate[(size_t) t * HV + hd]), b = beta[(size_t) t * HV + hd];
            std::vector<double> oc(S);
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += sd[((size_t) i * HV + hd) * S + j] * ht[qk + qh * S + i];
                const double delta = (ht[2 * qk + hd * S + j] - g * kv) * b;
                double o = 0;
                for (int i = 0; i < S; ++i) {
                    double& v = sd[((size_t) i * HV + hd) * S + j];
                    v = g * v + ht[qk + qh * S + i] * delta;
                    o += v * ht[qh * S + i];
                }
                oc[j] = o / std::sqrt((double) S);
            }
            double ss = 0;
            for (int j = 0; j < S; ++j) ss += oc[j] * oc[j];
            const double scale = 1.0 / std::sqrt(ss / S + EPS);
            for (int j = 0; j < S; ++j) {
                const double zz = z[(size_t) t * ZV + hd * S + j];
                r.y[(size_t) t * ZV + hd * S + j] = (float) (oc[j] * scale * gamma[j] * (1.0 / (1.0 + std::exp(-zz))));
            }
        }
    return r;
}

// the state after n_keep tokens of recorded inputs (h rows, gate, beta) and the conv history after n_keep rows of qkv
std::vector<float> cpu_commit(const std::vector<float>& st0, const float* h, const float* qkv, const float* gate,
                              const float* beta, int n_keep) {
    std::vector<float> r = st0;
    std::vector<double> sd(st0.begin(), st0.begin() + (size_t) S * HV * S);
    const int qk = S * HK;
    for (int t = 0; t < n_keep; ++t)
        for (int hd = 0; hd < HV; ++hd) {
            const int qh = hd % HK;
            const float* ht = h + (size_t) t * C;
            const double g = std::exp((double) gate[(size_t) t * HV + hd]), b = beta[(size_t) t * HV + hd];
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += sd[((size_t) i * HV + hd) * S + j] * ht[qk + qh * S + i];
                const double delta = (ht[2 * qk + hd * S + j] - g * kv) * b;
                for (int i = 0; i < S; ++i) {
                    double& v = sd[((size_t) i * HV + hd) * S + j];
                    v = g * v + ht[qk + qh * S + i] * delta;
                }
            }
        }
    if (n_keep > 0) {
        for (size_t i = 0; i < sd.size(); ++i) r[i] = (float) sd[i];
        const float* hist = st0.data() + (size_t) S * HV * S;
        for (int c = 0; c < C; ++c)
            for (int j = 0; j < 3; ++j) {
                const int src = n_keep + j;
                r[(size_t) S * HV * S + c * 3 + j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
            }
    }
    return r;
}

double max_err(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) a[i] - b[i]) / (1.0 + std::fabs((double) b[i])));
    return m;
}

bool parity_seg(cudaStream_t s) {
    std::mt19937 rng(97);
    bool ok = true;
    const size_t SF = (size_t) S * HV * S + (size_t) C * 3;
    Layer L = make_layer(14, 23, 14, rng);
    const std::vector<float> cw = down(L.conv_w, (size_t) C * 4), gam = down(L.gamma, S);
    constexpr int NL = 2;   // layers per state: the table points at layer 0, the kernels at layer 1
    const int layer = 1;
    const int counts[4] = {1, 2, 5, 8};
    for (int trial = 0; trial < 3; ++trial)
        for (int nseg : counts) {
            std::vector<int> T(nseg), keep(nseg), row0(nseg);
            int R = 0, tmax = 1;
            for (int k = 0; k < nseg; ++k) {
                T[k] = trial == 0 ? 1 : 1 + (int) (rng() % kDecMaxT);
                if (trial == 2 && k % 2) T[k] = kDecMaxT;
                keep[k] = trial == 0 ? 1 : (int) (rng() % (T[k] + 1));
                row0[k] = R;
                R += T[k];
                tmax = std::max(tmax, T[k]);
            }
            // per-segment states (NL layers each), random
            std::vector<float*> sbase(nseg);
            std::vector<std::vector<float>> st0(nseg);
            for (int k = 0; k < nseg; ++k) {
                st0[k] = randn(NL * SF, 0.1f, rng);
                sbase[k] = upload(st0[k]);
            }
            const std::vector<float> qkv = randn((size_t) R * C, 1.f, rng), z = randn((size_t) R * ZV, 1.f, rng),
                                     x = randn((size_t) R * N, 1.f, rng);
            float *d_qkv = upload(qkv), *d_z = upload(z), *d_x = upload(x);
            std::vector<SegRow> rows(R);
            for (int k = 0; k < nseg; ++k)
                for (int j = 0; j < T[k]; ++j)
                    rows[row0[k] + j] = SegRow{sbase[k], row0[k], j, T[k], keep[k], k, 100 + j};
            SegRow* d_rows = upload(rows);
            auto fbuf = [&](size_t n) { float* p; CK(cudaMalloc(&p, n * 4 + 64)); CK(cudaMemset(p, 0xff, n * 4)); return p; };
            // the current kernels, one call per segment
            float *h_r = fbuf((size_t) R * C), *g_r = fbuf((size_t) R * HV), *b_r = fbuf((size_t) R * HV),
                  *y_r = fbuf((size_t) R * ZV);
            float *h_f1 = fbuf((size_t) R * C), *g_f1 = fbuf((size_t) R * HV), *b_f1 = fbuf((size_t) R * HV),
                  *y_f1 = fbuf((size_t) R * ZV);
            void *yq_r, *yq_f;
            CK(cudaMalloc(&yq_r, (size_t) R * (ZV / 32) * 36 + 64)); CK(cudaMemset(yq_r, 0xff, (size_t) R * (ZV / 32) * 36));
            CK(cudaMalloc(&yq_f, (size_t) R * (ZV / 32) * 36 + 64)); CK(cudaMemset(yq_f, 0xff, (size_t) R * (ZV / 32) * 36));
            // the poison fills run on the legacy stream, which the non-blocking test stream does not wait for: finish them
            // before any kernel can write these buffers
            CK(cudaDeviceSynchronize());
            for (int k = 0; k < nseg; ++k) {
                const size_t r0 = row0[k];
                float* st = sbase[k] + layer * SF;
                gdn_conv_l2_multi(st + (size_t) S * HV * S, d_qkv + r0 * C, L.conv_w, h_r + r0 * C, C, 2 * HK, EPS, T[k], s, 0);
                gdn_ab_multi(d_x + r0 * N, L.wa, L.wb, L.dt, L.ssm_a, g_r + r0 * HV, b_r + r0 * HV, N, HV, T[k], s);
                gdn_step_norm_multi(st, h_r + r0 * C, C, g_r + r0 * HV, b_r + r0 * HV, d_z + r0 * ZV, L.gamma, EPS,
                                    y_r + r0 * ZV, HK, HV, T[k], nullptr, s, 0);
                DecRecArgs a;
                a.state = st; a.qkv = d_qkv + r0 * C; a.conv_w = L.conv_w; a.x = d_x + r0 * N; a.w_alpha = L.wa;
                a.w_beta = L.wb; a.dt = L.dt; a.ssm_a = L.ssm_a; a.z = d_z + r0 * ZV; a.gamma = L.gamma;
                a.h = h_f1 + r0 * C; a.gate = g_f1 + r0 * HV; a.beta = b_f1 + r0 * HV; a.y = y_f1 + r0 * ZV;
                a.yq = (char*) yq_r + r0 * (ZV / 32) * 36; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS;
                dec_gdn_rec(a, T[k], s);
            }
            // the segment kernels
            float *h_s = fbuf((size_t) R * C), *y_s = fbuf((size_t) R * ZV);
            float *h_f = fbuf((size_t) R * C), *g_f = fbuf((size_t) R * HV), *b_f = fbuf((size_t) R * HV),
                  *y_f = fbuf((size_t) R * ZV);
            CK(cudaDeviceSynchronize());   // the fills above, before the segment kernels write these
            gdn_conv_l2_seg(d_rows, layer * SF + (size_t) S * HV * S, d_qkv, L.conv_w, h_s, C, 2 * HK, EPS, R, s);
            gdn_step_norm_seg(d_rows, layer * SF, h_r, C, g_r, b_r, d_z, L.gamma, EPS, y_s, HK, HV, R, s);
            {
                DecRecArgs a;
                a.qkv = d_qkv; a.conv_w = L.conv_w; a.x = d_x; a.w_alpha = L.wa; a.w_beta = L.wb; a.dt = L.dt;
                a.ssm_a = L.ssm_a; a.z = d_z; a.gamma = L.gamma; a.h = h_f; a.gate = g_f; a.beta = b_f; a.y = y_f;
                a.yq = yq_f; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS;
                dec_gdn_rec_seg(a, d_rows, layer * SF, R, tmax, s);
            }
            CK(cudaStreamSynchronize(s));
            long long d = 0;
            double ma = 0;
            d += cmp_f("conv seg", h_r, h_s, (size_t) R * C, ma);
            d += cmp_f("step_norm seg", y_r, y_s, (size_t) R * ZV, ma);
            d += cmp_f("rec seg h", h_f1, h_f, (size_t) R * C, ma);
            d += cmp_f("rec seg gate", g_f1, g_f, (size_t) R * HV, ma);
            d += cmp_f("rec seg beta", b_f1, b_f, (size_t) R * HV, ma);
            d += cmp_f("rec seg y", y_f1, y_f, (size_t) R * ZV, ma);
            d += cmp_b("rec seg yq", yq_r, yq_f, (size_t) R * (ZV / 32) * 36);
            // the CPU reference, per segment
            double ec = 0, ey = 0;
            for (int k = 0; k < nseg; ++k) {
                const size_t r0 = row0[k];
                const std::vector<float> sl(st0[k].begin() + layer * SF, st0[k].begin() + (layer + 1) * SF);
                const CpuSeg c = cpu_seg(sl, std::vector<float>(qkv.begin() + r0 * C, qkv.begin() + (r0 + T[k]) * C),
                                         std::vector<float>(z.begin() + r0 * ZV, z.begin() + (r0 + T[k]) * ZV), cw, gam,
                                         down(g_r + r0 * HV, (size_t) T[k] * HV), down(b_r + r0 * HV, (size_t) T[k] * HV),
                                         T[k]);
                const std::vector<float> hs = down(h_s + r0 * C, (size_t) T[k] * C), ys = down(y_s + r0 * ZV, (size_t) T[k] * ZV);
                ec = std::max(ec, max_err(hs.data(), c.h.data(), c.h.size()));
                ey = std::max(ey, max_err(ys.data(), c.y.data(), c.y.size()));
            }
            // the commit: the current kernel per segment on a copy of the states against the segment kernel
            const int CNL = 2;
            const std::vector<float> hL = randn((size_t) CNL * R * C, 0.1f, rng), qL = randn((size_t) CNL * R * C, 1.f, rng),
                                     gL = randn((size_t) CNL * R * HV, 0.3f, rng), bL = randn((size_t) CNL * R * HV, 0.3f, rng);
            float *d_hL = upload(hL), *d_qL = upload(qL), *d_gL = upload(gL), *d_bL = upload(bL);
            std::vector<float*> sref(nseg);
            for (int k = 0; k < nseg; ++k) sref[k] = upload(st0[k]);
            // fresh states for the new kernel
            for (int k = 0; k < nseg; ++k) CK(cudaMemcpy(sbase[k], st0[k].data(), NL * SF * 4, cudaMemcpyHostToDevice));
            for (int k = 0; k < nseg; ++k) {
                int32_t* nk;
                CK(cudaMalloc(&nk, 4));
                CK(cudaMemcpy(nk, &keep[k], 4, cudaMemcpyHostToDevice));
                CK(cudaDeviceSynchronize());   // the stream is non-blocking: a small pageable copy can return before it lands
                GdnCommitLayers a;
                a.state = sref[k]; a.h = d_hL + (size_t) row0[k] * C; a.qkv = d_qL + (size_t) row0[k] * C;
                a.gate = d_gL + (size_t) row0[k] * HV; a.beta = d_bL + (size_t) row0[k] * HV;
                a.state_stride = SF; a.h_stride = (size_t) R * C; a.gb_stride = (size_t) R * HV;
                a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = CNL; a.n_keep = nk;
                gdn_commit_layers(a, s, 32);
            }
            {
                GdnCommitSeg a;
                a.rows = d_rows; a.state_off = 0; a.state_stride = SF; a.h_stride = (size_t) R * C;
                a.gb_stride = (size_t) R * HV; a.h = d_hL; a.qkv = d_qL; a.gate = d_gL; a.beta = d_bL;
                a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = CNL; a.n_rows = R;
                gdn_commit_layers_seg(a, s, 32);
            }
            CK(cudaStreamSynchronize(s));
            double es = 0;
            for (int k = 0; k < nseg; ++k) {
                d += cmp_f("commit seg", sref[k], sbase[k], NL * SF, ma);
                for (int l = 0; l < CNL; ++l) {
                    const std::vector<float> sl(st0[k].begin() + l * SF, st0[k].begin() + (l + 1) * SF);
                    const size_t r0 = row0[k], lr = (size_t) l * R;   // layer l's rows of the recorded inputs
                    const std::vector<float> c = cpu_commit(sl, hL.data() + (lr + r0) * C, qL.data() + (lr + r0) * C,
                                                            gL.data() + (lr + r0) * HV, bL.data() + (lr + r0) * HV, keep[k]);
                    const std::vector<float> got = down(sbase[k] + l * SF, SF);
                    es = std::max(es, max_err(got.data(), c.data(), SF));
                }
            }
            std::printf("seg S=%d R=%2d T={", nseg, R);
            for (int k = 0; k < nseg; ++k) std::printf("%d%s", T[k], k + 1 < nseg ? "," : "");
            std::printf("} keep={");
            for (int k = 0; k < nseg; ++k) std::printf("%d%s", keep[k], k + 1 < nseg ? "," : "");
            std::printf("}: vs current kernels %s; CPU max rel err conv %.2g recurrence %.2g commit %.2g\n",
                        d == 0 ? "bitwise" : "DIFF", ec, ey, es);
            ok = ok && d == 0 && ec < 1e-4 && ey < 1e-3 && es < 1e-3;
            for (int k = 0; k < nseg; ++k) { CK(cudaFree(sbase[k])); CK(cudaFree(sref[k])); }
            for (void* p : {(void*) d_qkv, (void*) d_z, (void*) d_x, (void*) d_rows, (void*) h_r, (void*) g_r, (void*) b_r,
                            (void*) y_r, (void*) h_f1, (void*) g_f1, (void*) b_f1, (void*) y_f1, yq_r, yq_f, (void*) h_s,
                            (void*) y_s, (void*) h_f, (void*) g_f, (void*) b_f, (void*) y_f, (void*) d_hL, (void*) d_qL,
                            (void*) d_gL, (void*) d_bL})
                CK(cudaFree(p));
        }
    return ok;
}

// ple_hist_gather / ple_hist_scatter: shuffled slots, per-segment keep counts (0 leaves the slot's history).
bool parity_ple_hist(cudaStream_t s) {
    const int64_t n = 4 * 1000 + 37;
    std::mt19937 rng(7);
    bool ok = true;
    for (int trial = 0; trial < 6; ++trial) {
        const int S = 1 + (int) (rng() % 8);
        std::vector<int> perm(8);
        for (int i = 0; i < 8; ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<float> hs((size_t) 8 * n), hw((size_t) S * n);
        for (auto& v : hs) v = (float) (rng() % 100000) / 7.0f;
        for (auto& v : hw) v = (float) (rng() % 100000) / 11.0f;
        float *d_hist, *d_work;
        float** d_tab;
        SegRow* d_rows;
        CK(cudaMalloc(&d_hist, hs.size() * 4)); CK(cudaMalloc(&d_work, hw.size() * 4));
        CK(cudaMalloc(&d_tab, 8 * sizeof(float*))); CK(cudaMalloc(&d_rows, 8 * sizeof(SegRow)));
        CK(cudaMemcpy(d_hist, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_work, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float*> tab(8);
        for (int i = 0; i < 8; ++i) tab[(size_t) i] = d_hist + (size_t) i * n;
        CK(cudaMemcpy(d_tab, tab.data(), 8 * sizeof(float*), cudaMemcpyHostToDevice));
        std::vector<SegRow> rows((size_t) S);
        for (int t = 0; t < S; ++t) {
            SegRow e{};
            e.row0 = t; e.j = 0; e.n = 1; e.n_keep = (int) (rng() % 2); e.slot = perm[(size_t) t];
            rows[(size_t) t] = e;
        }
        CK(cudaMemcpy(d_rows, rows.data(), (size_t) S * sizeof(SegRow), cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());   // the stream is non-blocking: pageable copies are not ordered before its kernels
        ple_hist_gather(d_work, d_tab, d_rows, n, S, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> got(hw.size());
        CK(cudaMemcpy(got.data(), d_work, got.size() * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < S; ++t)
            for (int64_t i = 0; i < n; ++i)
                if (got[(size_t) t * n + i] != hs[(size_t) rows[(size_t) t].slot * n + i]) ok = false;
        CK(cudaMemcpy(d_work, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        ple_hist_scatter(d_tab, d_work, d_rows, n, S, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> after(hs.size());
        CK(cudaMemcpy(after.data(), d_hist, after.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> want = hs;
        for (int t = 0; t < S; ++t)
            if (rows[(size_t) t].n_keep > 0)
                for (int64_t i = 0; i < n; ++i) want[(size_t) rows[(size_t) t].slot * n + i] = hw[(size_t) t * n + i];
        if (after != want) ok = false;
        cudaFree(d_hist); cudaFree(d_work); cudaFree(d_tab); cudaFree(d_rows);
    }
    std::printf("ple_hist gather/scatter: %s\n", ok ? "ok" : "MISMATCH");
    return ok;
}

void bench(cudaStream_t s) {
    std::mt19937 rng(7);
    constexpr int NL = 12;   // distinct layers per graph: far more weight than the L2 holds
    const int types[NL][3] = {{23, 21, 14}, {14, 14, 14}, {12, 12, 13}, {14, 23, 14}, {13, 23, 12}, {14, 21, 14},
                              {23, 12, 14}, {14, 13, 21}, {12, 21, 23}, {14, 14, 14}, {23, 23, 13}, {14, 21, 14}};
    std::vector<Layer> Ls;
    size_t tot = 0, totw[3] = {0, 0, 0};
    for (auto& t : types) {
        Ls.push_back(make_layer(t[0], t[1], t[2], rng));
        tot += Ls.back().bytes;
        totw[0] += mat_bytes(t[0], N, C) + mat_bytes(t[1], N, ZV);
        totw[1] += (size_t) S * HV * S * 4 + (size_t) C * 16 + 2 * (size_t) HV * N * 2;
        totw[2] += mat_bytes(t[2], ZV, N);
    }
    std::printf("bench: %d layers, %.1f MB per layer on average (in %.1f, rec %.1f, out %.1f)\n", NL,
                tot / 1e6 / NL, totw[0] / 1e6 / NL, totw[1] / 1e6 / NL, totw[2] / 1e6 / NL);
    Bufs b = make_bufs();
    float* x = upload(randn((size_t) kDecMaxT * N, 1.f, rng));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const char* names[4] = {"in-proj", "conv+ab+rec", "out-proj", "whole"};
    const int masks[4] = {1, 2, 4, 7};
    for (int T : {1, 2, 3, 4, 6, 8}) {
        for (int k = 0; k < 4; ++k) {
            double us[4];
            for (int path = 0; path < 4; ++path) {
                g_gdnf = path;
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                {
                    PdlScope ps(true, true);
                    for (const Layer& L : Ls) {
                        if (path == 0) run_ref(L, x, b, T, s, masks[k]);
                        else run_gdnf(L, x, b, T, s, masks[k]);
                    }
                }
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
                const int R = 20;
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                us[path] = ms * 1000.0 / R / NL;
                CK(cudaGraphExecDestroy(ge));
                CK(cudaGraphDestroy(g));
            }
            const double bytes = (k == 3 ? (double) tot : (double) totw[k]) / NL;
            std::printf("T=%d %-12s ref %6.2f us (%5.0f GB/s)  gdnf1 %6.2f  gdnf2 %6.2f  gdnf3 %6.2f us (%5.0f GB/s)\n",
                        T, names[k], us[0], bytes / us[0] / 1e3, us[1], us[2], us[3], bytes / us[3] / 1e3);
        }
    }
}

// Segmented windows of one-row segments (the batch window) over 12 layers' states: the per-segment calls the batch window
// made (conv, gates, recurrence per row) and dec_gdn_rec per row, against the segment kernels, as graphs; us per layer.
void bench_seg(cudaStream_t s) {
    std::mt19937 rng(21);
    constexpr int NL = 12;
    Layer L = make_layer(14, 23, 14, rng);
    const size_t SF = (size_t) S * HV * S + (size_t) C * 3;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int nseg : {1, 2, 5, 8}) {
        const int R = nseg;
        float* st = upload(randn(NL * nseg * SF, 0.1f, rng));
        float *d_qkv = upload(randn((size_t) R * C, 1.f, rng)), *d_z = upload(randn((size_t) R * ZV, 1.f, rng)),
              *d_x = upload(randn((size_t) R * N, 1.f, rng));
        std::vector<SegRow> rows(R);
        for (int k = 0; k < R; ++k) rows[k] = SegRow{st + (size_t) k * NL * SF, k, 0, 1, 1, k, 100};
        SegRow* d_rows = upload(rows);
        float *h, *g, *b, *y;
        void* yq;
        CK(cudaMalloc(&h, (size_t) R * C * 4)); CK(cudaMalloc(&g, (size_t) R * HV * 4)); CK(cudaMalloc(&b, (size_t) R * HV * 4));
        CK(cudaMalloc(&y, (size_t) R * ZV * 4)); CK(cudaMalloc(&yq, (size_t) R * (ZV / 32) * 36));
        auto stages = [&](int path, int l) {
            if (path == 0) {   // per row, as the batch window ran
                for (int k = 0; k < R; ++k) {
                    float* sl = st + ((size_t) k * NL + l) * SF;
                    gdn_conv_l2_multi(sl + (size_t) S * HV * S, d_qkv + (size_t) k * C, L.conv_w, h + (size_t) k * C, C, 2 * HK,
                                      EPS, 1, s, 0);
                }
                gdn_ab_multi(d_x, L.wa, L.wb, L.dt, L.ssm_a, g, b, N, HV, R, s);
                for (int k = 0; k < R; ++k)
                    gdn_step_norm_multi(st + ((size_t) k * NL + l) * SF, h + (size_t) k * C, C, g + (size_t) k * HV,
                                        b + (size_t) k * HV, d_z + (size_t) k * ZV, L.gamma, EPS, y + (size_t) k * ZV, HK, HV, 1,
                                        nullptr, s, 0);
            } else if (path == 1) {   // the segment kernels
                gdn_conv_l2_seg(d_rows, (size_t) l * SF + (size_t) S * HV * S, d_qkv, L.conv_w, h, C, 2 * HK, EPS, R, s);
                gdn_ab_multi(d_x, L.wa, L.wb, L.dt, L.ssm_a, g, b, N, HV, R, s);
                gdn_step_norm_seg(d_rows, (size_t) l * SF, h, C, g, b, d_z, L.gamma, EPS, y, HK, HV, R, s);
            } else {   // dec_gdn_rec per row, or once over the segments
                DecRecArgs a;
                a.qkv = d_qkv; a.conv_w = L.conv_w; a.x = d_x; a.w_alpha = L.wa; a.w_beta = L.wb; a.dt = L.dt;
                a.ssm_a = L.ssm_a; a.z = d_z; a.gamma = L.gamma; a.h = h; a.gate = g; a.beta = b; a.yq = yq; a.C = C;
                a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS; a.early_trigger = false;
                if (path == 3) { dec_gdn_rec_seg(a, d_rows, (size_t) l * SF, R, 1, s); return; }
                for (int k = 0; k < R; ++k) {
                    DecRecArgs r = a;
                    r.state = st + ((size_t) k * NL + l) * SF; r.qkv += (size_t) k * C; r.x += (size_t) k * N;
                    r.z += (size_t) k * ZV; r.h += (size_t) k * C; r.gate += (size_t) k * HV; r.beta += (size_t) k * HV;
                    r.yq = (char*) yq + (size_t) k * (ZV / 32) * 36;
                    dec_gdn_rec(r, 1, s);
                }
            }
        };
        double us[4];
        for (int path = 0; path < 4; ++path) {
            // per-row calls read state slice k*NL+l, the segment kernels (state_off l * SF) the table's: point the table at layer 0
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < NL; ++l) stages(path, l);
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
            constexpr int REPS = 50;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            us[path] = ms * 1000.0 / REPS / NL;
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
        }
        std::printf("bench seg S=%d x 1 row (us per layer, graph of %d layers): conv+ab+rec per row %.1f | segment kernels %.1f"
                    " || dec_gdn_rec per row %.1f | dec_gdn_rec_seg %.1f\n",
                    nseg, NL, us[0], us[1], us[2], us[3]);
        CK(cudaFree(st));
    }
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));
    if (p.major < 9) { std::printf("dec_gdn_parity: needs thread-block clusters (compute capability %d.%d)\n", p.major, p.minor); return 77; }
    std::printf("%s: %d SMs, L2 %d MB, smem/block opt-in %zu KB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20,
                p.sharedMemPerBlockOptin >> 10);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::string mode = argc > 1 ? argv[1] : "";
    CK(cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&g_join, cudaEventDisableTiming));
    const bool ok = mode == "--bench-only" || (parity_gdnf(s) && parity_seg(s) && parity_ple_hist(s));
    if (mode != "--bench-only") std::printf("dec_gdn_parity: %s\n", ok ? "PASS (bitwise, columns independent of T)" : "FAIL");
    if (ok && (mode == "--bench" || mode == "--bench-only")) { bench_seg(s); bench(s); }
    return ok ? 0 : 1;
}
