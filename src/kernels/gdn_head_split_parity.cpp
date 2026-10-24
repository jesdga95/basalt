// src/kernels/gdn_head_split_parity.cpp - a GDN layer split by key-head group across two cards against the whole
// layer on one card and a CPU reference.
//
//     build/gdn_head_split_parity [--bench]
//
// Each card runs its slice (core/gdn_head_split.hpp) with the current verify kernels: native q8_1 + native_mmvq for
// the qkv and z rows of its heads, dec_gdn_rec with the sliced head counts, native_mmvq over its out-projection
// columns.  Rank 0 owns the first key heads, the peer the last ones, in units of two key heads (one out-projection
// block per value-head run).  The two partial outputs add in a fixed order, rank 0 first.  Every per-head output
// (qkv, z, h, gate, beta, the q8_1 of y) must equal the whole layer's bit for bit and the state stays untouched;
// the combined output is compared with the whole layer's (max abs / rel, bit-equal share) and both with a CPU
// reference in double over the same q8_1 activations (RMS).  The slicing round trip of the state is checked on the
// CPU.  Each rank's heads also run in place on card 0 through dec_gdn_rec's head map (the layer's state and weights,
// packed qkv/z rows in, h, gates and raw qkv at the full layout, y packed; one-launch and split recurrence): every
// output equals the whole layer's bit for bit.  Real geometry, random weights (random bytes, sane fp16 scales), random activations and state.  With one
// card both ranks run on it.
//
// --bench: per card, the whole layer and each share's slice as CUDA graphs over eight layers' distinct weights, in
// microseconds per layer.
#include "basalt/artifact/dequant.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
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
using basalt::core::GdnHeadSlice;
using basalt::core::GdnRun;

namespace {

constexpr int N = 2560, HK = 16, HV = 48, S = 128, C = 2 * HK * S + HV * S, ZV = HV * S, HIST = 3;
constexpr float EPS = 1e-6f;
constexpr size_t Q81 = 36;   // bytes per 32 q8_1 values

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
    void (*deq)(const uint8_t*, float*);
};
const Fmt FMTS[] = {
    {12, "Q4_K", 144, {0, 2}, basalt::dequantize_q4_K},
    {13, "Q5_K", 176, {0, 2}, basalt::dequantize_q5_K},
    {14, "Q6_K", 210, {208, -1}, basalt::dequantize_q6_K},
    {23, "IQ4_XS", 136, {0, -1}, basalt::dequantize_iq4_xs},
};
const Fmt& fmt(int t) {
    for (const Fmt& f : FMTS)
        if (f.type == t) return f;
    std::printf("no format %d\n", t);
    std::exit(1);
}
size_t row_bytes(int t, int n_in) { return (size_t) (n_in / 256) * fmt(t).block_bytes; }

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
std::vector<uint8_t> rand_mat(int t, int n_in, int n_out, std::mt19937& rng) {
    const Fmt& f = fmt(t);
    const size_t nb = (size_t) n_out * (n_in / 256);
    std::vector<uint8_t> w(nb * f.block_bytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (size_t k = 0; k < nb; ++k)
        for (int at : f.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * f.block_bytes + at], &h, 2);
            }
    return w;
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

// one GDN layer on the host, whole
struct HostLayer {
    int tq, tz, to;
    std::vector<uint8_t> wqkv, wz, wo;
    std::vector<float> conv_w, dt, ssm_a, gamma, state;
    std::vector<uint16_t> wa, wb;
};
HostLayer make_layer(int tq, int tz, int to, std::mt19937& rng) {
    HostLayer L;
    L.tq = tq; L.tz = tz; L.to = to;
    L.wqkv = rand_mat(tq, N, C, rng);
    L.wz = rand_mat(tz, N, ZV, rng);
    L.wo = rand_mat(to, ZV, N, rng);
    L.conv_w = randn((size_t) C * 4, 0.5f, rng);
    L.dt = randn(HV, 1.f, rng);
    L.ssm_a.resize(HV);
    for (auto& v : L.ssm_a) v = -std::exp(randn(1, 1.f, rng)[0]);
    L.gamma.resize(S);
    for (auto& v : L.gamma) v = 1.f + randn(1, 0.1f, rng)[0];
    L.state = randn((size_t) S * HV * S + (size_t) C * HIST, 0.1f, rng);
    std::normal_distribution<float> nd(0.f, 0.03f);
    L.wa.resize((size_t) HV * N);
    L.wb.resize((size_t) HV * N);
    for (auto& v : L.wa) v = bf16(nd(rng));
    for (auto& v : L.wb) v = bf16(nd(rng));
    return L;
}

// the layer as one card holds it: the whole layer, or a slice packed by gdn_head_split
HostLayer slice_layer(const HostLayer& L, const GdnHeadSlice& g) {
    using namespace basalt::core;
    HostLayer R;
    R.tq = L.tq; R.tz = L.tz; R.to = L.to;
    const std::vector<GdnRun> qr = gdn_qkv_runs(g), v1 = gdn_vhead_runs(g, 1), vs = gdn_vhead_runs(g, S);
    R.wqkv.resize((size_t) g.c() * row_bytes(L.tq, N));
    gdn_gather_rows(L.wqkv.data(), row_bytes(L.tq, N), qr, R.wqkv.data());
    R.wz.resize((size_t) g.zv() * row_bytes(L.tz, N));
    gdn_gather_rows(L.wz.data(), row_bytes(L.tz, N), vs, R.wz.data());
    R.wo.resize((size_t) N * row_bytes(L.to, (int) g.zv()));
    gdn_gather_cols(L.wo.data(), N, ZV, 256, fmt(L.to).block_bytes, vs, R.wo.data());
    R.conv_w.resize((size_t) g.c() * 4);
    gdn_gather_rows(L.conv_w.data(), 4 * sizeof(float), qr, R.conv_w.data());
    R.dt.resize(g.nv()); R.ssm_a.resize(g.nv());
    gdn_gather_rows(L.dt.data(), sizeof(float), v1, R.dt.data());
    gdn_gather_rows(L.ssm_a.data(), sizeof(float), v1, R.ssm_a.data());
    R.wa.resize((size_t) g.nv() * N); R.wb.resize((size_t) g.nv() * N);
    gdn_gather_rows(L.wa.data(), (size_t) N * 2, v1, R.wa.data());
    gdn_gather_rows(L.wb.data(), (size_t) N * 2, v1, R.wb.data());
    R.gamma = L.gamma;
    R.state.resize(gdn_slice_state_floats(g, HIST));
    gdn_slice_state(g, HIST, L.state.data(), R.state.data());
    return R;
}

template<typename V>
V* upload(const std::vector<V>& h) {
    V* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    return d;
}
template<typename V>
std::vector<V> down(const V* d, size_t n) {
    std::vector<V> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(V), cudaMemcpyDeviceToHost));
    return h;
}

// a layer (whole or sliced) on one card, with its window buffers
struct Card {
    int dev = 0, hk = 0, hv = 0, c = 0, zv = 0;
    int tq = 0, tz = 0, to = 0;
    void *wqkv = nullptr, *wz = nullptr, *wo = nullptr, *xq = nullptr, *yq = nullptr;
    float *conv_w, *dt, *ssm_a, *gamma, *state, *x, *qkv, *z, *h, *gate, *beta, *out;
    uint16_t *wa, *wb;
    size_t state_n = 0;
};
Card make_card(int dev, const HostLayer& L, int hk, const std::vector<float>& x) {
    CK(cudaSetDevice(dev));
    Card k;
    k.dev = dev; k.hk = hk; k.hv = hk * HV / HK; k.c = (2 * hk + k.hv) * S; k.zv = k.hv * S;
    k.tq = L.tq; k.tz = L.tz; k.to = L.to;
    k.wqkv = upload(L.wqkv); k.wz = upload(L.wz); k.wo = upload(L.wo);
    k.conv_w = upload(L.conv_w); k.dt = upload(L.dt); k.ssm_a = upload(L.ssm_a); k.gamma = upload(L.gamma);
    k.state = upload(L.state); k.state_n = L.state.size();
    k.wa = upload(L.wa); k.wb = upload(L.wb);
    k.x = upload(x);
    CK(cudaMalloc(&k.qkv, sizeof(float) * kDecMaxT * k.c));
    CK(cudaMalloc(&k.z, sizeof(float) * kDecMaxT * k.zv));
    CK(cudaMalloc(&k.h, sizeof(float) * kDecMaxT * k.c));
    CK(cudaMalloc(&k.gate, sizeof(float) * kDecMaxT * k.hv));
    CK(cudaMalloc(&k.beta, sizeof(float) * kDecMaxT * k.hv));
    CK(cudaMalloc(&k.out, sizeof(float) * kDecMaxT * N));
    CK(cudaMalloc(&k.xq, native_q8_1_bytes(N, kDecMaxT)));
    CK(cudaMalloc(&k.yq, native_q8_1_bytes(k.zv, kDecMaxT)));
    CK(cudaDeviceSynchronize());
    return k;
}
void free_card(Card& k) {
    CK(cudaSetDevice(k.dev));
    for (void* p : {k.wqkv, k.wz, k.wo, k.xq, k.yq, (void*) k.conv_w, (void*) k.dt, (void*) k.ssm_a, (void*) k.gamma,
                    (void*) k.state, (void*) k.x, (void*) k.qkv, (void*) k.z, (void*) k.h, (void*) k.gate, (void*) k.beta,
                    (void*) k.out, (void*) k.wa, (void*) k.wb})
        CK(cudaFree(p));
}

// the verify window's GDN layer on this card's heads: q8_1, qkv and z rows, dec_gdn_rec, the out projection (a
// partial sum when the card holds a slice)
void run_card(const Card& k, int T, cudaStream_t s) {
    native_quantize_q8_1(k.x, k.xq, N, T, s);
    native_mmvq(k.tq, k.wqkv, k.xq, k.qkv, N, k.c, T, s);
    native_mmvq(k.tz, k.wz, k.xq, k.z, N, k.zv, T, s);
    DecRecArgs a;
    a.state = k.state; a.qkv = k.qkv; a.conv_w = k.conv_w; a.x = k.x; a.w_alpha = k.wa; a.w_beta = k.wb;
    a.dt = k.dt; a.ssm_a = k.ssm_a; a.z = k.z; a.gamma = k.gamma; a.h = k.h; a.gate = k.gate; a.beta = k.beta;
    a.yq = k.yq; a.C = k.c; a.n_embd = N; a.h_k = k.hk; a.h_v = k.hv; a.eps = EPS; a.early_trigger = false;
    dec_gdn_rec(a, T, s);
    native_mmvq(k.to, k.wo, k.yq, k.out, k.zv, N, T, s);
}
void clear_card(const Card& k, cudaStream_t s) {
    CK(cudaMemsetAsync(k.qkv, 0xff, sizeof(float) * kDecMaxT * k.c, s));
    CK(cudaMemsetAsync(k.z, 0xff, sizeof(float) * kDecMaxT * k.zv, s));
    CK(cudaMemsetAsync(k.h, 0xff, sizeof(float) * kDecMaxT * k.c, s));
    CK(cudaMemsetAsync(k.gate, 0xff, sizeof(float) * kDecMaxT * k.hv, s));
    CK(cudaMemsetAsync(k.beta, 0xff, sizeof(float) * kDecMaxT * k.hv, s));
    CK(cudaMemsetAsync(k.out, 0xff, sizeof(float) * kDecMaxT * N, s));
    CK(cudaMemsetAsync(k.yq, 0xff, native_q8_1_bytes(k.zv, kDecMaxT), s));
}

// bytes that differ between this slice's rows of a whole-layer output and the slice's own output, T tokens
long long diff_slice(const void* whole, size_t whole_row, const void* sl, size_t sl_row, size_t unit,
                     const std::vector<GdnRun>& runs, int T) {
    std::vector<uint8_t> g(sl_row);
    long long d = 0;
    for (int t = 0; t < T; ++t) {
        basalt::core::gdn_gather_rows((const uint8_t*) whole + (size_t) t * whole_row, unit, runs, g.data());
        const uint8_t* b = (const uint8_t*) sl + (size_t) t * sl_row;
        for (size_t i = 0; i < sl_row; ++i) d += g[i] != b[i];
    }
    return d;
}

// one card's slice outputs against the whole layer's, bitwise; the state untouched
long long check_slice(const Card& whole, const Card& k, const GdnHeadSlice& g, int T, const std::vector<float>& st0) {
    using namespace basalt::core;
    CK(cudaSetDevice(whole.dev));
    const auto wq = down(whole.qkv, (size_t) T * C), wz = down(whole.z, (size_t) T * ZV), wh = down(whole.h, (size_t) T * C),
               wg = down(whole.gate, (size_t) T * HV), wb = down(whole.beta, (size_t) T * HV);
    const auto wy = down((const uint8_t*) whole.yq, (size_t) T * (ZV / 32) * Q81);
    CK(cudaSetDevice(k.dev));
    const auto q = down(k.qkv, (size_t) T * k.c), z = down(k.z, (size_t) T * k.zv), h = down(k.h, (size_t) T * k.c),
               ga = down(k.gate, (size_t) T * k.hv), be = down(k.beta, (size_t) T * k.hv);
    const auto y = down((const uint8_t*) k.yq, (size_t) T * (k.zv / 32) * Q81);
    const auto st = down(k.state, k.state_n);
    const std::vector<GdnRun> qr = gdn_qkv_runs(g), v1 = gdn_vhead_runs(g, 1), vs = gdn_vhead_runs(g, S),
                              vb = gdn_vhead_runs(g, S / 32);
    long long d = 0;
    d += diff_slice(wq.data(), C * 4, q.data(), (size_t) k.c * 4, 4, qr, T);
    d += diff_slice(wh.data(), C * 4, h.data(), (size_t) k.c * 4, 4, qr, T);
    d += diff_slice(wz.data(), ZV * 4, z.data(), (size_t) k.zv * 4, 4, vs, T);
    d += diff_slice(wg.data(), HV * 4, ga.data(), (size_t) k.hv * 4, 4, v1, T);
    d += diff_slice(wb.data(), HV * 4, be.data(), (size_t) k.hv * 4, 4, v1, T);
    d += diff_slice(wy.data(), (ZV / 32) * Q81, y.data(), (size_t) (k.zv / 32) * Q81, Q81, vb, T);
    if (std::memcmp(st.data(), st0.data(), st0.size() * 4) != 0) {
        std::printf("    state written on card %d\n", k.dev);
        ++d;
    }
    return d;
}

// rows `runs` of each of T tokens of a row-major output, packed
template<typename V>
std::vector<V> pack_rows(const std::vector<V>& full, size_t row, const std::vector<GdnRun>& runs, int T) {
    const size_t n = (size_t) basalt::core::gdn_runs_len(runs);
    std::vector<V> out((size_t) T * n);
    for (int t = 0; t < T; ++t)
        basalt::core::gdn_gather_rows(full.data() + (size_t) t * row, sizeof(V), runs, out.data() + (size_t) t * n);
    return out;
}

// the slice run in place on the whole layer's card through dec_gdn_rec's head map (rank 0's engine path): the slice's
// packed qkv/z rows in; the layer's state, conv and gate weights; h, the gates and the raw qkv written at the full
// layout, y packed.  The one-launch and the split recurrence; bytes that differ from the whole layer's outputs.
long long check_mapped(const Card& whole, const GdnHeadSlice& g, int T, const std::vector<float>& st0, cudaStream_t s) {
    using namespace basalt::core;
    CK(cudaSetDevice(whole.dev));
    CK(cudaStreamSynchronize(s));
    const std::vector<GdnRun> qr = gdn_qkv_runs(g), v1 = gdn_vhead_runs(g, 1), vs = gdn_vhead_runs(g, S),
                              vb = gdn_vhead_runs(g, S / 32);
    const auto wq = down(whole.qkv, (size_t) T * C), wz = down(whole.z, (size_t) T * ZV), wh = down(whole.h, (size_t) T * C),
               wg = down(whole.gate, (size_t) T * HV), wb = down(whole.beta, (size_t) T * HV);
    const auto wy = down((const uint8_t*) whole.yq, (size_t) T * (ZV / 32) * Q81);
    float* pq = upload(pack_rows(wq, C, qr, T));
    float* pz = upload(pack_rows(wz, ZV, vs, T));
    float *h, *ga, *be, *qo, *o;
    void *y1, *y2;
    const size_t zv = (size_t) g.zv(), yb = (size_t) T * (zv / 32) * Q81;
    CK(cudaMalloc(&h, sizeof(float) * T * C)); CK(cudaMalloc(&qo, sizeof(float) * T * C));
    CK(cudaMalloc(&ga, sizeof(float) * T * HV)); CK(cudaMalloc(&be, sizeof(float) * T * HV));
    CK(cudaMalloc(&o, sizeof(float) * T * zv)); CK(cudaMalloc(&y1, yb)); CK(cudaMalloc(&y2, yb));
    long long d = 0;
    for (int split = 0; split < 2; ++split) {
        CK(cudaMemsetAsync(h, 0xff, sizeof(float) * T * C, s)); CK(cudaMemsetAsync(qo, 0xff, sizeof(float) * T * C, s));
        CK(cudaMemsetAsync(ga, 0xff, sizeof(float) * T * HV, s)); CK(cudaMemsetAsync(be, 0xff, sizeof(float) * T * HV, s));
        void* y = split ? y2 : y1;
        CK(cudaMemsetAsync(y, 0xff, yb, s));
        DecRecArgs a;
        a.state = whole.state; a.qkv = pq; a.conv_w = whole.conv_w; a.x = whole.x; a.w_alpha = whole.wa;
        a.w_beta = whole.wb; a.dt = whole.dt; a.ssm_a = whole.ssm_a; a.z = pz; a.gamma = whole.gamma; a.h = h;
        a.gate = ga; a.beta = be; a.yq = y; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS;
        a.early_trigger = false; a.k0 = g.k0; a.nk = g.nk; a.qkv_out = qo;
        if (split) {
            a.o = o;
            dec_gdn_rec_split(a, T, s);
            dec_gdn_rec_norm(a, T, s);
        } else {
            dec_gdn_rec(a, T, s);
        }
        CK(cudaStreamSynchronize(s));
        const auto mh = pack_rows(down(h, (size_t) T * C), C, qr, T), mq = pack_rows(down(qo, (size_t) T * C), C, qr, T);
        const auto mg = pack_rows(down(ga, (size_t) T * HV), HV, v1, T), mb = pack_rows(down(be, (size_t) T * HV), HV, v1, T);
        const auto my = down((const uint8_t*) y, yb);
        const size_t c = (size_t) g.c(), nv = (size_t) g.nv();
        d += diff_slice(wh.data(), C * 4, mh.data(), c * 4, 4, qr, T);
        d += diff_slice(wq.data(), C * 4, mq.data(), c * 4, 4, qr, T);
        d += diff_slice(wg.data(), HV * 4, mg.data(), nv * 4, 4, v1, T);
        d += diff_slice(wb.data(), HV * 4, mb.data(), nv * 4, 4, v1, T);
        d += diff_slice(wy.data(), (ZV / 32) * Q81, my.data(), (zv / 32) * Q81, Q81, vb, T);
    }
    if (std::memcmp(down(whole.state, st0.size()).data(), st0.data(), st0.size() * 4) != 0) {
        std::printf("    state written in place\n");
        ++d;
    }
    for (void* p : {(void*) pq, (void*) pz, (void*) h, (void*) ga, (void*) be, (void*) qo, (void*) o, y1, y2})
        CK(cudaFree(p));
    return d;
}

// the whole layer's out projection in double from its own q8_1 activations
std::vector<double> cpu_out(const HostLayer& L, const std::vector<uint8_t>& yq, int T) {
    const Fmt& f = fmt(L.to);
    std::vector<double> y((size_t) T * ZV), r((size_t) T * N);
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < ZV / 32; ++b) {
            const uint8_t* p = &yq[((size_t) t * (ZV / 32) + b) * Q81];
            const double d = basalt::fp16_to_fp32(basalt::read_u16(p));
            for (int i = 0; i < 32; ++i) y[(size_t) t * ZV + b * 32 + i] = d * (int8_t) p[4 + i];
        }
    std::vector<float> w(ZV);
    for (int i = 0; i < N; ++i) {
        const uint8_t* row = &L.wo[(size_t) i * row_bytes(L.to, ZV)];
        for (int b = 0; b < ZV / 256; ++b) f.deq(row + (size_t) b * f.block_bytes, &w[(size_t) b * 256]);
        for (int t = 0; t < T; ++t) {
            double acc = 0;
            for (int j = 0; j < ZV; ++j) acc += (double) w[j] * y[(size_t) t * ZV + j];
            r[(size_t) t * N + i] = acc;
        }
    }
    return r;
}

// the state slicing round trip on the CPU: both ranks' slices put back give the whole state
bool check_state_roundtrip(const HostLayer& L, const GdnHeadSlice& g0, const GdnHeadSlice& g1) {
    std::vector<float> full(L.state.size(), 0.f), a(gdn_slice_state_floats(g0, HIST)), b(gdn_slice_state_floats(g1, HIST));
    basalt::core::gdn_slice_state(g0, HIST, L.state.data(), a.data());
    basalt::core::gdn_slice_state(g1, HIST, L.state.data(), b.data());
    basalt::core::gdn_unslice_state(g0, HIST, a.data(), full.data());
    basalt::core::gdn_unslice_state(g1, HIST, b.data(), full.data());
    return std::memcmp(full.data(), L.state.data(), full.size() * 4) == 0;
}

int g_dev1 = 0;

bool parity() {
    bool ok = true;
    std::mt19937 rng(139);
    const int combos[3][3] = {{14, 14, 14}, {12, 23, 13}, {13, 12, 23}};
    for (const auto& cb : combos) {
        const HostLayer L = make_layer(cb[0], cb[1], cb[2], rng);
        const std::vector<float> x = randn((size_t) kDecMaxT * N, 1.f, rng);
        Card whole = make_card(0, L, HK, x);
        cudaStream_t s0;
        CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        for (int peer8 : {2, 4, 6, 7}) {
            const int nk1 = 2 * peer8, nk0 = HK - nk1;
            const GdnHeadSlice g0{HK, HV, S, 0, nk0}, g1{HK, HV, S, nk0, nk1};
            if (!basalt::core::gdn_slice_aligned(g0, 256) || !basalt::core::gdn_slice_aligned(g1, 256)) {
                std::printf("share %d/8: out-projection runs off the block grid\n", peer8);
                ok = false;
                continue;
            }
            const bool rt = check_state_roundtrip(L, g0, g1);
            const HostLayer L0 = slice_layer(L, g0), L1 = slice_layer(L, g1);
            Card k0 = make_card(0, L0, nk0, x), k1 = make_card(g_dev1, L1, nk1, x);
            cudaStream_t s1;
            CK(cudaSetDevice(g_dev1));
            CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
            for (int T : {1, 2, 3, 5, 8}) {
                CK(cudaSetDevice(0));
                clear_card(whole, s0); clear_card(k0, s0);
                run_card(whole, T, s0);
                run_card(k0, T, s0);
                CK(cudaSetDevice(g_dev1));
                clear_card(k1, s1);
                run_card(k1, T, s1);
                CK(cudaStreamSynchronize(s1));
                CK(cudaSetDevice(0));
                CK(cudaStreamSynchronize(s0));
                const long long d = check_slice(whole, k0, g0, T, L0.state) + check_slice(whole, k1, g1, T, L1.state);
                // both ranks' heads in place on card 0 through the head map
                const long long dm = check_mapped(whole, g0, T, L.state, s0) + check_mapped(whole, g1, T, L.state, s0);
                CK(cudaSetDevice(0));
                const auto wo = down(whole.out, (size_t) T * N), p0 = down(k0.out, (size_t) T * N);
                const auto wyq = down((const uint8_t*) whole.yq, (size_t) T * (ZV / 32) * Q81);
                CK(cudaSetDevice(g_dev1));
                const auto p1 = down(k1.out, (size_t) T * N);
                const std::vector<double> ref = cpu_out(L, wyq, T);
                double max_abs = 0, max_rel = 0, se_split = 0, se_whole = 0, se_ref = 0, mc_s = 0, mc_w = 0;
                long long eq = 0;
                for (size_t i = 0; i < wo.size(); ++i) {
                    const float c = p0[i] + p1[i];   // rank 0 first
                    const double e = std::fabs((double) c - wo[i]);
                    max_abs = std::max(max_abs, e);
                    if (std::fabs(wo[i]) >= 1e-3) max_rel = std::max(max_rel, e / std::fabs(wo[i]));
                    eq += std::memcmp(&c, &wo[i], 4) == 0;
                    se_split += (c - ref[i]) * (c - ref[i]);
                    se_whole += (wo[i] - ref[i]) * (wo[i] - ref[i]);
                    se_ref += ref[i] * ref[i];
                    mc_s = std::max(mc_s, std::fabs(c - ref[i]) / (1.0 + std::fabs(ref[i])));
                    mc_w = std::max(mc_w, std::fabs(wo[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
                }
                const double rms_s = std::sqrt(se_split / wo.size()), rms_w = std::sqrt(se_whole / wo.size()),
                             rms_r = std::sqrt(se_ref / wo.size());
                // the split only reorders the out projection's sum: its error against the CPU stays the whole layer's
                const bool pass = d == 0 && dm == 0 && rt && rms_s <= 1.5 * rms_w + 1e-9 && mc_s <= 2.0 * mc_w + 1e-9;
                std::printf("%-6s %-6s %-6s share %d/8 T=%d: slices %s, in place %s, state round trip %s; out (RMS %.3g) vs whole max "
                            "abs %.3g max rel %.3g bit-equal %.0f%%; vs CPU RMS split %.3g whole %.3g, max rel split %.3g "
                            "whole %.3g%s\n",
                            fmt(cb[0]).name, fmt(cb[1]).name, fmt(cb[2]).name, peer8, T, d == 0 ? "bitwise" : "DIFF", dm == 0 ? "bitwise" : "DIFF",
                            rt ? "ok" : "FAIL", rms_r, max_abs, max_rel, 100.0 * eq / wo.size(), rms_s, rms_w, mc_s, mc_w,
                            pass ? "" : "  FAIL");
                ok = ok && pass;
            }
            CK(cudaStreamDestroy(s1));
            free_card(k0); free_card(k1);
        }
        CK(cudaSetDevice(0));
        CK(cudaStreamDestroy(s0));
        free_card(whole);
    }
    return ok;
}

// us per layer of a graph over eight layers' distinct weights on one card
double time_layers(const std::vector<Card>& ks, int T) {
    CK(cudaSetDevice(ks[0].dev));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (const Card& k : ks) run_card(k, T, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    constexpr int R = 30;
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms * 1000.0 / R / ks.size();
}

void bench(int ndev) {
    std::mt19937 rng(7);
    constexpr int NL = 8;
    std::vector<HostLayer> Ls;
    for (int i = 0; i < NL; ++i) Ls.push_back(make_layer(14, 14, 14, rng));
    const std::vector<float> x = randn((size_t) kDecMaxT * N, 1.f, rng);
    for (int dev = 0; dev < ndev; ++dev) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, dev));
        std::vector<Card> whole;
        for (const HostLayer& L : Ls) whole.push_back(make_card(dev, L, HK, x));
        for (int T : {1, 3, 8}) {
            std::printf("bench card %d (%s) Q6_K T=%d: whole %.1f us/layer", dev, p.name, T, time_layers(whole, T));
            for (int k8 : {1, 2, 4, 6, 7}) {
                std::vector<Card> ks;
                const GdnHeadSlice g{HK, HV, S, 0, 2 * k8};
                for (const HostLayer& L : Ls) ks.push_back(make_card(dev, slice_layer(L, g), g.nk, x));
                std::printf(" | %d/8 %.1f", k8, time_layers(ks, T));
                for (Card& k : ks) free_card(k);
            }
            std::printf("\n");
        }
        for (Card& k : whole) free_card(k);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    if (p.major < 9) {
        std::printf("gdn_head_split_parity: needs thread-block clusters (compute capability %d.%d)\n", p.major, p.minor);
        return 77;
    }
    g_dev1 = ndev > 1 ? 1 : 0;
    std::printf("gdn_head_split_parity: rank 0 on card 0, the peer on card %d (%d cards)\n", g_dev1, ndev);
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool ok = parity();
    std::printf("gdn_head_split_parity: %s\n", ok ? "PASS" : "FAIL");
    if (ok && mode == "--bench") bench(ndev);
    return ok ? 0 : 1;
}
