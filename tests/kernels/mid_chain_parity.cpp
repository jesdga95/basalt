// tests/kernels/mid_chain_parity.cpp - mid_chain against the kernels it replaces (dense_tc_pair, dec_gdn_rec, dense_tc), bit
// for bit on every output, between the first hc read and the second one with the router (so the whole chain, under
// programmatic dependent launch, is compared), with real geometry, random inputs and every projection-type triple of the
// packs, the recurrence's q8_1 output aliasing the read's rows as the engine has it and apart; a token's outputs must not
// depend on the window width (a T-row launch against its T'-row prefix); the projections against a double-precision CPU
// reference (Q6_K and Q8_0); then the time per layer with the weights hot in L2 and cold (distinct weights over more layers than
// L2 holds), next to the three kernels, in a graph with PDL.  The replaced kernels are checked against CPU references by
// dense_tc_parity and dec_gdn_parity.
//
//     build/mid_chain_parity [iters]
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/mid_chain.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = basalt::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, KT = 10, S = 128, HV = 48, HK = 16;
constexpr int C = 2 * HK * S + HV * S, ZV = HV * S, NQ = 10240, NZ = 6144, NO_IN = 6144, TM = K::kMidChainMaxT;
constexpr size_t XQ_BYTES = (size_t) TM * (ZV / 32) * 36;   // holds the read's rows and the recurrence's output rows

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
uint16_t sane_half(std::mt19937& r) { const uint32_t v = r(); return (uint16_t) (((v >> 31) << 15) | ((5u + (v >> 10) % 5u) << 10) | (v & 0x3ffu)); }
float half_to_float(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m + 1024), e - 25);
    return (h >> 15) ? -v : v;
}
template <typename V> V* upload(const std::vector<V>& h) {
    V* d;
    CK(cudaMalloc(&d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    return d;
}
std::vector<float> randn(size_t n, float sd, std::mt19937& r) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(r);
    return v;
}
int sb_bytes(int type) { return type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 21 ? 110 : type == 8 ? 272 : 136; }
// random super-blocks of a type with finite scales (d, and dmin for Q4_K / Q5_K; Q6_K's d is its last 2 bytes)
std::vector<uint8_t> rand_type_host(int type, size_t nb, std::mt19937& r) {
    const int sb = sb_bytes(type);
    std::vector<uint8_t> w(nb * sb);
    for (auto& v : w) v = (uint8_t) (r() & 0xff);
    for (size_t k = 0; k < nb; ++k) {
        const uint16_t h = sane_half(r), h2 = sane_half(r);
        if (type == 8) { for (int j = 0; j < 8; ++j) { const uint16_t hj = sane_half(r); std::memcpy(&w[k * sb + 34 * j], &hj, 2); } }   // eight Q8_0 blocks
        else if (type == 14) std::memcpy(&w[k * sb + 208], &h, 2);
        else { std::memcpy(&w[k * sb], &h, 2); if (type == 12 || type == 13) std::memcpy(&w[k * sb + 2], &h2, 2); }
    }
    return w;
}
// one Q8_0 super-block (eight blocks of a half scale and 32 values)'s 256 values
void deq_q80(const uint8_t* b, float* y) {
    for (int j = 0; j < 8; ++j) {
        uint16_t dh;
        std::memcpy(&dh, b + 34 * j, 2);
        for (int i = 0; i < 32; ++i) y[32 * j + i] = half_to_float(dh) * (int8_t) b[34 * j + 2 + i];
    }
}
// one Q6_K super-block's 256 values
void deq_q6k(const uint8_t* b, float* y) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t* sc = (const int8_t*) (b + 192);
    uint16_t dh;
    std::memcpy(&dh, b + 208, 2);
    const float d = half_to_float(dh);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l] = d * sc[is] * q1; y[n + l + 32] = d * sc[is + 2] * q2; y[n + l + 64] = d * sc[is + 4] * q3; y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

struct Hc {   // the two reads' weights (shared by every layer)
    uint16_t *wd[2], *wu[2], *wi[2], *wr;
    float* wn[2];
};
struct Layer {
    uint8_t *wq, *wz, *wo;
    K::DecRecArgs rec;
    int tq, tz, to;
    std::vector<uint8_t> hq, hz, ho;   // host copies (CPU reference)
};
struct Bufs {
    float *R0, *R, *boM, *injB0, *injA, *injB, *mixed1, *mixed2, *lo, *rs, *xn, *qkv, *z, *hh, *gate, *beta, *yrec, *bo2, *logits, *wts;
    uint8_t *xq1, *xq2, *yq;
    int32_t* ids;
};
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    K::hc_fused_init();
    K::mid_chain_init();
    if (!K::hc_fused_supported()) { std::printf("mid_chain_parity: no fused hc read on this device\n"); return 77; }
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    Bufs b{};
    auto zf = [&](size_t n) { return upload(std::vector<float>(n)); };
    b.R0 = upload(randn((size_t) TM * D, 1.f, rng)); b.R = zf((size_t) TM * D);
    b.boM = upload(randn((size_t) TM * N, 0.5f, rng)); b.injB0 = upload(randn((size_t) TM * HC, 1.f, rng));
    b.injA = zf(TM * HC); b.injB = zf(TM * HC);
    b.mixed1 = zf((size_t) TM * N); b.mixed2 = zf((size_t) TM * N); b.lo = zf((size_t) TM * LR); b.rs = zf(TM * HC); b.xn = zf((size_t) TM * D);
    b.qkv = zf((size_t) TM * C); b.z = zf((size_t) TM * ZV); b.hh = zf((size_t) TM * C); b.gate = zf((size_t) TM * HV); b.beta = zf((size_t) TM * HV);
    b.yrec = zf((size_t) TM * ZV); b.bo2 = zf((size_t) TM * N); b.logits = zf((size_t) TM * NE); b.wts = zf((size_t) TM * KT);
    b.xq1 = upload(std::vector<uint8_t>(XQ_BYTES)); b.xq2 = upload(std::vector<uint8_t>((size_t) TM * (N / 32) * 36));
    b.yq = upload(std::vector<uint8_t>(XQ_BYTES));
    CK(cudaMalloc(&b.ids, (size_t) TM * KT * 4));

    Hc hc{};
    for (int o = 0; o < 2; ++o) {
        std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
        std::vector<float> wn(D);
        for (auto& x : wd) x = bf16(0.02f * nd(rng));
        for (auto& x : wu) x = bf16(0.05f * nd(rng));
        for (auto& x : wi) x = bf16(0.02f * nd(rng));
        for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
        hc.wd[o] = upload(wd); hc.wu[o] = upload(wu); hc.wi[o] = upload(wi); hc.wn[o] = upload(wn);
        K::hc8_prepare(hc.wd[o], LR, D); K::hc8_prepare(hc.wu[o], D, LR); K::hc8_prepare(hc.wi[o], HC, D);
    }
    {
        std::vector<uint16_t> wr((size_t) NE * N);
        for (auto& x : wr) x = bf16(0.05f * nd(rng));
        hc.wr = upload(wr);
    }
    // a layer: projections of the given types and the recurrence's constants; `keep`: host copies for the CPU reference
    auto make_layer = [&](int tq, int tz, int to, bool keep) {
        Layer w{};
        w.tq = tq; w.tz = tz; w.to = to;
        auto hq = rand_type_host(tq, (size_t) NQ * 10, rng), hz = rand_type_host(tz, (size_t) NZ * 10, rng), ho = rand_type_host(to, (size_t) 2560 * (NO_IN / 256), rng);
        w.wq = upload(hq); w.wz = upload(hz); w.wo = upload(ho);
        if (keep) { w.hq = std::move(hq); w.hz = std::move(hz); w.ho = std::move(ho); }
        K::DecRecArgs& a = w.rec;
        a.state = upload(randn((size_t) S * HV * S + (size_t) C * 3, 0.1f, rng));
        a.conv_w = upload(randn((size_t) C * 4, 0.5f, rng));
        std::vector<float> dt = randn(HV, 1.f, rng), sa(HV), gm(S);
        for (auto& v : sa) v = -std::exp(randn(1, 1.f, rng)[0]);
        for (auto& v : gm) v = 1.f + randn(1, 0.1f, rng)[0];
        a.dt = upload(dt); a.ssm_a = upload(sa); a.gamma = upload(gm);
        std::vector<uint16_t> wa((size_t) HV * N), wbt((size_t) HV * N);
        std::normal_distribution<float> nd3(0.f, 0.03f);
        for (auto& v : wa) v = bf16(nd3(rng));
        for (auto& v : wbt) v = bf16(nd3(rng));
        a.w_alpha = upload(wa); a.w_beta = upload(wbt);
        a.qkv = b.qkv; a.x = b.mixed1; a.z = b.z; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = 1e-6f;
        a.h = b.hh; a.gate = b.gate; a.beta = b.beta; a.y = b.yrec; a.yq = b.yq;
        return w;
    };
    auto hcargs = [&](int ord, int T) {
        K::HcFusedArgs a;
        a.T = T; a.R = b.R; a.R_out = b.R; a.apply = true; a.w_norm = hc.wn[ord]; a.w_down = hc.wd[ord]; a.w_up = hc.wu[ord]; a.w_inject = hc.wi[ord];
        a.lo = b.lo; a.rs = b.rs; a.xn = b.xn;
        if (ord == 0) { a.bo_prev = b.boM; a.inj_prev = b.injB; a.inject_out = b.injA; a.mixed = b.mixed1; a.xq = b.xq1; a.next_early = true; }
        else {
            a.bo_prev = b.bo2; a.inj_prev = b.injA; a.inject_out = b.injB; a.mixed = b.mixed2; a.xq = b.xq2;
            a.w_router = hc.wr; a.logits = b.logits; a.ids = b.ids; a.weights = b.wts;
        }
        return a;
    };
    int grid = 0;   // the mid chain's grid (0: the device's)
    auto mid_args = [&](const Layer& w, int T, bool alias) {
        K::MidChainArgs a;
        a.ctas = grid;
        a.T = T; a.xq = b.xq1; a.wq = w.wq; a.wz = w.wz; a.wo = w.wo; a.type_q = w.tq; a.type_z = w.tz; a.type_o = w.to;
        a.rec = w.rec; a.rec.yq = alias ? b.xq1 : b.yq; a.bo = b.bo2;
        return a;
    };
    auto reset = [&] {
        CK(cudaMemcpy(b.R, b.R0, (size_t) TM * D * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(b.injB, b.injB0, (size_t) TM * HC * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(b.bo2, b.boM, (size_t) TM * N * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemset(b.mixed1, 0xff, (size_t) TM * N * 4));
        CK(cudaMemset(b.xq1, 0, XQ_BYTES));
        CK(cudaMemset(b.yq, 0, XQ_BYTES));
    };
    int mid_ord = 0;
    // the chain around the middle, in a PDL scope: read 0, the middle (the three kernels, or mid_chain), read 1 + router
    auto run = [&](const Layer& w, int T, bool alias, bool mid, cudaStream_t s) {
        K::PdlScope ps(true, false);
        K::DecRecArgs ra = w.rec;
        ra.yq = alias ? (void*) b.xq1 : (void*) b.yq;
        K::hc_fused_read(hcargs(0, T), s);
        if (mid) {
            K::MidChainArgs a = mid_args(w, T, alias);
            a.ord = mid_ord++;
            K::PdlScope pm(true, a.ord == 0);
            K::mid_chain(a, s);
        } else {
            K::dense_tc_pair(w.tq, w.wq, b.qkv, NQ, w.tz, w.wz, b.z, NZ, b.xq1, N, T, s);
            K::dec_gdn_rec(ra, T, s);
            K::dense_tc(w.to, w.wo, ra.yq, b.bo2, NO_IN, 2560, T, s);
        }
        K::hc_fused_read(hcargs(1, T), s);
    };
    struct Out { const char* name; const void* p; size_t per_row; bool first_read; };   // per token row
    const std::vector<Out> outs = {{"R", b.R, (size_t) D * 4, false}, {"mixed1", b.mixed1, (size_t) N * 4, true}, {"qkv", b.qkv, (size_t) C * 4, false},
                                   {"z", b.z, (size_t) ZV * 4, false}, {"h", b.hh, (size_t) C * 4, false}, {"gate", b.gate, (size_t) HV * 4, false}, {"beta", b.beta, (size_t) HV * 4, false},
                                   {"y", b.yrec, (size_t) ZV * 4, false}, {"bo2", b.bo2, (size_t) N * 4, false}, {"mixed2", b.mixed2, (size_t) N * 4, false},
                                   {"xq2", b.xq2, (size_t) N / 32 * 36, false}, {"logits", b.logits, (size_t) NE * 4, false}, {"ids", b.ids, (size_t) KT * 4, false}, {"weights", b.wts, (size_t) KT * 4, false},
                                   {"injA", b.injA, (size_t) HC * 4, false}, {"injB", b.injB, (size_t) HC * 4, false}, {"lo2", b.lo, (size_t) LR * 4, false}};
    // the q8_1 rows of the recurrence's output (apart, or in the read's buffer), then the read's rows themselves when apart
    auto snap = [&](int T, bool alias) {
        std::vector<std::vector<uint8_t>> v;
        for (const Out& o : outs) { std::vector<uint8_t> h(o.per_row * T); CK(cudaMemcpy(h.data(), o.p, h.size(), cudaMemcpyDeviceToHost)); v.push_back(h); }
        std::vector<uint8_t> yq((size_t) T * (ZV / 32) * 36), xq((size_t) T * (N / 32) * 36);
        CK(cudaMemcpy(yq.data(), alias ? b.xq1 : b.yq, yq.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(xq.data(), b.xq1, xq.size(), cudaMemcpyDeviceToHost));
        v.push_back(yq);
        v.push_back(alias ? std::vector<uint8_t>() : xq);
        return v;
    };
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const bool big = sms >= 160;   // the output projection has its one-tile kernel
    int bad = 0;
    auto ref_names_n = [](size_t n) { return n + 1; };

    // the pair ring alternates its tiles between two groups: its slot count is even for every type triple
    for (int q : {8, 12, 13, 14, 21, 23}) for (int z : {8, 12, 13, 14, 21, 23}) for (int o : {8, 12, 13, 14, 21, 23}) {
        const int ns = K::mid_chain_ring_slots(q, z, o);
        if (ns < 2 || (ns & 1)) { std::printf("types %d/%d/%d: ring of %d slots\n", q, z, o, ns); ++bad; }
    }

    // the type triples (qkv, z, out) of the GDN layers of the packs, plus the one the kernel was first built for
    const std::vector<std::array<int, 3>> triples = {{14, 14, 14}, {23, 21, 12}, {14, 12, 14}, {13, 14, 14}, {12, 12, 14}, {14, 21, 23},
                                                     {23, 23, 14}, {13, 13, 13}, {12, 21, 21}, {14, 23, 12}, {23, 14, 13}, {13, 23, 14},
                                                     {13, 12, 14}, {13, 13, 14}, {13, 14, 13}, {14, 12, 12}, {14, 13, 12}, {14, 13, 14}, {14, 14, 13}, {14, 23, 13}, {14, 23, 14}, {23, 14, 14},
                                                     {21, 21, 21}, {8, 8, 8}};
    std::vector<Layer> layers;
    for (const auto& t : triples) layers.push_back(make_layer(t[0], t[1], t[2], (t[0] == 14 && t[1] == 14 && t[2] == 14) || (t[0] == 8 && t[1] == 8 && t[2] == 8)));
    // the device's grid with every triple; then smaller grids (the fewer-SM devices' split: tiles round-robin, several rec
    // heads and out tiles per CTA) with a few triples
    for (const int gr : {0, 120, 84, 70, 48, 36, 20, 16}) for (size_t l = 0; l < layers.size(); ++l) {
        const Layer& w = layers[l];
        grid = gr;
        if (gr != 0 && l != 0 && l != 1 && l + 1 != layers.size()) continue;
        if (gr != 0 && !K::mid_chain_supported(mid_args(w, 1, true))) continue;
        if (!K::mid_chain_supported(mid_args(w, 1, true))) { std::printf("types %d/%d/%d: not supported by the mid chain\n", w.tq, w.tz, w.to); ++bad; continue; }
        for (int alias = 1; alias >= 0; --alias) {
            std::vector<std::vector<std::vector<uint8_t>>> got(TM + 1);
            size_t tot = 0;
            std::vector<size_t> perk(outs.size() + 2, 0);
            // the reference: today's kernels on the device's grid; a smaller grid, or a device whose output projection is not
            // the one-tile kernel's, takes the mid chain on the device's grid (a tile's bits do not depend on its CTA).  A
            // device without the one-tile output projection has another reduction order there: its comparison with today's
            // kernels covers what comes before the output projection, the CPU reference below the projection itself.
            const bool unf = gr == 0 && (big || w.to != 21);
            std::vector<bool> use(outs.size() + 2, true);
            if (unf && !big)
                for (size_t k = 0; k < outs.size(); ++k) {
                    const std::string nm = outs[k].name;
                    use[k] = nm == "mixed1" || nm == "qkv" || nm == "z" || nm == "h" || nm == "gate" || nm == "beta" || nm == "y";
                }
            for (int T = 1; T <= TM; ++T) {
                reset();
                if (unf) run(w, T, alias, false, s);
                else { grid = 0; run(w, T, alias, true, s); grid = gr; }
                CK(cudaStreamSynchronize(s));
                const auto ref = snap(T, alias);
                for (int it = 0; it < iters; ++it) {
                    reset(); run(w, T, alias, true, s); CK(cudaStreamSynchronize(s));
                    const auto g = snap(T, alias);
                    for (size_t k = 0; k < ref.size(); ++k) { if (!use[k]) continue; for (size_t j = 0; j < ref[k].size(); ++j) { const size_t d = ref[k][j] != g[k][j]; tot += d; perk[k] += d; } }
                    if (it == 0) got[T] = g;
                }
            }
            // a row's outputs do not depend on the window width: the T = 4 launch against each shorter launch's rows
            size_t prefix = 0;
            for (int T = 1; T < TM; ++T)
                for (size_t k = 0; k < outs.size(); ++k) {
                    if (outs[k].first_read) continue;   // overwritten by the second read's mixed? (mixed1 is the first read's: compared as is)
                    for (size_t j = 0; j < outs[k].per_row * T; ++j) prefix += got[T][k][j] != got[TM][k][j];
                }
            for (int T = 1; T < TM; ++T)   // the recurrence's q8_1 rows (token-major)
                for (size_t j = 0; j < (size_t) T * (ZV / 32) * 36; ++j) prefix += got[T][outs.size()][j] != got[TM][outs.size()][j];
            std::printf("grid %3d types %2d/%2d/%2d %s: T1-T4 x%d vs %s %zu differing bytes, T-prefix %zu\n", gr, w.tq, w.tz, w.to, alias ? "yq in xq" : "yq apart", iters, unf ? "today's kernels" : "the device's grid", tot, prefix);
            if (tot) { std::printf("  differing bytes by output:"); for (size_t k = 0; k < ref_names_n(outs.size()); ++k) std::printf(" %s %zu", k < outs.size() ? outs[k].name : "yq", perk[k]); std::printf("\n"); }
            bad += (int) (tot + prefix);
        }
    }

    grid = 0;
    // the projections against a double-precision CPU reference (Q6_K, then the Q8_0 triple), from the q8_1 rows the kernels read
    for (const size_t li : {(size_t) 0, layers.size() - 1}) {
        const Layer& w = layers[li];
        const bool q8 = w.tq == 8;
        const int sbb = q8 ? 272 : 210;
        const int T = 3;
        reset(); run(w, T, true, true, s); CK(cudaStreamSynchronize(s));
        std::vector<uint8_t> xq((size_t) T * (N / 32) * 36);
        CK(cudaMemcpy(xq.data(), b.xq1, 0, cudaMemcpyDeviceToHost));   // (rows are overwritten by the recurrence's output: rebuild below)
        reset(); K::hc_fused_read(hcargs(0, T), s); CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(xq.data(), b.xq1, xq.size(), cudaMemcpyDeviceToHost));
        auto deqx = [&](const std::vector<uint8_t>& q, int nin, int t, std::vector<double>& x) {
            x.assign(nin, 0.0);
            for (int k = 0; k < nin / 32; ++k) {
                const uint8_t* blk = q.data() + ((size_t) t * (nin / 32) + k) * 36;
                uint16_t dh; std::memcpy(&dh, blk, 2);
                for (int i = 0; i < 32; ++i) x[k * 32 + i] = (double) half_to_float(dh) * (int8_t) blk[4 + i];
            }
        };
        auto proj = [&](const std::vector<uint8_t>& wt, int rows, int nin, const std::vector<uint8_t>& q, int t, std::vector<double>& y) {
            std::vector<double> x; deqx(q, nin, t, x);
            y.assign(rows, 0.0);
            std::vector<float> row(nin);
            for (int r = 0; r < rows; ++r) {
                for (int k = 0; k < nin / 256; ++k) { const uint8_t* sbp = &wt[((size_t) r * (nin / 256) + k) * sbb]; if (q8) deq_q80(sbp, &row[k * 256]); else deq_q6k(sbp, &row[k * 256]); }
                double a = 0; for (int i = 0; i < nin; ++i) a += (double) row[i] * x[i];
                y[r] = a;
            }
        };
        reset(); run(w, T, false, true, s); CK(cudaStreamSynchronize(s));   // yq apart: the CPU out projection reads it
        std::vector<float> qkv((size_t) T * C), z((size_t) T * ZV), bo((size_t) T * N);
        std::vector<uint8_t> yq((size_t) T * (ZV / 32) * 36);
        CK(cudaMemcpy(qkv.data(), b.qkv, qkv.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(z.data(), b.z, z.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bo.data(), b.bo2, bo.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yq.data(), b.yq, yq.size(), cudaMemcpyDeviceToHost));
        double e_q = 0, e_z = 0, e_o = 0, n_q = 0, n_z = 0, n_o = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<double> y;
            proj(w.hq, NQ, N, xq, t, y);
            for (int r = 0; r < NQ; ++r) { e_q = std::max(e_q, std::fabs(y[r] - qkv[(size_t) t * C + r])); n_q = std::max(n_q, std::fabs(y[r])); }
            proj(w.hz, NZ, N, xq, t, y);
            for (int r = 0; r < NZ; ++r) { e_z = std::max(e_z, std::fabs(y[r] - z[(size_t) t * ZV + r])); n_z = std::max(n_z, std::fabs(y[r])); }
            proj(w.ho, 2560, NO_IN, yq, t, y);
            for (int r = 0; r < 2560; ++r) { e_o = std::max(e_o, std::fabs(y[r] - bo[(size_t) t * N + r])); n_o = std::max(n_o, std::fabs(y[r])); }
        }
        std::printf("CPU double reference (%s, T=3, vs the q8_1 rows the kernel reads): max abs error qkv %.3g (max |y| %.3g), z %.3g (%.3g), out %.3g (%.3g)\n", q8 ? "Q8_0" : "Q6_K", e_q, n_q, e_z, n_z, e_o, n_o);
        if (e_q > 1e-3 * n_q || e_z > 1e-3 * n_z || e_o > 1e-3 * n_o) { std::printf("CPU reference: too far\n"); ++bad; }
    }

    // time per layer: hot (one layer repeated) and cold (distinct weights over more layers than L2 holds), in a graph of
    // `nl` layers with PDL; the middle is the three kernels, mid_chain, nothing (the two reads alone), the pair alone of
    // each (the risk gate's pair phase)
    for (const int tt : {14, 8}) {
        const int T = 3;
        const int NCOLD = 48;
        std::vector<Layer> cold;
        for (int i = 0; i < NCOLD; ++i) cold.push_back(make_layer(tt, tt, tt, false));
        const Layer& hot = tt == 8 ? layers.back() : layers[0];
        std::printf("%s triple:\n", tt == 8 ? "Q8_0" : "Q6_K");
        const double bytes = (double) NQ * 10 * sb_bytes(tt) + (double) NZ * 10 * sb_bytes(tt) + 2560.0 * (NO_IN / 256) * sb_bytes(tt);
        const double state_bytes = (double) S * HV * S * 4 + (double) C * 3 * 4;
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        enum Mode { kNone, kOld, kMid, kOldPair, kMidPair };
        const char* names[] = {"the two reads alone", "today's 3 kernels", "mid_chain", "today's pair alone", "mid_chain's pair alone"};
        auto layer_seq = [&](const Layer& w, Mode m, int idx, cudaStream_t st) {
            K::PdlScope ps(true, false);
            K::hc_fused_read(hcargs(0, T), st);
            if (m == kOld) {
                K::DecRecArgs ra = w.rec; ra.yq = b.xq1;
                K::dense_tc_pair(w.tq, w.wq, b.qkv, NQ, w.tz, w.wz, b.z, NZ, b.xq1, N, T, st);
                K::dec_gdn_rec(ra, T, st);
                K::dense_tc(w.to, w.wo, b.xq1, b.bo2, NO_IN, 2560, T, st);
            } else if (m == kOldPair) {
                K::dense_tc_pair(w.tq, w.wq, b.qkv, NQ, w.tz, w.wz, b.z, NZ, b.xq1, N, T, st);
            } else if (m == kMid || m == kMidPair) {
                K::MidChainArgs a = mid_args(w, T, true);
                a.ord = idx; a.pair_only = m == kMidPair;
                K::PdlScope pm(true, idx == 0);
                K::mid_chain(a, st);
            }
            K::hc_fused_read(hcargs(1, T), st);
        };
        for (int cold_run = 0; cold_run < 2; ++cold_run) {
            const std::vector<Layer>& set = cold_run ? cold : std::vector<Layer>(NCOLD, hot);
            double us[5] = {};
            for (int m = 0; m < 5; ++m) {
                reset();
                cudaGraph_t g; cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
                for (int i = 0; i < NCOLD; ++i) layer_seq(set[i], (Mode) m, i, s);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                const int reps = 20;
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                us[m] = 1e3 * ms / reps / NCOLD;
                CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
            }
            std::printf("T=%d %s (graph of %d layers, PDL, per layer incl. the two reads):\n", T, cold_run ? "cold" : "hot ", NCOLD);
            for (int m = 0; m < 5; ++m)
                std::printf("  %-24s %7.2f us   increment over the reads %7.2f us%s\n", names[m], us[m], us[m] - us[kNone],
                            m == 0 ? "" : (m == kOld || m == kMid) ? "" : "");
            const double inc_old = us[kOld] - us[kNone], inc_mid = us[kMid] - us[kNone];
            std::printf("  3 kernels %.2f us -> %.1f GB/s of %.1f MB (+%.1f MB state); mid_chain %.2f us -> %.1f GB/s; mid / 3 kernels %.1f%%; pair phase mid / today %.1f%%\n", inc_old,
                        bytes / inc_old / 1e3, bytes / 1e6, state_bytes / 1e6, inc_mid, bytes / inc_mid / 1e3, 100.0 * inc_mid / inc_old,
                        100.0 * (us[kMidPair] - us[kNone]) / (us[kOldPair] - us[kNone]));
        }
    }
    // the phase ledger of mid_chain (cold weights, a graph of distinct layers with PDL, the stamping instances): times in us
    // from the first CTA's pass of the grid-dependency wait, medians over the layers; first / median / last over the CTAs
    if (sms >= K::kMidStampCtas) {   // the ledger's roles are those of the full-size grid
        using K::kMidStampCtas;
        using K::kMidStampPts;
        const int NL = 48;
        std::vector<Layer> cold;
        for (int i = 0; i < NL; ++i) cold.push_back(make_layer(14, 14, 14, false));
        unsigned long long* sd;
        const size_t cnt = (size_t) NL * kMidStampCtas * kMidStampPts;
        CK(cudaMalloc(&sd, cnt * 8));
        // the stamp points (the kernel's enum)
        const int E_ENTRY = 0, E_WAIT = 1, E_TILE0 = 2, E_PAIR_REL = 9, E_PZ_REL = 11, E_PAIR_END = 13, E_REQ0 = 14, E_REQ1 = 15, E_OFULL = 16, E_RPAIR = 17,
                  E_RSTATE = 18, E_RTOK = 19, E_RPZ = 20, E_RREC = 21, E_OREC = 22, E_ODONE = 23;
        std::printf("phase ledger (cold, 48 layers, PDL; us after the first CTA's wait; first/median/last over CTAs, median over layers):\n");
        std::printf("  T  pair-end f/m/l        C_PAIR  C_PZ   C_REC  | rec: pair->state state->tok tok->PZ(slack) | last-out CTA: req0 req1 C_REC-seen ofull  done | k_mid end\n");
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        for (int Tl = 1; Tl <= TM; ++Tl) {
            reset();
            CK(cudaMemset(sd, 0, cnt * 8));
            cudaGraph_t g; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < NL; ++i) {
                K::PdlScope ps(true, false);
                K::hc_fused_read(hcargs(0, Tl), s);
                K::MidChainArgs a = mid_args(cold[i], Tl, true);
                a.ord = i; a.stamps = sd;
                K::PdlScope pm(true, i == 0);
                K::mid_chain(a, s);
                K::hc_fused_read(hcargs(1, Tl), s);
            }
            CK(cudaStreamEndCapture(s, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            std::vector<unsigned long long> st(cnt);
            CK(cudaMemcpy(st.data(), sd, cnt * 8, cudaMemcpyDeviceToHost));
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
            std::vector<double> pe_f, pe_m, pe_l, cpair, cpz, crec, r1, r2, r3, qreq0, qreq1, qrec, qfull, qdone, kend;
            for (int l = 0; l < NL; ++l) {
                auto at = [&](int cta, int pt) { return (double) st[((size_t) l * kMidStampCtas + cta) * kMidStampPts + pt]; };
                double t0 = 1e30;
                for (int c = 0; c < 170; ++c) t0 = std::min(t0, at(c, E_WAIT));
                std::vector<double> pe;
                double lp = 0, lz = 0, lr = 0, lastdone = -1; int lc = 0;
                for (int c = 0; c < 170; ++c) {
                    pe.push_back((at(c, E_PAIR_END) - t0) / 1e3);
                    for (int gi = 0; gi < 2; ++gi) { lp = std::max(lp, at(c, E_PAIR_REL + gi)); lz = std::max(lz, at(c, E_PZ_REL + gi)); }
                    if (c >= 122) lr = std::max(lr, at(c, E_RREC));
                    if (c < 160 && at(c, E_ODONE) > lastdone) { lastdone = at(c, E_ODONE); lc = c; }
                }
                std::sort(pe.begin(), pe.end());
                pe_f.push_back(pe.front()); pe_m.push_back(pe[pe.size() / 2]); pe_l.push_back(pe.back());
                cpair.push_back((lp - t0) / 1e3); cpz.push_back((lz - t0) / 1e3); crec.push_back((lr - t0) / 1e3);
                std::vector<double> a1, a2, a3;
                for (int c = 122; c < 170; ++c) { a1.push_back((at(c, E_RSTATE) - at(c, E_RPAIR)) / 1e3); a2.push_back((at(c, E_RTOK) - at(c, E_RSTATE)) / 1e3); a3.push_back((at(c, E_RPZ) - at(c, E_RTOK)) / 1e3); }
                r1.push_back(med(a1)); r2.push_back(med(a2)); r3.push_back(med(a3));
                qreq0.push_back((at(lc, E_REQ0) - t0) / 1e3); qreq1.push_back((at(lc, E_REQ1) - t0) / 1e3);
                qrec.push_back((at(lc, E_OREC) - t0) / 1e3); qfull.push_back((at(lc, E_OFULL) - t0) / 1e3); qdone.push_back((at(lc, E_ODONE) - t0) / 1e3);
                kend.push_back((lastdone - t0) / 1e3);
            }
            std::printf("  %d  %5.1f/%5.1f/%5.1f   %5.1f  %5.1f  %5.1f  |      %5.1f      %5.1f     %5.1f          |        %5.1f %5.1f   %5.1f      %5.1f %5.1f | %5.1f\n", Tl,
                        med(pe_f), med(pe_m), med(pe_l), med(cpair), med(cpz), med(crec), med(r1), med(r2), med(r3), med(qreq0), med(qreq1), med(qrec), med(qfull), med(qdone), med(kend));
        }
        CK(cudaFree(sd));
    }
    std::printf(bad == 0 ? "mid_chain_parity: bitwise\n" : "mid_chain_parity: %d differing\n", bad);
    return bad != 0;
}
