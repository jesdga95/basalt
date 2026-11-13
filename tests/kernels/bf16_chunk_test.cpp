// tests/kernels/bf16_chunk_test.cpp - the prompt chunk's BF16 entry points (the BF16 residual stream and activations)
// give, bitwise, what the FP32 kernels give on the same values widened, rounded once to BF16 where the output is BF16:
// fp8 rms_norm_quantize and the StoreBf16 / AddBf16 epilogues, fp4_dense rms_norm_quantize and product_add_bf16,
// gdn_gating_proj, gdn_conv, gdn_recurrence_bf16, attn_qk_prepare_bf16 and attn_gate_e4m3.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;
namespace fp8 = basalt::prefill::fp8;
namespace fd = basalt::prefill::fp4_dense;

namespace {
std::mt19937 rng(5);
template <class T> T* dev(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> T* dev_n(size_t n) {
    T* d = nullptr;
    cudaMalloc(&d, n * sizeof(T));
    cudaMemset(d, 0, n * sizeof(T));
    return d;
}
std::vector<uint8_t> host(const void* d, size_t n) {
    cudaDeviceSynchronize();
    std::vector<uint8_t> h(n);
    cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
    return h;
}
uint16_t to_bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float from_bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// n random BF16 values (sd) and the same values as FP32
void pair(size_t n, float sd, std::vector<uint16_t>& b, std::vector<float>& f, float mean = 0.0f) {
    std::normal_distribution<float> nd(mean, sd);
    b.resize(n);
    f.resize(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = to_bf(nd(rng));
        f[i] = from_bf(b[i]);
    }
}
// FP32 rows rounded to BF16 on the host (the reference for a BF16 output)
std::vector<uint8_t> round_rows(const std::vector<uint8_t>& f32) {
    std::vector<uint8_t> out(f32.size() / 2);
    for (size_t i = 0; i < f32.size() / 4; ++i) {
        float v;
        std::memcpy(&v, &f32[4 * i], 4);
        const uint16_t b = to_bf(v);
        std::memcpy(&out[2 * i], &b, 2);
    }
    return out;
}
bool report(const char* what, bool same) {
    std::printf("%s: %s\n", what, same ? "bitwise" : "DIFFER");
    return same;
}
}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    bool ok = true;
    const int64_t T = 300, H = 5120;
    std::vector<uint16_t> xb;
    std::vector<float> xf;
    pair((size_t) T * H, 1.0f, xb, xf);
    uint16_t* dxb = dev(xb);
    float* dxf = dev(xf);
    std::vector<uint16_t> nwb;
    std::vector<float> nw;
    pair(H, 0.1f, nwb, nw, 1.0f);
    float* dnw = dev(nw);

    {   // FP8: norm + quantize, then the segmented product's Store and Add epilogues
        void *a1 = dev_n<uint8_t>(fp8::act_bytes(T, H)), *a2 = dev_n<uint8_t>(fp8::act_bytes(T, H));
        float* xn1 = dev_n<float>((size_t) T * H);
        uint16_t* xn2 = dev_n<uint16_t>((size_t) T * H);
        fp8::rms_norm_quantize(dxf, dnw, 1e-6f, T, H, a1, xn1, nullptr);
        fp8::rms_norm_quantize(dxb, dnw, 1e-6f, T, H, a2, xn2, nullptr);
        ok &= report("fp8 rms_norm_quantize of BF16 rows (codes, scales)",
                     host(a1, fp8::act_bytes(T, H)) == host(a2, fp8::act_bytes(T, H)));
        ok &= report("fp8 rms_norm_quantize BF16 normed rows", round_rows(host(xn1, (size_t) T * H * 4)) == host(xn2, (size_t) T * H * 2));
        const int64_t R = 512;
        std::vector<uint8_t> w((size_t) R * H);
        for (auto& c : w) { c = (uint8_t) rng(); if ((c & 0x7f) == 0x7f) c = 0x31; }
        uint8_t* dw = dev(w);
        const float sc = 0.02f;
        float* dsc = dev(std::vector<float>{sc});
        float* y1 = dev_n<float>((size_t) T * R);
        uint16_t* y2 = dev_n<uint16_t>((size_t) T * R);
        fp8::gemm(dw, R, H, dsc, a1, T, y1, R, fp8::Epilogue::Store, nullptr);
        fp8::gemm(dw, R, H, dsc, a1, T, y2, R, fp8::Epilogue::StoreBf16, nullptr);
        ok &= report("fp8 StoreBf16", round_rows(host(y1, (size_t) T * R * 4)) == host(y2, (size_t) T * R * 2));
        std::vector<uint16_t> rb;
        std::vector<float> rf;
        pair((size_t) T * R, 1.0f, rb, rf);
        float* r1 = dev(rf);
        uint16_t* r2 = dev(rb);
        fp8::gemm(dw, R, H, dsc, a1, T, r1, R, fp8::Epilogue::Add, nullptr);
        fp8::gemm(dw, R, H, dsc, a1, T, r2, R, fp8::Epilogue::AddBf16, nullptr);
        ok &= report("fp8 AddBf16", round_rows(host(r1, (size_t) T * R * 4)) == host(r2, (size_t) T * R * 2));
    }
    {   // FP4: norm + quantize, then the down product's Add into a BF16 residual
        void *a1 = dev_n<uint8_t>(fd::act_bytes(T, H)), *a2 = dev_n<uint8_t>(fd::act_bytes(T, H));
        float *g1 = dev_n<float>(T), *g2 = dev_n<float>(T);
        fd::rms_norm_quantize(dxf, dnw, 1e-6f, T, H, a1, g1, nullptr, nullptr);
        fd::rms_norm_quantize(dxb, dnw, 1e-6f, T, H, a2, g2, nullptr);
        ok &= report("fp4 rms_norm_quantize of BF16 rows", host(a1, fd::act_bytes(T, H)) == host(a2, fd::act_bytes(T, H)) &&
                                                                host(g1, T * 4) == host(g2, T * 4));
        const int64_t R = 512;
        std::vector<uint8_t> w((size_t) R * H / 64 * 36);
        for (size_t i = 0; i < (size_t) R * H / 2; ++i) w[i] = (uint8_t) rng();
        for (size_t i = (size_t) R * H / 2; i < w.size(); ++i) w[i] = (uint8_t) (0x28 + rng() % 16);   // UE4M3 ~0.06 .. 0.12
        uint8_t* dw = dev(w);
        float* dsc = dev(std::vector<float>{0.5f});
        std::vector<uint16_t> rb;
        std::vector<float> rf;
        pair((size_t) T * R, 1.0f, rb, rf);
        float* r1 = dev(rf);
        uint16_t* r2 = dev(rb);
        fd::product(dw, dsc, R, H, a1, g1, T, r1, R, true, nullptr);
        fd::product_add_bf16(dw, dsc, R, H, a1, g1, T, r2, R, nullptr);
        ok &= report("fp4 product_add_bf16", round_rows(host(r1, (size_t) T * R * 4)) == host(r2, (size_t) T * R * 2));
    }
    {   // GDN: the gating projection, the conv and the recurrence on BF16 rows
        const int64_t HV = 48, S = 128, C = 2 * 16 * S + HV * S, VD = HV * S;
        if (!k::gdn_prompt_supported((int) S, 16, (int) HV, 4)) return 77;
        std::vector<uint16_t> wab;
        std::vector<float> wabf;
        pair((size_t) 2 * HV * H, 0.02f, wab, wabf);
        uint16_t* dwab = dev(wab);
        std::vector<float> dt(HV, 0.1f), sa(HV, -0.5f);
        float *ddt = dev(dt), *dsa = dev(sa);
        float *ga1 = dev_n<float>((size_t) T * HV), *be1 = dev_n<float>((size_t) T * HV);
        float *ga2 = dev_n<float>((size_t) T * HV), *be2 = dev_n<float>((size_t) T * HV);
        k::gdn_gating_proj(dxf, dwab, ddt, dsa, ga1, be1, T, HV, H, nullptr);
        k::gdn_gating_proj(dxb, dwab, ddt, dsa, ga2, be2, T, HV, H, nullptr);
        ok &= report("gdn_gating_proj of BF16 rows", host(ga1, T * HV * 4) == host(ga2, T * HV * 4) && host(be1, T * HV * 4) == host(be2, T * HV * 4));
        std::vector<uint16_t> qb;
        std::vector<float> qf;
        pair((size_t) T * C, 0.5f, qb, qf);
        std::vector<float> cw((size_t) C * 4), hist((size_t) C * 3);
        std::normal_distribution<float> nd(0.0f, 0.3f);
        for (auto& v : cw) v = nd(rng);
        for (auto& v : hist) v = nd(rng);
        float *h1 = dev(hist), *h2 = dev(hist), *dcw = dev(cw);
        float* hc1 = dev_n<float>((size_t) T * C);
        uint16_t* hc2 = dev_n<uint16_t>((size_t) T * C);
        k::gdn_conv(h1, dev(qf), dcw, hc1, T, nullptr);
        k::gdn_conv(h2, dev(qb), dcw, hc2, T, nullptr);
        const auto hc1h = host(hc1, (size_t) T * C * 4);
        ok &= report("gdn_conv of BF16 rows", round_rows(hc1h) == host(hc2, (size_t) T * C * 2) && host(h1, C * 12) == host(h2, C * 12));
        // the recurrence: FP32 inputs = the BF16 conv output and z widened
        std::vector<float> hcw((size_t) T * C);
        {
            const auto hb = host(hc2, (size_t) T * C * 2);
            for (size_t i = 0; i < hcw.size(); ++i) { uint16_t b; std::memcpy(&b, &hb[2 * i], 2); hcw[i] = from_bf(b); }
        }
        std::vector<uint16_t> zb;
        std::vector<float> zf;
        pair((size_t) T * VD, 1.0f, zb, zf);
        std::vector<float> gate((size_t) T * HV), beta((size_t) T * HV), st0((size_t) S * HV * S);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t i = 0; i < gate.size(); ++i) { gate[i] = -0.5f * u(rng); beta[i] = u(rng); }
        std::normal_distribution<float> ns(0.0f, 0.05f);
        for (auto& v : st0) v = ns(rng);
        float *s1 = dev(st0), *s2 = dev(st0), *dg = dev(gate), *db = dev(beta), *gm = dev(std::vector<float>(S, 1.0f));
        float* y = dev_n<float>((size_t) T * VD);
        void* work = dev_n<uint8_t>(k::gdn_work_bytes(T));
        void *e1 = dev_n<uint8_t>(fp8::act_bytes(T, VD)), *e2 = dev_n<uint8_t>(fp8::act_bytes(T, VD));
        k::gdn_recurrence(s1, dev(hcw), dg, db, dev(zf), gm, 1e-6f, work, y, nullptr, T, nullptr, k::GdnGate::Silu, nullptr,
                          (uint8_t*) e1, fp8::act_scales(e1, T, VD));
        k::gdn_recurrence_bf16(s2, hc2, dg, db, dev(zb), gm, 1e-6f, work, y, T, nullptr, k::GdnGate::Silu, (uint8_t*) e2,
                               fp8::act_scales(e2, T, VD));
        ok &= report("gdn_recurrence_bf16 (E4M3 out, state)", host(e1, fp8::act_bytes(T, VD)) == host(e2, fp8::act_bytes(T, VD)) &&
                                                                  host(s1, st0.size() * 4) == host(s2, st0.size() * 4));
    }
    {   // attention: q/k norm + rope into the int8 cache from BF16 q_full / k / v, and the gate to E4M3
        const int NH = 24, NKV = 4, HD = 256;
        std::vector<uint16_t> qb, kb, vb;
        std::vector<float> qf, kf, vf;
        pair((size_t) T * NH * 2 * HD, 1.0f, qb, qf);
        pair((size_t) T * NKV * HD, 1.0f, kb, kf);
        pair((size_t) T * NKV * HD, 1.0f, vb, vf);
        std::vector<float> qn(HD, 1.0f), kn(HD, 1.0f);
        float *dqn = dev(qn), *dkn = dev(kn);
        k::RopeSpec rope;
        rope.rot = 64;
        rope.theta = 1e7f;
        std::vector<int32_t> pos((size_t) T * 3);
        for (int64_t t = 0; t < T; ++t) pos[t * 3] = pos[t * 3 + 1] = pos[t * 3 + 2] = (int32_t) (100 + t);
        int32_t* dpos = dev(pos);
        const int64_t cells = 512;
        k::KvInt8 c1, c2;
        for (k::KvInt8* c : {&c1, &c2}) {
            c->k = dev_n<int8_t>((size_t) cells * NKV * HD);
            c->v = dev_n<int8_t>((size_t) cells * NKV * HD);
            c->ks = dev_n<uint16_t>((size_t) cells * NKV * HD / 64);
            c->vs = dev_n<uint16_t>((size_t) cells * NKV * HD / 64);
        }
        float* q1 = dev_n<float>((size_t) T * NH * HD);
        uint16_t* q2 = dev_n<uint16_t>((size_t) T * NH * HD);
        k::attn_qk_prepare(dev(qf), dev(kf), dev(vf), dqn, dkn, 1e-6f, rope, dpos, T, NH, NKV, q1, nullptr, nullptr, 100, nullptr,
                           256, true, 0, nullptr, c1);
        k::attn_qk_prepare_bf16(dev(qb), dev(kb), dev(vb), dqn, dkn, 1e-6f, rope, dpos, T, NH, NKV, q2, 100, nullptr, c2, nullptr,
                                nullptr);
        const size_t cb = (size_t) cells * NKV * HD;
        ok &= report("attn_qk_prepare_bf16 (q rounded once, int8 K/V and scales)",
                     round_rows(host(q1, (size_t) T * NH * HD * 4)) == host(q2, (size_t) T * NH * HD * 2) && host(c1.k, cb) == host(c2.k, cb) &&
                         host(c1.v, cb) == host(c2.v, cb) && host(c1.ks, cb / 32) == host(c2.ks, cb / 32) &&
                         host(c1.vs, cb / 32) == host(c2.vs, cb / 32));
        // the prompt attention over the cache just written, from the BF16 queries widened and as BF16
        {
            const auto qh = host(q2, (size_t) T * NH * HD * 2);
            std::vector<float> qw((size_t) T * NH * HD);
            for (size_t i = 0; i < qw.size(); ++i) { uint16_t b; std::memcpy(&b, &qh[2 * i], 2); qw[i] = from_bf(b); }
            const size_t sf = k::gqa_prompt_split_floats(T, NH, 3);
            float* scr = dev_n<float>(sf);
            float* o1 = dev_n<float>((size_t) T * NH * HD);
            uint16_t* o2 = dev_n<uint16_t>((size_t) T * NH * HD);
            k::gqa_prompt_attn(dev(qw), nullptr, nullptr, NH, NKV, 100, T, o1, nullptr, c2, scr, sf);
            k::gqa_prompt_attn_bf16(q2, NH, NKV, 100, T, o2, nullptr, c2, scr, sf);
            ok &= report("gqa_prompt_attn_bf16", round_rows(host(o1, (size_t) T * NH * HD * 4)) == host(o2, (size_t) T * NH * HD * 2));
        }
        std::vector<uint16_t> ab;
        std::vector<float> at;
        pair((size_t) T * NH * HD, 1.0f, ab, at);
        float* dat = dev(at);
        void *e1 = dev_n<uint8_t>(fp8::act_bytes(T, NH * HD)), *e2 = dev_n<uint8_t>(fp8::act_bytes(T, NH * HD));
        void* e3 = dev_n<uint8_t>(fp8::act_bytes(T, NH * HD));
        k::attn_gate_e4m3(dat, dev(qf), T, NH, (uint8_t*) e1, fp8::act_scales(e1, T, NH * HD), nullptr);
        k::attn_gate_e4m3(dat, dev(qb), T, NH, (uint8_t*) e2, fp8::act_scales(e2, T, NH * HD), nullptr);
        k::attn_gate_e4m3(dev(ab), dev(qb), T, NH, (uint8_t*) e3, fp8::act_scales(e3, T, NH * HD), nullptr);
        ok &= report("attn_gate_e4m3 of BF16 q_full and BF16 rows", host(e1, fp8::act_bytes(T, NH * HD)) == host(e2, fp8::act_bytes(T, NH * HD)) &&
                                                                     host(e1, fp8::act_bytes(T, NH * HD)) == host(e3, fp8::act_bytes(T, NH * HD)));
    }
    std::printf("%s\n", ok ? "bf16 chunk entry points ok" : "FAIL");
    return ok ? 0 : 1;
}
