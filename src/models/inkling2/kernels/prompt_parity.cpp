// src/models/inkling2/kernels/prompt_parity.cpp - the Inkling 2 prompt path's kernels at the model's geometry, each against
// a plain CPU reference (double accumulation) and against the decode kernel it stands in for: the prompt attention (window
// ring with a staged chunk, full cache; fp16, E4M3 and INT8), the router's scores as one product with the shared selection, the
// experts' sort by expert, the softcapped SwiGLU of the MMQ path, and the routed experts as MMQ products (NVFP4 and Q8_0)
// against native_expert_grouped, and the NVFP4 experts on the FP4 tensor cores (moe_fp4, activations as two FP4 terms)
// against a CPU reference that rounds the activations the same way. Rows are checked for not depending on the chunk's width or their place in it, and timed.
// Exit 77 without a GPU.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/prompt_kernels.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/q8_gemm.hpp"

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
#include <numeric>
#include <random>
#include <vector>

using namespace basalt::models::inkling2;
namespace bk = basalt::kernels;
namespace mmq = basalt::prefill::mmq;
namespace fp4 = basalt::prefill::fp4;

namespace {

constexpr int E = kWidth;
std::mt19937 rng(29);
int failures = 0;
cudaStream_t S = nullptr;   // every launch's stream (a blocking one: the copies back wait for it)

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
// max |a - b| over max |b|, a and b over the same entries
void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-44s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void report(const char* what, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    report(what, got, std::vector<double>(ref.begin(), ref.end()), tol);
}
void same_bits(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    std::printf("%-44s %s\n", what, ok ? "bitwise" : "FAIL");
    failures += !ok;
}
double time_us(const std::function<void()>& f, int n = 20) {
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
uint8_t f2e4(float f) {
    const __nv_fp8_e4m3 q(f);
    return *reinterpret_cast<const uint8_t*>(&q);
}
double e42d(uint8_t c) {
    __nv_fp8_e4m3 q;
    *reinterpret_cast<uint8_t*>(&q) = c;
    return (double) (float) q;
}
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---- attention -----------------------------------------------------------------------------------------------------------

// The cache formats: fp16, E4M3 codes (fp8), INT8 codes with an fp16 scale per 64 channels (int8).
enum Kv { kF16, kE4M3, kI8 };
size_t kv_bytes(Kv kv) { return (size_t) kv_cache_row_bytes(kv == kE4M3, kv == kI8); }
const char* kv_name(Kv kv) { return kv == kE4M3 ? "e4m3" : kv == kI8 ? "int8" : "fp16"; }
// One head row of kHeadDim values `x` written in format kv to `row` (kv_bytes(kv) bytes); `got` takes the values the row
// holds. INT8: each 64 channels' scale the fp16 rounding of their max |x| / 127, codes x / scale rounded to nearest even.
void put_row(Kv kv, const float* x, uint8_t* row, double* got) {
    if (kv == kI8) {
        for (int grp = 0; grp < 2; ++grp) {
            float m = 0.f;
            for (int d = grp * 64; d < grp * 64 + 64; ++d) m = std::max(m, std::fabs(x[d]));
            const uint16_t sh = f2h(m / 127.f);
            const float sc = (float) h2d(sh);
            std::memcpy(row + kHeadDim + 2 * grp, &sh, 2);
            for (int d = grp * 64; d < grp * 64 + 64; ++d) {
                const int q = sc > 0.f ? std::clamp((int) std::nearbyint(x[d] / sc), -127, 127) : 0;
                row[d] = (uint8_t) (int8_t) q;
                got[d] = (double) q * sc;
            }
        }
    } else {
        for (int d = 0; d < kHeadDim; ++d) {
            if (kv == kE4M3) {
                row[d] = f2e4(x[d]);
                got[d] = e42d(row[d]);
            } else {
                const uint16_t h = f2h(x[d]);
                std::memcpy(row + 2 * d, &h, 2);
                got[d] = h2d(h);
            }
        }
    }
}

// A cache of `cap` cells ([cell][kKvHeads] rows) holding positions [lo, hi) in cell p % cap: a row per (head, position),
// the same at every place a position is stored, in format kv.
struct KeyBank {
    Kv kv = kF16;
    std::vector<uint8_t> k, v;   // [kKvHeads][positions] rows for positions [0, hi)
    std::vector<double> kd, vd;  // the values they hold, [kKvHeads][positions][kHeadDim]
    int64_t hi = 0;
    void make(Kv f, int64_t n) {
        kv = f;
        hi = n;
        const size_t rows = (size_t) kKvHeads * n;
        k.assign(rows * kv_bytes(kv), 0);
        v.assign(rows * kv_bytes(kv), 0);
        kd.resize(rows * kHeadDim);
        vd.resize(rows * kHeadDim);
        for (size_t i = 0; i < rows; ++i) {
            const std::vector<float> kf = randn(kHeadDim, 0.6f), vf = randn(kHeadDim, 1.0f);
            put_row(kv, kf.data(), &k[i * kv_bytes(kv)], &kd[i * kHeadDim]);
            put_row(kv, vf.data(), &v[i * kv_bytes(kv)], &vd[i * kHeadDim]);
        }
    }
    double key(int h, int64_t p, int d) const { return kd[((size_t) h * hi + p) * kHeadDim + d]; }
    double val(int h, int64_t p, int d) const { return vd[((size_t) h * hi + p) * kHeadDim + d]; }
    // the cache bytes of positions [lo, hi2) at p % cap
    std::vector<uint8_t> cache(bool keys, int64_t cap, int64_t lo, int64_t hi2) const {
        const size_t rb = kv_bytes(kv);
        std::vector<uint8_t> out((size_t) kKvHeads * cap * rb, 0);
        const std::vector<uint8_t>& src = keys ? k : v;
        for (int h = 0; h < kKvHeads; ++h)
            for (int64_t p = lo; p < hi2; ++p)
                std::memcpy(&out[((size_t) (p % cap) * kKvHeads + h) * rb], &src[((size_t) h * hi + p) * rb], rb);
        return out;
    }
};

void test_attention_prompt() {
    for (const Kv kv : {kF16, kE4M3, kI8})
        for (const bool window : {true, false}) {
            const int R = 300;
            const int64_t pos0 = window ? 2500 : 3000;
            KeyBank kb;
            kb.make(kv, pos0 + R);
            const std::vector<float> q = randn((size_t) R * kHeads * kHeadDim, 0.8f), rs = randn((size_t) R * kHeads * kRel, 1.0f);
            const std::vector<float> sink = randn((size_t) R * kHeads, 2.0f), table = randn((size_t) kRel * kExtent, 0.3f);
            // the window layer: a ring of the window and 64 cells holding the keys before the chunk (and the chunk's first 37,
            // for the second call below), the chunk's own staged at p % 512; the full layer: one cache of every position
            const int64_t ring_cap = kExtent + 64, stage_cap = 512, full_cap = 4096;
            std::vector<uint8_t> kc, vc, ks, vs;
            if (window) {
                kc = kb.cache(true, ring_cap, std::max<int64_t>(0, pos0 + 37 - ring_cap), pos0 + 37);
                vc = kb.cache(false, ring_cap, std::max<int64_t>(0, pos0 + 37 - ring_cap), pos0 + 37);
                ks = kb.cache(true, stage_cap, pos0, pos0 + R);
                vs = kb.cache(false, stage_cap, pos0, pos0 + R);
            } else {
                kc = kb.cache(true, full_cap, 0, pos0 + R);
                vc = kb.cache(false, full_cap, 0, pos0 + R);
            }
            uint8_t *dkc = dev(kc), *dvc = dev(vc), *dks = window ? dev(ks) : dkc, *dvs = window ? dev(vs) : dvc;
            float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table);
            float* out = zeros<float>((size_t) R * kHeads * kHeadDim);
            AttnPromptArgs a;
            a.q = dq;
            a.r = drs;
            a.sink = dsink;
            a.rel_table = dt;
            a.pos0 = pos0;
            a.k_cache = dkc;
            a.v_cache = dvc;
            a.cap = window ? ring_cap : full_cap;
            a.k_chunk = dks;
            a.v_chunk = dvs;
            a.chunk_cap = window ? stage_cap : full_cap;
            a.fp8 = kv == kE4M3;
            a.int8 = kv == kI8;
            a.window = window ? kExtent : 0;
            attention_prompt(a, R, out, S);
            const std::vector<float> got = host(out, (size_t) R * kHeads * kHeadDim);
            // the CPU reference on every 7th row
            std::vector<float> gsub;
            std::vector<double> ref;
            for (int r = 0; r < R; r += 7)
                for (int h = 0; h < kHeads; ++h) {
                    const int g = h / 3;
                    const int64_t p = pos0 + r, lo = window ? std::max<int64_t>(0, p - kExtent + 1) : 0;
                    std::vector<double> sc;
                    double m = sink[(size_t) r * kHeads + h];
                    for (int64_t key = lo; key <= p; ++key) {
                        double d = 0;
                        for (int i = 0; i < kHeadDim; ++i) d += (double) q[((size_t) r * kHeads + h) * kHeadDim + i] * kb.key(g, key, i);
                        d *= a.scale;
                        const int64_t dist = p - key;
                        if (dist < kExtent)
                            for (int i = 0; i < kRel; ++i) d += (double) rs[((size_t) r * kHeads + h) * kRel + i] * table[(size_t) i * kExtent + dist];
                        sc.push_back(d);
                        m = std::max(m, d);
                    }
                    double den = std::exp(sink[(size_t) r * kHeads + h] - m);
                    for (double& s : sc) den += (s = std::exp(s - m));
                    for (int i = 0; i < kHeadDim; ++i) {
                        double acc = 0;
                        for (int64_t key = lo; key <= p; ++key) acc += sc[(size_t) (key - lo)] * kb.val(g, key, i);
                        ref.push_back(acc / den);
                        gsub.push_back(got[((size_t) r * kHeads + h) * kHeadDim + i]);
                    }
                }
            char name[96];
            std::snprintf(name, sizeof name, "attention_prompt %s %s vs CPU", window ? "window" : "full", kv_name(kv));
            report(name, gsub, ref, 4e-3);
            {   // the output's hash: a change meant to keep the bits keeps it
                uint64_t h = 1469598103934665603ull;
                const uint8_t* b = reinterpret_cast<const uint8_t*>(got.data());
                for (size_t i = 0; i < got.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
                std::printf("  output hash %016llx\n", (unsigned long long) h);
            }
            // the decode kernel on the same rows, over one cache that holds every position
            {
                const int64_t cap = window ? kExtent + R + 8 : full_cap;
                std::vector<uint8_t> dk = kb.cache(true, cap, window ? std::max<int64_t>(0, pos0 - kExtent) : 0, pos0 + R);
                std::vector<uint8_t> dv = kb.cache(false, cap, window ? std::max<int64_t>(0, pos0 - kExtent) : 0, pos0 + R);
                uint8_t *ddk = dev(dk), *ddv = dev(dv);
                std::vector<int32_t> pos(R);
                for (int r = 0; r < R; ++r) pos[r] = (int32_t) (pos0 + r);
                int32_t* dpos = dev(pos);
                AttnArgs d;
                d.q = dq;
                d.r = drs;
                d.sink = dsink;
                d.rel_table = dt;
                d.pos = dpos;
                d.k_cache = ddk;
                d.v_cache = ddv;
                d.cap = cap;
                d.fp8 = a.fp8;
                d.int8 = a.int8;
                d.window = a.window;
                const int64_t max_keys = window ? kExtent : pos0 + R;
                void* scr = zeros<uint8_t>(attention_decode_scratch(8, max_keys));
                float* dout = zeros<float>((size_t) R * kHeads * kHeadDim);
                for (int r0 = 0; r0 < R; r0 += 8) {
                    AttnArgs w = d;
                    w.q = dq + (size_t) r0 * kHeads * kHeadDim;
                    w.r = drs + (size_t) r0 * kHeads * kRel;
                    w.sink = dsink + (size_t) r0 * kHeads;
                    w.pos = dpos + r0;
                    attention_decode(w, std::min(8, R - r0), max_keys, scr, dout + (size_t) r0 * kHeads * kHeadDim, S);
                }
                const std::vector<float> dec = host(dout, (size_t) R * kHeads * kHeadDim);
                std::snprintf(name, sizeof name, "  vs attention_decode (all %d rows)", R);
                report(name, got, dec, 4e-3);
                {
                    uint64_t h = 1469598103934665603ull;
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(dec.data());
                    for (size_t i = 0; i < dec.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
                    std::printf("  attention_decode output hash %016llx\n", (unsigned long long) h);
                }
                cudaFree(ddk);
                cudaFree(ddv);
                cudaFree(dpos);
                cudaFree(scr);
                cudaFree(dout);
            }
            // rows 37.. as their own chunk: other blocks, other key sources for positions pos0 .. pos0 + 36
            {
                AttnPromptArgs b = a;
                b.pos0 = pos0 + 37;
                b.q = dq + (size_t) 37 * kHeads * kHeadDim;
                b.r = drs + (size_t) 37 * kHeads * kRel;
                b.sink = dsink + (size_t) 37 * kHeads;
                float* out2 = zeros<float>((size_t) (R - 37) * kHeads * kHeadDim);
                attention_prompt(b, R - 37, out2, S);
                const std::vector<float> g2 = host(out2, (size_t) (R - 37) * kHeads * kHeadDim);
                same_bits("  rows 37.. as a chunk of their own", g2.data(), &got[(size_t) 37 * kHeads * kHeadDim], g2.size() * 4);
                cudaFree(out2);
            }
            if (window) {
                cudaFree(dks);
                cudaFree(dvs);
            }
            cudaFree(dkc);
            cudaFree(dvc);
            cudaFree(dq);
            cudaFree(drs);
            cudaFree(dsink);
            cudaFree(dt);
            cudaFree(out);
        }
    // timing: a 2,048-row chunk of a window layer, and of a full layer after 8K, 32K and 128K positions, in each format
    for (const Kv kv : {kF16, kE4M3, kI8})
        for (const int64_t ctx : {(int64_t) 0, (int64_t) 8192, (int64_t) 32768, (int64_t) 131072}) {
            const int R = 2048;
            const bool window = ctx == 0;
            const int64_t pos0 = window ? 8192 : ctx, cap = window ? kExtent + 8 : ctx + R;
            uint8_t* kc = zeros<uint8_t>((size_t) kKvHeads * cap * kv_bytes(kv));
            uint8_t* vc = zeros<uint8_t>((size_t) kKvHeads * cap * kv_bytes(kv));
            uint8_t* st = zeros<uint8_t>((size_t) kKvHeads * R * kv_bytes(kv));
            uint8_t* sv = zeros<uint8_t>((size_t) kKvHeads * R * kv_bytes(kv));
            const std::vector<float> q = randn((size_t) R * kHeads * kHeadDim, 0.8f), rs = randn((size_t) R * kHeads * kRel, 1.0f);
            float *dq = dev(q), *drs = dev(rs), *dsink = zeros<float>((size_t) R * kHeads), *dt = dev(randn((size_t) kRel * kExtent, 0.3f));
            float* out = zeros<float>((size_t) R * kHeads * kHeadDim);
            AttnPromptArgs a;
            a.q = dq;
            a.r = drs;
            a.sink = dsink;
            a.rel_table = dt;
            a.pos0 = pos0;
            a.k_cache = kc;
            a.v_cache = vc;
            a.cap = cap;
            a.k_chunk = window ? st : kc;
            a.v_chunk = window ? sv : vc;
            a.chunk_cap = window ? R : cap;
            a.fp8 = kv == kE4M3;
            a.int8 = kv == kI8;
            a.window = window ? kExtent : 0;
            const double us = time_us([&] { attention_prompt(a, R, out, S); }, 10);
            double pairs = 0;   // (query, key) pairs
            for (int r = 0; r < R; ++r) pairs += window ? kExtent : (double) (pos0 + r + 1);
            const double flop = pairs * kHeads * kHeadDim * 4;
            const char* where = window ? "window" : ctx == 8192 ? "full at 8K" : ctx == 32768 ? "full at 32K" : "full at 128K";
            std::printf("  attention_prompt %s %s, 2048 rows: %.0f us (%.1f TFLOP/s of QK and PV)\n", where, kv_name(kv), us,
                        flop / us / 1e6);
            cudaFree(kc);
            cudaFree(vc);
            cudaFree(st);
            cudaFree(sv);
            cudaFree(dq);
            cudaFree(drs);
            cudaFree(dsink);
            cudaFree(dt);
            cudaFree(out);
        }
}

// A chunk's staged rows into the window ring (stage_to_ring): the last `keep` positions of the chunk, cell for cell, in each
// format, past the ring's wrap; the ring's other cells untouched.
void test_stage_to_ring() {
    for (const Kv kv : {kF16, kE4M3, kI8}) {
        const int R = 3000, keep = kExtent;
        const int64_t pos0 = 5000, stage_cap = 4096, ring_cap = kExtent + 8;
        KeyBank kb;
        kb.make(kv, pos0 + R);
        const std::vector<uint8_t> ks = kb.cache(true, stage_cap, pos0, pos0 + R), vs = kb.cache(false, stage_cap, pos0, pos0 + R);
        std::vector<uint8_t> kr = kb.cache(true, ring_cap, pos0 - ring_cap, pos0), vr = kb.cache(false, ring_cap, pos0 - ring_cap, pos0);
        uint8_t *dks = dev(ks), *dvs = dev(vs), *dkr = dev(kr), *dvr = dev(vr);
        stage_to_ring(dks, dvs, stage_cap, dkr, dvr, ring_cap, pos0, R, keep, (int) kv_bytes(kv), S);
        const std::vector<uint8_t> gk = host(dkr, kr.size()), gv = host(dvr, vr.size());
        // the ring before, its cells of the chunk's last `keep` positions replaced
        std::vector<uint8_t> wk = kr, wv = vr;
        const std::vector<uint8_t> nk = kb.cache(true, ring_cap, pos0 + R - keep, pos0 + R), nv = kb.cache(false, ring_cap, pos0 + R - keep, pos0 + R);
        const size_t cell = (size_t) kKvHeads * kv_bytes(kv);
        for (int64_t p = pos0 + R - keep; p < pos0 + R; ++p) {
            std::memcpy(&wk[(size_t) (p % ring_cap) * cell], &nk[(size_t) (p % ring_cap) * cell], cell);
            std::memcpy(&wv[(size_t) (p % ring_cap) * cell], &nv[(size_t) (p % ring_cap) * cell], cell);
        }
        char name[64];
        std::snprintf(name, sizeof name, "stage_to_ring %s (keys)", kv_name(kv));
        same_bits(name, gk.data(), wk.data(), wk.size());
        std::snprintf(name, sizeof name, "stage_to_ring %s (values)", kv_name(kv));
        same_bits(name, gv.data(), wv.data(), wv.size());
        cudaFree(dks);
        cudaFree(dvs);
        cudaFree(dkr);
        cudaFree(dvr);
    }
}

// ---- router ----------------------------------------------------------------------------------------------------------------

void test_router() {
    const int R = 512, NE = 128, K = 8;
    const std::vector<float> x = randn((size_t) R * E, 1.0f), gate = randn((size_t) NE * E, 0.02f), bias = randn(NE, 0.05f);
    float *dx = dev(x), *dg = dev(gate), *db = dev(bias);
    float *lg_a = zeros<float>((size_t) R * NE), *lg_b = zeros<float>((size_t) R * NE), *wa = zeros<float>((size_t) R * K), *wb = zeros<float>((size_t) R * K);
    int32_t *ia = zeros<int32_t>((size_t) R * K), *ib = zeros<int32_t>((size_t) R * K);
    bk::RouterArgs ra;
    ra.n_expert = NE;
    ra.n_embd = E;
    ra.k = K;
    ra.weight_scale = 8.f;
    ra.gate = bk::RouterGate::Sigmoid;
    bk::router(dx, dg, db, R, ra, lg_a, ia, wa, S);
    bk::f32_rows_dot(dx, dg, lg_b, R, NE, E, S);
    bk::router_select(lg_b, db, R, ra, ib, wb, S);
    cudaDeviceSynchronize();
    const std::vector<float> la = host(lg_a, (size_t) R * NE), lb = host(lg_b, (size_t) R * NE);
    std::vector<double> ref((size_t) R * NE);
    for (int r = 0; r < R; ++r)
        for (int e = 0; e < NE; ++e) {
            double s = 0;
            for (int c = 0; c < E; ++c) s += (double) x[(size_t) r * E + c] * gate[(size_t) e * E + c];
            ref[(size_t) r * NE + e] = s;
        }
    report("f32_rows_dot (router scores) vs CPU", lb, ref, 1e-5);
    report("  router's own scores vs CPU", la, ref, 1e-5);
    const std::vector<int32_t> ha = host(ia, (size_t) R * K), hb = host(ib, (size_t) R * K);
    int differ = 0;
    for (int r = 0; r < R; ++r) differ += std::memcmp(&ha[(size_t) r * K], &hb[(size_t) r * K], K * 4) != 0;
    std::printf("%-44s %d of %d rows choose other experts%s\n", "  f32_rows_dot + router_select vs router", differ, R, differ ? "" : " (none)");
    failures += differ > R / 100;
    const double us = time_us([&] { bk::f32_rows_dot(dx, dg, lg_b, R, NE, E, S); });
    std::printf("  f32_rows_dot %d x %d x %d: %.1f us (%.1f TFLOP/s)\n", R, NE, E, us, 2.0 * R * NE * E / us / 1e6);
}

// ---- the experts' sort -----------------------------------------------------------------------------------------------------

void test_sort() {
    const int R = 2048, NE = 128, K = 8, P = R * K;
    std::vector<int32_t> ids((size_t) P);
    std::vector<int> perm(NE);
    std::iota(perm.begin(), perm.end(), 0);
    for (int r = 0; r < R; ++r) {   // distinct experts a row, skewed toward the low ones
        std::shuffle(perm.begin(), perm.end(), rng);
        std::partial_sort(perm.begin(), perm.begin() + K, perm.end(), [&](int a, int b) { return (a % 37) < (b % 37); });
        for (int s = 0; s < K; ++s) ids[(size_t) r * K + s] = perm[s];
    }
    int32_t* dids = dev(ids);
    int32_t *bounds = zeros<int32_t>(NE + 1), *tok = zeros<int32_t>(P), *dst = zeros<int32_t>(P), *ex = zeros<int32_t>(P);
    void* scr = zeros<uint8_t>(moe_sort_scratch(P, NE));
    moe_sort(dids, R, K, NE, scr, bounds, tok, dst, ex, S);
    std::vector<int32_t> rb(NE + 1, 0), rt, rd, re;
    for (int e = 0; e < NE; ++e) {
        rb[e] = (int32_t) rt.size();
        for (int i = 0; i < P; ++i)
            if (ids[(size_t) i] == e) {
                rt.push_back(i / K);
                rd.push_back(i);
                re.push_back(e);
            }
    }
    rb[NE] = P;
    const bool ok = host(bounds, NE + 1) == rb && host(tok, P) == rt && host(dst, P) == rd && host(ex, P) == re;
    std::printf("%-44s %s\n", "moe_sort vs CPU stable sort (2,048 rows)", ok ? "exact" : "FAIL");
    failures += !ok;
    const double us = time_us([&] { moe_sort(dids, R, K, NE, scr, bounds, tok, dst, ex, S); });
    std::printf("  moe_sort 2,048 rows x 8: %.1f us\n", us);

    // pairs another card computes (id -1) are left out; bounds[NE] counts the rest
    std::vector<int32_t> part = ids;
    for (size_t i = 0; i < part.size(); ++i)
        if (part[i] % 3 == 0) part[i] = -1;
    moe_sort(dev(part), R, K, NE, scr, bounds, tok, dst, ex, S);
    std::vector<int32_t> pb(NE + 1, 0), pt, pd, pe;
    for (int e = 0; e < NE; ++e) {
        pb[e] = (int32_t) pt.size();
        for (int i = 0; i < P; ++i)
            if (part[(size_t) i] == e) {
                pt.push_back(i / K);
                pd.push_back(i);
                pe.push_back(e);
            }
    }
    pb[NE] = (int32_t) pt.size();
    const std::vector<int32_t> gt = host(tok, P), gd = host(dst, P), ge = host(ex, P);
    const bool ok2 = host(bounds, NE + 1) == pb && std::equal(pt.begin(), pt.end(), gt.begin()) &&
                     std::equal(pd.begin(), pd.end(), gd.begin()) && std::equal(pe.begin(), pe.end(), ge.begin());
    std::printf("%-44s %s\n", "moe_sort leaving pairs out (id -1)", ok2 ? "exact" : "FAIL");
    failures += !ok2;
}

// ---- the softcapped SwiGLU of the MMQ path ---------------------------------------------------------------------------------

void test_swiglu() {
    const int P = 4096, F = 1152, NE = 128;
    const std::vector<float> gu = randn((size_t) P * 2 * F, 3.0f);
    std::vector<float> tails((size_t) NE * 4);
    for (int e = 0; e < NE; ++e) {
        tails[(size_t) e * 4] = 0.5f + 0.01f * e;
        tails[(size_t) e * 4 + 1] = 1.5f - 0.005f * e;
        tails[(size_t) e * 4 + 2] = 0.8f + 0.002f * e;
    }
    std::vector<int32_t> rexp((size_t) P);
    for (int i = 0; i < P; ++i) rexp[(size_t) i] = (int32_t) (rng() % NE);
    float *dgu = dev(gu), *dt = dev(tails), *h = zeros<float>((size_t) P * F);
    int32_t* dre = dev(rexp);
    mmq::SwigluForm form;
    form.gate_softcap = 4.f;
    form.up_softcap = 25.f;
    form.row_expert = dre;
    mmq::swiglu(dgu, h, P, F, false, S, dt, nullptr, 0, form);
    const std::vector<float> got = host(h, (size_t) P * F);
    std::vector<double> ref((size_t) P * F);
    for (int i = 0; i < P; ++i) {
        const float* t = &tails[(size_t) rexp[(size_t) i] * 4];
        for (int f = 0; f < F; ++f) {
            const double g = (double) gu[(size_t) i * 2 * F + f] * t[0], u = (double) gu[(size_t) i * 2 * F + F + f] * t[1];
            ref[(size_t) i * F + f] = 4.0 * std::tanh(g / 4.0) * sig(g) * 25.0 * std::tanh(u / 25.0) * t[2];
        }
    }
    report("mmq::swiglu softcapped, tail by row", got, ref, 5e-5);   // the mmq library builds with -use_fast_math
}

// ---- the routed experts ------------------------------------------------------------------------------------------------------

// n_expert blobs of `type` (NVFP4 with a random scale tail, or Q8_0) at `stride` bytes, and their dequantized weights.
struct Experts {
    bk::NativeExpertLayout L;
    size_t stride = 0;
    std::vector<uint8_t> blobs;
    std::vector<std::vector<float>> wg, wu, wd;
    std::vector<float> tail;   // [n][4]
    void make(int type, int n, int F) {
        const auto* tr = ggml_get_type_traits((ggml_type) type);
        L = bk::native_expert_layout(type, type, E, F);
        L.gate_softcap = 4.f;
        L.up_softcap = 25.f;
        const size_t unit = type == 40 ? 144 : 272;
        stride = (L.bytes + unit - 1) / unit * unit;
        blobs.assign((size_t) n * stride, 0);
        wg.resize(n);
        wu.resize(n);
        wd.resize(n);
        tail.assign((size_t) n * 4, 1.f);
        for (int e = 0; e < n; ++e) {
            uint8_t* b = blobs.data() + (size_t) e * stride;
            auto put = [&](std::vector<float>& deq, int rows, int k, size_t row_bytes, uint8_t* dst) {
                const std::vector<float> w = randn((size_t) rows * k, 0.05f);
                deq.resize((size_t) rows * k);
                for (int r = 0; r < rows; ++r) {
                    tr->from_float_ref(&w[(size_t) r * k], dst + (size_t) r * row_bytes, k);
                    tr->to_float(dst + (size_t) r * row_bytes, &deq[(size_t) r * k], k);
                }
            };
            put(wg[e], F, E, L.gu_row, b);
            put(wu[e], F, E, L.up_row, b + L.up_off);
            put(wd[e], E, F, L.d_row, b + L.down_off);
            if (type == 40) {
                float* t = &tail[(size_t) e * 4];
                t[0] = 0.7f + 0.004f * e;
                t[1] = 1.3f - 0.003f * e;
                t[2] = 0.9f + 0.001f * e;
                t[3] = 0.f;
                std::memcpy(b + L.bytes - 16, t, 16);
            }
        }
    }
};

void test_experts(int type) {
    const int NE = 128, K = 8, F = 1152, R = 256, P = R * K;
    Experts ex;
    ex.make(type, NE, F);
    std::vector<int32_t> ids((size_t) P);
    std::vector<int> perm(NE);
    std::iota(perm.begin(), perm.end(), 0);
    for (int r = 0; r < R; ++r) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int s = 0; s < K; ++s) ids[(size_t) r * K + s] = perm[s];
    }
    const std::vector<float> w = randn((size_t) P, 0.5f, 1.f), x = randn((size_t) R * E, 1.0f);
    uint8_t* blobs = dev(ex.blobs);
    int32_t* dids = dev(ids);
    float *dw = dev(w), *dx = dev(x);
    // the prompt path: sort, q8_1 rows by expert, gate/up, softcapped SwiGLU, q8_1 H, down, the weighted sum
    int32_t *bounds = zeros<int32_t>(NE + 1), *tok = zeros<int32_t>(P), *dst = zeros<int32_t>(P), *rexp = zeros<int32_t>(P), *ident = zeros<int32_t>(P);
    void* scr = zeros<uint8_t>(moe_sort_scratch(P, NE));
    void* xq = zeros<uint8_t>(mmq::q8_bytes(P, E));
    float *gu = zeros<float>((size_t) P * 2 * F), *h = zeros<float>((size_t) P * F), *y = zeros<float>((size_t) R * E);
    mmq::iota(ident, P, S);
    mmq::load(type, 2 * F);
    mmq::load(type, E);
    mmq::Context ctx;
    const auto prompt = [&](int rows, const int32_t* ids_r, const float* x_r, const float* w_r, float* y_r) {
        const int pairs = rows * K;
        moe_sort(ids_r, rows, K, NE, scr, bounds, tok, dst, rexp, S);
        mmq::quantize(x_r, tok, xq, type, E, E, pairs, S);
        mmq::Product p;
        p.w = blobs;
        p.type = type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = ex.stride;
        p.n = NE;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = pairs;
        p.max_rows = rows;
        p.opt_rows = (pairs + NE - 1) / NE;
        p.dst = gu;
        p.ld_dst = 2 * F;
        ctx.run(p, S);
        mmq::SwigluForm form;
        form.gate_softcap = 4.f;
        form.up_softcap = 25.f;
        form.row_expert = rexp;
        form.tail_stride = ex.stride;   // the tails read in the blobs, as the engine reads them
        mmq::swiglu(gu, h, pairs, F, false, S, type == 40 ? (const float*) (blobs + ex.L.bytes - 16) : nullptr, nullptr, 0, form);
        mmq::quantize(h, nullptr, xq, type, F, F, pairs, S);
        mmq::Product d = p;
        d.w = blobs + ex.L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.ids = dst;
        d.dst = gu;
        d.ld_dst = E;
        ctx.run(d, S);
        moe_combine(gu, w_r, rows, K, y_r, S);
    };
    prompt(R, dids, dx, dw, y);
    const std::vector<float> got = host(y, (size_t) R * E);
    // the decode path, 8 rows a window: moe_plan, native_expert_grouped, moe_combine
    std::vector<float> dec;
    {
        const int W = 8, WP = W * K;
        int32_t *start = zeros<int32_t>(WP + 2), *ng = zeros<int32_t>(4), *ddst = zeros<int32_t>(WP), *dtok = zeros<int32_t>(WP);
        unsigned long long* ptr = zeros<unsigned long long>(WP);
        std::vector<unsigned long long> table((size_t) NE);
        for (int e = 0; e < NE; ++e) table[(size_t) e] = (unsigned long long) (blobs + (size_t) e * ex.stride);
        unsigned long long* dtable = dev(table);
        void* dxq = zeros<uint8_t>(bk::native_q8_1_bytes(E, W));
        void* dscr = zeros<uint8_t>(bk::native_expert_scratch_bytes(WP, F));
        float *rows = zeros<float>((size_t) WP * E), *yd = zeros<float>((size_t) R * E);
        for (int r0 = 0; r0 < R; r0 += W) {
            moe_plan(dids + (size_t) r0 * K, W, K, NE, dtable, ptr, start, ng, ddst, dtok, S);
            bk::native_quantize_q8_1(dx + (size_t) r0 * E, dxq, E, W, S);
            bk::native_expert_grouped(ex.L, ptr, start, ng, ddst, dtok, std::min(WP, NE), WP, dxq, dscr, rows, S);
            moe_combine(rows, dw + (size_t) r0 * K, W, K, yd + (size_t) r0 * E, S);
        }
        dec = host(yd, (size_t) R * E);
        cudaFree(start);
        cudaFree(ng);
        cudaFree(ddst);
        cudaFree(dtok);
        cudaFree(ptr);
        cudaFree(dtable);
        cudaFree(dxq);
        cudaFree(dscr);
        cudaFree(rows);
        cudaFree(yd);
    }
    // the CPU reference on the first 8 rows (exact weights, fp32 activations)
    std::vector<double> ref((size_t) 8 * E, 0.0);
    for (int r = 0; r < 8; ++r)
        for (int s = 0; s < K; ++s) {
            const int e = ids[(size_t) r * K + s];
            const float* t = &ex.tail[(size_t) e * 4];
            std::vector<double> hh(F);
            for (int f = 0; f < F; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < E; ++c) {
                    g += (double) ex.wg[e][(size_t) f * E + c] * x[(size_t) r * E + c];
                    u += (double) ex.wu[e][(size_t) f * E + c] * x[(size_t) r * E + c];
                }
                g *= t[0];
                u *= t[1];
                hh[f] = 4.0 * std::tanh(g / 4.0) * sig(g) * 25.0 * std::tanh(u / 25.0);
            }
            for (int c = 0; c < E; ++c) {
                double o = 0;
                for (int f = 0; f < F; ++f) o += (double) ex.wd[e][(size_t) c * F + f] * hh[f];
                ref[(size_t) r * E + c] += (double) w[(size_t) r * K + s] * o * t[2];
            }
        }
    char name[96];
    std::snprintf(name, sizeof name, "experts by MMQ (%s) vs CPU, 8 rows", type == 40 ? "NVFP4" : "Q8_0");
    report(name, std::vector<float>(got.begin(), got.begin() + 8 * E), ref, 2e-2);
    report("  vs the decode path's native_expert_grouped", got, dec, 5e-3);   // two q8_1 roundings of x and H
    std::vector<double> dref(ref);
    report("  (decode path vs CPU, 8 rows)", std::vector<float>(dec.begin(), dec.begin() + 8 * E), dref, 2e-2);
    // the first 100 rows as a chunk of their own
    float* y2 = zeros<float>((size_t) 100 * E);
    prompt(100, dids, dx, dw, y2);
    const std::vector<float> g2 = host(y2, (size_t) 100 * E);
    const bool same = std::memcmp(g2.data(), got.data(), g2.size() * 4) == 0;
    std::printf("%-44s %s\n", "  rows 0..99 as a chunk of their own", same ? "bitwise" : "differ (MMQ's split of K follows the chunk)");
    if (!same) report("    the difference", g2, std::vector<float>(got.begin(), got.begin() + 100 * E), 1e-5);
    // timing at a 2,048-row chunk's shape: 16,384 pairs over the 128 experts
    {
        const int RR = 2048, PP = RR * K;
        std::vector<int32_t> big((size_t) PP);
        for (int r = 0; r < RR; ++r) {
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int s = 0; s < K; ++s) big[(size_t) r * K + s] = perm[s];
        }
        int32_t *bids = dev(big), *bb = zeros<int32_t>(NE + 1), *bt = zeros<int32_t>(PP), *bd = zeros<int32_t>(PP), *be = zeros<int32_t>(PP), *bi = zeros<int32_t>(PP);
        void* bscr = zeros<uint8_t>(moe_sort_scratch(PP, NE));
        void* bxq = zeros<uint8_t>(mmq::q8_bytes(PP, E));
        float *bgu = zeros<float>((size_t) PP * 2 * F), *bh = zeros<float>((size_t) PP * F), *bx = dev(randn((size_t) RR * E, 1.f));
        mmq::iota(bi, PP, S);
        moe_sort(bids, RR, K, NE, bscr, bb, bt, bd, be, S);
        mmq::quantize(bx, bt, bxq, type, E, E, PP, S);
        mmq::Product p;
        p.w = blobs;
        p.type = type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = ex.stride;
        p.n = NE;
        p.xq = bxq;
        p.bounds = bb;
        p.ids = bi;
        p.total_rows = PP;
        p.max_rows = RR;
        p.opt_rows = PP / NE;
        p.dst = bgu;
        p.ld_dst = 2 * F;
        const double us_gu = time_us([&] { ctx.run(p, S); }, 10);
        mmq::Product d = p;
        d.w = blobs + ex.L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.ids = bd;
        d.dst = bgu;
        d.ld_dst = E;
        mmq::quantize(bh, nullptr, bxq, type, F, F, PP, S);
        const double us_d = time_us([&] { ctx.run(d, S); }, 10);
        const double fl = 2.0 * PP * 2 * F * E, fd = 2.0 * PP * F * E;
        std::printf("  MMQ %s experts, 2,048 rows: gate/up %.0f us (%.0f TOP/s), down %.0f us (%.0f TOP/s)\n", type == 40 ? "NVFP4" : "Q8_0",
                    us_gu, fl / us_gu / 1e6, us_d, fd / us_d / 1e6);
        cudaFree(bids);
        cudaFree(bb);
        cudaFree(bt);
        cudaFree(bd);
        cudaFree(be);
        cudaFree(bi);
        cudaFree(bscr);
        cudaFree(bxq);
        cudaFree(bgu);
        cudaFree(bh);
        cudaFree(bx);
    }
    cudaFree(blobs);
    cudaFree(dids);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(bounds);
    cudaFree(tok);
    cudaFree(dst);
    cudaFree(rexp);
    cudaFree(ident);
    cudaFree(scr);
    cudaFree(xq);
    cudaFree(gu);
    cudaFree(h);
    cudaFree(y);
    cudaFree(y2);
}

// ---- the NVFP4 experts on the FP4 tensor cores (moe_fp4) ------------------------------------------------------------------------------------

float e4m3_round(float v) {
    const __nv_fp8_storage_t b = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}
// x to the E2M1 grid, nearest, ties to the even code, saturating at 6
float e2m1_round(float x) {
    static const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = std::fabs(x);
    int best = 7;
    for (int i = 0; i < 7; ++i) {
        const float mid = 0.5f * (grid[i] + grid[i + 1]);
        if (a < mid || (a == mid && (i % 2) == 0)) { best = i; break; }
    }
    return std::copysign(grid[best], x);
}
// a row as the FP4 experts round it, dequantized: token scale amax / (6 * 448), per 16 values an E4M3 scale of their
// largest |value| / 6, then the same for what the first term leaves (two terms)
std::vector<float> fp4_two_round(const float* x, int cols) {
    float amax = 0.f;
    for (int i = 0; i < cols; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float g = amax * (1.0f / (6.0f * 448.0f)), inv_g = amax > 0.f ? 1.0f / g : 0.f;
    std::vector<float> out((size_t) cols);
    for (int b = 0; b < cols / 16; ++b) {
        float v[16];
        std::memcpy(v, x + 16 * b, sizeof v);
        for (int term = 0; term < 2; ++term) {
            float bm = 0.f;
            for (int i = 0; i < 16; ++i) bm = std::max(bm, std::fabs(v[i]));
            const float sd = e4m3_round(bm * inv_g * (1.0f / 6.0f));
            const float inv = sd > 0.f ? 1.0f / (sd * g) : 0.f;
            for (int i = 0; i < 16; ++i) {
                const float q = e2m1_round(v[i] * inv) * (sd * g);
                out[(size_t) 16 * b + i] = term == 0 ? q : out[(size_t) 16 * b + i] + q;
                v[i] -= q;
            }
        }
    }
    return out;
}

void test_experts_fp4() {
    const int NE = 128, K = 8, F = 1152, R = 256, P = R * K;
    Experts ex;
    ex.make(40, NE, F);
    std::vector<int32_t> ids((size_t) P);
    std::vector<int> perm(NE);
    std::iota(perm.begin(), perm.end(), 0);
    for (int r = 0; r < R; ++r) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int s = 0; s < K; ++s) ids[(size_t) r * K + s] = perm[s];
    }
    const std::vector<float> w = randn((size_t) P, 0.5f, 1.f), x = randn((size_t) R * E, 1.0f);
    uint8_t* blobs = dev(ex.blobs);
    int32_t* dids = dev(ids);
    float *dw = dev(w), *dx = dev(x);
    const int RMAX = 2048, PMAX = RMAX * K;
    int32_t *bounds = zeros<int32_t>(NE + 1), *tok = zeros<int32_t>(PMAX), *dst = zeros<int32_t>(PMAX), *rexp = zeros<int32_t>(PMAX);
    int32_t *toff = zeros<int32_t>(NE + 1), *boff = zeros<int32_t>(NE + 1);
    void* scr = zeros<uint8_t>(moe_sort_scratch(PMAX, NE));
    void* a4x = zeros<uint8_t>(fp4::token_bytes(RMAX, 2 * E));   // two terms a row
    void* a4h = zeros<uint8_t>(fp4::a4_bytes(PMAX, 2 * F, NE));
    float *gsx = zeros<float>(RMAX), *gsh = zeros<float>(fp4::scale_count(PMAX, NE));
    float *h = zeros<float>((size_t) PMAX * F), *o = zeros<float>((size_t) PMAX * E), *y = zeros<float>((size_t) R * E);
    const auto run = [&](int rows, const int32_t* ids_r, const float* x_r, const float* w_r, float* y_r, bool combine) {
        const int pairs = rows * K;
        moe_sort(ids_r, rows, K, NE, scr, bounds, tok, dst, rexp, S);
        fp4::plan_blocks(bounds, NE, toff, boff, S);
        fp4::quantize_tokens(x_r, rows, E, E, a4x, gsx, S, true);
        fp4::Product gu;
        gu.base = blobs;
        gu.stride = ex.stride;
        gu.blk_off = boff;
        gu.blocks = fp4::max_blocks(pairs, NE);
        gu.n = NE;
        gu.w_rows = 2 * F;
        gu.w_cols = E;
        gu.a4 = a4x;
        gu.gs = gsx;
        gu.bounds = bounds;
        gu.ids = tok;
        gu.tokens = rows;
        gu.swiglu = true;
        gu.tail_off = ex.L.bytes - 16;
        gu.gate_softcap = 4.f;
        gu.up_softcap = 25.f;
        gu.dst = h;
        gu.ld_dst = F;
        gu.two = true;
        fp4::run(gu, S);
        fp4::quantize(h, nullptr, bounds, toff, NE, a4h, gsh, F, F, pairs, S, true);
        fp4::Product d = gu;
        d.swiglu = false;
        d.ids = nullptr;
        d.w_off = ex.L.down_off;
        d.w_rows = E;
        d.w_cols = F;
        d.a4 = a4h;
        d.gs = gsh;
        d.tile_off = toff;
        d.dst = o;
        d.dst_ids = dst;
        d.ld_dst = E;
        fp4::run(d, S);
        if (combine) moe_combine(o, w_r, rows, K, y_r, S);
    };
    run(R, dids, dx, dw, y, true);
    const std::vector<float> got = host(y, (size_t) R * E);
    // the CPU reference on the first 8 rows: exact weights, x and H rounded to two FP4 terms
    std::vector<double> ref((size_t) 8 * E, 0.0);
    for (int r = 0; r < 8; ++r) {
        const std::vector<float> xq = fp4_two_round(&x[(size_t) r * E], E);
        for (int s = 0; s < K; ++s) {
            const int e = ids[(size_t) r * K + s];
            const float* t = &ex.tail[(size_t) e * 4];
            std::vector<float> hh(F);
            for (int f = 0; f < F; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < E; ++c) {
                    g += (double) ex.wg[e][(size_t) f * E + c] * xq[(size_t) c];
                    u += (double) ex.wu[e][(size_t) f * E + c] * xq[(size_t) c];
                }
                g *= t[0];
                u *= t[1];
                hh[f] = (float) (4.0 * std::tanh(g / 4.0) * sig(g) * 25.0 * std::tanh(u / 25.0) * t[2]);
            }
            const std::vector<float> hq = fp4_two_round(hh.data(), F);
            for (int c = 0; c < E; ++c) {
                double ov = 0;
                for (int f = 0; f < F; ++f) ov += (double) ex.wd[e][(size_t) c * F + f] * hq[(size_t) f];
                ref[(size_t) r * E + c] += (double) w[(size_t) r * K + s] * ov;
            }
        }
    }
    // a value of H on a rounding edge can take the other code, so the tolerance is a few FP4 steps of one value
    report("experts FP4 (NVFP4, two terms) vs CPU, 8 rows", std::vector<float>(got.begin(), got.begin() + 8 * E), ref, 2e-2);
    double se = 0, sr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        se += (got[i] - ref[i]) * (got[i] - ref[i]);
        sr += ref[i] * ref[i];
    }
    const double rms = std::sqrt(se / std::max(sr, 1e-30));
    std::printf("%-44s %.3e %s\n", "  relative rms vs CPU", rms, rms < 3e-3 ? "ok" : "FAIL");
    failures += rms >= 3e-3;
    // the first 100 rows as a chunk of their own: every row's codes, chain over K and H are its own
    float* y2 = zeros<float>((size_t) 100 * E);
    run(100, dids, dx, dw, y2, true);
    const std::vector<float> g2 = host(y2, (size_t) 100 * E);
    same_bits("  rows 0..99 as a chunk of their own", g2.data(), got.data(), g2.size() * 4);
    // deterministic fp16 intermediate pipeline matches fp32 down + moe_combine and CPU double
    {
        float* yf = zeros<float>((size_t) R * E);
        moe_sort(dids, R, K, NE, scr, bounds, tok, dst, rexp, S);
        fp4::plan_blocks(bounds, NE, toff, boff, S);
        fp4::quantize_tokens(dx, R, E, E, a4x, gsx, S, true);
        fp4::Product gu;
        gu.base = blobs; gu.stride = ex.stride; gu.blk_off = boff; gu.blocks = fp4::max_blocks(R * K, NE); gu.n = NE;
        gu.w_rows = 2 * F; gu.w_cols = E; gu.a4 = a4x; gu.gs = gsx; gu.bounds = bounds; gu.ids = tok; gu.tokens = R;
        auto* h16 = reinterpret_cast<uint16_t*>(h);
        gu.swiglu = true; gu.tail_off = ex.L.bytes - 16; gu.gate_softcap = 4.f; gu.up_softcap = 25.f; gu.dst_f16 = h16; gu.ld_dst = F; gu.two = true;
        fp4::run(gu, S);
        fp4::quantize_f16(h16, bounds, toff, NE, a4h, gsh, F, F, R * K, S);
        fp4::Product d = gu;
        d.swiglu = false; d.ids = nullptr; d.w_off = ex.L.down_off; d.w_rows = E; d.w_cols = F;
        auto* o16 = reinterpret_cast<uint16_t*>(o);
        d.a4 = a4h; d.gs = gsh; d.tile_off = toff; d.dst = nullptr; d.dst_f16 = o16; d.dst_ids = dst; d.ld_dst = E;
        fp4::run(d, S);
        moe_combine_f16(o16, dw, R, K, yf, S);
        const std::vector<float> gf = host(yf, (size_t) R * E);
        report("  fp16 H + fp16 down + combine_f16 vs fp32", gf, std::vector<double>(got.begin(), got.end()), 5e-3);
        report("  fp16 H + fp16 down + combine_f16 vs CPU", std::vector<float>(gf.begin(), gf.begin() + 8 * E), ref, 2e-2);
        // Chunk invariance of the fp16 pipeline: rows 0..99 alone must be bitwise identical
        float* yf2 = zeros<float>((size_t) 100 * E);
        moe_sort(dids, 100, K, NE, scr, bounds, tok, dst, rexp, S);
        fp4::plan_blocks(bounds, NE, toff, boff, S);
        fp4::quantize_tokens(dx, 100, E, E, a4x, gsx, S, true);
        gu.blocks = fp4::max_blocks(100 * K, NE); gu.tokens = 100;
        fp4::run(gu, S);
        fp4::quantize_f16(h16, bounds, toff, NE, a4h, gsh, F, F, 100 * K, S);
        d.blocks = gu.blocks; d.tokens = 100;
        fp4::run(d, S);
        moe_combine_f16(o16, dw, 100, K, yf2, S);
        same_bits("  fp16 pipeline rows 0..99 alone", host(yf2, (size_t) 100 * E).data(), gf.data(), (size_t) 100 * E * 4);
        cudaFree(yf2);
        cudaFree(yf);
    }
    // timing at a 2,048-row chunk's shape: 16,384 pairs over the 128 experts
    {
        std::vector<int32_t> big((size_t) PMAX);
        for (int r = 0; r < RMAX; ++r) {
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int s = 0; s < K; ++s) big[(size_t) r * K + s] = perm[s];
        }
        int32_t* bids = dev(big);
        float *bx = dev(randn((size_t) RMAX * E, 1.f)), *bw = dev(randn((size_t) PMAX, 0.5f, 1.f)), *by = zeros<float>((size_t) RMAX * E);
        const double us = time_us([&] { run(RMAX, bids, bx, bw, by, true); }, 10);
        const double fl = 2.0 * PMAX * 3 * F * E;
        std::printf("  FP4 experts, 2,048 rows: sort + quantize + gate/up + H + down + sum %.0f us (%.0f TOP/s)\n", us, fl / us / 1e6);
        cudaFree(bids);
        cudaFree(bx);
        cudaFree(bw);
        cudaFree(by);
    }
    for (void* p : {(void*) blobs, (void*) dids, (void*) dw, (void*) dx, (void*) bounds, (void*) tok, (void*) dst, (void*) rexp,
                    (void*) toff, (void*) boff, scr, a4x, a4h, (void*) gsx, (void*) gsh, (void*) h, (void*) o, (void*) y, (void*) y2})
        cudaFree(p);
}

// ---- the dense products ------------------------------------------------------------------------------------------------------

// A Q8_0 matrix [n_out][cols] times a chunk's rows by MMQ (one group, the prompt path's form) against the decode path's
// native_q8_0_mmvq over 8 rows a call, and the controllers' BF16 products by bf16_rows_dot against bf16_gemv_fp32_mmvf_multi.
void test_dense() {
    const int R = 300;
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (const int n_out : {kProjQ, kProjKV, E}) {
        const int cols = n_out == E ? kHeads * kHeadDim : E;
        const size_t row_bytes = (size_t) cols / 32 * 34;
        std::vector<uint8_t> wq((size_t) n_out * row_bytes);
        std::vector<float> wd((size_t) n_out * cols);
        {
            const std::vector<float> w = randn((size_t) n_out * cols, 0.03f);
            for (int r = 0; r < n_out; ++r) {
                tr->from_float_ref(&w[(size_t) r * cols], wq.data() + (size_t) r * row_bytes, cols);
                tr->to_float(wq.data() + (size_t) r * row_bytes, &wd[(size_t) r * cols], cols);
            }
        }
        const std::vector<float> x = randn((size_t) R * cols, 1.0f);
        uint8_t* dw = dev(wq);
        float *dx = dev(x), *y = zeros<float>((size_t) R * n_out), *yd = zeros<float>((size_t) R * n_out);
        void* xq = zeros<uint8_t>(mmq::q8_bytes(R, cols));
        int32_t *ident = zeros<int32_t>(R), *bounds = dev(std::vector<int32_t>{0, R});
        mmq::iota(ident, R, S);
        mmq::load(8, n_out);
        mmq::Context ctx;
        mmq::quantize(dx, nullptr, xq, 8, cols, cols, R, S);
        mmq::Product p;
        p.w = dw;
        p.type = 8;
        p.w_rows = n_out;
        p.w_cols = cols;
        p.expert_bytes = mmq::matrix_bytes(8, n_out, cols);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = R;
        p.max_rows = R;
        p.dst = y;
        p.ld_dst = n_out;
        ctx.run(p, S);
        void* xq8 = zeros<uint8_t>(bk::native_q8_1_bytes(cols, 8));
        for (int r0 = 0; r0 < R; r0 += 8) {
            const int n = std::min(8, R - r0);
            bk::native_quantize_q8_1(dx + (size_t) r0 * cols, xq8, cols, n, S);
            bk::native_q8_0_mmvq(dw, xq8, yd + (size_t) r0 * n_out, cols, n_out, n, S);
        }
        const std::vector<float> got = host(y, (size_t) R * n_out), dec = host(yd, (size_t) R * n_out);
        std::vector<double> ref((size_t) 4 * n_out);
        for (int r = 0; r < 4; ++r)
            for (int o = 0; o < n_out; ++o) {
                double a = 0;
                for (int c = 0; c < cols; ++c) a += (double) wd[(size_t) o * cols + c] * x[(size_t) r * cols + c];
                ref[(size_t) r * n_out + o] = a;
            }
        char name[96];
        std::snprintf(name, sizeof name, "Q8_0 product by MMQ, %d x %d, vs CPU (4 rows)", n_out, cols);
        report(name, std::vector<float>(got.begin(), got.begin() + 4 * n_out), ref, 1e-2);
        report("  vs native_q8_0_mmvq (300 rows)", got, dec, 1e-2);
        const double us = time_us([&] { ctx.run(p, S); });
        std::printf("  %.1f us at 300 rows (%.0f TOP/s)\n", us, 2.0 * R * n_out * cols / us / 1e6);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(y);
        cudaFree(yd);
        cudaFree(xq);
        cudaFree(xq8);
        cudaFree(ident);
        cudaFree(bounds);
    }
    // the controllers: [rank][2E] then [2E][rank], BF16
    {
        const int rank = 64, K2 = 2 * E;
        std::vector<uint16_t> wdn((size_t) rank * K2), wup((size_t) K2 * rank);
        const std::vector<float> a = randn(wdn.size(), 0.02f), b = randn(wup.size(), 0.1f);
        auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16); };
        for (size_t i = 0; i < wdn.size(); ++i) wdn[i] = bf(a[i]);
        for (size_t i = 0; i < wup.size(); ++i) wup[i] = bf(b[i]);
        const std::vector<float> x = randn((size_t) R * K2, 1.0f);
        uint16_t *ddn = dev(wdn), *dup = dev(wup);
        float *dx = dev(x), *h1 = zeros<float>((size_t) R * rank), *h2 = zeros<float>((size_t) R * rank);
        float *o1 = zeros<float>((size_t) R * K2), *o2 = zeros<float>((size_t) R * K2);
        bk::bf16_rows_dot(dx, ddn, h1, R, rank, K2, S);
        bk::bf16_rows_dot(h1, dup, o1, R, K2, rank, S);
        for (int r0 = 0; r0 < R; r0 += 8) {
            const int n = std::min(8, R - r0);
            bk::bf16_gemv_fp32_mmvf_multi(dx + (size_t) r0 * K2, K2, ddn, h2 + (size_t) r0 * rank, rank, K2, rank, n, S);
        }
        for (int r0 = 0; r0 < R; r0 += 8) {
            const int n = std::min(8, R - r0);
            bk::bf16_gemv_fp32_mmvf_multi(h1 + (size_t) r0 * rank, rank, dup, o2 + (size_t) r0 * K2, K2, rank, K2, n, S);
        }
        report("controller down: bf16_rows_dot vs mmvf_multi", host(h1, (size_t) R * rank), host(h2, (size_t) R * rank), 5e-5);   // x as two tf32 terms
        report("controller up: bf16_rows_dot vs mmvf_multi", host(o1, (size_t) R * K2), host(o2, (size_t) R * K2), 1e-5);
        {   // the 8-output tiles against the one-warp form: the same chain per output
            float* h3 = zeros<float>((size_t) R * rank);
            bk::bf16_rows_dot(dx, ddn, h3, R, rank, K2, S, false);
            const std::vector<float> a = host(h1, (size_t) R * rank), b = host(h3, (size_t) R * rank);
            same_bits("controller down: tiles vs one warp", a.data(), b.data(), a.size() * 4);
            const double t_tiles = time_us([&] { bk::bf16_rows_dot(dx, ddn, h1, R, rank, K2, S); });
            const double t_warp = time_us([&] { bk::bf16_rows_dot(dx, ddn, h3, R, rank, K2, S, false); });
            std::printf("  controller down at %d rows: tiles %.1f us, one warp %.1f us\n", R, t_tiles, t_warp);
            cudaFree(h3);
        }
        {   // fused rms_norm_q8_planes_prompt and attn_gate_q8_planes_prompt vs unfused through prefill::q8
            namespace q8 = basalt::prefill::q8;
            const int Rq = 2048;
            constexpr int OCols = kHeads * kHeadDim;
            const std::vector<float> hx = randn((size_t) Rq * E, 1.0f), hw = randn(E, 0.2f, 1.0f);
            const std::vector<float> ha = randn((size_t) Rq * OCols, 1.0f), hyq = randn((size_t) Rq * kProj, 1.0f);
            float *dx2 = dev(hx), *dw2 = dev(hw), *dn = zeros<float>((size_t) Rq * E);
            float *da2 = dev(ha), *dyq2 = dev(hyq), *dg2 = zeros<float>((size_t) Rq * OCols);
            void *act_ref = zeros<uint8_t>(q8::act_bytes(Rq, OCols));
            void *act_new = zeros<uint8_t>(q8::act_bytes(Rq, OCols));
            const int n_out = 256;
            const auto* tr2 = ggml_get_type_traits(GGML_TYPE_Q8_0);
            struct Q8Mat { uint8_t* planes; uint16_t* sc_t; };
            const auto make_q8 = [&](int cols, std::vector<float>& w_deq) -> Q8Mat {
                const size_t rb = (size_t) cols / 32 * 34;
                std::vector<uint8_t> wq((size_t) n_out * rb);
                const std::vector<float> wf = randn((size_t) n_out * cols, 0.03f);
                w_deq.resize((size_t) n_out * cols);
                for (int r = 0; r < n_out; ++r) {
                    tr2->from_float_ref(&wf[(size_t) r * cols], &wq[(size_t) r * rb], cols);
                    tr2->to_float(&wq[(size_t) r * rb], &w_deq[(size_t) r * cols], cols);
                }
                uint8_t* raw = dev(wq);
                uint8_t* pl = zeros<uint8_t>(wq.size());
                uint16_t* sc_t = zeros<uint16_t>(q8::scale_plane_bytes(n_out, cols) / 2);
                bk::wgemm::q8_planes(raw, n_out, cols, pl, S);
                q8::transpose_scales(reinterpret_cast<const uint16_t*>(pl + (size_t) n_out * cols), n_out, cols, sc_t, S);
                cudaStreamSynchronize(S);
                cudaFree(raw);
                return {pl, sc_t};
            };
            std::vector<float> wE_deq, wO_deq;
            Q8Mat wE = make_q8(E, wE_deq), wO = make_q8(OCols, wO_deq);
            float *yref = zeros<float>((size_t) Rq * n_out), *ynew = zeros<float>((size_t) Rq * n_out);

            bk::rms_norm_rows(dx2, dw2, dn, Rq, E, 1e-6f, S);
            q8::quantize(dn, Rq, E, E, act_ref, S);
            q8::gemm(reinterpret_cast<const int8_t*>(wE.planes), wE.sc_t, n_out, E, act_ref, Rq, yref, n_out, q8::Epilogue::Store, S);
            rms_norm_q8_planes_prompt(dx2, dw2, Rq, 1e-6f, act_new, S);
            q8::gemm(reinterpret_cast<const int8_t*>(wE.planes), wE.sc_t, n_out, E, act_new, Rq, ynew, n_out, q8::Epilogue::Store, S);
            const auto yn_host = host(ynew, (size_t) Rq * n_out);
            report("rms_norm_q8_planes_prompt vs unfused (via q8::gemm)", yn_host, host(yref, (size_t) Rq * n_out), 2e-3);
            {   // CPU double reference (8 rows) + position invariance (rows 37..100 alone)
                std::vector<double> ref_n((size_t) 8 * n_out, 0.0);
                for (int r = 0; r < 8; ++r) {
                    double ss = 0.0;
                    for (int c = 0; c < E; ++c) ss += (double) hx[(size_t) r * E + c] * hx[(size_t) r * E + c];
                    const double inv = 1.0 / std::sqrt(ss / E + 1e-6);
                    for (int o = 0; o < n_out; ++o) {
                        double acc = 0.0;
                        for (int c = 0; c < E; ++c) acc += (double) hx[(size_t) r * E + c] * inv * hw[c] * wE_deq[(size_t) o * E + c];
                        ref_n[(size_t) r * n_out + o] = acc;
                    }
                }
                report("rms_norm_q8_planes_prompt vs CPU double (8 rows)", std::vector<float>(yn_host.begin(), yn_host.begin() + 8 * n_out), ref_n, 2e-2);
                const int off_r = 37, sub_r = 64;
                void* act_sub = zeros<uint8_t>(q8::act_bytes(sub_r, E));
                float* ysub = zeros<float>((size_t) sub_r * n_out);
                rms_norm_q8_planes_prompt(dx2 + (size_t) off_r * E, dw2, sub_r, 1e-6f, act_sub, S);
                q8::gemm(reinterpret_cast<const int8_t*>(wE.planes), wE.sc_t, n_out, E, act_sub, sub_r, ysub, n_out, q8::Epilogue::Store, S);
                same_bits("  rms_norm_q8_planes rows 37..100 alone", yn_host.data() + (size_t) off_r * n_out,
                          host(ysub, (size_t) sub_r * n_out).data(), (size_t) sub_r * n_out * 4);
                cudaFree(act_sub);
                cudaFree(ysub);
            }
            const double t_n_old = time_us([&] { bk::rms_norm_rows(dx2, dw2, dn, Rq, E, 1e-6f, S); q8::quantize(dn, Rq, E, E, act_ref, S); });
            const double t_n_new = time_us([&] { rms_norm_q8_planes_prompt(dx2, dw2, Rq, 1e-6f, act_new, S); });
            std::printf("  rms_norm_q8_planes_prompt %d rows: %.1f us vs unfused %.1f us (%.2fx)\n", Rq, t_n_new, t_n_old, t_n_old / t_n_new);

            attn_gate(da2, dyq2, Rq, dg2, S, kProj);
            q8::quantize(dg2, Rq, OCols, OCols, act_ref, S);
            q8::gemm(reinterpret_cast<const int8_t*>(wO.planes), wO.sc_t, n_out, OCols, act_ref, Rq, yref, n_out, q8::Epilogue::Store, S);
            attn_gate_q8_planes_prompt(da2, dyq2, kProj, Rq, act_new, S);
            q8::gemm(reinterpret_cast<const int8_t*>(wO.planes), wO.sc_t, n_out, OCols, act_new, Rq, ynew, n_out, q8::Epilogue::Store, S);
            const auto yg_host = host(ynew, (size_t) Rq * n_out);
            report("attn_gate_q8_planes_prompt vs unfused (via q8::gemm)", yg_host, host(yref, (size_t) Rq * n_out), 2e-3);
            {   // CPU double reference (8 rows) + position invariance (rows 37..100 alone)
                std::vector<double> ref_g((size_t) 8 * n_out, 0.0);
                for (int r = 0; r < 8; ++r) {
                    for (int o = 0; o < n_out; ++o) {
                        double acc = 0.0;
                        for (int c = 0; c < OCols; ++c) {
                            const double gv = 1.0 / (1.0 + std::exp(-(double) hyq[(size_t) r * kProj + kOffGate + c]));
                            acc += (double) ha[(size_t) r * OCols + c] * gv * wO_deq[(size_t) o * OCols + c];
                        }
                        ref_g[(size_t) r * n_out + o] = acc;
                    }
                }
                report("attn_gate_q8_planes_prompt vs CPU double (8 rows)", std::vector<float>(yg_host.begin(), yg_host.begin() + 8 * n_out), ref_g, 2e-2);
                const int off_r = 37, sub_r = 64;
                void* act_sub = zeros<uint8_t>(q8::act_bytes(sub_r, OCols));
                float* ysub = zeros<float>((size_t) sub_r * n_out);
                attn_gate_q8_planes_prompt(da2 + (size_t) off_r * OCols, dyq2 + (size_t) off_r * kProj, kProj, sub_r, act_sub, S);
                q8::gemm(reinterpret_cast<const int8_t*>(wO.planes), wO.sc_t, n_out, OCols, act_sub, sub_r, ysub, n_out, q8::Epilogue::Store, S);
                same_bits("  attn_gate_q8_planes rows 37..100 alone", yg_host.data() + (size_t) off_r * n_out,
                          host(ysub, (size_t) sub_r * n_out).data(), (size_t) sub_r * n_out * 4);
                cudaFree(act_sub);
                cudaFree(ysub);
            }
            const double t_g_old = time_us([&] { attn_gate(da2, dyq2, Rq, dg2, S, kProj); q8::quantize(dg2, Rq, OCols, OCols, act_ref, S); });
            const double t_g_new = time_us([&] { attn_gate_q8_planes_prompt(da2, dyq2, kProj, Rq, act_new, S); });
            std::printf("  attn_gate_q8_planes_prompt %d rows: %.1f us vs unfused %.1f us (%.2fx)\n", Rq, t_g_new, t_g_old, t_g_old / t_g_new);

            for (void* q : {(void*) dx2, (void*) dw2, (void*) dn, (void*) da2, (void*) dyq2, (void*) dg2,
                            act_ref, act_new, (void*) wE.planes, (void*) wE.sc_t, (void*) wO.planes, (void*) wO.sc_t,
                            (void*) yref, (void*) ynew})
                cudaFree(q);
        }
        {   // fused ctrl_down_prompt + ctrl_up_blend_prompt vs unfused (ctrl_pair_norm + bf16_rows_dot + ctrl_blend) and CPU
            const int Rc = 2048;
            const std::vector<float> hd = randn((size_t) Rc * E, 1.0f), hm = randn((size_t) Rc * E, 1.0f);
            float *d_dest = dev(hd), *d_msg = dev(hm), *d_pair = zeros<float>((size_t) Rc * K2);
            float *d_h_ref = zeros<float>((size_t) Rc * rank), *d_h_new = zeros<float>((size_t) Rc * rank);
            float *d_o_ref = zeros<float>((size_t) Rc * E), *d_o_new = zeros<float>((size_t) Rc * E);
            ctrl_pair_norm(d_dest, d_msg, Rc, d_pair, S);
            bk::bf16_rows_dot(d_pair, ddn, d_h_ref, Rc, rank, K2, S);
            bk::bf16_rows_dot(d_h_ref, dup, d_pair, Rc, K2, rank, S);
            ctrl_blend(d_dest, d_msg, d_pair, Rc, d_o_ref, S);
            ctrl_down_prompt(d_dest, d_msg, ddn, Rc, rank, d_h_new, S);
            ctrl_up_blend_prompt(d_dest, d_msg, d_h_new, dup, Rc, rank, d_o_new, S);
            report("ctrl_down_prompt vs unfused", host(d_h_new, (size_t) Rc * rank), host(d_h_ref, (size_t) Rc * rank), 5e-5);
            report("ctrl_up_blend_prompt vs unfused", host(d_o_new, (size_t) Rc * E), host(d_o_ref, (size_t) Rc * E), 5e-5);
            // Position invariance: slice rows [37..100) alone must give identical bits
            const int off_r = 37, sub_r = 64;
            float *d_h_sub = zeros<float>((size_t) sub_r * rank), *d_o_sub = zeros<float>((size_t) sub_r * E);
            ctrl_down_prompt(d_dest + (size_t) off_r * E, d_msg + (size_t) off_r * E, ddn, sub_r, rank, d_h_sub, S);
            ctrl_up_blend_prompt(d_dest + (size_t) off_r * E, d_msg + (size_t) off_r * E, d_h_sub, dup, sub_r, rank, d_o_sub, S);
            const auto full_o = host(d_o_new, (size_t) Rc * E), sub_o = host(d_o_sub, (size_t) sub_r * E);
            same_bits("ctrl_prompt rows 37..100 alone", full_o.data() + (size_t) off_r * E, sub_o.data(), sub_o.size() * 4);
            // CPU double reference on first 8 rows
            std::vector<double> ref_o((size_t) 8 * E);
            for (int r = 0; r < 8; ++r) {
                double sd = 0, sm = 0;
                for (int c = 0; c < E; ++c) { sd += (double) hd[(size_t) r * E + c] * hd[(size_t) r * E + c]; sm += (double) hm[(size_t) r * E + c] * hm[(size_t) r * E + c]; }
                const double id = 1.0 / std::sqrt(sd / E + 1e-6), im = 1.0 / std::sqrt(sm / E + 1e-6);
                std::vector<double> hrow(rank, 0.0);
                auto bf2d = [](uint16_t u) { uint32_t v = (uint32_t) u << 16; float f; std::memcpy(&f, &v, 4); return (double) f; };
                for (int o = 0; o < rank; ++o) {
                    double s = 0;
                    for (int c = 0; c < E; ++c) s += bf2d(wdn[(size_t) o * K2 + c]) * hd[(size_t) r * E + c] * id;
                    for (int c = 0; c < E; ++c) s += bf2d(wdn[(size_t) o * K2 + E + c]) * hm[(size_t) r * E + c] * im;
                    hrow[o] = s;
                }
                for (int c = 0; c < E; ++c) {
                    double sg = 0, sf = 0;
                    for (int o = 0; o < rank; ++o) {
                        sg += bf2d(wup[(size_t) c * rank + o]) * hrow[o];
                        sf += bf2d(wup[(size_t) (E + c) * rank + o]) * hrow[o];
                    }
                    ref_o[(size_t) r * E + c] = (1.0 - std::tanh(sf)) * hd[(size_t) r * E + c] + std::tanh(sg) * hm[(size_t) r * E + c];
                }
            }
            report("ctrl_prompt vs CPU double (8 rows)", host(d_o_new, (size_t) 8 * E), ref_o, 5e-5);
            const double t_old = time_us([&] {
                ctrl_pair_norm(d_dest, d_msg, Rc, d_pair, S);
                bk::bf16_rows_dot(d_pair, ddn, d_h_ref, Rc, rank, K2, S);
                bk::bf16_rows_dot(d_h_ref, dup, d_pair, Rc, K2, rank, S);
                ctrl_blend(d_dest, d_msg, d_pair, Rc, d_o_ref, S);
            });
            const double t_new = time_us([&] {
                ctrl_down_prompt(d_dest, d_msg, ddn, Rc, rank, d_h_new, S);
                ctrl_up_blend_prompt(d_dest, d_msg, d_h_new, dup, Rc, rank, d_o_new, S);
            });
            std::printf("  controller fused at %d rows: %.1f us vs unfused %.1f us (%.2fx)\n", Rc, t_new, t_old, t_old / t_new);
            cudaFree(d_dest); cudaFree(d_msg); cudaFree(d_pair);
            cudaFree(d_h_ref); cudaFree(d_h_new); cudaFree(d_o_ref); cudaFree(d_o_new);
            cudaFree(d_h_sub); cudaFree(d_o_sub);
        }
        cudaFree(ddn);
        cudaFree(dup);
        cudaFree(dx);
        cudaFree(h1);
        cudaFree(h2);
        cudaFree(o1);
        cudaFree(o2);
    }
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    cudaStreamCreate(&S);
    test_dense();
    test_attention_prompt();
    test_stage_to_ring();
    test_router();
    test_sort();
    test_swiglu();
    test_experts(40);
    test_experts_fp4();
    test_experts(8);
    std::printf("inkling2_prompt_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
