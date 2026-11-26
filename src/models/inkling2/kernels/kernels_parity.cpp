// src/models/inkling2/kernels/kernels_parity.cpp - the Inkling 2 kernels against plain CPU references (double accumulation)
// at the model's geometry: the block-residual mixes, the loop controllers' norm and blend, the attention preparation
// (short convolutions, clips, cache writes), the attention with its relative bias and sink over a window ring and a full
// cache (fp16, E4M3 and INT8 caches), the post-sublayer convolutions, the experts with the softcapped SwiGLU and their weighted sum, and the softcap;
// and the decode window's fused kernels against the references and the kernels they replace, their q8_1 blocks against a
// CPU rounding of their own fp32 rows. Every kernel is checked for a row's bits not depending on the rows beside it, and
// timed. Exit 77 without a GPU.
#include "basalt/core/device_copy.hpp"
#include "../cpu_expert_drain.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
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
std::vector<float> fixed_values(size_t n, float scale, uint32_t state) {
    std::vector<float> v(n);
    for (float& x : v) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        x = ((float) (state >> 8) * (1.f / 16777216.f) * 2.f - 1.f) * scale;
    }
    return v;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    basalt::core::device_copy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    basalt::core::device_copy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
// Device memory of `n` bytes filled with 0xa5: what a kernel leaves unwritten shows.
uint8_t* poisoned(size_t n) {
    uint8_t* p = nullptr;
    cudaMalloc(&p, n + 16);
    cudaMemset(p, 0xa5, n + 16);
    return p;
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
// An FNV-1a hash of a result's bytes: a change meant to keep the bits keeps it.
uint64_t print_hash(const char* what, const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : b) h = (h ^ x) * 1099511628211ull;
    std::printf("  %s hash %016llx\n", what, (unsigned long long) h);
    return h;
}
template <class T> std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
    return std::vector<uint8_t>(p, p + v.size() * sizeof(T));
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

// The cache formats: fp16, E4M3 codes (fp8), INT8 codes with an fp16 scale per 64 channels (int8).
enum Kv { kF16, kE4M3, kI8 };
const Kv kAllKv[] = {kF16, kE4M3, kI8};
size_t kv_bytes(Kv kv) { return (size_t) kv_cache_row_bytes(kv == kE4M3, kv == kI8); }
const char* kv_tag(Kv kv) { return kv == kE4M3 ? ", E4M3" : kv == kI8 ? ", INT8" : ""; }
void kv_set(Kv kv, AttnPrep& a) { a.fp8 = kv == kE4M3; a.int8 = kv == kI8; }
void kv_set(Kv kv, AttnArgs& a) { a.fp8 = kv == kE4M3; a.int8 = kv == kI8; }
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
                const __nv_fp8_e4m3 q(x[d]);
                row[d] = q.__x;
                got[d] = (double) (float) q;
            } else {
                const uint16_t h = f2h(x[d]);
                std::memcpy(row + 2 * d, &h, 2);
                got[d] = h2d(h);
            }
        }
    }
}
// `rows` head rows of N(0, s) values in format kv; `vals` (when set) the values they hold, kHeadDim a row.
std::vector<uint8_t> kv_rows(Kv kv, size_t rows, float s, std::vector<double>* vals = nullptr) {
    std::vector<uint8_t> b(rows * kv_bytes(kv), 0);
    std::vector<double> got(kHeadDim);
    if (vals) vals->resize(rows * kHeadDim);
    for (size_t i = 0; i < rows; ++i) {
        const std::vector<float> x = randn(kHeadDim, s);
        put_row(kv, x.data(), &b[i * kv_bytes(kv)], vals ? &(*vals)[i * kHeadDim] : got.data());
    }
    return b;
}
// The values row `i` of a cache in format kv holds.
void get_row(Kv kv, const uint8_t* rows, size_t i, double* out) {
    const uint8_t* r = rows + i * kv_bytes(kv);
    for (int d = 0; d < kHeadDim; ++d) {
        if (kv == kI8) {
            uint16_t sh;
            std::memcpy(&sh, r + kHeadDim + 2 * (d / 64), 2);
            out[d] = (double) (int8_t) r[d] * h2d(sh);
        } else if (kv == kE4M3) {
            __nv_fp8_e4m3 q;
            q.__x = r[d];
            out[d] = (double) (float) q;
        } else {
            uint16_t h;
            std::memcpy(&h, r + 2 * d, 2);
            out[d] = h2d(h);
        }
    }
}

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
        float *q = zeros<float>((size_t) R * kHeads * kHeadDim), *rr = zeros<float>((size_t) R * kHeads * kRel), *sink = zeros<float>((size_t) R * kHeads);
        AttnPrep a;
        a.yq = dyq; a.yk = dyk; a.yv = dyv; a.pos = dpos; a.full = full; a.k_conv = dkc; a.v_conv = dvc; a.q_scale = dqs; a.r_scale = drs;
        a.k_ring = dkr; a.v_ring = dvr; a.cap = cap; a.q = q; a.r = rr; a.sink = sink;
        // keys and values: the reference
        std::vector<double> rk, rv;
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
                    for (int d = 0; d < kHeadDim; ++d) (which ? rv : rk).push_back(val[d] * f);
                }
            }
        // each cache format from the same rings, into poisoned caches: the values its cells hold against the reference
        // (INT8: within half a step of its group's scale, and the row's padding written as zeros)
        for (const Kv kv : kAllKv) {
            const size_t kvb = (size_t) kKvHeads * cap * kv_bytes(kv);
            uint8_t *kcache = poisoned(kvb), *vcache = poisoned(kvb);
            basalt::core::device_copy(dkr, kr.data(), kr.size() * 4, cudaMemcpyHostToDevice);
            basalt::core::device_copy(dvr, vr.data(), vr.size() * 4, cudaMemcpyHostToDevice);
            a.k_cache = kcache; a.v_cache = vcache;
            kv_set(kv, a);
            attn_prepare(a, R, nullptr);
            cudaDeviceSynchronize();
            const std::vector<uint8_t> hk = host(kcache, kvb), hv = host(vcache, kvb);
            std::vector<float> gk, gv;
            bool pad = true;
            std::vector<double> row(kHeadDim);
            for (int r = 0; r < R; ++r)
                for (int which = 0; which < 2; ++which)
                    for (int h = 0; h < kKvHeads; ++h) {
                        const size_t cell = (size_t) (pos[r] % cap) * kKvHeads + h;
                        get_row(kv, (which ? hv : hk).data(), cell, row.data());
                        for (int d = 0; d < kHeadDim; ++d) (which ? gv : gk).push_back((float) row[d]);
                        for (size_t b = kHeadDim + 4; kv == kI8 && b < kv_bytes(kv); ++b) pad = pad && (which ? hv : hk)[cell * kv_bytes(kv) + b] == 0;
                    }
            const double tol = kv == kI8 ? 4e-3 : kv == kE4M3 ? 7e-2 : 1e-3;
            char tag[64];
            std::snprintf(tag, sizeof tag, "attn_prepare keys (%s%s)", full ? "full" : "window", kv == kF16 ? ", fp16" : kv_tag(kv));
            report(tag, gk, rk, tol);
            std::snprintf(tag, sizeof tag, "attn_prepare values (%s%s)", full ? "full" : "window", kv == kF16 ? ", fp16" : kv_tag(kv));
            report(tag, gv, rv, tol);
            if (kv == kI8) {
                std::printf("%-34s %s\n", "  INT8 rows' padding zeroed", pad ? "ok" : "FAIL");
                failures += !pad;
            }
            if (!full) std::printf("  %.1f us a call (1 row%s)\n", time_us([&] { attn_prepare(a, 1, nullptr); }), kv_tag(kv));
            cudaFree(kcache);
            cudaFree(vcache);
        }
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
    }
}

// ---- attention ---------------------------------------------------------------------------------------------------------

void test_attention() {
    // a window ring, a full cache, a full cache of E4M3 codes, a full cache and a window ring of INT8 codes
    for (int mode = 0; mode < 5; ++mode) {
        const bool window = mode == 0 || mode == 4;
        const Kv kv = mode == 2 ? kE4M3 : mode >= 3 ? kI8 : kF16;
        const int R = 4;
        const int p0 = window ? 2900 : 2995;
        const int64_t cap = window ? kExtent + 8 : 3000;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        std::vector<double> kd, vd;   // the cached values as the kernel reads them, [cell][kKvHeads][kHeadDim]
        std::vector<uint8_t> kb, vb;
        {
            const size_t rows = (size_t) kKvHeads * cap;
            kb.assign(rows * kv_bytes(kv), 0);
            vb.assign(rows * kv_bytes(kv), 0);
            kd.resize(rows * kHeadDim);
            vd.resize(rows * kHeadDim);
            const std::vector<float> ksrc = fixed_values(rows * kHeadDim, 0.6f, 0x21a48d3bu);
            const std::vector<float> vsrc = fixed_values(rows * kHeadDim, 1.0f, 0x7b51c2e9u);
            for (size_t i = 0; i < rows; ++i) {
                put_row(kv, &ksrc[i * kHeadDim], &kb[i * kv_bytes(kv)], &kd[i * kHeadDim]);
                put_row(kv, &vsrc[i * kHeadDim], &vb[i * kv_bytes(kv)], &vd[i * kHeadDim]);
            }
        }
        const std::vector<float> q = fixed_values((size_t) R * kHeads * kHeadDim, 0.8f, 0x44b13a97u);
        const std::vector<float> rs = fixed_values((size_t) R * kHeads * kRel, 1.0f, 0x2f9d08c5u);
        const std::vector<float> sink = fixed_values((size_t) R * kHeads, 2.0f, 0xd650a317u);
        const std::vector<float> table = fixed_values((size_t) kRel * kExtent, 0.3f, 0x94ce725bu);
        const std::vector<float> gate = fixed_values((size_t) R * kProjQ, 1.5f, 0xa17e4c03u);
        void* dk = dev(kb);
        void* dv = dev(vb);
        float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table), *out = zeros<float>((size_t) R * kHeads * kHeadDim);
        float* dgate = dev(gate);
        int* dpos = dev(pos);
        AttnArgs a;
        a.q = dq; a.r = drs; a.sink = dsink; a.rel_table = dt; a.pos = dpos; a.k_cache = dk; a.v_cache = dv; a.cap = cap;
        kv_set(kv, a);
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
        char tag[64];
        std::snprintf(tag, sizeof tag, "attention (%s%s)", window ? "window ring" : "full, 3k keys", kv_tag(kv));
        report(tag, got, ref, 2e-3);
        const uint64_t hash = print_hash("output", bytes_of(got));
        // Frozen outputs from the FP16/FP8 kernels on the fixed attention fixture.
        static constexpr uint64_t kBaseHashes[] = {0x72538aec1379b868ull, 0x59167d639fa4a77bull, 0x3460547e2a0576d1ull};
        if (mode < 3) {
            const bool baseline = hash == kBaseHashes[mode];
            std::printf("  previous FP16/FP8 baseline %s\n", baseline ? "matches" : "FAIL");
            failures += !baseline;
        }
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
                    keys * kKvHeads * kv_bytes(kv) * 2 / us / 1e3);
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

// heat_fold: the kept rows' lookups of every visit, against a host count; only the first `rows` rows of each visit count,
// whatever the row stride, and two folds add up
void test_heat_fold() {
    std::printf("heat_fold\n");
    constexpr int V = 48, R = kDecodeRows, K = 8, NE = 128, L = 24;
    std::uniform_int_distribution<int> pick_e(0, NE - 1), pick_l(0, L - 1);
    std::vector<int32_t> ids((size_t) V * R * K), layer(V);
    for (auto& e : ids) e = pick_e(rng);
    for (auto& l : layer) l = pick_l(rng);
    int32_t* di = dev(ids);
    int32_t* dl = dev(layer);
    bool ok = true;
    for (int rows = 1; rows <= R && ok; ++rows) {
        uint32_t* tally = zeros<uint32_t>((size_t) L * NE);
        heat_fold(di, dl, V, R, rows, K, NE, tally, nullptr);
        heat_fold(di, dl, V, R, rows, K, NE, tally, nullptr);
        const std::vector<uint32_t> got = host(tally, (size_t) L * NE);
        std::vector<uint32_t> want((size_t) L * NE, 0);
        for (int v = 0; v < V; ++v)
            for (int i = 0; i < rows * K; ++i) want[(size_t) layer[v] * NE + ids[(size_t) v * R * K + i]] += 2;
        ok = got == want;
        cudaFree(tally);
    }
    std::printf("%-34s %s\n", "  kept rows' lookups (1..8 rows)", ok ? "counted" : "FAIL");
    failures += !ok;
    uint32_t* tally = zeros<uint32_t>((size_t) L * NE);
    const double us = time_us([&] { heat_fold(di, dl, V, R, 4, K, NE, tally, nullptr); });
    std::printf("  %.1f us a call (48 visits, 4 rows: %.1f GB/s of ids)\n", us, (double) V * 4 * K * 4 / us / 1e3);
    cudaFree(tally);
    cudaFree(di);
    cudaFree(dl);
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
    for (const Kv kv : kAllKv)
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
        kv_set(kv, a);
        const size_t kvb = (size_t) kKvHeads * cap * kv_bytes(kv);
        AttnPrep b = a;
        for (AttnPrep* x : {&a, &b}) {
            x->k_ring = dev(kr); x->v_ring = dev(vr);
            x->k_cache = poisoned(kvb); x->v_cache = poisoned(kvb);
            x->q = zeros<float>((size_t) R * kHeads * kHeadDim); x->r = zeros<float>((size_t) R * kHeads * kRel); x->sink = zeros<float>((size_t) R * kHeads);
        }
        decode_prepare(a, R, nullptr);
        attn_prepare(b, R, nullptr);
        cudaDeviceSynchronize();
        char tag[64];
        std::snprintf(tag, sizeof tag, "decode_prepare (%s%s)", full ? "full" : "window", kv_tag(kv));
        bool same = !std::memcmp(host((uint8_t*) a.k_cache, kvb).data(), host((uint8_t*) b.k_cache, kvb).data(), kvb) &&
                    !std::memcmp(host((uint8_t*) a.v_cache, kvb).data(), host((uint8_t*) b.v_cache, kvb).data(), kvb) &&
                    host(a.q, (size_t) R * kHeads * kHeadDim) == host(b.q, (size_t) R * kHeads * kHeadDim) &&
                    host(a.r, (size_t) R * kHeads * kRel) == host(b.r, (size_t) R * kHeads * kRel) &&
                    host(a.sink, (size_t) R * kHeads) == host(b.sink, (size_t) R * kHeads) &&
                    host(a.k_ring, (size_t) kConvRing * 1024) == host(b.k_ring, (size_t) kConvRing * 1024) &&
                    host(a.v_ring, (size_t) kConvRing * 1024) == host(b.v_ring, (size_t) kConvRing * 1024);
        std::printf("%-34s %s\n", tag, same ? "bitwise = attn_prepare" : "FAIL");
        failures += !same;
        if (!full) std::printf("  %.1f us a call (1 row%s)\n", time_us([&] { decode_prepare(a, 1, nullptr); }), kv_tag(kv));
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
    for (const Kv kv : kAllKv)
        for (const Case& k : cases) {
            const int R = k.R;
            std::vector<int> pos(R);
            for (int r = 0; r < R; ++r) pos[r] = k.p0 + r;
            std::vector<uint8_t> kc = kv_rows(kv, (size_t) kKvHeads * k.cap, 0.6f), vc = kv_rows(kv, (size_t) kKvHeads * k.cap, 1.0f);
            if (kv == kI8)   // the padding poisoned: the rows the window writes must come out zero-padded
                for (size_t i = 0; i < kc.size(); ++i)
                    if (i % kv_bytes(kv) >= kHeadDim + 4) kc[i] = vc[i] = 0xa5;
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
                p.k_ring = dev(kr); p.v_ring = dev(vr); p.k_cache = dev(kc); p.v_cache = dev(vc); p.cap = k.cap;
                kv_set(kv, p);
                p.q = zeros<float>((size_t) R * kHeads * kHeadDim); p.r = zeros<float>((size_t) R * kHeads * kRel);
                p.sink = zeros<float>((size_t) R * kHeads);
                AttnArgs& a = aa[i];
                a.q = p.q; a.r = p.r; a.sink = p.sink; a.rel_table = dev(table); a.pos = dpos; a.k_cache = p.k_cache; a.v_cache = p.v_cache;
                a.cap = k.cap;
                kv_set(kv, a);
                a.window = k.window ? kExtent : 0; a.gate = dy + kOffGate; a.gate_ld = kProj;
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
            const size_t q8b = (size_t) R * kHeads * kHeadDim / 32 * 36, kvb = kc.size(), ringb = (size_t) kConvRing * 1024;
            const bool same = host((uint8_t*) aa[0].q8, q8b) == host((uint8_t*) aa[1].q8, q8b) &&
                              host((uint8_t*) pa[0].k_cache, kvb) == host((uint8_t*) pa[1].k_cache, kvb) &&
                              host((uint8_t*) pa[0].v_cache, kvb) == host((uint8_t*) pa[1].v_cache, kvb) &&
                              host(pa[0].k_ring, ringb) == host(pa[1].k_ring, ringb) && host(pa[0].v_ring, ringb) == host(pa[1].v_ring, ringb);
            char tag[96];
            std::snprintf(tag, sizeof tag, "attention_window (%s%s, %d rows at %d)", k.window ? "window" : "full", kv_tag(kv), R, k.p0);
            std::printf("%-46s %s\n", tag, same ? "bitwise = prepare + attention" : "FAIL");
            failures += !same;
            if (kv == kI8) {
                bool pad = true;
                for (const void* c : {pa[1].k_cache, pa[1].v_cache}) {
                    const std::vector<uint8_t> b = host((const uint8_t*) c, kvb);
                    for (int r = 0; r < R; ++r)
                        for (int h = 0; h < kKvHeads; ++h)
                            for (size_t x = kHeadDim + 4; x < kv_bytes(kv); ++x)
                                pad = pad && b[((size_t) (pos[r] % k.cap) * kKvHeads + h) * kv_bytes(kv) + x] == 0;
                }
                std::printf("%-46s %s\n", "  the written rows' padding zeroed", pad ? "ok" : "FAIL");
                failures += !pad;
            }
            std::vector<uint8_t> all = host((uint8_t*) aa[1].q8, q8b);
            for (const void* c : {pa[1].k_cache, pa[1].v_cache}) {
                std::vector<uint8_t> b = host((const uint8_t*) c, kvb);
                if (kv == kI8)   // the hash covers codes and scales: the padding is checked above
                    for (size_t i = 0; i < b.size(); ++i)
                        if (i % kv_bytes(kv) >= kHeadDim + 4) b[i] = 0;
                all.insert(all.end(), b.begin(), b.end());
            }
            print_hash("q8_1 output and caches", all);
            if (k.window && R == 1) {
                std::printf("  %.1f us a call (1 row, 1 key, fused)  %.1f us (prepare + attention)%s\n",
                            time_us([&] { attention_window(pa[1], aa[1], 1, max_keys, scr, nullptr); }),
                            time_us([&] { decode_prepare(pa[0], 1, nullptr); attention_decode(aa[0], 1, max_keys, scr, nullptr, nullptr); }), kv_tag(kv));
            }
            if (k.window && R == 3) {
                std::printf("  %.1f us a call (1 row, 301 keys, fused)  %.1f us (prepare + attention)%s\n",
                            time_us([&] { attention_window(pa[1], aa[1], 1, max_keys, scr, nullptr); }),
                            time_us([&] { decode_prepare(pa[0], 1, nullptr); attention_decode(aa[0], 1, max_keys, scr, nullptr, nullptr); }), kv_tag(kv));
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
    // experts in RAM (0 in the table, the CPU's) are skipped: some of them, then all
    for (int all : {0, 1}) {
        for (int e = 0; e < NE; ++e) tab[e] = all || e % 2 ? 0ull : (unsigned long long) (blobs + (size_t) e * blob);
        basalt::core::device_copy(dtab, tab.data(), NE * sizeof(tab[0]), cudaMemcpyHostToDevice);
        expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr);
        const bool skip_ok = cudaDeviceSynchronize() == cudaSuccess;
        std::printf("%-34s %s\n", all ? "expert_prefetch: every one in RAM" : "expert_prefetch: some in RAM", skip_ok ? "ok" : "FAIL");
        failures += !skip_ok;
    }
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

// The CPU experts' rows into the experts' output (moe_cpu_merge): from mapped host memory a host function filled, in a
// captured graph replayed twice; the pairs whose expert is in RAM (0 in the table) take their rows and their lookups, the
// card's pairs keep theirs. Every expert on the card, none, and every other one, for 1 to 8 rows.
void test_cpu_merge() {
    struct Fill {
        float* data = nullptr;
        size_t n = 0;
        const float* src = nullptr;
        int replay = 0;
        static void CUDART_CB run(void* p) {
            const Fill& f = *static_cast<const Fill*>(p);
            for (size_t i = 0; i < f.n; ++i) f.data[i] = f.src[i] + (float) f.replay * 0.125f;
        }
    };
    const int K = 8, NE = 128;
    bool ok = true;
    for (int rows = 1; rows <= kDecodeRows; ++rows)
        for (int mode : {0, 1, 2}) {
            const int pairs = rows * K;
            std::vector<int32_t> ids((size_t) pairs);
            for (int p = 0; p < pairs; ++p) ids[(size_t) p] = (p % K * 7 + p / K) % NE;
            std::vector<unsigned long long> tab(NE);
            for (int e = 0; e < NE; ++e) tab[(size_t) e] = mode == 0 ? 1 : mode == 1 ? 0 : e % 2;
            const std::vector<float> before((size_t) pairs * E, -17.f), src = randn((size_t) pairs * E);
            int32_t* di = dev(ids);
            unsigned long long* dt = dev(tab);
            float* dy = dev(before);
            uint32_t* tally = zeros<uint32_t>(NE);
            Fill fill;
            fill.n = (size_t) pairs * E;
            fill.src = src.data();
            float* mapped = nullptr;
            cudaHostAlloc((void**) &fill.data, fill.n * 4, cudaHostAllocMapped);
            cudaHostGetDevicePointer((void**) &mapped, fill.data, 0);
            cudaStream_t st;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            cudaLaunchHostFunc(st, Fill::run, &fill);
            moe_cpu_merge(di, rows, K, dt, mapped, dy, tally, st);
            cudaStreamEndCapture(st, &graph);
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            for (int replay = 1; replay <= 2; ++replay) {
                fill.replay = replay;
                cudaGraphLaunch(exec, st);
                ok &= cudaStreamSynchronize(st) == cudaSuccess;
                const std::vector<float> y = host(dy, (size_t) pairs * E);
                const std::vector<uint32_t> counts = host(tally, NE);
                std::vector<uint32_t> want(NE, 0);
                for (int p = 0; p < pairs; ++p) {
                    const bool ram = !tab[(size_t) ids[(size_t) p]];
                    if (ram) want[(size_t) ids[(size_t) p]] += replay;
                    for (int c = 0; c < E; ++c) {
                        const size_t i = (size_t) p * E + c;
                        const float ref = ram ? src[i] + (float) replay * 0.125f : -17.f;
                        ok &= std::memcmp(&y[i], &ref, 4) == 0;
                    }
                }
                ok &= counts == want;
            }
            if (mode == 1 && rows == kDecodeRows)
                std::printf("  %.1f us a call (%d rows from mapped memory)\n",
                            time_us([&] { moe_cpu_merge(di, rows, K, dt, mapped, dy, nullptr, nullptr); }), rows);
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(st);
            cudaFreeHost(fill.data);
        }
    std::printf("%-34s %s\n", "moe_cpu_merge: graph, mapped rows", ok ? "ok" : "FAIL");
    failures += !ok;
}

// A host thread serving the doorbell as cpu_experts.cpp does: a raised bell taken (cleared), then after `delay_us` each
// pair whose expert is off the card gets row (x_m row * 0.5 + its expert + the replay) in `out`, then `done`.
struct BellServer {
    CpuBell* bell = nullptr;
    const float* x_m = nullptr;
    const int32_t* ids_m = nullptr;
    float* out = nullptr;
    const unsigned long long* tab = nullptr;   // host copy of the blob table
    int k = 0, delay_us = 0, layer = -1, rows = -1;
    std::atomic<int> replay{0}, rings{0};
    std::atomic<bool> stop{false};
    std::thread t;
    void start() {
        t = std::thread([this] {
            std::atomic_ref<uint32_t> ring(bell->ring), ticket(bell->ticket), fin(bell->done);
            while (!stop.load(std::memory_order_relaxed)) {
                const uint32_t w = ring.load(std::memory_order_acquire);
                if (!w) continue;
                const uint32_t visit = ticket.load(std::memory_order_relaxed);
                ring.store(0, std::memory_order_relaxed);
                layer = (int) (w >> 8 & 0xffu);
                rows = (int) (w >> 16);
                if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
                for (int p = 0; p < rows * k; ++p)
                    if (!tab[ids_m[p]])
                        for (int c = 0; c < E; ++c)
                            out[(size_t) p * E + c] = x_m[(size_t) (p / k) * E + c] * 0.5f + (float) ids_m[p] + (float) replay.load();
                rings.fetch_add(1);
                fin.store(visit, std::memory_order_release);
            }
        });
    }
    void end() {
        stop.store(true);
        t.join();
    }
};

// Replay a ring/merge graph with a delayed host. Only RAM visits advance the ticket and publish their choices and
// RAM rows. The merge takes the host's rows only for the matching ticket in each replay.
void test_cpu_ring() {
    const int K = 8, NE = 128, L = 37;
    CpuBell *bell = nullptr, *bell_d = nullptr;
    float *x_m = nullptr, *x_md = nullptr, *out_m = nullptr, *out_md = nullptr;
    int32_t *ids_m = nullptr, *ids_md = nullptr;
    cudaHostAlloc((void**) &bell, sizeof(CpuBell), cudaHostAllocMapped);
    cudaHostAlloc((void**) &x_m, (size_t) kDecodeRows * E * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &ids_m, (size_t) kDecodeRows * K * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &out_m, (size_t) kDecodeRows * K * E * 4, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**) &bell_d, bell, 0);
    cudaHostGetDevicePointer((void**) &x_md, x_m, 0);
    cudaHostGetDevicePointer((void**) &ids_md, ids_m, 0);
    cudaHostGetDevicePointer((void**) &out_md, out_m, 0);
    uint32_t* seq = nullptr;
    cudaMalloc(&seq, sizeof(*seq));
    bool ok = true;
    for (int rows = 1; rows <= kDecodeRows; ++rows)
        for (int mode : {0, 1, 2, 3}) {
            const int pairs = rows * K;
            std::vector<int32_t> ids((size_t) pairs);
            for (int p = 0; p < pairs; ++p) ids[(size_t) p] = (p % K * 7 + p / K) % NE;
            std::vector<unsigned long long> tab(NE);
            for (int e = 0; e < NE; ++e) tab[(size_t) e] = mode == 0 ? 1 : mode == 1 ? 0 : mode == 2 ? e % 2 : 1;
            if (mode == 3) tab[(size_t) ids[(size_t) pairs - 5]] = 0;
            const std::vector<float> x = randn((size_t) rows * E), before((size_t) pairs * E, -17.f);
            int32_t* di = dev(ids);
            unsigned long long* dt = dev(tab);
            float *dx = dev(x), *dy = dev(before);
            uint32_t* tally = zeros<uint32_t>(NE);
            auto poison = [&] {
                std::memset(x_m, 0xff, (size_t) kDecodeRows * E * 4);
                std::memset(ids_m, 0xff, (size_t) kDecodeRows * K * 4);
                std::memset(out_m, 0xff, (size_t) kDecodeRows * K * E * 4);
            };
            *bell = CpuBell{};
            bell->done = UINT32_MAX;   // a stale completion, not a ticket of any replay below
            BellServer sv;
            sv.bell = bell;
            sv.x_m = x_m;
            sv.ids_m = ids_m;
            sv.out = out_m;
            sv.tab = tab.data();
            sv.k = K;
            sv.delay_us = 200;
            sv.start();
            cudaStream_t st;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaMemsetAsync(seq, 0, sizeof(*seq), st);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            moe_cpu_ring(di, rows, K, dt, dx, L, x_md, ids_md, bell_d, seq, st);
            moe_cpu_merge(di, rows, K, dt, out_md, dy, tally, st, bell_d, seq);
            cudaStreamEndCapture(st, &graph);
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            std::vector<uint32_t> want(NE, 0);
            int served = 0, last_rows = 0;
            for (int replay = 1; replay <= 4; ++replay) {
                // a different input every replay: fresh activations and choices (the same experts off the card, a rotated
                // pairing), the mapped buffers and the output poisoned first, so a stale payload or a stale row shows
                std::vector<int32_t> gids((size_t) pairs);
                for (int p = 0; p < pairs; ++p) gids[(size_t) p] = (ids[(size_t) p] + 5 * (replay - 1) * (p % 3)) % NE;
                const std::vector<float> gx = randn((size_t) rows * E);
                uint32_t grows = 0;
                for (int p = 0; p < pairs; ++p)
                    if (!tab[(size_t) gids[(size_t) p]]) grows |= 1u << (p / K);
                basalt::core::device_copy(di, gids.data(), gids.size() * 4, cudaMemcpyHostToDevice);
                basalt::core::device_copy(dx, gx.data(), gx.size() * 4, cudaMemcpyHostToDevice);
                basalt::core::device_copy(dy, before.data(), before.size() * 4, cudaMemcpyHostToDevice);
                poison();
                sv.replay.store(replay);
                cudaGraphLaunch(exec, st);
                ok &= cudaStreamSynchronize(st) == cudaSuccess;
                if (grows) {
                    ++served;
                    last_rows = rows;
                }
                ok &= sv.rings.load() == served;
                const std::vector<float> y = host(dy, (size_t) pairs * E);
                const std::vector<uint32_t> counts = host(tally, NE);
                for (int p = 0; p < pairs; ++p) {
                    const int e = gids[(size_t) p];
                    const bool ram = !tab[(size_t) e];
                    if (ram) ++want[(size_t) e];
                    for (int c = 0; c < E; ++c) {
                        const float ref = ram ? gx[(size_t) (p / K) * E + c] * 0.5f + (float) e + (float) replay : -17.f;
                        ok &= std::memcmp(&y[(size_t) p * E + c], &ref, 4) == 0;
                    }
                }
                ok &= counts == want;
                for (int r = 0; r < kDecodeRows; ++r)   // the payload is this replay's, rows and choices, bit for bit
                    for (int c = 0; c < E; ++c) {
                        const float got = x_m[(size_t) r * E + c];
                        uint32_t bits;
                        std::memcpy(&bits, &got, 4);
                        ok &= grows >> r & 1 ? std::memcmp(&got, &gx[(size_t) r * E + c], 4) == 0 : bits == 0xffffffffu;
                    }
                for (int p = 0; p < kDecodeRows * K; ++p) ok &= ids_m[p] == (grows && p < pairs ? gids[(size_t) p] : -1);

                // The merge without a bell takes the same completed host rows without the hand-off.
                cudaMemcpyAsync(dy, before.data(), before.size() * sizeof(float), cudaMemcpyHostToDevice, st);
                moe_cpu_merge(di, rows, K, dt, out_md, dy, nullptr, st);
                ok &= cudaStreamSynchronize(st) == cudaSuccess;
                const auto direct = host(dy, y.size());
                ok &= std::memcmp(direct.data(), y.data(), y.size() * sizeof(float)) == 0;

                if (replay == 4 && rows > 1) {
                    // Replay the same inputs one row at a time, independent of their position in the window.
                    for (int r = 0; r < rows; ++r) {
                        constexpr size_t row_values = (size_t) K * E;
                        poison();
                        cudaMemcpyAsync(dy, before.data(), row_values * sizeof(float), cudaMemcpyHostToDevice, st);
                        moe_cpu_ring(di + r * K, 1, K, dt, dx + (size_t) r * E, L, x_md, ids_md, bell_d, seq, st);
                        moe_cpu_merge(di + r * K, 1, K, dt, out_md, dy, nullptr, st, bell_d, seq);
                        ok &= cudaStreamSynchronize(st) == cudaSuccess;
                        const auto single = host(dy, row_values);
                        ok &= std::memcmp(single.data(), y.data() + (size_t) r * row_values, row_values * sizeof(float)) == 0;
                        if (grows >> r & 1) {
                            ++served;
                            last_rows = 1;
                        }
                    }
                }
            }
            sv.end();
            const uint32_t expected = (uint32_t) served;
            ok &= bell->ring == 0 && bell->ticket == expected && bell->abort == 0;
            ok &= bell->done == (served ? expected : UINT32_MAX) && host(seq, 1)[0] == expected;
            ok &= sv.rings.load() == served;
            if (served) ok &= sv.layer == L && sv.rows == last_rows;
            if (rows == 1 && mode != 2) {
                std::printf("  moe_cpu_ring %.1f us a call (1 row, %s)\n",
                            time_us([&] { moe_cpu_ring(di, 1, K, dt, dx, L, x_md, ids_md, bell_d, seq, nullptr); }),
                            mode == 0 ? "every expert on the card: no bell" : mode == 1 ? "every pair off the card" : "one pair off the card");
                bell->ring = 0;
            }
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(st);
            for (void* q : {(void*) di, (void*) dt, (void*) dx, (void*) dy, (void*) tally}) cudaFree(q);
        }
    std::printf("%-34s %s\n", "moe_cpu_ring + merge: doorbell", ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(seq);
    cudaFreeHost(bell);
    cudaFreeHost(x_m);
    cudaFreeHost(ids_m);
    cudaFreeHost(out_m);
}

// The bell aborted (what cpu_experts.cpp does for an engine that ends): a merge waiting for a visit nobody serves runs out
// within a bounded time with nothing taken, and the visits queued behind it, or after an abort, publish no payload or bell
// and take nothing; an abort after a served window changes nothing already taken.
void test_cpu_abort() {
    const int K = 8, NE = 128, L = 3, rows = 2, pairs = rows * K, visits = 3;
    CpuBell *bell = nullptr, *bell_d = nullptr;
    float *x_m = nullptr, *x_md = nullptr, *out_m = nullptr, *out_md = nullptr;
    int32_t *ids_m = nullptr, *ids_md = nullptr;
    cudaHostAlloc((void**) &bell, sizeof(CpuBell), cudaHostAllocMapped);
    cudaHostAlloc((void**) &x_m, (size_t) kDecodeRows * E * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &ids_m, (size_t) kDecodeRows * K * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &out_m, (size_t) kDecodeRows * K * E * 4, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**) &bell_d, bell, 0);
    cudaHostGetDevicePointer((void**) &x_md, x_m, 0);
    cudaHostGetDevicePointer((void**) &ids_md, ids_m, 0);
    cudaHostGetDevicePointer((void**) &out_md, out_m, 0);
    std::vector<int32_t> ids((size_t) pairs);
    for (int p = 0; p < pairs; ++p) ids[(size_t) p] = (p % K * 7 + p / K) % NE;
    std::vector<unsigned long long> tab(NE);
    for (int e = 0; e < NE; ++e) tab[(size_t) e] = e % 2;   // the even experts are off the card
    const std::vector<float> x = randn((size_t) rows * E), before((size_t) pairs * E, -17.f);
    int32_t* di = dev(ids);
    unsigned long long* dt = dev(tab);
    float *dx = dev(x), *dy = dev(before);
    uint32_t* tally = zeros<uint32_t>(NE);
    uint32_t* seq = nullptr;
    cudaMalloc(&seq, sizeof(*seq));
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    struct LateCompletion {
        CpuBell* bell;
        std::atomic<uint32_t> ticket{0};
        void publish() const {
            std::atomic_ref<uint32_t>(bell->done).store(ticket.load(std::memory_order_acquire), std::memory_order_release);
        }
    } late{bell};
    auto capture = [&](int n, cudaGraph_t& graph, cudaGraphExec_t& exec, LateCompletion* completion = nullptr) {
        cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        for (int v = 0; v < n; ++v) {
            moe_cpu_ring(di, rows, K, dt, dx, L, x_md, ids_md, bell_d, seq, st);
            if (completion)
                cudaLaunchHostFunc(st, +[](void* p) { static_cast<LateCompletion*>(p)->publish(); }, completion);
            moe_cpu_merge(di, rows, K, dt, out_md, dy, tally, st, bell_d, seq);
        }
        cudaStreamEndCapture(st, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    };
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    capture(visits, graph, exec);
    auto poison_payload = [&] {
        std::memset(x_m, 0xff, (size_t) kDecodeRows * E * 4);
        std::memset(ids_m, 0xff, (size_t) kDecodeRows * K * 4);
    };
    auto payload_untouched = [&] {
        const auto* bytes = reinterpret_cast<const unsigned char*>(x_m);
        return std::all_of(bytes, bytes + (size_t) kDecodeRows * E * sizeof(*x_m),
                           [](unsigned char v) { return v == 0xff; }) &&
               std::all_of(ids_m, ids_m + kDecodeRows * K, [](int32_t v) { return v == -1; });
    };
    auto reset = [&] {
        *bell = CpuBell{};
        poison_payload();
        std::memset(out_m, 0xff, (size_t) kDecodeRows * K * E * 4);
        cudaMemcpyAsync(dy, before.data(), before.size() * 4, cudaMemcpyHostToDevice, st);
        cudaMemsetAsync(tally, 0, NE * 4, st);
        cudaMemsetAsync(seq, 0, sizeof(*seq), st);
    };
    auto require_drain = [&](const char* where) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        do {
            const cudaError_t status = cudaStreamQuery(st);
            if (status == cudaSuccess) return;
            if (status != cudaErrorNotReady) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        std::printf("moe_cpu_ring + merge: abort FAIL (%s did not drain)\n", where);
        std::fflush(stdout);
        std::_Exit(1);
    };
    auto abort_bell = [&] { std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release); };
    auto wait_for_ring = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        do {
            if (std::atomic_ref<uint32_t>(bell->ring).load(std::memory_order_acquire)) return true;
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        return false;
    };
    auto untouched = [&] {
        const std::vector<float> y = host(dy, (size_t) pairs * E);
        const std::vector<uint32_t> counts = host(tally, NE);
        bool same = std::all_of(counts.begin(), counts.end(), [](uint32_t c) { return c == 0; });
        for (float v : y) same = same && v == -17.f;
        return same;
    };
    bool ok = true;
    // 1. no one serves the bell: the first visit's merge waits; the abort ends it and the visits behind it
    reset();
    cudaGraphLaunch(exec, st);
    ok &= wait_for_ring() && cudaStreamQuery(st) == cudaErrorNotReady;
    abort_bell();
    require_drain("unanswered visit");
    ok &= untouched();
    ok &= bell->ring != 0 && bell->ticket == 1 && bell->done == 0;   // the first visit's bell, never taken
    ok &= host(seq, 1)[0] == visits;
    // 2. aborted before queueing: no payload or bell; every merge leaves when it polls abort
    reset();
    abort_bell();
    cudaGraphLaunch(exec, st);
    require_drain("pre-aborted visits");
    ok &= untouched() && payload_untouched() && bell->ring == 0 && bell->ticket == 0 && bell->done == 0;
    ok &= host(seq, 1)[0] == visits;
    // 3. a served window, then the abort: what was taken stays, the next window takes nothing
    reset();
    BellServer sv;
    sv.bell = bell;
    sv.x_m = x_m;
    sv.ids_m = ids_m;
    sv.out = out_m;
    sv.tab = tab.data();
    sv.k = K;
    sv.replay.store(1);
    sv.start();
    cudaGraphLaunch(exec, st);
    require_drain("served visits");
    const std::vector<uint32_t> served = host(tally, NE);
    uint32_t total = 0;
    for (uint32_t c : served) total += c;
    int off = 0;
    for (int p = 0; p < pairs; ++p) off += !tab[(size_t) ids[(size_t) p]];
    ok &= total == (uint32_t) off * visits && sv.rings.load() == visits;
    abort_bell();
    const int rung = sv.rings.load();
    cudaGraphLaunch(exec, st);
    require_drain("visits queued after a served graph");
    sv.end();
    ok &= sv.rings.load() == rung && host(tally, NE) == served;
    ok &= bell->ring == 0 && bell->ticket == visits && bell->done == visits && host(seq, 1)[0] == 2 * visits;
    std::printf("%-34s %s\n", "moe_cpu_ring + merge: abort", ok ? "ok" : "FAIL");
    failures += !ok;

    cudaGraph_t single_graph, ordered_graph;
    cudaGraphExec_t single_exec, ordered_exec;
    capture(1, single_graph, single_exec);
    capture(1, ordered_graph, ordered_exec, &late);
    enum class Publication { before_launch, delayed, after_ring };
    bool late_ok = true;
    for (Publication when : {Publication::before_launch, Publication::delayed, Publication::after_ring}) {
        reset();
        cudaGraphLaunch(single_exec, st);
        const bool rang = wait_for_ring();
        const uint32_t t_a = rang ? std::atomic_ref<uint32_t>(bell->ticket).load(std::memory_order_relaxed) : 0;
        if (rang) std::atomic_ref<uint32_t>(bell->ring).store(0, std::memory_order_relaxed);
        abort_bell();
        require_drain("claimed visit A");
        late_ok &= rang && t_a == 1 && untouched() && bell->done == 0 && host(seq, 1)[0] == t_a;
        if (!rang) continue;

        poison_payload();
        late.ticket.store(t_a, std::memory_order_release);
        std::fill_n(out_m, (size_t) pairs * E, 11.f);   // A's completed rows
        if (when == Publication::before_launch) late.publish();
        // The host node in ordered_exec publishes after B's ring and before its first merge poll.
        cudaGraphLaunch(when == Publication::after_ring ? ordered_exec : single_exec, st);
        std::thread writer;
        if (when == Publication::delayed) {
            const auto publish_at = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            writer = std::thread([&] {
                std::this_thread::sleep_until(publish_at);
                late.publish();
            });
            writer.join();
        }
        require_drain("visit B after late completion");
        late_ok &= untouched() && payload_untouched() && bell->ring == 0 && bell->ticket == t_a;
        late_ok &= bell->done == t_a && host(seq, 1)[0] == t_a + 1;
    }
    std::printf("%-34s %s\n", "moe_cpu_ring + merge: late done", late_ok ? "ok" : "FAIL");
    failures += !late_ok;

    // The abort through the admission gate CpuExperts keeps: a drain in its wait while a call holds admission behind a
    // merge waiting for an unserved visit returns only after the bell's release lets the call sync and leave (a drain
    // that returned earlier would report the call in flight), the call's results are discarded, and no call enters after
    // the close; a call still in flight at the bound, one whose sync failed, or one that left a fork unjoined leaves
    // completion unconfirmed.
    auto& unconfirmed = basalt::core::gpu_left_running();
    const bool previous = unconfirmed.exchange(false);
    bool gate_ok = true;
    {
        WindowGate gate;
        reset();
        cudaStreamSynchronize(st);   // the reset is queued on st
        gate_ok &= gate.enter();
        cudaGraphLaunch(exec, st);
        gate_ok &= wait_for_ring();
        bool kept = true;
        std::thread call([&] {
            const cudaError_t synced = cudaStreamSynchronize(st);   // the merge waits until the bell is aborted
            kept = gate.leave(synced, true);
        });
        std::atomic<bool> drained{false};
        std::thread drain([&] {
            gate.close();
            gate.drain(5000);
            drained.store(true);
        });
        while (!gate.draining()) std::this_thread::yield();
        const bool waited = !drained.load();   // the call cannot leave before the release below
        abort_bell();
        drain.join();
        call.join();
        gate_ok &= waited && !kept && !gate.enter() && !unconfirmed.exchange(false);
    }
    {
        WindowGate gate;
        gate_ok &= gate.enter();
        gate.close();
        gate.drain(20);
        gate_ok &= unconfirmed.exchange(false);
        gate.leave(cudaSuccess, true);
    }
    {
        WindowGate gate;
        gate_ok &= gate.enter();
        gate.close();
        gate.leave(cudaErrorLaunchFailure, true);
        gate.drain(20);
        gate_ok &= unconfirmed.exchange(false);
    }
    {
        WindowGate gate;
        gate_ok &= gate.enter();
        gate.close();
        gate.leave(cudaSuccess, false);
        gate.drain(20);
        gate_ok &= unconfirmed.exchange(false);
    }
    unconfirmed.store(previous);
    std::printf("%-34s %s\n", "CPU abort gate: calls in flight", gate_ok ? "ok" : "FAIL");
    failures += !gate_ok;
    cudaGraphExecDestroy(ordered_exec);
    cudaGraphDestroy(ordered_graph);
    cudaGraphExecDestroy(single_exec);
    cudaGraphDestroy(single_graph);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(st);
    for (void* q : {(void*) di, (void*) dt, (void*) dx, (void*) dy, (void*) tally}) cudaFree(q);
    cudaFree(seq);
    cudaFreeHost(bell);
    cudaFreeHost(x_m);
    cudaFreeHost(ids_m);
    cudaFreeHost(out_m);
}

// Hand-off time from graphs of visits with no CPU expert work: copies plus a host callback versus the doorbell,
// and doorbell-only timings for visits with no RAM pairs or several RAM pairs.
void time_cpu_handoff() {
    const int K = 8, NE = 128, V = 48, R = 1, wide_rows = 3;
    std::vector<int32_t> ids(wide_rows * K);
    for (int p = 0; p < wide_rows * K; ++p) ids[(size_t) p] = (p % K) * 9;
    ids.back() = NE - 1;
    std::vector<unsigned long long> tab(NE, 1), all(NE, 1);
    tab[(size_t) ids[3]] = 0;
    auto wide_tab = tab;
    wide_tab[(size_t) ids.back()] = 0;   // one RAM choice per row and one more in the last row
    int32_t* di = dev(ids);
    unsigned long long *dt = dev(tab), *da = dev(all), *dw = dev(wide_tab);
    auto x = randn(E);
    x.resize((size_t) wide_rows * E);
    for (int r = 1; r < wide_rows; ++r) std::copy_n(x.data(), E, x.data() + (size_t) r * E);
    float *dx = dev(x), *dy = zeros<float>((size_t) wide_rows * K * E);
    CpuBell *bell = nullptr, *bell_d = nullptr;
    float *x_m = nullptr, *x_md = nullptr, *out_m = nullptr, *out_md = nullptr, *x_h = nullptr;
    int32_t *ids_m = nullptr, *ids_md = nullptr, *ids_h = nullptr;
    cudaHostAlloc((void**) &bell, sizeof(CpuBell), cudaHostAllocMapped);
    cudaHostAlloc((void**) &x_m, (size_t) wide_rows * E * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &ids_m, (size_t) wide_rows * K * 4, cudaHostAllocMapped);
    cudaHostAlloc((void**) &out_m, (size_t) wide_rows * K * E * 4, cudaHostAllocMapped);
    cudaMallocHost((void**) &x_h, (size_t) E * 4);
    cudaMallocHost((void**) &ids_h, (size_t) K * 4);
    cudaHostGetDevicePointer((void**) &bell_d, bell, 0);
    cudaHostGetDevicePointer((void**) &x_md, x_m, 0);
    cudaHostGetDevicePointer((void**) &ids_md, ids_m, 0);
    cudaHostGetDevicePointer((void**) &out_md, out_m, 0);
    *bell = CpuBell{};
    std::memset(out_m, 0, (size_t) wide_rows * K * E * 4);
    cudaStream_t st, side;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking);
    uint32_t* seq = nullptr;
    cudaMalloc(&seq, sizeof(*seq));
    cudaMemsetAsync(seq, 0, sizeof(*seq), side);
    cudaStreamSynchronize(side);
    cudaEvent_t ready, done;
    cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
    auto noop = [](void*) {};
    auto graph_us = [&](const std::function<void()>& visit) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        for (int v = 0; v < V; ++v) visit();
        cudaStreamEndCapture(st, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        const double us = time_us([&] { cudaGraphLaunch(exec, st); }, 40) / V;
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return us;
    };
    const double copies = graph_us([&] {
        cudaEventRecord(ready, st);
        cudaStreamWaitEvent(side, ready, 0);
        cudaMemcpyAsync(x_h, dx, (size_t) R * E * 4, cudaMemcpyDeviceToHost, side);
        cudaMemcpyAsync(ids_h, di, (size_t) R * K * 4, cudaMemcpyDeviceToHost, side);
        cudaLaunchHostFunc(side, +noop, nullptr);
        cudaEventRecord(done, side);
        cudaStreamWaitEvent(st, done, 0);
        moe_cpu_merge(di, R, K, dt, out_md, dy, nullptr, st);
    });
    BellServer sv;
    sv.bell = bell;
    sv.x_m = x_m;
    sv.ids_m = ids_m;
    sv.out = out_m;
    sv.tab = all.data();   // no rows to make: the bell's round trip alone
    sv.k = K;
    sv.start();
    auto bell_visit = [&](const unsigned long long* t, int rows) {
        cudaEventRecord(ready, st);
        cudaStreamWaitEvent(side, ready, 0);
        moe_cpu_ring(di, rows, K, t, dx, 5, x_md, ids_md, bell_d, seq, side);
        cudaEventRecord(done, side);
        cudaStreamWaitEvent(st, done, 0);
        moe_cpu_merge(di, rows, K, t, out_md, dy, nullptr, st, bell_d, seq);
    };
    const double rung = graph_us([&] { bell_visit(dt, R); });
    const double quiet = graph_us([&] { bell_visit(da, R); });
    const double wide = graph_us([&] { bell_visit(dw, wide_rows); });
    sv.end();
    std::printf("  hand-off a visit (1 row, 1 pair off the card, no CPU work): copies + host function %.1f us, doorbell %.1f us, "
                "doorbell with no pair off the card %.1f us\n", copies, rung, quiet);
    std::printf("  hand-off a visit (3 rows, 4 pairs off the card, no CPU work): doorbell %.1f us\n", wide);
    cudaFree(seq);
    cudaEventDestroy(ready);
    cudaEventDestroy(done);
    cudaStreamDestroy(st);
    cudaStreamDestroy(side);
    for (void* q : {(void*) bell, (void*) x_m, (void*) ids_m, (void*) out_m, (void*) x_h, (void*) ids_h}) cudaFreeHost(q);
    for (void* q : {(void*) di, (void*) dt, (void*) da, (void*) dw, (void*) dx, (void*) dy}) cudaFree(q);
}

// A batch window (several conversations' rows in one call, docs/design/batching.md) against each conversation's own call:
// the attention's q8_1 output, caches and rings, post_attn's and moe_finish's outputs and rings, bitwise. Three segments
// of 3, 1 and 2 rows (one at position 1, one past the window ring's wrap), over a window ring and a full cache, fp16 and E4M3.
void test_batch_window() {
    struct Seg { int p0, T; };
    const Seg segs[3] = {{300, 3}, {1, 1}, {2900, 2}};
    const int S = 3, R = 6, row0[S + 1] = {0, 3, 4, 6};
    Segments sg;
    sg.at.n = S;
    for (int i = 0; i <= S; ++i) sg.at.row0[i] = row0[i];
    std::vector<int> pos(R);
    for (int i = 0; i < S; ++i)
        for (int t = 0; t < segs[i].T; ++t) pos[row0[i] + t] = segs[i].p0 + t;
    int* dpos = dev(pos);
    const size_t q8_row = (size_t) kHeads * kHeadDim / 32 * 36;
    for (const Kv kv : kAllKv)
        for (bool window : {true, false}) {
            const int64_t cap = window ? kExtent + 8 : 4096;
            const size_t kvb = (size_t) kKvHeads * cap * kv_bytes(kv), ringb = (size_t) kConvRing * 1024;
            std::vector<std::vector<uint8_t>> kc(S), vc(S);
            std::vector<std::vector<float>> kr(S), vr(S);
            for (int i = 0; i < S; ++i) {
                kc[i] = kv_rows(kv, (size_t) kKvHeads * cap, 0.6f);
                vc[i] = kv_rows(kv, (size_t) kKvHeads * cap, 1.0f);
                kr[i] = randn(ringb, 2.0f);
                vr[i] = randn(ringb, 0.6f);
            }
            const std::vector<float> y = randn((size_t) R * kProj, 1.5f);
            const std::vector<float> tk = randn(1024 * kConv, 0.3f), tv = randn(1024 * kConv, 0.3f);
            const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rsc = randn(kHeads * kRel, 0.2f, 1.f);
            const std::vector<float> table = randn((size_t) kRel * kExtent, 0.3f);
            float* dy = dev(y);
            const auto prep_of = [&](int r0, int rows, const int* p) {
                AttnPrep a;
                a.yq = dy + (size_t) r0 * kProj; a.yk = a.yq + kProjQ; a.yv = a.yq + kProjQ + kProjKV; a.ldq = a.ldk = a.ldv = kProj;
                a.pos = p; a.full = !window; a.k_conv = dev(tk); a.v_conv = dev(tv); a.q_scale = dev(qs); a.r_scale = dev(rsc);
                a.cap = cap;
                kv_set(kv, a);
                a.q = zeros<float>((size_t) rows * kHeads * kHeadDim); a.r = zeros<float>((size_t) rows * kHeads * kRel);
                a.sink = zeros<float>((size_t) rows * kHeads);
                return a;
            };
            const auto args_of = [&](const AttnPrep& p, int rows) {
                AttnArgs a;
                a.q = p.q; a.r = p.r; a.sink = p.sink; a.rel_table = dev(table); a.pos = p.pos; a.k_cache = p.k_cache; a.v_cache = p.v_cache;
                a.cap = cap;
                kv_set(kv, a);
                a.window = window ? kExtent : 0; a.gate = p.yq + kOffGate; a.gate_ld = kProj;
                a.q8 = zeros<uint8_t>((size_t) rows * q8_row);
                return a;
            };
            const auto keys_of = [&](int64_t last) {
                int64_t mk = 1024;
                while (mk < last + 1) mk <<= 1;
                return window ? (int64_t) kExtent : mk;
            };
            // each conversation alone (its own call: the fused kernel for these rows), then all of them in one call
            std::vector<AttnPrep> ps(S);
            std::vector<AttnArgs> as(S);
            for (int i = 0; i < S; ++i) {
                ps[i] = prep_of(row0[i], segs[i].T, dpos + row0[i]);
                ps[i].k_ring = dev(kr[i]); ps[i].v_ring = dev(vr[i]); ps[i].k_cache = dev(kc[i]); ps[i].v_cache = dev(vc[i]);
                as[i] = args_of(ps[i], segs[i].T);
                const int64_t mk = keys_of(segs[i].p0 + segs[i].T - 1);
                attention_window(ps[i], as[i], segs[i].T, mk, zeros<uint8_t>(attention_decode_scratch(segs[i].T, mk)), nullptr);
            }
            AttnPrep pb = prep_of(0, R, dpos);
            pb.seg = sg;
            for (int i = 0; i < S; ++i) {
                SegCaches& c = pb.seg.c[i];
                c.k = dev(kc[i]); c.v = dev(vc[i]); c.k_ring = dev(kr[i]); c.v_ring = dev(vr[i]);
            }
            pb.k_cache = pb.seg.c[0].k; pb.v_cache = pb.seg.c[0].v; pb.k_ring = pb.seg.c[0].k_ring; pb.v_ring = pb.seg.c[0].v_ring;
            AttnArgs ab = args_of(pb, R);
            ab.seg = pb.seg;
            const int64_t mkb = keys_of(2901);
            attention_window(pb, ab, R, mkb, zeros<uint8_t>(attention_decode_scratch(R, mkb)), nullptr);
            cudaDeviceSynchronize();
            bool same = true;
            const std::vector<uint8_t> q8b = host((uint8_t*) ab.q8, (size_t) R * q8_row);
            for (int i = 0; i < S; ++i) {
                const std::vector<uint8_t> q8s = host((uint8_t*) as[i].q8, (size_t) segs[i].T * q8_row);
                same = same && !std::memcmp(q8s.data(), q8b.data() + (size_t) row0[i] * q8_row, q8s.size()) &&
                       host((uint8_t*) ps[i].k_cache, kvb) == host((uint8_t*) pb.seg.c[i].k, kvb) &&
                       host((uint8_t*) ps[i].v_cache, kvb) == host((uint8_t*) pb.seg.c[i].v, kvb) &&
                       host(ps[i].k_ring, ringb) == host(pb.seg.c[i].k_ring, ringb) && host(ps[i].v_ring, ringb) == host(pb.seg.c[i].v_ring, ringb);
            }
            char tag[96];
            std::snprintf(tag, sizeof tag, "batch window: attention (%s%s)", window ? "window" : "full", kv_tag(kv));
            std::printf("%-46s %s\n", tag, same ? "bitwise = each conversation alone" : "FAIL");
            failures += !same;
            print_hash("q8_1 output", q8b);
        }
    // post_attn and moe_finish: each conversation's ring and its own rows before it
    const int nb = 2, K = 8;
    const std::vector<float> o = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f), hid = randn((size_t) R * E, 1.0f),
                             q = randn(E, 0.05f), w = randn(E, 0.1f, 1.f), xr = randn((size_t) R * K * E, 0.3f), xw = randn((size_t) R * K, 0.3f, 1.f),
                             mt = randn((size_t) E * kConv, 0.3f), mres = randn((size_t) R * E, 1.0f);
    std::vector<std::vector<float>> blocks(nb), ar(S), mr(S);
    for (int i = 0; i < nb; ++i) blocks[i] = randn((size_t) R * E, 0.7f + 0.5f * i);
    for (int i = 0; i < S; ++i) {
        ar[i] = randn((size_t) kConvRing * E, 0.4f);
        mr[i] = randn((size_t) kConvRing * E, 0.4f);
    }
    float *d_o = dev(o), *d_hid = dev(hid), *d_xr = dev(xr), *d_xw = dev(xw), *d_mres = dev(mres);
    std::vector<float*> d_blocks;
    for (auto& b : blocks) d_blocks.push_back(dev(b));
    const auto post_of = [&](int r0, int rows) {
        PostAttn a;
        for (int i = 0; i < nb; ++i) a.blocks[a.n_blocks++] = d_blocks[i] + (size_t) r0 * E;
        a.o = d_o + (size_t) r0 * E; a.pos = dpos + r0; a.taps = dev(taps); a.hid = d_hid + (size_t) r0 * E;
        a.resid = zeros<float>((size_t) rows * E); a.query = dev(q); a.norm = dev(w);
        a.q8 = zeros<uint8_t>((size_t) rows * E / 32 * 36); a.norm_out = zeros<float>((size_t) rows * E);
        return a;
    };
    const auto moe_of = [&](int r0, int rows) {
        MoeFinish a;
        a.rows = d_xr + (size_t) r0 * K * E; a.w = d_xw + (size_t) r0 * K; a.k = K; a.pos = dpos + r0; a.taps = dev(mt);
        a.resid = d_mres + (size_t) r0 * E; a.out = zeros<float>((size_t) rows * E);
        return a;
    };
    std::vector<PostAttn> pa(S);
    std::vector<MoeFinish> ma(S);
    for (int i = 0; i < S; ++i) {
        pa[i] = post_of(row0[i], segs[i].T);
        pa[i].ring = dev(ar[i]);
        post_attn(pa[i], segs[i].T, nullptr);
        ma[i] = moe_of(row0[i], segs[i].T);
        ma[i].ring = dev(mr[i]);
        moe_finish(ma[i], segs[i].T, nullptr);
    }
    PostAttn pb = post_of(0, R);
    MoeFinish mb = moe_of(0, R);
    pb.seg = mb.seg = sg;
    for (int i = 0; i < S; ++i) {
        pb.seg.c[i].a_ring = dev(ar[i]);
        mb.seg.c[i].m_ring = dev(mr[i]);
    }
    pb.ring = pb.seg.c[0].a_ring;
    mb.ring = mb.seg.c[0].m_ring;
    post_attn(pb, R, nullptr);
    moe_finish(mb, R, nullptr);
    cudaDeviceSynchronize();
    bool post_same = true, moe_same = true;
    const size_t ring_e = (size_t) kConvRing * E;
    const std::vector<float> nb_all = host(pb.norm_out, (size_t) R * E), res_all = host(pb.resid, (size_t) R * E), out_all = host(mb.out, (size_t) R * E);
    const std::vector<uint8_t> q8_all = host((uint8_t*) pb.q8, (size_t) R * E / 32 * 36);
    for (int i = 0; i < S; ++i) {
        const size_t n = (size_t) segs[i].T * E, at = (size_t) row0[i] * E;
        post_same = post_same && !std::memcmp(host(pa[i].norm_out, n).data(), nb_all.data() + at, n * 4) &&
                    !std::memcmp(host(pa[i].resid, n).data(), res_all.data() + at, n * 4) &&
                    !std::memcmp(host((uint8_t*) pa[i].q8, n / 32 * 36).data(), q8_all.data() + at / 32 * 36, n / 32 * 36) &&
                    host(pa[i].ring, ring_e) == host(pb.seg.c[i].a_ring, ring_e);
        moe_same = moe_same && !std::memcmp(host(ma[i].out, n).data(), out_all.data() + at, n * 4) &&
                   host(ma[i].ring, ring_e) == host(mb.seg.c[i].m_ring, ring_e);
    }
    std::printf("%-46s %s\n", "batch window: post_attn", post_same ? "bitwise = each conversation alone" : "FAIL");
    std::printf("%-46s %s\n", "batch window: moe_finish", moe_same ? "bitwise = each conversation alone" : "FAIL");
    failures += !post_same + !moe_same;
}


// Multi-row sliding-window decode across a 64-key chunk boundary at position 2,556..2,563 (where early rows read the tail of
// chunk 23 while later rows advance p_max into chunk 40): parity with CPU double and bitwise identity with single-row decode.
void test_attention_window_boundary() {
    for (const Kv kv : {kF16, kE4M3}) {
        const int R = 8, p0 = 2556, cap = kExtent + 8;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        const size_t rows = (size_t) kKvHeads * cap;
        std::vector<uint8_t> kb(rows * kv_bytes(kv), 0), vb(rows * kv_bytes(kv), 0);
        std::vector<double> kd(rows * kHeadDim), vd(rows * kHeadDim);
        const std::vector<float> ksrc = fixed_values(rows * kHeadDim, 0.6f, 0x31c59a7du);
        const std::vector<float> vsrc = fixed_values(rows * kHeadDim, 1.0f, 0x58e27b41u);
        for (size_t i = 0; i < rows; ++i) {
            put_row(kv, &ksrc[i * kHeadDim], &kb[i * kv_bytes(kv)], &kd[i * kHeadDim]);
            put_row(kv, &vsrc[i * kHeadDim], &vb[i * kv_bytes(kv)], &vd[i * kHeadDim]);
        }
        const std::vector<float> q = fixed_values((size_t) R * kHeads * kHeadDim, 0.8f, 0x62d48f19u);
        const std::vector<float> rs = fixed_values((size_t) R * kHeads * kRel, 1.0f, 0x19a35c87u);
        const std::vector<float> sink = fixed_values((size_t) R * kHeads, 2.0f, 0x8b41e623u);
        const std::vector<float> table = fixed_values((size_t) kRel * kExtent, 0.3f, 0x74f0b2d5u);
        void *dk = dev(kb), *dv = dev(vb);
        float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table);
        float* out = zeros<float>((size_t) R * kHeads * kHeadDim);
        int* dpos = dev(pos);
        AttnArgs a;
        a.q = dq; a.r = drs; a.sink = dsink; a.rel_table = dt; a.pos = dpos; a.k_cache = dk; a.v_cache = dv; a.cap = cap;
        kv_set(kv, a);
        a.window = kExtent;
        void* scr = zeros<uint8_t>(attention_decode_scratch(R, kExtent));
        attention_decode(a, R, kExtent, scr, out, nullptr);
        const std::vector<float> got = host(out, (size_t) R * kHeads * kHeadDim);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r)
            for (int h = 0; h < kHeads; ++h) {
                const int g = h / 3, lo = std::max(0, pos[r] - kExtent + 1);
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
                for (double& sv : sc) den += (sv = std::exp(sv - m));
                for (int i = 0; i < kHeadDim; ++i) {
                    double acc = 0;
                    for (int k = lo; k <= pos[r]; ++k) acc += sc[(size_t) (k - lo)] * vd[((size_t) (k % cap) * kKvHeads + g) * kHeadDim + i];
                    ref[((size_t) r * kHeads + h) * kHeadDim + i] = acc / den;
                }
            }
        char tag[64];
        std::snprintf(tag, sizeof tag, "attention (window boundary, 8 rows%s)", kv_tag(kv));
        report(tag, got, ref, 2e-3);
        float* out1 = zeros<float>((size_t) R * kHeads * kHeadDim);
        for (int r = 0; r < R; ++r) {
            AttnArgs one = a;
            one.q = dq + (size_t) r * kHeads * kHeadDim;
            one.r = drs + (size_t) r * kHeads * kRel;
            one.sink = dsink + (size_t) r * kHeads;
            one.pos = dpos + r;
            attention_decode(one, 1, kExtent, scr, out1 + (size_t) r * kHeads * kHeadDim, nullptr);
        }
        const std::vector<float> solo = host(out1, got.size());
        same_bits("  8-row batch vs each row alone", solo.data(), got.data(), got.size() * 4);
        cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(drs); cudaFree(dsink); cudaFree(dt);
        cudaFree(out); cudaFree(out1); cudaFree(dpos); cudaFree(scr);
    }
}

void test_router_select_plan() {
    std::printf("router_select_plan (rows = 1..64)\n");
    constexpr int H = kWidth, NE = 128, K = 8, RMAX = kBatchRows;
    const std::vector<float> x = randn((size_t) RMAX * H, 0.5f), gate = randn((size_t) NE * H, 0.05f), bias = randn(NE, 0.1f);
    std::vector<unsigned long long> table(NE);
    for (int e = 0; e < NE; ++e) table[(size_t) e] = e == 17 ? 0ull : (unsigned long long) (0x100000ull + (uint64_t) e * 4096ull);
    float *dx = dev(x), *dg = dev(gate), *db = dev(bias), *dlog = zeros<float>(RMAX * NE), *dlog1 = zeros<float>(RMAX * NE);
    float *dw_ref = zeros<float>(RMAX * K), *dw_got = zeros<float>(RMAX * K);
    int32_t *di_ref = zeros<int32_t>(RMAX * K), *di_got = zeros<int32_t>(RMAX * K);
    unsigned long long *dt = dev(table), *p_ref = zeros<unsigned long long>(NE), *p_got = zeros<unsigned long long>(NE);
    unsigned long long *rb_ref = zeros<unsigned long long>(kAheadExperts), *rb_got = zeros<unsigned long long>(kAheadExperts);
    int32_t *s_ref = zeros<int32_t>(NE + 2), *s_got = zeros<int32_t>(NE + 2), *ng_ref = zeros<int32_t>(1), *ng_got = zeros<int32_t>(1);
    int32_t *dst_ref = zeros<int32_t>(RMAX * K), *dst_got = zeros<int32_t>(RMAX * K);
    int32_t *tok_ref = zeros<int32_t>(RMAX * K), *tok_got = zeros<int32_t>(RMAX * K);
    int32_t *rn_ref = zeros<int32_t>(1), *rn_got = zeros<int32_t>(1);
    uint32_t *tl_ref = zeros<uint32_t>(NE), *tl_got = zeros<uint32_t>(NE);
    bk::RouterArgs ra;
    ra.n_expert = NE;
    ra.n_embd = H;
    ra.k = K;
    ra.weight_scale = 2.5f;
    ra.gate = bk::RouterGate::Sigmoid;
    for (int r = 0; r < RMAX; ++r) bk::router_logits(dx + (size_t) r * H, dg, 1, ra, dlog1 + (size_t) r * NE, nullptr);
    const std::vector<float> ref_log = host(dlog1, RMAX * NE);
    bool ok = true;
    for (int R : {1, 2, 7, 16, 35, 64}) {
        bk::router_logits(dx, dg, R, ra, dlog, nullptr);
        const std::vector<float> got_log = host(dlog, R * NE);
        ok = ok && std::memcmp(ref_log.data(), got_log.data(), (size_t) R * NE * 4) == 0;
        cudaMemset(tl_ref, 0, NE * 4);
        cudaMemset(tl_got, 0, NE * 4);
        bk::router_select(dlog1, db, R, ra, di_ref, dw_ref, nullptr);
        moe_plan(di_ref, R, K, NE, dt, p_ref, s_ref, ng_ref, dst_ref, tok_ref, nullptr, tl_ref, rb_ref, rn_ref);
        router_select_plan(dlog, db, R, K, NE, ra.weight_scale, di_got, dw_got, dt, p_got, s_got, ng_got, dst_got, tok_got, nullptr,
                           tl_got, rb_got, rn_got);
        cudaDeviceSynchronize();
        const int g_r = host(ng_ref, 1)[0], g_g = host(ng_got, 1)[0];
        const auto sr = host(s_ref, g_r + 2), sg = host(s_got, g_g + 2);
        const int tot = sr[(size_t) g_r];
        ok = ok && g_r == g_g && sr == sg &&
             host(di_ref, R * K) == host(di_got, R * K) &&
             std::memcmp(host(dw_ref, R * K).data(), host(dw_got, R * K).data(), (size_t) R * K * 4) == 0 &&
             host(p_ref, g_r) == host(p_got, g_g) &&
             host(dst_ref, tot) == host(dst_got, tot) &&
             host(tok_ref, tot) == host(tok_got, tot) &&
             host(tl_ref, NE) == host(tl_got, NE) &&
             host(rb_ref, kAheadExperts) == host(rb_got, kAheadExperts) &&
             host(rn_ref, 1) == host(rn_got, 1);
    }
    std::printf("%-34s %s\n", "  logits + select + plan (1..64)", ok ? "bitwise" : "FAIL");
    failures += !ok;
    cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dlog); cudaFree(dlog1);
    cudaFree(dw_ref); cudaFree(dw_got); cudaFree(di_ref); cudaFree(di_got);
    cudaFree(dt); cudaFree(p_ref); cudaFree(p_got); cudaFree(rb_ref); cudaFree(rb_got);
    cudaFree(s_ref); cudaFree(s_got); cudaFree(ng_ref); cudaFree(ng_got);
    cudaFree(dst_ref); cudaFree(dst_got); cudaFree(tok_ref); cudaFree(tok_got);
    cudaFree(rn_ref); cudaFree(rn_got); cudaFree(tl_ref); cudaFree(tl_got);
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
    test_cpu_merge();
    test_cpu_ring();
    test_cpu_abort();
    time_cpu_handoff();
    test_heat_fold();
    test_batch_window();
    test_attention_window_boundary();
    test_router_select_plan();
    std::printf("inkling2_kernels_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
