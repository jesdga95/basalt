// tests/kernels/iq1s_parity.cpp - IQ1_S (and the IQ2_XXS / IQ1_S and NVFP4 expert pairs of a limited-SwiGLU MoE) on the GPU
// and CPU expert paths, at the real expert geometry (hidden 5120, FF 2304), against a CPU double reference.
//
//     build/iq1s_parity            the checks (a ctest; needs the GPU)
//     build/iq1s_parity --cpu      the CPU tier only (no GPU needed)
//     build/iq1s_parity --bench    the checks, then a microbench line per kernel
//
// Random bytes are valid IQ1_S / IQ2_XXS codes (every grid index is in range); only the fp16 block scales are set.  NVFP4
// experts are ggml's NVFP4 rounding of normal values with an FP32 scale tail {s_gate, s_up, s_down, 0}, which the
// reference applies to the dequantized weights.
// Reference: the weights dequantized with the ggml row dequantizer, the activation as the kernel sees it (q8_1 or
// Q8_K blocks dequantized), accumulated in double; the SwiGLU limit applied in double.  The GPU / CPU outputs of one
// entry must not depend on the window: the same entry alone, in a window and at another position is bitwise equal.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"

#ifdef BASALT_IQ1S_GPU
#include "basalt/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace kc = basalt::kernels::cpu;
#ifdef BASALT_IQ1S_GPU
#endif

constexpr int H = 5120, FF = 2304, K = 6;
constexpr float kLimit = 10.0f;
int g_fail = 0;

struct Err {
    double sq = 0, ref_sq = 0, max_abs = 0, max_rel = 0, ref_max = 0;
    void add(double got, double ref) {
        const double d = got - ref;
        sq += d * d;
        ref_sq += ref * ref;
        max_abs = std::max(max_abs, std::fabs(d));
        ref_max = std::max(ref_max, std::fabs(ref));
    }
    double rms() const { return std::sqrt(sq / (ref_sq + 1e-30)); }
    // worst error relative to the largest reference value (single-element relative error is meaningless near 0)
    double rel_max() const { return max_abs / (ref_max + 1e-30); }
};

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

uint16_t f16_bits(float lo_exp_scale, std::mt19937& rng) {
    std::uniform_int_distribution<int> man(0, 1023);
    const int e = 15 + (int) std::floor(std::log2(lo_exp_scale));   // exponent field of 2^k
    return (uint16_t) ((e << 10) | man(rng));
}

// rows of n values: random bytes, finite fp16 scale at the start of each block (IQ1_S and IQ2_XXS both start with d)
std::vector<uint8_t> random_rows(int type, int64_t rows, int64_t n, float scale, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, n), bs = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = f16_bits(scale, rng);
        if (type == GGML_TYPE_IQ1_M) {   // the fp16 scale sits in the top nibbles of the block's last four int16
            for (int i = 0; i < 4; ++i) {
                uint16_t sc;
                std::memcpy(&sc, &w[o + bs - 8 + 2 * (size_t) i], 2);
                sc = (uint16_t) ((sc & 0x0FFF) | (((h >> (4 * i)) & 0xF) << 12));
                std::memcpy(&w[o + bs - 8 + 2 * (size_t) i], &sc, 2);
            }
            continue;
        }
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

std::vector<float> dequant(int type, const uint8_t* w, int64_t rows, int64_t n) {
    std::vector<float> o((size_t) rows * n);
    const size_t rb = ggml_row_size((ggml_type) type, n);
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(w + (size_t) r * rb, o.data() + (size_t) r * n, n);
    return o;
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }

// the expert in double from dequantized weights and an already-rounded activation xh (H); the limit in double
std::vector<double> ref_expert(const std::vector<float>& wg, const std::vector<float>& wu, const std::vector<float>& wd,
                               const std::vector<double>& xh, std::vector<double>* h_out) {
    std::vector<double> h(FF), out(H, 0.0);
    for (int r = 0; r < FF; ++r) {
        double g = 0, u = 0;
        for (int k = 0; k < H; ++k) {
            g += (double) wg[(size_t) r * H + k] * xh[k];
            u += (double) wu[(size_t) r * H + k] * xh[k];
        }
        g = std::min(g, (double) kLimit);
        u = std::max(std::min(u, (double) kLimit), -(double) kLimit);
        h[r] = silu(g) * u;
    }
    for (int r = 0; r < H; ++r) {
        double s = 0;
        for (int k = 0; k < FF; ++k) s += (double) wd[(size_t) r * FF + k] * h[k];
        out[r] = s;
    }
    if (h_out) *h_out = h;
    return out;
}

const char* nm(int t) {
    return t == 19 ? "IQ1_S" : t == 16 ? "IQ2_XXS" : t == 17 ? "IQ2_XS" : t == 22 ? "IQ2_S" : t == 29 ? "IQ1_M" : t == 40 ? "NVFP4" : "?";
}

struct Expert {
    std::vector<uint8_t> blob;          // [gate | up | down] (NVFP4: and the scale tail)
    std::vector<float> wg, wu, wd;      // dequantized (NVFP4: times the tail's scales)
    float tail[3] = {1.f, 1.f, 1.f};
};

// NVFP4 rows: ggml's rounding of N(0, sd) values
std::vector<uint8_t> nvfp4_rows(int64_t rows, int64_t n, float sd, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) rows * ggml_row_size(GGML_TYPE_NVFP4, n));
    std::vector<float> x((size_t) n);
    std::normal_distribution<float> nd(0.f, sd);
    const size_t rb = ggml_row_size(GGML_TYPE_NVFP4, n);
    for (int64_t r = 0; r < rows; ++r) {
        for (auto& v : x) v = nd(rng);
        ggml_get_type_traits(GGML_TYPE_NVFP4)->from_float_ref(x.data(), w.data() + (size_t) r * rb, n);
    }
    return w;
}

Expert make_expert(int gu, int d, std::mt19937& rng) {
    Expert e;
    if (gu == GGML_TYPE_NVFP4) {   // gate/up outputs around the limit; s_gate, s_up and s_down in the tail
        auto g = nvfp4_rows(FF, H, 0.1f, rng), u = nvfp4_rows(FF, H, 0.1f, rng), dn = nvfp4_rows(H, FF, 0.1f, rng);
        std::uniform_real_distribution<float> ts(0.9f, 1.6f);
        const float t4[4] = {ts(rng), ts(rng), 0.02f * ts(rng), 0.f};
        e.wg = dequant(gu, g.data(), FF, H);
        e.wu = dequant(gu, u.data(), FF, H);
        e.wd = dequant(d, dn.data(), H, FF);
        for (auto& v : e.wg) v *= t4[0];
        for (auto& v : e.wu) v *= t4[1];
        for (auto& v : e.wd) v *= t4[2];
        e.blob = g;
        e.blob.insert(e.blob.end(), u.begin(), u.end());
        e.blob.insert(e.blob.end(), dn.begin(), dn.end());
        e.blob.insert(e.blob.end(), (const uint8_t*) t4, (const uint8_t*) t4 + 16);
        std::copy(t4, t4 + 3, e.tail);
        return e;
    }
    const float gs = gu == 19 ? 0.02f : 0.002f;       // block scales: gate/up outputs that reach the SwiGLU limit
    const float ds = d == 19 ? 0.002f : 0.0005f;
    auto g = random_rows(gu, FF, H, gs, rng), u = random_rows(gu, FF, H, gs, rng), dn = random_rows(d, H, FF, ds, rng);
    e.wg = dequant(gu, g.data(), FF, H);
    e.wu = dequant(gu, u.data(), FF, H);
    e.wd = dequant(d, dn.data(), H, FF);
    e.blob = g;
    e.blob.insert(e.blob.end(), u.begin(), u.end());
    e.blob.insert(e.blob.end(), dn.begin(), dn.end());
    return e;
}

// ---------------------------------------------------------------- CPU tier
void cpu_pair(int gu, int d, std::mt19937& rng) {
    const Expert e = make_expert(gu, d, rng);
    kc::NativeFmt f;
    std::string err;
    if (!kc::native_fmt(gu, d, H, FF, f, err)) {
        std::printf("FAIL: native_fmt %s/%s: %s\n", nm(gu), nm(d), err.c_str());
        ++g_fail;
        return;
    }
    f.swiglu_limit = kLimit;
    check(f.bytes == e.blob.size(), "native_fmt blob size");
    const int NT = 3;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> x(NT, std::vector<float>(H));
    for (auto& v : x) for (auto& t : v) t = nd(rng);
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(kc::kNativeActBytes));
    std::vector<const void*> ap(NT);
    for (int t = 0; t < NT; ++t) {
        kc::native_quant_act(f, x[t].data(), act[t].data());
        ap[t] = act[t].data();
    }
    // the dequantized Q8_K activation (4-byte scale, 256 int8 values, 16 int16 sums per block)
    auto act_deq = [&](int t) {
        std::vector<double> o(H);
        if (f.gu_act == GGML_TYPE_Q8_K) {
            for (int b = 0; b < H / 256; ++b) {
                const uint8_t* p = act[t].data() + (size_t) b * 292;
                float dd;
                std::memcpy(&dd, p, 4);
                for (int i = 0; i < 256; ++i) o[b * 256 + i] = (double) dd * (int8_t) p[4 + i];
            }
        } else {   // Q8_0: fp16 scale + 32 int8
            for (int b = 0; b < H / 32; ++b) {
                const uint8_t* p = act[t].data() + (size_t) b * 34;
                uint16_t hb;
                std::memcpy(&hb, p, 2);
                const double dd = ggml_fp16_to_fp32(hb);
                for (int i = 0; i < 32; ++i) o[b * 32 + i] = dd * (int8_t) p[2 + i];
            }
        }
        return o;
    };
    std::vector<std::vector<float>> ff(NT, std::vector<float>(FF)), out(NT, std::vector<float>(H));
    std::vector<float*> ffp(NT), op(NT);
    for (int t = 0; t < NT; ++t) { ffp[t] = ff[t].data(); op[t] = out[t].data(); }
    kc::native_gu_rows(f, e.blob.data(), ap.data(), NT, ffp.data(), 0, FF);
    bool same_ggml = true;   // gate/up rows and the down rows bit for bit what the ggml dot gives
    {
        const auto dot = ggml_get_type_traits_cpu((ggml_type) gu)->vec_dot;
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < FF; r += 7) {
                float g = 0, u = 0;
                dot(H, &g, 0, e.blob.data() + (size_t) r * f.gu_row, 0, act[t].data(), 0, 1);
                dot(H, &u, 0, e.blob.data() + f.up_off + (size_t) r * f.gu_row, 0, act[t].data(), 0, 1);
                if (f.tail) {
                    g *= e.tail[0];
                    u *= e.tail[1];
                }
                g = std::fmin(g, kLimit);
                u = std::fmax(std::fmin(u, kLimit), -kLimit);
                const float want = (g / (1.f + std::exp(-g))) * u;
                same_ggml = same_ggml && std::memcmp(&want, &ff[t][r], 4) == 0;
            }
    }
    std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(kc::kNativeHBytes));
    std::vector<const void*> hp(NT);
    for (int t = 0; t < NT; ++t) {
        kc::native_quant_h(f, ff[t].data(), hq[t].data());
        hp[t] = hq[t].data();
    }
    kc::native_down_rows(f, e.blob.data(), hp.data(), NT, op.data(), 0, H);
    {
        const auto dot = ggml_get_type_traits_cpu((ggml_type) d)->vec_dot;
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < H; r += 11) {
                float s = 0;
                dot(FF, &s, 0, e.blob.data() + f.down_off + (size_t) r * f.d_row, 0, hq[t].data(), 0, 1);
                if (f.tail) s *= e.tail[2];
                same_ggml = same_ggml && std::memcmp(&s, &out[t][r], 4) == 0;
            }
    }
    Err eh, eo;
    for (int t = 0; t < NT; ++t) {
        std::vector<double> h;
        const auto ref = ref_expert(e.wg, e.wu, e.wd, act_deq(t), &h);
        for (int r = 0; r < FF; ++r) eh.add(ff[t][r], h[r]);
        for (int r = 0; r < H; ++r) eo.add(out[t][r], ref[r]);
    }
    // alone and at another position: bitwise equal to the window's
    bool same = true;
    for (int t = 0; t < NT; ++t) {
        std::vector<float> f1(FF), o1(H);
        float* f1p = f1.data();
        float* o1p = o1.data();
        kc::native_gu_rows(f, e.blob.data(), &ap[t], 1, &f1p, 0, FF);
        const void* hp1 = hp[t];
        kc::native_down_rows(f, e.blob.data(), &hp1, 1, &o1p, 0, H);
        same = same && std::memcmp(f1.data(), ff[t].data(), FF * 4) == 0 && std::memcmp(o1.data(), out[t].data(), H * 4) == 0;
    }
    const bool ok = eh.rms() < 1e-5 && eo.rms() < 2e-2 && same && same_ggml;
    std::printf("cpu %-7s/%-7s ffn rel RMS %.2e (max abs %.2e)  out rel RMS %.2e (max abs %.2e, max/|ref|max %.2e)  window-independent %s  ggml-identical %s  %s\n",
                nm(gu), nm(d), eh.rms(), eh.max_abs, eo.rms(), eo.max_abs, eo.rel_max(), same ? "yes" : "NO", same_ggml ? "yes" : "NO", ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// IQ1_S dot against the CPU dequantizer on many rows: the ggml AVX2 dot vs double (isolates the new dot)
void cpu_dot() {
    std::mt19937 rng(11);
    const int rows = 64;
    auto w = random_rows(19, rows, H, 0.01f, rng);
    const auto deq = dequant(19, w.data(), rows, H);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(H);
    for (auto& v : x) v = nd(rng);
    kc::NativeFmt f;
    std::string err;
    if (!kc::native_fmt(19, 19, H, FF, f, err)) { check(false, "native_fmt 19/19"); return; }
    std::vector<uint8_t> act(kc::kNativeActBytes);
    kc::native_quant_act(f, x.data(), act.data());
    const auto* tr = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S);
    Err e;
    for (int r = 0; r < rows; ++r) {
        float s = 0;
        tr->vec_dot(H, &s, 0, w.data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
        double ref = 0;
        for (int b = 0; b < H / 256; ++b) {
            const uint8_t* p = act.data() + (size_t) b * 292;
            float dd;
            std::memcpy(&dd, p, 4);
            for (int i = 0; i < 256; ++i) ref += (double) deq[(size_t) r * H + b * 256 + i] * dd * (int8_t) p[4 + i];
        }
        e.add(s, ref);
    }
    bool ok = e.rms() < 1e-5;
    // the AVX-512 kernel is bit for bit the ggml dot (rows, tokens and the clamped SwiGLU)
    if (kc::cpu_avx512_ok()) {
        const void* ap = act.data();
        std::vector<float> got(rows);
        float* gp = got.data();
        kc::iq1s_512_rows(w.data(), f.gu_row, H, &ap, 1, &gp, 0, rows);
        int diff = 0;
        for (int r = 0; r < rows; ++r) {
            float s = 0;
            tr->vec_dot(H, &s, 0, w.data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
            diff += std::memcmp(&s, &got[r], 4) != 0;
        }
        std::printf("cpu IQ1_S avx512 vs ggml dot: %d of %d rows differ\n", diff, rows);
        ok = ok && diff == 0;
    }
    std::printf("cpu IQ1_S vec_dot (%d rows x %d): rel RMS %.2e, max abs %.2e  %s\n", rows, H, e.rms(), e.max_abs, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

#ifdef BASALT_IQ1S_GPU
namespace kg = basalt::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template<typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template<typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d"); }
template<typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

// q8_1 block (36 bytes): half2 {scale, sum}, 32 int8
std::vector<double> q81_deq(const std::vector<uint8_t>& q, int tok) {
    std::vector<double> o(H);
    for (int b = 0; b < H / 32; ++b) {
        const uint8_t* p = q.data() + ((size_t) tok * (H / 32) + b) * 36;
        uint16_t hb;
        std::memcpy(&hb, p, 2);
        const double dd = ggml_fp16_to_fp32(hb);
        for (int i = 0; i < 32; ++i) o[b * 32 + i] = dd * (int8_t) p[4 + i];
    }
    return o;
}

// the single-matrix kernel against the dequantized rows
void gpu_mmvq(cudaStream_t s) {
    std::mt19937 rng(3);
    const int rows = 256, NC = 4;
    auto w = random_rows(19, rows, H, 0.01f, rng);
    const auto deq = dequant(19, w.data(), rows, H);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NC * H);
    for (auto& v : x) v = nd(rng);
    auto* dw = dalloc<uint8_t>(w.size());
    up(dw, w);
    auto* dx = dalloc<float>(x.size());
    up(dx, x);
    auto* xq = dalloc<uint8_t>((size_t) NC * H / 32 * 36);
    auto* dy = dalloc<float>((size_t) NC * rows);
    kg::quantize_q8_1_rows(dx, NC, H, xq, s);
    kg::iq_mmvq(19, dw, xq, dy, H, rows, NC, s);
    ck(cudaStreamSynchronize(s), "sync");
    const auto y = down(dy, (size_t) NC * rows);
    const auto q = down(xq, (size_t) NC * H / 32 * 36);
    Err e;
    for (int c = 0; c < NC; ++c) {
        const auto xh = q81_deq(q, c);
        for (int r = 0; r < rows; ++r) {
            double ref = 0;
            for (int k = 0; k < H; ++k) ref += (double) deq[(size_t) r * H + k] * xh[k];
            e.add(y[(size_t) c * rows + r], ref);
        }
    }
    // each column bitwise equal to a one-column call
    bool same = true;
    for (int c = 0; c < NC; ++c) {
        kg::iq_mmvq(19, dw, xq + (size_t) c * H / 32 * 36, dy, H, rows, 1, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto y1 = down(dy, rows);
        same = same && std::memcmp(y1.data(), &y[(size_t) c * rows], rows * 4) == 0;
    }
    const bool ok = e.rms() < 1e-5 && same;
    std::printf("gpu IQ1_S iq_mmvq (%d rows x %d, ncols %d): rel RMS %.2e, max abs %.2e, max/|ref|max %.2e  column-independent %s  %s\n",
                rows, H, NC, e.rms(), e.max_abs, e.rel_max(), same ? "yes" : "NO", ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
    // dequantizer
    auto* dq = dalloc<float>((size_t) rows * H);
    kg::iq_dequant_f32(19, dw, (int64_t) rows * H, dq, s);
    ck(cudaStreamSynchronize(s), "sync");
    const auto dqh = down(dq, (size_t) rows * H);
    Err ed;
    for (size_t i = 0; i < dqh.size(); ++i) ed.add(dqh[i], deq[i]);
    const bool okd = ed.max_abs == 0.0 || ed.rms() < 1e-6;
    std::printf("gpu IQ1_S dequant vs ggml CPU dequantizer: rel RMS %.2e, max abs %.2e  %s\n", ed.rms(), ed.max_abs, okd ? "ok" : "FAIL");
    if (!okd) ++g_fail;
    cudaFree(dw); cudaFree(dx); cudaFree(xq); cudaFree(dy); cudaFree(dq);
}

// the grouped expert call: T tokens x K experts of NB distinct blobs, against double; then the window checks
struct Window {
    int T;
    std::vector<int> ids;   // T*K expert ids
};

void gpu_pair(int gu, int d, bool bench, cudaStream_t s) {
    std::mt19937 rng(100 + gu * 7 + d);
    const int NB = 8;
    std::vector<Expert> ex;
    for (int b = 0; b < NB; ++b) ex.push_back(make_expert(gu, d, rng));
    const kg::NativeExpertLayout L = kg::native_expert_layout(gu, d, H, FF);
    if (!kg::native_expert_supported(gu, d, H, FF)) {
        std::printf("FAIL: native_expert_supported %s/%s\n", nm(gu), nm(d));
        ++g_fail;
        return;
    }
    check(L.bytes == ex[0].blob.size(), "layout bytes");
    const size_t slot = (L.bytes + 255) & ~(size_t) 255;
    auto* blobs = dalloc<uint8_t>(slot * NB);
    for (int b = 0; b < NB; ++b)
        ck(cudaMemcpy(blobs + slot * b, ex[b].blob.data(), L.bytes, cudaMemcpyHostToDevice), "blob");
    const int TMAX = 6;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) TMAX * H);
    for (auto& v : x) v = nd(rng);
    auto* dx = dalloc<float>(x.size());
    up(dx, x);
    auto* xq = dalloc<uint8_t>((size_t) TMAX * H / 32 * 36);
    kg::quantize_q8_1_rows(dx, TMAX, H, xq, s);
    ck(cudaStreamSynchronize(s), "sync");
    const auto xqh = down(xq, (size_t) TMAX * H / 32 * 36);

    // one call: the (token, expert) entries given; groups by expert
    struct Call { std::vector<int> tok, exp; };
    auto run = [&](const Call& c, int mode, std::vector<float>& out_rows) {   // 0 plain, 1 explicit limit, 2 limit in the layout
        const int n = (int) c.tok.size();
        std::vector<int> order(NB);
        std::vector<unsigned long long> gptr;
        std::vector<int32_t> start{0}, dst(n), tk(n);
        int e = 0;
        for (int b = 0; b < NB; ++b) {
            bool any = false;
            for (int i = 0; i < n; ++i)
                if (c.exp[i] == b) {
                    any = true;
                    dst[e] = i;
                    tk[e] = c.tok[i];
                    ++e;
                }
            if (any) {
                gptr.push_back((unsigned long long) (blobs + slot * b));
                start.push_back(e);
            }
        }
        const int ng = (int) gptr.size();
        auto* dptr = dalloc<unsigned long long>(ng);
        auto* dstart = dalloc<int32_t>(ng + 1);
        auto* dng = dalloc<int32_t>(1);
        auto* ddst = dalloc<int32_t>(n);
        auto* dtok = dalloc<int32_t>(n);
        up(dptr, gptr); up(dstart, start); up(ddst, dst); up(dtok, tk);
        const int32_t ngv = ng;
        ck(cudaMemcpy(dng, &ngv, 4, cudaMemcpyHostToDevice), "ng");
        const size_t sb = kg::native_expert_scratch_bytes(n, FF);
        auto* scr = dalloc<uint8_t>(sb);
        auto* dout = dalloc<float>((size_t) n * H);
        if (mode == 1)
            kg::native_expert_grouped_clamped(L, kLimit, dptr, dstart, dng, ddst, dtok, ng, n, xq, scr, dout, s);
        else if (mode == 2)
            kg::native_expert_grouped(kg::native_expert_layout(gu, d, H, FF, kLimit), dptr, dstart, dng, ddst, dtok, ng, n, xq, scr, dout, s);
        else
            kg::native_expert_grouped(L, dptr, dstart, dng, ddst, dtok, ng, n, xq, scr, dout, s);
        ck(cudaStreamSynchronize(s), "sync");
        out_rows = down(dout, (size_t) n * H);
        cudaFree(dptr); cudaFree(dstart); cudaFree(dng); cudaFree(ddst); cudaFree(dtok); cudaFree(scr); cudaFree(dout);
    };

    // the window: 6 tokens x 6 distinct experts of 8, different expert sets per token
    Call win;
    for (int t = 0; t < TMAX; ++t)
        for (int k = 0; k < K; ++k) {
            win.tok.push_back(t);
            win.exp.push_back((t + k * 3 + (k > 2 ? 1 : 0)) % NB);
        }
    // make sure each token's experts are distinct
    for (int t = 0; t < TMAX; ++t) {
        std::vector<int> seen;
        for (int k = 0; k < K; ++k) {
            int& v = win.exp[t * K + k];
            while (std::find(seen.begin(), seen.end(), v) != seen.end()) v = (v + 1) % NB;
            seen.push_back(v);
        }
    }
    std::vector<float> got;
    run(win, 1, got);
    Err e;
    double worst_rms = 0;
    for (int i = 0; i < (int) win.tok.size(); ++i) {
        const auto xh = q81_deq(xqh, win.tok[i]);
        const auto& E = ex[win.exp[i]];
        const auto ref = ref_expert(E.wg, E.wu, E.wd, xh, nullptr);
        Err one;
        for (int r = 0; r < H; ++r) { e.add(got[(size_t) i * H + r], ref[r]); one.add(got[(size_t) i * H + r], ref[r]); }
        worst_rms = std::max(worst_rms, one.rms());
    }
    // entry alone, and in a window shifted by one token (other position, other groups)
    bool same = true;
    for (int i : {0, 7, 20, 35}) {
        Call one;
        one.tok = {win.tok[i]};
        one.exp = {win.exp[i]};
        std::vector<float> g1;
        run(one, 1, g1);
        same = same && std::memcmp(g1.data(), &got[(size_t) i * H], H * 4) == 0;
    }
    {
        Call rev = win;
        std::reverse(rev.tok.begin(), rev.tok.end());
        std::reverse(rev.exp.begin(), rev.exp.end());
        std::vector<float> gr;
        run(rev, 1, gr);
        const int n = (int) win.tok.size();
        for (int i = 0; i < n; ++i)
            same = same && std::memcmp(&gr[(size_t) (n - 1 - i) * H], &got[(size_t) i * H], H * 4) == 0;
    }
    // the unclamped entry point with a limit that never binds is the same arithmetic but for expf vs __expf
    std::vector<float> ung;
    run(win, 0, ung);
    std::vector<float> lay;
    run(win, 2, lay);
    const bool lay_same = std::memcmp(lay.data(), got.data(), got.size() * 4) == 0;
    // clamp is active: count entries whose gate/up exceed the limit by comparing clamped vs unclamped
    int differ = 0;
    for (size_t i = 0; i < got.size(); ++i) differ += got[i] != ung[i];
    const bool ok = e.rms() < 2e-2 && same && (differ > 0 || gu == GGML_TYPE_IQ1_M) && lay_same;   // IQ1_M's random rows stay below the limit
    std::printf("gpu %-7s/%-7s grouped clamped (%zu entries, %dx%d): out rel RMS %.2e (worst entry %.2e), max abs %.2e, max/|ref|max %.2e  window/position-independent %s  clamp active %s  limit-in-layout bitwise same %s  %s\n",
                nm(gu), nm(d), win.tok.size(), H, FF, e.rms(), worst_rms, e.max_abs, e.rel_max(), same ? "yes" : "NO",
                differ > 0 ? "yes" : "NO", lay_same ? "yes" : "NO", ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;

    if (bench && (gu == 19 || gu == GGML_TYPE_IQ1_M || d == GGML_TYPE_IQ1_M || gu == GGML_TYPE_NVFP4)) {
        Call one;   // a decode step: one token, six experts
        for (int k = 0; k < K; ++k) { one.tok.push_back(0); one.exp.push_back(k); }
        const int n = K;
        std::vector<unsigned long long> gptr;
        std::vector<int32_t> start{0}, dst(n), tk(n, 0);
        for (int k = 0; k < K; ++k) { gptr.push_back((unsigned long long) (blobs + slot * k)); start.push_back(k + 1); dst[k] = k; }
        auto* dptr = dalloc<unsigned long long>(n);
        auto* dstart = dalloc<int32_t>(n + 1);
        auto* dng = dalloc<int32_t>(1);
        auto* ddst = dalloc<int32_t>(n);
        auto* dtok = dalloc<int32_t>(n);
        up(dptr, gptr); up(dstart, start); up(ddst, dst); up(dtok, tk);
        const int32_t ngv = n;
        ck(cudaMemcpy(dng, &ngv, 4, cudaMemcpyHostToDevice), "ng");
        auto* scr = dalloc<uint8_t>(kg::native_expert_scratch_bytes(n, FF));
        auto* dout = dalloc<float>((size_t) n * H);
        for (int i = 0; i < 20; ++i)
            kg::native_expert_grouped_clamped(L, kLimit, dptr, dstart, dng, ddst, dtok, n, n, xq, scr, dout, s);
        ck(cudaStreamSynchronize(s), "sync");
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        const int it = 200;
        cudaEventRecord(a, s);
        for (int i = 0; i < it; ++i)
            kg::native_expert_grouped_clamped(L, kLimit, dptr, dstart, dng, ddst, dtok, n, n, xq, scr, dout, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double us = ms * 1000.0 / it;
        const double bytes = (double) K * L.bytes;
        std::printf("bench gpu %s/%s grouped clamped, 1 token x %d experts: %.1f us/call, %.0f MB read, %.0f GB/s (weights only)\n",
                    nm(gu), nm(d), K, us, bytes / 1e6, bytes / (us * 1e3));
        cudaFree(dptr); cudaFree(dstart); cudaFree(dng); cudaFree(ddst); cudaFree(dtok); cudaFree(scr); cudaFree(dout);
    }
    cudaFree(blobs); cudaFree(dx); cudaFree(xq);
}
#endif

}  // namespace

// The exact AVX-512 kernels against ggml's dot, bit for bit: both row lengths of an expert, random and extreme codes,
// ordinary and saturating activations, five tokens in one call (a group of four and one alone).  A row's bits do not
// depend on the row range of the call or on the tokens beside it.
void cpu_exact_dots() {
    if (!kc::cpu_avx512_ok()) return;
    constexpr int NT = 5;
    for (int type : {19, 16, 17, 29}) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        const size_t bs = ggml_type_size((ggml_type) type);
        const auto rows_of = [type](const uint8_t* w, size_t rb, int n, const void* const* a, int nt, float* const* o, int r0, int r1) {
            if (type == 19) kc::iq1s_512_rows(w, rb, n, a, nt, o, r0, r1);
            else kc::iq_exact_512_rows(type, w, rb, n, a, nt, o, r0, r1);
        };
        for (int n : {H, FF}) {
            std::mt19937 rng(40 + type + n);
            const int rows = 400;
            const size_t rb = (size_t) n / 256 * bs;
            size_t diff = 0, total = 0;
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<uint8_t> w(rb * rows);
                for (auto& b : w) b = mode == 1 ? 0 : mode == 2 ? 0xff : (uint8_t) rng();
                // a finite fp16 block scale wherever the format stores one
                for (int r = 0; r < rows; ++r)
                    for (int b = 0; b < n / 256; ++b) {
                        uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * bs;
                        const uint16_t h = (uint16_t) (0x3800 + (rng() & 0x3ff));
                        if (type == 29) {
                            for (int k = 0; k < 4; ++k) blk[48 + 2 * k + 1] = (uint8_t) ((blk[48 + 2 * k + 1] & 0x0f) | (((h >> (4 * k)) & 15) << 4));
                        } else {
                            std::memcpy(blk, &h, 2);
                        }
                    }
                std::normal_distribution<float> nd(0.f, 1.f);
                std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>((size_t) n / 256 * ggml_type_size(GGML_TYPE_Q8_K)));
                std::vector<const void*> ap(NT);
                for (int t = 0; t < NT; ++t) {
                    std::vector<float> x(n);
                    for (auto& v : x) v = mode == 3 ? ((rng() & 1) ? 1e4f : -1e4f) : nd(rng);
                    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), act[t].data(), n);
                    ap[t] = act[t].data();
                }
                std::vector<std::vector<float>> o(NT, std::vector<float>(rows));
                std::vector<float*> op(NT);
                for (int t = 0; t < NT; ++t) op[t] = o[t].data();
                rows_of(w.data(), rb, n, ap.data(), NT, op.data(), 0, rows);
                for (int t = 0; t < NT; ++t) {
                    std::vector<float> one(rows), ref(rows);
                    float* p1 = one.data();
                    rows_of(w.data(), rb, n, &ap[t], 1, &p1, 17, 18);
                    diff += std::memcmp(&one[17], &o[t][17], 4) != 0;
                    for (int r = 0; r < rows; ++r) {
                        tr->vec_dot(n, &ref[r], 0, w.data() + (size_t) r * rb, 0, act[t].data(), 0, 1);
                        diff += std::memcmp(&o[t][r], &ref[r], 4) != 0;
                        ++total;
                    }
                }
            }
            std::printf("cpu %-7s avx512 dot vs ggml, n %d, %d tokens: %zu of %zu rows differ\n", nm(type), n, NT, diff, total);
            check(diff == 0, "exact avx512 dot");
        }
    }
}

// single-thread streaming rate of the IQ1_S row dot, ggml's against the AVX-512 kernel, over a buffer well past L2
void cpu_bench() {
    std::mt19937 rng(3);
    const int rows = 131072;   // 131 MB of 1000-byte rows
    auto w = random_rows(19, rows, H, 0.01f, rng);
    kc::NativeFmt f;
    std::string err;
    kc::native_fmt(19, 19, H, FF, f, err);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(H);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> act(kc::kNativeActBytes);
    kc::native_quant_act(f, x.data(), act.data());
    const auto* tr = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S);
    std::vector<float> o(rows);
    const void* ap = act.data();
    float* op = o.data();
    auto run = [&](const char* name, auto&& fn) {
        double best = 1e30;
        for (int it = 0; it < 12; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        const double bytes = (double) rows * f.gu_row;
        std::printf("bench cpu IQ1_S row dot, 1 thread, %s: %.2f us/row, %.1f GB/s\n", name, best * 1e6 / rows, bytes / best / 1e9);
    };
    run("ggml avx2", [&] { for (int r = 0; r < rows; ++r) tr->vec_dot(H, &o[r], 0, w.data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1); });
    if (kc::cpu_avx512_ok()) run("avx512", [&] { kc::iq1s_512_rows(w.data(), f.gu_row, H, &ap, 1, &op, 0, rows); });
    // three tokens in one call (the same activation three times): the rate is per row of weights, all three tokens
    const void* ap3[3] = {ap, ap, ap};
    std::vector<float> o3b(rows), o3c(rows);
    float* op3[3] = {op, o3b.data(), o3c.data()};
    if (kc::cpu_avx512_ok()) run("avx512, 3 tokens", [&] { kc::iq1s_512_rows(w.data(), f.gu_row, H, ap3, 3, op3, 0, rows); });
    for (int type : {16, 17, 29}) {
        const size_t rb = (size_t) H / 256 * ggml_type_size((ggml_type) type);
        const int nr = (int) (131072000ull / rb);
        auto wt = random_rows(type, nr, H, 0.01f, rng);
        const auto* tt = ggml_get_type_traits_cpu((ggml_type) type);
        std::vector<float> ot(nr);
        float* otp = ot.data();
        auto timed = [&](const char* name, auto&& fn) {
            double best = 1e30;
            for (int it = 0; it < 12; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::printf("bench cpu %s row dot, 1 thread, %s: %.2f us/row, %.1f GB/s\n", nm(type), name, best * 1e6 / nr, (double) nr * rb / best / 1e9);
        };
        timed("ggml avx2", [&] { for (int r = 0; r < nr; ++r) tt->vec_dot(H, &ot[r], 0, wt.data() + (size_t) r * rb, 0, act.data(), 0, 1); });
        if (kc::cpu_avx512_ok()) timed("avx512", [&] { kc::iq_exact_512_rows(type, wt.data(), rb, H, &ap, 1, &otp, 0, nr); });
        std::vector<float> ot2(nr), ot3(nr);
        float* otp3[3] = {otp, ot2.data(), ot3.data()};
        if (kc::cpu_avx512_ok()) timed("avx512, 3 tokens", [&] { kc::iq_exact_512_rows(type, wt.data(), rb, H, ap3, 3, otp3, 0, nr); });
    }
}

int main(int argc, char** argv) {
    bool cpu_only = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu")) cpu_only = true;
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
    }
    std::mt19937 rng(5);
    cpu_dot();
    cpu_exact_dots();
    if (bench) cpu_bench();
    cpu_pair(19, 19, rng);
    cpu_pair(16, 16, rng);
    cpu_pair(19, 16, rng);
    cpu_pair(16, 22, rng);   // the other pairs of the mixed-format packs
    cpu_pair(16, 17, rng);
    cpu_pair(17, 17, rng);
    cpu_pair(29, 29, rng);   // the GSQ-RCO non-uniform layers
    cpu_pair(16, 29, rng);
    cpu_pair(17, 29, rng);
    cpu_pair(29, 16, rng);
    cpu_pair(17, 16, rng);
    cpu_pair(40, 40, rng);   // NVFP4 experts
#ifdef BASALT_IQ1S_GPU
    if (!cpu_only) {
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        gpu_mmvq(s);
        gpu_pair(19, 19, bench, s);
        gpu_pair(16, 16, bench, s);
        gpu_pair(19, 16, bench, s);
        gpu_pair(16, 22, bench, s);
        gpu_pair(16, 17, bench, s);
        gpu_pair(17, 17, bench, s);
        gpu_pair(29, 29, bench, s);
        gpu_pair(16, 29, bench, s);
        gpu_pair(17, 29, bench, s);
        gpu_pair(29, 16, bench, s);
        gpu_pair(17, 16, bench, s);
        gpu_pair(40, 40, bench, s);
    }
#else
    (void) cpu_only;
    (void) bench;
#endif
    std::printf("iq1s_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
