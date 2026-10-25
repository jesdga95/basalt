// tests/kernels/dense_tc_parity.cpp - dense_tc (DEC_DTC) against native_mmvq, and its column contract.
//
//     build/dense_tc_parity [--bench]
//
// Random GGUF blocks (fp16 scales rewritten as normal values) and normal activations, one q8_1 quantization shared
// by both paths, at the model's projection shapes.  Checks:
//   - every column of a T-column dense_tc call is bitwise equal to the same column of the 8-column call (T = 1..8);
//   - the error against native_mmvq's single-column outputs (the current numerics): max |diff| / max |ref| and the
//     RMS of the diff over the RMS of ref, per case; it fails above 1e-3 RMS.
// --bench: native_mmvq and dense_tc timed per T as CUDA graphs over enough weight copies that the L2 holds none
// (launched as programmatic dependents under PDL=1).
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"

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

namespace {

struct Case {
    const char* name;
    int type, n_in, n_out;
    int block_bytes;
    int scale_at[2];
};

const Case CASES[] = {
    {"qkv Q6_K", 14, 2560, 10240, 210, {208, -1}},
    {"qkv Q4_K", 12, 2560, 10240, 144, {0, 2}},
    {"qkv Q5_K", 13, 2560, 10240, 176, {0, 2}},
    {"qkv IQ4_XS", 23, 2560, 10240, 136, {0, -1}},
    {"z Q6_K", 14, 2560, 6144, 210, {208, -1}},
    {"z IQ4_XS", 23, 2560, 6144, 136, {0, -1}},
    {"z IQ3_S", 21, 2560, 6144, 110, {0, -1}},
    {"out Q6_K", 14, 6144, 2560, 210, {208, -1}},
    {"out IQ4_XS", 23, 6144, 2560, 136, {0, -1}},
    {"out Q4_K", 12, 6144, 2560, 144, {0, 2}},
    {"out Q5_K", 13, 6144, 2560, 176, {0, 2}},
    {"out IQ3_S", 21, 6144, 2560, 110, {0, -1}},
    {"q Q6_K", 14, 2560, 12288, 210, {208, -1}},
    {"kv Q6_K", 14, 2560, 512, 210, {208, -1}},
    {"tail Q4_K", 12, 1024, 1000, 144, {0, 2}},
    {"tail Q6_K", 14, 1536, 999, 210, {208, -1}},
    {"qkv Q3_K", 11, 2560, 10240, 110, {108, -1}},
    {"z Q3_K", 11, 2560, 6144, 110, {108, -1}},
    {"out Q3_K", 11, 6144, 2560, 110, {108, -1}},
    {"q Q3_K", 11, 2560, 12288, 110, {108, -1}},
    {"qkv Q2_0", 42, 2560, 10240, 18, {0, -1}},
    {"z Q2_0", 42, 2560, 6144, 18, {0, -1}},
    {"q Q2_0", 42, 2560, 12288, 18, {0, -1}},
    {"tail Q3_K", 11, 1536, 999, 110, {108, -1}},
    {"tail Q2_0", 42, 1024, 1000, 18, {0, -1}},
    {"qkv Q8_0", 8, 2560, 10240, 34, {0, -1}},
    {"z Q8_0", 8, 2560, 6144, 34, {0, -1}},
    {"out Q8_0", 8, 6144, 2560, 34, {0, -1}},
    {"q Q8_0", 8, 2560, 12288, 34, {0, -1}},
    {"kv Q8_0", 8, 2560, 512, 34, {0, -1}},
    {"tail Q8_0", 8, 1024, 1000, 34, {0, -1}},
    {"tail2 Q8_0", 8, 1536, 999, 34, {0, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

std::vector<uint8_t> make_weights(const Case& c, std::mt19937& rng) {
    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t nb = wbytes / (std::size_t) c.block_bytes;
    for (std::size_t k = 0; k < nb; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    return w;
}

// float64 reference for Q8_0: sum over blocks of d_w * d_x * (exact integer dot), from the same q8_1 columns
std::vector<double> cpu_q8_0(const Case& c, const std::vector<uint8_t>& w, const std::vector<uint8_t>& xq, int cols) {
    const int nb = c.n_in / 32;
    const std::size_t qcol = (std::size_t) nb * 36;
    auto h2f = [](uint16_t h) {
        const int e = (h >> 10) & 31, m = h & 1023;
        const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (1024 + m), e - 25);
        return (h >> 15) ? -v : v;
    };
    std::vector<double> y((std::size_t) cols * c.n_out);
    for (int j = 0; j < cols; ++j)
        for (int r = 0; r < c.n_out; ++r) {
            double sum = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* wb = &w[((std::size_t) r * nb + b) * 34];
                const uint8_t* xb = &xq[j * qcol + (std::size_t) b * 36];
                uint16_t dwh, dxh;
                std::memcpy(&dwh, wb, 2);
                std::memcpy(&dxh, xb, 2);
                long dot = 0;
                for (int k = 0; k < 32; ++k) dot += (long) (int8_t) wb[2 + k] * (int8_t) xb[4 + k];
                sum += h2f(dwh) * h2f(dxh) * (double) dot;
            }
            y[(std::size_t) j * c.n_out + r] = sum;
        }
    return y;
}

bool parity(const Case& c, cudaStream_t s) {
    constexpr int TM = 8;
    std::mt19937 rng(777u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    const std::vector<uint8_t> w = make_weights(c, rng);
    std::vector<float> x((std::size_t) TM * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(c.n_in, 1);
    void *dw = nullptr, *xq = nullptr;
    float *dx = nullptr, *yref = nullptr, *yt = nullptr;
    const std::size_t ybytes = (std::size_t) TM * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, w.size()), "malloc") || !ck(cudaMalloc(&xq, qcol * TM), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc") || !ck(cudaMalloc(&yref, ybytes), "malloc") ||
        !ck(cudaMalloc(&yt, ybytes * (TM + 1)), "malloc"))
        return false;
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();   // pageable copies may still be in flight for a non-blocking stream
    basalt::kernels::native_quantize_q8_1(dx, xq, c.n_in, TM, s);
    for (int j = 0; j < TM; ++j)
        basalt::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + j * qcol, yref + (std::size_t) j * c.n_out,
                                     c.n_in, c.n_out, 1, s);
    cudaMemsetAsync(yt, 0xff, ybytes * (TM + 1), s);
    for (int T = 1; T <= TM; ++T)
        basalt::kernels::dense_tc(c.type, dw, xq, yt + (std::size_t) T * TM * c.n_out, c.n_in, c.n_out, T, s);
    if (!ck(cudaStreamSynchronize(s), "run")) return false;
    std::vector<float> ref((std::size_t) TM * c.n_out), got((std::size_t) (TM + 1) * TM * c.n_out);
    cudaMemcpy(ref.data(), yref, ybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), yt, got.size() * 4, cudaMemcpyDeviceToHost);
    long long bits = 0, nonfinite = 0;
    const float* g8 = got.data() + (std::size_t) TM * TM * c.n_out;
    for (int T = 1; T <= TM; ++T)
        for (int j = 0; j < T; ++j)
            for (int r = 0; r < c.n_out; ++r) {
                const float a = got[((std::size_t) T * TM + j) * c.n_out + r], b = g8[(std::size_t) j * c.n_out + r];
                if (std::memcmp(&a, &b, 4) != 0) ++bits;
                if (!std::isfinite(a)) ++nonfinite;
            }
    double maxd = 0, maxr = 0, sd = 0, sr = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) g8[i] - (double) ref[i]);
        maxd = std::max(maxd, d);
        maxr = std::max(maxr, std::fabs((double) ref[i]));
        sd += d * d;
        sr += (double) ref[i] * ref[i];
    }
    const double rms = std::sqrt(sd / std::max(sr, 1e-300));
    bool ok = bits == 0 && nonfinite == 0 && rms < 1e-3;
    if (c.type == 8) {   // against float64, for every T's columns; native_mmvq's own error alongside
        std::vector<uint8_t> xh(qcol * TM);
        cudaMemcpy(xh.data(), xq, xh.size(), cudaMemcpyDeviceToHost);
        const std::vector<double> ex = cpu_q8_0(c, w, xh, TM);
        double dd = 0, dn = 0, rr = 0, mx = 0;
        for (std::size_t i = 0; i < ex.size(); ++i) {
            dd += (g8[i] - ex[i]) * (g8[i] - ex[i]);
            dn += (ref[i] - ex[i]) * (ref[i] - ex[i]);
            rr += ex[i] * ex[i];
            mx = std::max(mx, std::fabs(ex[i]));
        }
        const double rd = std::sqrt(dd / std::max(rr, 1e-300)), rn = std::sqrt(dn / std::max(rr, 1e-300));
        double tmax = 0;   // every T's column j against float64 too
        for (int T = 1; T <= TM; ++T)
            for (int j = 0; j < T; ++j)
                for (int r = 0; r < c.n_out; ++r)
                    tmax = std::max(tmax, std::fabs(got[((std::size_t) T * TM + j) * c.n_out + r] -
                                                    ex[(std::size_t) j * c.n_out + r]));
        const bool k = rd < 1e-5 && tmax < 1e-4 * mx;
        std::printf("   %s float64: dense_tc rms %.3e  native_mmvq rms %.3e  worst |d| over all T %.3e  %s\n", c.name, rd,
                    rn, tmax, k ? "PASS" : "FAIL");
        ok = ok && k;
    }
    std::printf("%-12s %5d x %5d  T-columns bitwise: %s (%lld diff, %lld nonfinite)  vs native: max|d|/max|ref| "
                "%.3e  rms %.3e  %s\n",
                c.name, c.n_in, c.n_out, bits == 0 ? "yes" : "NO", bits, nonfinite, maxd / std::max(maxr, 1e-30), rms,
                ok ? "PASS" : "FAIL");
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yt);
    return ok;
}

// dense_tc_pair of a qkv and a z case: each output's T columns bitwise equal to the 8-column call, and its error
// against native_mmvq
bool pair_parity(const Case& a, const Case& b, cudaStream_t s) {
    constexpr int TM = 8;
    std::mt19937 rng(4242u + (unsigned) a.type * 7u + (unsigned) b.type);
    const std::vector<uint8_t> wa = make_weights(a, rng), wb = make_weights(b, rng);
    std::vector<float> x((std::size_t) TM * a.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(a.n_in, 1);
    void *dwa = nullptr, *dwb = nullptr, *xq = nullptr;
    float *dx = nullptr, *ra = nullptr, *rb = nullptr, *ya = nullptr, *yb = nullptr;
    cudaMalloc(&dwa, wa.size()); cudaMalloc(&dwb, wb.size()); cudaMalloc(&xq, qcol * TM);
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&ra, (std::size_t) TM * a.n_out * 4); cudaMalloc(&rb, (std::size_t) TM * b.n_out * 4);
    cudaMalloc(&ya, (std::size_t) (TM + 1) * TM * a.n_out * 4); cudaMalloc(&yb, (std::size_t) (TM + 1) * TM * b.n_out * 4);
    cudaMemcpy(dwa, wa.data(), wa.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dwb, wb.data(), wb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    basalt::kernels::native_quantize_q8_1(dx, xq, a.n_in, TM, s);
    for (int j = 0; j < TM; ++j) {
        basalt::kernels::native_mmvq(a.type, dwa, (const uint8_t*) xq + j * qcol, ra + (std::size_t) j * a.n_out, a.n_in,
                                     a.n_out, 1, s);
        basalt::kernels::native_mmvq(b.type, dwb, (const uint8_t*) xq + j * qcol, rb + (std::size_t) j * b.n_out, b.n_in,
                                     b.n_out, 1, s);
    }
    for (int T = 1; T <= TM; ++T)
        basalt::kernels::dense_tc_pair(a.type, dwa, ya + (std::size_t) T * TM * a.n_out, a.n_out, b.type, dwb,
                                       yb + (std::size_t) T * TM * b.n_out, b.n_out, xq, a.n_in, T, s);
    if (!ck(cudaStreamSynchronize(s), "pair run")) return false;
    bool ok = true;
    for (int m = 0; m < 2; ++m) {
        const Case& c = m == 0 ? a : b;
        float* dy = m == 0 ? ya : yb;
        float* dr = m == 0 ? ra : rb;
        std::vector<float> ref((std::size_t) TM * c.n_out), got((std::size_t) (TM + 1) * TM * c.n_out);
        cudaMemcpy(ref.data(), dr, ref.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
        long long bits = 0;
        const float* g8 = got.data() + (std::size_t) TM * TM * c.n_out;
        for (int T = 1; T <= TM; ++T)
            for (int j = 0; j < T; ++j)
                for (int r = 0; r < c.n_out; ++r)
                    if (std::memcmp(&got[((std::size_t) T * TM + j) * c.n_out + r], &g8[(std::size_t) j * c.n_out + r], 4))
                        ++bits;
        double maxd = 0, maxr = 0, sd = 0, sr = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs((double) g8[i] - (double) ref[i]);
            maxd = std::max(maxd, d);
            maxr = std::max(maxr, std::fabs((double) ref[i]));
            sd += d * d;
            sr += (double) ref[i] * ref[i];
        }
        const double rms = std::sqrt(sd / std::max(sr, 1e-300));
        const bool k = bits == 0 && rms < 1e-3;
        uint64_t h = 1469598103934665603ull;   // FNV-1a of the 8-column output
        const uint8_t* hb = reinterpret_cast<const uint8_t*>(g8);
        for (std::size_t i = 0; i < (std::size_t) TM * c.n_out * 4; ++i) h = (h ^ hb[i]) * 1099511628211ull;
        std::printf("pair %-10s + %-10s [%d] T-columns bitwise: %s (%lld)  vs native: max|d|/max|ref| %.3e  rms %.3e  "
                    "hash %016llx  %s\n",
                    a.name, b.name, m, bits == 0 ? "yes" : "NO", bits, maxd / std::max(maxr, 1e-30), rms,
                    (unsigned long long) h, k ? "PASS" : "FAIL");
        ok = ok && k;
    }
    cudaFree(dwa); cudaFree(dwb); cudaFree(xq); cudaFree(dx); cudaFree(ra); cudaFree(rb); cudaFree(ya); cudaFree(yb);
    return ok;
}

// µs per call: NL calls on NL distinct weight copies (> L2), captured once as a graph, best of `reps` replays
template<typename F>
double time_graph(F&& call, int NL, cudaStream_t s, int reps = 7) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    {
        basalt::kernels::PdlScope ps(true);   // with PDL=1, as the verify graphs launch them
        for (int l = 0; l < NL; ++l) call(l);
    }
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaGraphLaunch(ge, s);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float m = 0;
        cudaEventElapsedTime(&m, e0, e1);
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms[ms.size() / 2] / NL;
}

// --bench-cases NAMES (comma-separated substrings) picks the cases; --bench-warm replays one weight copy (L2-warm)
void bench(cudaStream_t s, const char* pick, bool warm) {
    std::vector<const Case*> sel;
    for (const Case& c : CASES) {
        bool on = pick == nullptr ? (&c == &CASES[0] || &c == &CASES[1] || &c == &CASES[3] || &c == &CASES[4] ||
                                     &c == &CASES[7] || &c == &CASES[12])
                                  : false;
        if (pick != nullptr) {
            std::string all(pick), name(c.name);
            for (std::size_t a = 0; a <= all.size();) {
                std::size_t b = all.find(',', a);
                if (b == std::string::npos) b = all.size();
                if (b > a && name.find(all.substr(a, b - a)) != std::string::npos) on = true;
                a = b + 1;
            }
        }
        if (on) sel.push_back(&c);
    }
    std::printf("\nbench: us per call (median of 7 graph replays), GB/s of the weight bytes%s\n", warm ? ", L2-warm" : "");
    for (const Case* c : sel) {
        const std::size_t wb = basalt::kernels::native_mmvq_weight_bytes(c->type, c->n_in, c->n_out);
        const int NL = warm ? 16 : (int) std::max<std::size_t>(8, (400u << 20) / wb);
        std::mt19937 rng(99u);
        const std::vector<uint8_t> w = make_weights(*c, rng);
        uint8_t* dw = nullptr;
        void* xq = nullptr;
        float *dx = nullptr, *y = nullptr;
        cudaMalloc(&dw, wb * NL);
        for (int l = 0; l < (warm ? 1 : NL); ++l) cudaMemcpy(dw + wb * l, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMalloc(&xq, basalt::kernels::native_q8_1_bytes(c->n_in, 8));
        cudaMalloc(&dx, (std::size_t) 8 * c->n_in * 4);
        cudaMalloc(&y, (std::size_t) 8 * c->n_out * 4);
        std::vector<float> x((std::size_t) 8 * c->n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        basalt::kernels::native_quantize_q8_1(dx, xq, c->n_in, 8, s);
        std::printf("%-12s %5d x %5d  %.1f MB  floor %.2f us at 1.79 TB/s\n", c->name, c->n_in, c->n_out, wb / 1e6,
                    wb / 1.79e6);
        std::printf("   T   native us   GB/s   dtc us   GB/s   speedup\n");
        for (int T = 1; T <= 8; ++T) {
            const double tn = time_graph([&](int l) {
                basalt::kernels::native_mmvq(c->type, dw + wb * (warm ? 0 : l), xq, y, c->n_in, c->n_out, T, s);
            }, NL, s);
            const double td = time_graph([&](int l) {
                basalt::kernels::dense_tc(c->type, dw + wb * (warm ? 0 : l), xq, y, c->n_in, c->n_out, T, s);
            }, NL, s);
            std::printf("  %2d  %8.2f %7.0f %8.2f %7.0f   %.2fx\n", T, tn, wb / tn / 1e3, td, wb / td / 1e3, tn / td);
        }
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(y);
    }
}

// --bench-pair: dense_tc_pair of qkv + z against dense_tc qkv then native z, L2-cold (--bench-warm: L2-warm)
void bench_pair(cudaStream_t s, bool warm) {
    const int pairs[][2] = {{0, 6}, {1, 5}, {3, 5}, {0, 4}, {1, 6}, {3, 6}, {25, 26}};
    std::printf("\npair bench: us per call (median of 7 graph replays)\n");
    for (const auto& pr : pairs) {
        const Case &a = CASES[pr[0]], &b = CASES[pr[1]];
        const std::size_t wba = basalt::kernels::native_mmvq_weight_bytes(a.type, a.n_in, a.n_out);
        const std::size_t wbb = basalt::kernels::native_mmvq_weight_bytes(b.type, b.n_in, b.n_out);
        const std::size_t wb = ((wba + 255) & ~(std::size_t) 255) + ((wbb + 255) & ~(std::size_t) 255);
        const int NL = warm ? 16 : (int) std::max<std::size_t>(8, (400u << 20) / wb), NW = warm ? 1 : NL;
        std::mt19937 rng(5u);
        const std::vector<uint8_t> w0 = make_weights(a, rng), w1 = make_weights(b, rng);
        uint8_t* dw = nullptr;
        void* xq = nullptr;
        float *dx = nullptr, *y0 = nullptr, *y1 = nullptr;
        cudaMalloc(&dw, wb * NW);
        const std::size_t off1 = (wba + 255) & ~(std::size_t) 255;
        for (int l = 0; l < NW; ++l) {
            cudaMemcpy(dw + wb * l, w0.data(), wba, cudaMemcpyHostToDevice);
            cudaMemcpy(dw + wb * l + off1, w1.data(), wbb, cudaMemcpyHostToDevice);
        }
        cudaMalloc(&xq, basalt::kernels::native_q8_1_bytes(a.n_in, 8));
        cudaMalloc(&dx, (std::size_t) 8 * a.n_in * 4);
        cudaMalloc(&y0, (std::size_t) 8 * a.n_out * 4);
        cudaMalloc(&y1, (std::size_t) 8 * b.n_out * 4);
        std::vector<float> x((std::size_t) 8 * a.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        basalt::kernels::native_quantize_q8_1(dx, xq, a.n_in, 8, s);
        std::printf("%-10s + %-10s %.1f MB  floor %.2f us at 1.79 TB/s\n   T   serial us   pair us   GB/s\n", a.name,
                    b.name, (wba + wbb) / 1e6, (wba + wbb) / 1.79e6);
        for (int T = 1; T <= 4; ++T) {
            const double ts = time_graph([&](int l0) {
                const int l = warm ? 0 : l0;
                basalt::kernels::dense_tc(a.type, dw + wb * l, xq, y0, a.n_in, a.n_out, T, s);
                basalt::kernels::native_mmvq(b.type, dw + wb * l + off1, xq, y1, b.n_in, b.n_out, T, s);
            }, NL, s);
            const double tp = time_graph([&](int l0) {
                const int l = warm ? 0 : l0;
                basalt::kernels::dense_tc_pair(a.type, dw + wb * l, y0, a.n_out, b.type, dw + wb * l + off1, y1, b.n_out,
                                               xq, a.n_in, T, s);
            }, NL, s);
            std::printf("  %2d  %8.2f  %8.2f  %6.0f\n", T, ts, tp, (wba + wbb) / tp / 1e3);
        }
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(y0); cudaFree(y1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 9) {
        std::printf("dense_tc needs thread-block clusters; skipped\n");
        return 77;
    }
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    bool only_bench = false, run_bench = false, pair_only = false, bench_pair_on = false, warm = false;
    const char* pick = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench-only") only_bench = run_bench = true;
        else if (a == "--bench") run_bench = true;
        else if (a == "--pair-only") pair_only = true;          // only the dense_tc_pair checks
        else if (a == "--bench-pair") bench_pair_on = true;
        else if (a == "--bench-warm") warm = true;
        else if (a == "--bench-cases" && i + 1 < argc) pick = argv[++i];
        else {
            std::printf("usage: dense_tc_parity [--bench | --bench-only] [--bench-cases NAMES] [--bench-warm] "
                        "[--bench-pair] [--pair-only]\n");
            return 2;
        }
    }
    bool ok = true;
    if (!only_bench) {
        if (!pair_only)
            for (const Case& c : CASES)
                if (c.type != 21 || c.n_in % 2048 == 0) ok = parity(c, s) && ok;
        const int pairs[][2] = {{0, 6}, {1, 5}, {2, 4}, {3, 6}, {0, 5}, {16, 17}, {3, 17}, {20, 21}, {16, 21}, {1, 17},
                                {20, 17}, {25, 26}, {25, 4}, {0, 26}};
        for (const auto& pr : pairs) ok = pair_parity(CASES[pr[0]], CASES[pr[1]], s) && ok;
    }
    if (run_bench) {
        if (bench_pair_on) bench_pair(s, warm);
        else bench(s, pick, warm);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
