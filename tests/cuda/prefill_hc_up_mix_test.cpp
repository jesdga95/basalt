// tests/cuda/prefill_hc_up_mix_test.cpp - gr_up_mix (the hc up product and the mix in one kernel) against the two
// steps it replaces (Gemm::bf16_w, then gr_mix_r) and a double-precision CPU product and mix (GPU, synthetic, no model):
//
//   1. every output (FP32, BF16 and its low part, FP16) bitwise equal to the two steps' at chunk widths from
//      kUpMixMinT up to a long prompt's chunk, with the weights as a BF16 matrix and as an 8-bit copy;
//   2. the mixed rows against the CPU reference (max abs and rel error, for the kernel and for the two steps);
//   3. a token's outputs do not depend on the chunk width or on the token's place in the chunk;
//   4. output hashes of both paths (FNV-1a over the three images);
//   5. unless --no-bench: us per call, TFLOP/s of the product and GB/s of the bytes moved, against the two steps.
//
//     build/prefill_hc_up_mix_test [--no-bench]
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace P = basalt::prefill;
namespace K = basalt::kernels;

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
double f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
int g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++g_fail; }
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// the outputs of one path over T tokens
struct Out {
    float* mixed = nullptr;
    uint16_t *m16 = nullptr, *m16lo = nullptr, *mh = nullptr;
    void alloc(int64_t T) {
        cudaMalloc((void**) &mixed, (size_t) T * N * 4);
        cudaMalloc((void**) &m16, (size_t) T * N * 2);
        cudaMalloc((void**) &m16lo, (size_t) T * N * 2);
        cudaMalloc((void**) &mh, (size_t) T * N * 2);
    }
    void clear(int64_t T) {
        cudaMemset(mixed, 0xff, (size_t) T * N * 4);
        cudaMemset(m16, 0xff, (size_t) T * N * 2);
        cudaMemset(m16lo, 0xff, (size_t) T * N * 2);
        cudaMemset(mh, 0xff, (size_t) T * N * 2);
    }
};
// rows [t0, t0 + n) of a and b bitwise equal, b's rows starting at b_t0
size_t diff_rows(const Out& a, int64_t a_t0, const Out& b, int64_t b_t0, int64_t n) {
    const size_t e = (size_t) n * N;
    std::vector<uint32_t> x(e), y(e);
    std::vector<uint16_t> u(e), v(e);
    size_t bad = 0;
    cudaMemcpy(x.data(), a.mixed + a_t0 * N, e * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), b.mixed + b_t0 * N, e * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < e; ++i) bad += x[i] != y[i];
    const uint16_t* pa[3] = {a.m16, a.m16lo, a.mh};
    const uint16_t* pb[3] = {b.m16, b.m16lo, b.mh};
    for (int k = 0; k < 3; ++k) {
        cudaMemcpy(u.data(), pa[k] + a_t0 * N, e * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), pb[k] + b_t0 * N, e * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < e; ++i) bad += u[i] != v[i];
    }
    return bad;
}
uint64_t hash_out(const Out& o, int64_t T) {
    std::vector<uint8_t> h((size_t) T * N * 4);
    uint64_t x = 1469598103934665603ull;
    cudaMemcpy(h.data(), o.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToHost);
    x = fnv(h.data(), (size_t) T * N * 4, x);
    for (const uint16_t* p : {o.m16, o.mh}) {
        cudaMemcpy(h.data(), p, (size_t) T * N * 2, cudaMemcpyDeviceToHost);
        x = fnv(h.data(), (size_t) T * N * 2, x);
    }
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = !(argc > 1 && std::string(argv[1]) == "--no-bench");
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
    const int64_t TM = 27798;   // one chunk of a long prompt
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    // inputs as the prompt path has them: lo16 = bf16(silu(lo / 4)), R the residual rows, rs their norms' scales,
    // w_norm about 1, w_up with a spread of magnitudes (and the 8-bit copy of it)
    std::vector<uint16_t> lo((size_t) TM * LR), wu((size_t) D * LR);
    for (auto& x : lo) { const float v = 2.0f * nd(rng) / HC; x = bf16(v / (1.0f + std::exp(-v))); }
    for (auto& x : wu) x = bf16(0.05f * nd(rng) * std::exp2((float) (int) (rng() % 6) - 3.0f));
    std::vector<float> R((size_t) TM * D), rs((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng) * 0.7f;
    for (auto& x : rs) x = 1.0f + 0.25f * std::fabs(nd(rng));
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);

    uint16_t *dlo, *dwu;
    float *dR, *drs, *dwn, *gated;
    cudaMalloc((void**) &dlo, lo.size() * 2); cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dR, R.size() * 4); cudaMalloc((void**) &drs, rs.size() * 4); cudaMalloc((void**) &dwn, D * 4);
    cudaMalloc((void**) &gated, (size_t) TM * D * 4);
    cudaMemcpy(dlo, lo.data(), lo.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(drs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), D * 4, cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreate(&st);
    P::Gemm gm;
    std::string err;
    if (!gm.init(st, 32ll << 20, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    const void* wq = K::hc8_adopt(dwu, D, LR);   // the 8-bit copy, as the engine holds the hc matrices
    expect(wq != nullptr, "hc8_adopt");
    if (g_fail) return 1;
    Out a, b;
    a.alloc(TM); b.alloc(TM);
    // the two steps, and the kernel on the same BF16 weights
    auto two_steps = [&](const uint16_t* W, int64_t t0, int64_t T, const Out& o) {
        gm.bf16_w(dlo + t0 * LR, W, gated, T, D, LR);
        P::gr_mix_r(dR + t0 * D, drs + t0 * HC, dwn, gated, o.mixed, o.m16, T, st, o.mh, o.m16lo);
    };
    auto fused = [&](const uint16_t* W, int64_t t0, int64_t T, const Out& o, int64_t o_t0) {
        const uint16_t* w = gm.bf16_rows(W, D, LR);
        const bool ok = w != nullptr && P::gr_up_mix(dlo + t0 * LR, w, dR + t0 * D, drs + t0 * HC, dwn,
                                                     o.mixed + o_t0 * N, o.m16 + o_t0 * N, T, st, o.mh + o_t0 * N,
                                                     o.m16lo + o_t0 * N);
        expect(ok, "gr_up_mix launched");
    };

    // ---------------------------------------------------------------- 1. bitwise vs the two steps
    const int64_t widths[] = {P::kUpMixMinT, 64, 100, 127, 128, 129, 1000, 4099, 16384, TM};
    for (int wsel = 0; wsel < 2; ++wsel) {
        const uint16_t* W = wsel == 0 ? dwu : (const uint16_t*) wq;
        for (int64_t T : widths) {
            a.clear(T); b.clear(T);
            two_steps(W, 0, T, a);
            fused(W, 0, T, b, 0);
            cudaStreamSynchronize(st);
            const size_t bad = diff_rows(a, 0, b, 0, T);
            std::printf("%s T=%-6lld outputs differing from the two steps: %zu of %lld\n", wsel ? "8-bit copy" : "bf16      ",
                        (long long) T, bad, (long long) T * N * 4);
            expect(bad == 0, "bitwise vs the two steps at T=" + std::to_string(T));
        }
    }
    expect(cudaGetLastError() == cudaSuccess, "no CUDA error");

    // ---------------------------------------------------------------- 2. CPU reference (sampled tokens of the last run)
    {
        const int64_t T = TM;
        std::vector<float> mk((size_t) T * N), m2((size_t) T * N);
        cudaMemcpy(mk.data(), b.mixed, mk.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(m2.data(), a.mixed, m2.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<uint16_t> wd(wu.size());
        const uint16_t* wdev = gm.bf16_rows((const uint16_t*) wq, D, LR);
        cudaMemcpy(wd.data(), wdev, wd.size() * 2, cudaMemcpyDeviceToHost);
        double ea = 0, eb = 0, ref = 0;
        for (int s = 0; s < 64; ++s) {
            const int64_t t = s == 0 ? 0 : s == 63 ? T - 1 : (int64_t) (rng() % T);
            for (int d = 0; d < N; ++d) {
                double m = 0;
                for (int c = 0; c < HC; ++c) {
                    const size_t j = (size_t) c * N + d;
                    double g = 0;
                    for (int k = 0; k < LR; ++k) g += f32(lo[(size_t) t * LR + k]) * f32(wd[j * LR + k]);
                    const double x = (double) R[(size_t) t * D + j] * rs[(size_t) t * HC + c] * wn[j];
                    m += x / (1.0 + std::exp(-g));
                }
                m /= HC;
                ea = std::max(ea, std::fabs(mk[(size_t) t * N + d] - m));
                eb = std::max(eb, std::fabs(m2[(size_t) t * N + d] - m));
                ref = std::max(ref, std::fabs(m));
            }
        }
        std::printf("vs CPU double (64 tokens of T=%lld): gr_up_mix max abs %.3e rel %.3e; two steps max abs %.3e rel %.3e\n",
                    (long long) T, ea, ea / ref, eb, eb / ref);
        expect(ea / ref < 1e-5, "gr_up_mix near the CPU reference");
    }

    // ---------------------------------------------------------------- 3. width and position independence
    {
        const int64_t T = 1000;
        b.clear(T);
        fused(dwu, 0, T, b, 0);
        Out c;
        c.alloc(T);
        c.clear(T);
        struct Part { int64_t t0, n; } parts[] = {{0, 33}, {33, 300}, {333, 129}, {462, 538}};
        for (const Part& p : parts) fused(dwu, p.t0, p.n, c, p.t0);
        cudaStreamSynchronize(st);
        size_t bad = diff_rows(b, 0, c, 0, T);
        // a token at a different place in its chunk: tokens 77.. as the first rows of a chunk
        c.clear(T);
        fused(dwu, 77, 200, c, 0);
        cudaStreamSynchronize(st);
        bad += diff_rows(b, 77, c, 0, 200);
        std::printf("width / position independence: %zu outputs differ\n", bad);
        expect(bad == 0, "outputs independent of the chunk width and the token's place");
    }

    // ---------------------------------------------------------------- 4. hashes
    {
        const int64_t T = 4099;
        a.clear(T); b.clear(T);
        two_steps((const uint16_t*) wq, 0, T, a);
        fused((const uint16_t*) wq, 0, T, b, 0);
        cudaStreamSynchronize(st);
        const uint64_t ha = hash_out(a, T), hb = hash_out(b, T);
        std::printf("hash T=%lld: two steps %016llx, gr_up_mix %016llx\n", (long long) T, (unsigned long long) ha,
                    (unsigned long long) hb);
        expect(ha == hb, "hashes equal");
    }

    // ---------------------------------------------------------------- 5. microbench
    if (bench) {
        const int64_t T = TM;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto time_us = [&](auto&& f) {
            for (int i = 0; i < 3; ++i) f();
            cudaEventRecord(e0, st);
            const int reps = 20;
            for (int i = 0; i < reps; ++i) f();
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms / reps;
        };
        const double t2 = time_us([&] { two_steps((const uint16_t*) wq, 0, T, a); });
        const double tf = time_us([&] { fused((const uint16_t*) wq, 0, T, b, 0); });
        const double fl = 2.0 * T * D * LR;
        const double by2 = (double) T * D * 4 * 3 + (double) T * N * (4 + 2 + 2 + 2);   // gate write + read, R, outputs
        const double byf = (double) T * D * 4 + (double) T * N * (4 + 2 + 2 + 2);
        std::printf("bench T=%lld: two steps %.1f us (%.1f TFLOP/s, %.0f GB/s); gr_up_mix %.1f us (%.1f TFLOP/s, %.0f GB/s), "
                    "x%.2f\n",
                    (long long) T, t2, fl / t2 / 1e6, by2 / t2 / 1e3, tf, fl / tf / 1e6, byf / tf / 1e3, t2 / tf);
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
