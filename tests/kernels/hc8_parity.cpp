// tests/kernels/hc8_parity.cpp - the reads and products that take a hyper-connection matrix held only as its 8-bit copy
// (DEC_HC8, hc_fused.hpp's hc8_adopt), real geometry and random inputs (GPU, synthetic, no model):
//
//   1. hc8_gr_read_multi (what fused_gr_read, fused_gr_read_multi and gr_read run for such matrices) against
//      fused_gr_read_multi on the bf16 matrices (the error the copies add) and on the copies dequantized to bf16 (the
//      kernels' own error), 1..8 tokens, with and without the pending write and the injection; every token's outputs
//      bitwise the same whatever the token count and column; gr_read bitwise the one-token read.
//   2. Gemm::bf16_w (the prompt path's product on a copy) against Gemm::bf16 on the bf16 matrix and against a
//      double-precision product with the copy's exact values, on the prompt path's hc shapes, with a scratch that holds
//      the whole matrix and with one that slices it (within fp32 rounding: cuBLAS's kernel can change with the width).
//   3. --bench: the read's time (weights cold in L2) and the products' time against the bf16 ones.
//
//     build/hc8_parity [--bench]
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 1024), e - 25);
    return s ? -v : v;
}
std::vector<float> get(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}
struct Err {
    double abs = 0, ref = 0;
    void add(float a, float b) { abs = std::max(abs, (double) std::fabs(a - b)); ref = std::max(ref, (double) std::fabs(b)); }
    double rel() const { return ref > 0 ? abs / ref : abs; }
};
int g_fail = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
    const float EPS = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    for (int k = 0; k < 64; ++k) wd[(size_t) (k * 5) * D + k * 101] = bf16(0.6f);   // outliers that set a group's scale

    uint16_t *dwd, *dwu, *dwi, *dqd, *dqu, *dqi;   // bf16, and the copies dequantized to bf16
    float *dwn, *dbo, *dinj;
    cudaMalloc((void**) &dwd, wd.size() * 2); cudaMalloc((void**) &dwu, wu.size() * 2); cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMalloc((void**) &dqd, wd.size() * 2); cudaMalloc((void**) &dqu, wu.size() * 2); cudaMalloc((void**) &dqi, wi.size() * 2);
    cudaMalloc((void**) &dwn, D * 4); cudaMalloc((void**) &dbo, bo.size() * 4); cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    const void* kd = K::hc8_adopt(dwd, LR, D);
    const void* ku = K::hc8_adopt(dwu, D, LR);
    const void* ki = K::hc8_adopt(dwi, HC, D);
    K::Hc8View vd, vu, vi;
    expect(kd && ku && ki && K::hc8_view(kd, vd) && K::hc8_view(ku, vu) && K::hc8_view(ki, vi) && vd.owned, "adopt");
    if (g_fail) return 1;
    K::hc8_dequant_bf16(vd, 0, LR, dqd, nullptr);
    K::hc8_dequant_bf16(vu, 0, D, dqu, nullptr);
    K::hc8_dequant_bf16(vi, 0, HC, dqi, nullptr);
    cudaDeviceSynchronize();
    {   // the copies' quantization error
        std::vector<uint16_t> q(wd.size());
        cudaMemcpy(q.data(), dqd, q.size() * 2, cudaMemcpyDeviceToHost);
        double e2 = 0, w2 = 0;
        for (size_t i = 0; i < q.size(); ++i) {
            const double a = f32(wd[i]), b = f32(q[i]);
            e2 += (a - b) * (a - b); w2 += a * a;
        }
        std::printf("w_down copy (dequantized to bf16): relative rms error %.3e\n", std::sqrt(e2 / w2));
    }

    // ---------------------------------------------------------------- 1. the read
    struct Out { float *R, *xn, *lo, *rs, *io, *mix; };
    auto alloc_out = [&](Out& o) {
        cudaMalloc((void**) &o.R, (size_t) TM * D * 4); cudaMalloc((void**) &o.xn, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.lo, (size_t) TM * LR * 4); cudaMalloc((void**) &o.rs, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.io, (size_t) TM * HC * 4); cudaMalloc((void**) &o.mix, (size_t) TM * N * 4);
    };
    Out o[3];
    for (auto& x : o) alloc_out(x);
    // run `which` (0: hc8 copies, 1: bf16, 2: dequantized copies through the bf16 read) on tokens `tok` in columns 0..n-1
    auto run = [&](int which, Out& out, const std::vector<int>& tok, bool apply, bool inject) {
        cudaMemcpy(out.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        K::FusedGrArgs a[K::kFusedGrMaxT];
        for (size_t c = 0; c < tok.size(); ++c) {
            const int t = tok[c];
            K::FusedGrArgs& x = a[c];
            x.R = out.R + (size_t) t * D; x.R_out = out.R + (size_t) t * D; x.apply = apply;
            x.bo_prev = dbo + (size_t) t * N; x.inj_prev = dinj + (size_t) t * HC;
            x.w_norm = dwn;
            x.w_down = (const uint16_t*) (which == 0 ? kd : which == 1 ? (const void*) dwd : (const void*) dqd);
            x.w_up = (const uint16_t*) (which == 0 ? ku : which == 1 ? (const void*) dwu : (const void*) dqu);
            x.w_inject = !inject ? nullptr : (const uint16_t*) (which == 0 ? ki : which == 1 ? (const void*) dwi : (const void*) dqi);
            x.eps = EPS; x.lo = out.lo + (size_t) t * LR; x.rs = out.rs + (size_t) t * HC;
            x.inject_out = out.io + (size_t) t * HC;
            x.mixed = out.mix + (size_t) t * N;
        }
        K::fused_gr_read_multi(a, (int) tok.size(), out.xn, nullptr);
        cudaDeviceSynchronize();
    };
    std::vector<int> all(TM);
    for (int t = 0; t < TM; ++t) all[t] = t;
    for (int inject = 1; inject >= 0; --inject)
        for (int apply = 0; apply < 2; ++apply) {
            run(0, o[0], all, apply, inject);
            run(1, o[1], all, apply, inject);
            run(2, o[2], all, apply, inject);
            Err e[2][5];
            const float* dev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
            const size_t n[5] = {(size_t) TM * N, (size_t) TM * LR, (size_t) TM * HC, (size_t) TM * HC, (size_t) TM * D};
            const char* nm[5] = {"mixed", "lo", "inject", "rs", "R'"};
            std::vector<float> h0[5], h1[5], h2[5];
            for (int k = 0; k < 5; ++k) {
                if (k == 2 && !inject) continue;
                float* p0[5] = {o[0].mix, o[0].lo, o[0].io, o[0].rs, o[0].R};
                float* p1[5] = {o[1].mix, o[1].lo, o[1].io, o[1].rs, o[1].R};
                float* p2[5] = {o[2].mix, o[2].lo, o[2].io, o[2].rs, o[2].R};
                h0[k] = get(p0[k], n[k]); h1[k] = get(p1[k], n[k]); h2[k] = get(p2[k], n[k]);
                for (size_t i = 0; i < n[k]; ++i) { e[0][k].add(h0[k][i], h1[k][i]); e[1][k].add(h0[k][i], h2[k][i]); }
            }
            (void) dev;
            std::printf("read, 8 tokens, %s, %s:\n", apply ? "pending write" : "no write", inject ? "injection" : "no injection");
            for (int k = 0; k < 5; ++k) {
                if (k == 2 && !inject) continue;
                std::printf("  %-6s vs bf16: max abs %.3e (rel to max %.3e) | vs dequantized copy: max abs %.3e (rel %.3e)\n",
                            nm[k], e[0][k].abs, e[0][k].rel(), e[1][k].abs, e[1][k].rel());
                expect(e[1][k].rel() < 2e-3, "the 8-bit read against the dequantized copy");
                expect(e[0][k].rel() < 5e-2, "the 8-bit read against bf16");
            }
            // every token's outputs the same bits whatever the token count and column
            const std::vector<std::vector<int>> sets = {{0}, {5}, {7, 2, 4}, {3, 1}, {6, 0, 2, 5, 1, 7, 4}};
            bool same = true;
            for (const auto& s : sets) {
                run(0, o[2], s, apply, inject);
                float* p2[5] = {o[2].mix, o[2].lo, o[2].io, o[2].rs, o[2].R};
                for (int k = 0; k < 5; ++k) {
                    if (k == 2 && !inject) continue;
                    const std::vector<float> h = get(p2[k], n[k]);
                    const size_t per = n[k] / TM;
                    for (int t : s) same = same && std::memcmp(h.data() + t * per, h0[k].data() + t * per, per * 4) == 0;
                }
            }
            std::printf("  token count / column independence: %s\n", same ? "bitwise" : "DIFFERS");
            expect(same, "token count / column independence");
        }
    {   // gr_read on the copies: the one-token read
        const K::GrShapes gs{N, HC, LR};
        void* wsb = nullptr;
        cudaMalloc(&wsb, K::gr_workspace_bytes(gs));
        K::GrWorkspace ws;
        K::gr_workspace_init(gs, wsb, ws);
        float *mix, *io;
        cudaMalloc((void**) &mix, N * 4); cudaMalloc((void**) &io, HC * 4);
        cudaMemcpy(o[2].R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        K::gr_read(o[2].R + 3 * D, dwn, (const uint16_t*) kd, (const uint16_t*) ku, (const uint16_t*) ki, EPS, gs, ws, mix, io,
                   nullptr);
        run(0, o[0], {3}, false, true);
        const std::vector<float> a = get(mix, N), b = get(o[0].mix + 3 * N, N), c = get(io, HC), d = get(o[0].io + 3 * HC, HC);
        const bool same = std::memcmp(a.data(), b.data(), N * 4) == 0 && std::memcmp(c.data(), d.data(), HC * 4) == 0;
        std::printf("gr_read on the copies vs the one-token read: %s\n", same ? "bitwise" : "DIFFERS");
        expect(same, "gr_read");
    }

    // ---------------------------------------------------------------- 2. the prompt path's product
    cudaStream_t st;
    cudaStreamCreate(&st);
    std::string err;
    basalt::prefill::Gemm big, small;
    if (!big.init(st, 32ll << 20, err) || !small.init(st, 1ll << 20, err)) { std::printf("FAIL: gemm init: %s\n", err.c_str()); return 1; }
    struct Shape { int64_t T, N, K; const void* key; const uint16_t* w; const char* what; };
    const Shape shapes[] = {
        {512, LR, D, kd, dwd, "hc down 512 x 320 x 10240"},
        {512, D, LR, ku, dwu, "hc up   512 x 10240 x 320"},
        {512, HC, D, ki, dwi, "inject  512 x 4 x 10240"},
        {37, LR, D, kd, dwd, "hc down  37 x 320 x 10240"},
        {1, D, LR, ku, dwu, "hc up     1 x 10240 x 320"},
    };
    std::vector<int8_t> hq;
    std::vector<uint16_t> hs;
    for (const Shape& s : shapes) {
        K::Hc8View v;
        K::hc8_view(s.key, v);
        hq.resize((size_t) v.rows * v.cols);
        hs.resize((size_t) v.rows * v.srow);
        cudaMemcpy(hq.data(), v.q, hq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), v.s, hs.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<uint16_t> X((size_t) s.T * s.K);
        for (size_t i = 0; i < X.size(); ++i) {
            X[i] = bf16(nd(rng));
        }
        uint16_t* dX;
        float *Y0, *Y1, *Y2;
        const int64_t ldy = s.N + 8;
        cudaMalloc((void**) &dX, X.size() * 2);
        cudaMalloc((void**) &Y0, (size_t) s.T * ldy * 4); cudaMalloc((void**) &Y1, (size_t) s.T * ldy * 4);
        cudaMalloc((void**) &Y2, (size_t) s.T * ldy * 4);
        cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice);
        big.bf16_w(dX, s.w, Y0, s.T, s.N, s.K, ldy);                         // bf16 matrix
        big.bf16_w(dX, (const uint16_t*) s.key, Y1, s.T, s.N, s.K, ldy);     // copy, one slice
        small.bf16_w(dX, (const uint16_t*) s.key, Y2, s.T, s.N, s.K, ldy);   // copy, row slices
        cudaStreamSynchronize(st);
        const std::vector<float> y0 = get(Y0, (size_t) s.T * ldy), y1 = get(Y1, (size_t) s.T * ldy),
                                 y2 = get(Y2, (size_t) s.T * ldy);
        Err eb, ex;
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t n = 0; n < s.N; ++n) eb.add(y1[t * ldy + n], y0[t * ldy + n]);
        std::uniform_int_distribution<int64_t> pt(0, s.T - 1), pn(0, s.N - 1);
        for (int k = 0; k < 256; ++k) {   // the copy's exact values in double
            const int64_t t = pt(rng), n = pn(rng);
            double ref = 0;
            for (int64_t i = 0; i < s.K; ++i) {
                const double x = (double) f32(X[t * s.K + i]);
                ref += x * (double) hq[n * s.K + i] * (double) f16f(hs[n * v.srow + i / 32]);
            }
            ex.add(y1[t * ldy + n], (float) ref);
        }
        Err es;   // row slices: cuBLAS may pick another kernel for the narrower product (another summation order)
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t n = 0; n < s.N; ++n) es.add(y2[t * ldy + n], y1[t * ldy + n]);
        const bool sliced = es.rel() < 1e-5;
        std::printf("%s: vs bf16 max abs %.3e (rel %.3e) | vs the copy in double max abs %.3e (rel %.3e) | sliced vs whole "
                    "rel %.1e\n", s.what, eb.abs, eb.rel(), ex.abs, ex.rel(), es.rel());
        expect(ex.rel() < 5e-3, "the product against the copy in double");
        expect(eb.rel() < 5e-2, "the product against bf16");
        expect(sliced, "row slices");
        cudaFree(dX); cudaFree(Y0); cudaFree(Y1); cudaFree(Y2);
    }

    // ---------------------------------------------------------------- 3. times
    if (bench) {
        void* evict = nullptr;
        const size_t eb = (size_t) 256 << 20;
        cudaMalloc(&evict, eb);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto time_us = [&](auto&& fn, int iters, bool cold) {
            double total = 0;
            for (int i = 0; i < iters + 3; ++i) {
                if (cold) cudaMemsetAsync(evict, i & 0xff, eb, st);
                cudaEventRecord(e0, st);
                fn();
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 3) total += ms;
            }
            return total * 1e3 / iters;
        };
        const double b8 = (double) (vd.bytes + vu.bytes + vi.bytes), b16 = (double) (wd.size() + wu.size() + wi.size()) * 2;
        for (int T : {1, 3, 8}) {
            K::FusedGrArgs a[2][K::kFusedGrMaxT];
            for (int w = 0; w < 2; ++w)
                for (int t = 0; t < T; ++t) {
                    K::FusedGrArgs& x = a[w][t];
                    x.R = o[w].R + (size_t) t * D; x.R_out = o[w].R + (size_t) t * D; x.apply = true;
                    x.bo_prev = dbo + (size_t) t * N; x.inj_prev = dinj + (size_t) t * HC; x.w_norm = dwn;
                    x.w_down = (const uint16_t*) (w ? (const void*) dwd : kd);
                    x.w_up = (const uint16_t*) (w ? (const void*) dwu : ku);
                    x.w_inject = (const uint16_t*) (w ? (const void*) dwi : ki);
                    x.eps = EPS; x.lo = o[w].lo + (size_t) t * LR; x.rs = o[w].rs + (size_t) t * HC;
                    x.inject_out = o[w].io + (size_t) t * HC; x.mixed = o[w].mix + (size_t) t * N;
                }
            const double u8 = time_us([&] { K::fused_gr_read_multi(a[0], T, o[0].xn, st); }, 50, true);
            const double u16 = time_us([&] { K::fused_gr_read_multi(a[1], T, o[1].xn, st); }, 50, true);
            std::printf("read T %d (cold L2): 8-bit %.1f us (%.0f GB/s of %.1f MB) | bf16 fused_gr %.1f us (%.0f GB/s of %.1f MB)\n",
                        T, u8, b8 / u8 / 1e3, b8 / 1e6, u16, b16 / u16 / 1e3, b16 / 1e6);
        }
        for (int64_t T : {512, 2048, 8192}) {
            uint16_t* dX;
            float* Y;
            cudaMalloc((void**) &dX, (size_t) T * D * 2);
            cudaMalloc((void**) &Y, (size_t) T * D * 4);
            cudaMemset(dX, 0, (size_t) T * D * 2);
            const double dn16 = time_us([&] { big.bf16(dX, dwd, Y, T, LR, D); }, 20, false);
            const double dn8 = time_us([&] { big.bf16_w(dX, (const uint16_t*) kd, Y, T, LR, D); }, 20, false);
            const double up16 = time_us([&] { big.bf16(dX, dwu, Y, T, D, LR); }, 20, false);
            const double up8 = time_us([&] { big.bf16_w(dX, (const uint16_t*) ku, Y, T, D, LR); }, 20, false);
            const double dq = time_us([&] { K::hc8_dequant_bf16(vd, 0, LR, big.scratch(), st); }, 20, false);
            std::printf("product T %lld: down bf16 %.1f us, copy %.1f us | up bf16 %.1f us, copy %.1f us | dequantize one "
                        "matrix %.1f us (%.0f GB/s of %.1f MB)\n",
                        (long long) T, dn16, dn8, up16, up8, dq, (vd.bytes + wd.size() * 2.0) / dq / 1e3,
                        (vd.bytes + wd.size() * 2.0) / 1e6);
            cudaFree(dX); cudaFree(Y);
        }
    }
    std::printf("%s\n", g_fail ? "hc8_parity: FAILED" : "hc8_parity: ok");
    return g_fail ? 1 : 0;
}
