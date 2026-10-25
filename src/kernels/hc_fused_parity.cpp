// src/kernels/hc_fused_parity.cpp - hc_fused_read against the kernels it replaces (fused_gr_read_multi on the 8-bit
// copies, then the MMVF router GEMV and the native top-10), with real geometry and random inputs.  The fused read
// has its own reduction order and fast exponentials, so against those kernels each output must agree within
// kRelTol of its largest value and pick the same experts; against itself it must be bitwise: repeated launches,
// every window width, reversed token order and overlapped launches (--stress).  A read that counts its progress
// against one that does not, its q8_1 rows against a CPU quantizer.  Then the time of both with the weights hot in L2
// and cold (L2 overwritten before every read).
//
//     build/hc_fused_parity [iters] [T min] [T max] [--tol] [--check-only] [--stress N]
//
// --tol prints every case's differences against the plain kernels, not only the failing ones.
#include "basalt/core/diag.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/shared_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace K = basalt::kernels;


// the fused read's largest deviation from the plain kernels, relative to the output's largest value
constexpr double kRelTol = 2e-6;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // hc_fused_parity [ITERS [T_LO [T_HI]]] [--tol] [--stamp-reps N] [--block T:K,...]
    std::vector<std::string> pos;
    bool tol = false;
    int stamp_reps = 1;
    bool check_only = false;   // --check-only: stop after the output checks (for the sanitizers)
    int rot = 0, rotpad = 0, stress = 0;   // --rot N: the timing and stamp runs cycle through N weight sets (every read starts with its matrices out of L2)
    std::string block_list = "2:4,2:100,2:165,6:4,6:100,8:169";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tol") tol = true;
        else if (a == "--stamp-reps" && i + 1 < argc) stamp_reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--rot" && i + 1 < argc) rot = std::max(0, std::atoi(argv[++i]));
        else if (a == "--rotpad" && i + 1 < argc) rotpad = std::atoi(argv[++i]);
        else if (a == "--stress" && i + 1 < argc) stress = std::atoi(argv[++i]);
        else if (a == "--check-only") check_only = true;
        else if (a == "--block" && i + 1 < argc) block_list = argv[++i];
        else pos.push_back(a);
    }
    const int iters = pos.size() > 0 ? std::atoi(pos[0].c_str()) : 200;
    const int t_lo = pos.size() > 1 ? std::atoi(pos[1].c_str()) : 1, t_hi = pos.size() > 2 ? std::atoi(pos[2].c_str()) : 8;
    if (!K::hc_fused_supported()) {
        std::printf("hc_fused_parity: the fused read does not run on this device\n");
        return 1;
    }
    K::fused_gr_check();
    const int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, KT = 10, TM = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D), wr((size_t) NE * N);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    for (auto& x : wr) x = bf16(0.05f * nd(rng));
    // a few exact ties in the router rows: two experts with the same weights
    std::memcpy(wr.data() + (size_t) 77 * N, wr.data() + (size_t) 300 * N, (size_t) N * 2);

    struct Out {
        float *R, *xn, *lo, *rs, *io, *mix, *logits, *w;
        int32_t* ids;
    };
    auto alloc_out = [&](Out& o) {
        cudaMalloc((void**) &o.R, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.xn, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.lo, (size_t) TM * LR * 4);
        cudaMalloc((void**) &o.rs, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.io, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.mix, (size_t) TM * N * 4);
        cudaMalloc((void**) &o.logits, (size_t) TM * NE * 4);
        cudaMalloc((void**) &o.w, (size_t) TM * KT * 4);
        cudaMalloc((void**) &o.ids, (size_t) TM * KT * 4);
    };
    Out o[2];
    alloc_out(o[0]);
    alloc_out(o[1]);
    float *dbo, *dinj, *dwn;
    uint16_t *dwd, *dwu, *dwi, *dwr;
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMalloc((void**) &dwr, wr.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwr, wr.data(), wr.size() * 2, cudaMemcpyHostToDevice);
    const bool q8 = K::hc8_enabled();
    if (q8) {
        K::hc8_prepare(dwd, LR, D);
        K::hc8_prepare(dwu, D, LR);
        K::hc8_prepare(dwi, HC, D);
    }
    const size_t evict_bytes = (size_t) 192 << 20;
    void* devict = nullptr;
    if (cudaMalloc(&devict, evict_bytes) != cudaSuccess) {
        std::printf("hc_fused_parity: no room for the L2 eviction buffer\n");
        return 1;
    }
    float* dRbak = nullptr;   // R and bo rewritten after the eviction: in the decode they come from the previous kernel
    cudaMalloc((void**) &dRbak, R.size() * 4);
    cudaMemcpy(dRbak, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    // w_norm too: the previous read prefetches it (HcFusedArgs::pf) in the decode
    float* dwnbak = nullptr;
    cudaMalloc((void**) &dwnbak, wn.size() * 4);
    cudaMemcpy(dwnbak, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    auto evict = [&](int f, int i) {
        cudaMemsetAsync(devict, i & 0xff, evict_bytes, s);
        cudaMemcpyAsync(o[f].R, dRbak, R.size() * 4, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dbo, dRbak, bo.size() * 4, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dwn, dwnbak, wn.size() * 4, cudaMemcpyDeviceToDevice, s);
    };
    struct WSet {
        uint16_t *wd, *wu, *wi, *wr;
    };
    std::vector<WSet> ws;
    int wcur = -1;   // >= 0: f = 1 reads this weight set
    for (int i = 0; i < rot; ++i) {
        WSet w{};
        if (rotpad > 0) {   // address space between the sets: every read starts on pages its predecessors did not touch
            void* pad = nullptr;
            cudaMalloc(&pad, (size_t) rotpad << 20);
        }
        cudaMalloc((void**) &w.wd, wd.size() * 2);
        cudaMalloc((void**) &w.wu, wu.size() * 2);
        cudaMalloc((void**) &w.wi, wi.size() * 2);
        cudaMalloc((void**) &w.wr, wr.size() * 2);
        cudaMemcpy(w.wd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(w.wu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(w.wi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(w.wr, wr.data(), wr.size() * 2, cudaMemcpyHostToDevice);
        if (q8) {
            K::hc8_prepare(w.wd, LR, D);
            K::hc8_prepare(w.wu, D, LR);
            K::hc8_prepare(w.wi, HC, D);
        }
        ws.push_back(w);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // f = 0: the reference kernels; f = 1: the fused read
    auto run = [&](int f, int T, bool apply, bool inject, bool router) {
        Out& x = o[f];

        K::PdlScope ps(true);
        if (f == 0) {
            K::FusedGrArgs a[TM];
            for (int t = 0; t < T; ++t) {
                a[t].R = x.R + (size_t) t * D; a[t].R_out = x.R + (size_t) t * D; a[t].apply = apply;
                a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                a[t].lo = x.lo + (size_t) t * LR; a[t].rs = x.rs + (size_t) t * HC;
                a[t].inject_out = x.io + (size_t) t * HC; a[t].mixed = x.mix + (size_t) t * N;
            }
            K::fused_gr_read_multi(a, T, x.xn, s);
            if (router) {
                K::bf16_gemv_fp32_mmvf_multi(x.mix, N, dwr, x.logits, NE, N, NE, T, s);
                K::native_router_top10_multi(x.logits, x.ids, x.w, T, s);
            }
        } else {
            K::HcFusedArgs a;
            const WSet w = wcur >= 0 ? ws[wcur] : WSet{dwd, dwu, dwi, dwr};
            a.T = T; a.R = x.R; a.R_out = x.R; a.apply = apply; a.bo_prev = dbo; a.inj_prev = dinj;
            a.w_norm = dwn; a.w_down = w.wd; a.w_up = w.wu; a.w_inject = inject ? w.wi : nullptr;
            a.lo = x.lo; a.rs = x.rs; a.inject_out = x.io; a.mixed = x.mix; a.xn = x.xn;
            if (router) { a.w_router = w.wr; a.logits = x.logits; a.ids = x.ids; a.weights = x.w; }
            K::hc_fused_read(a, s);
        }
    };
    auto reset = [&](int f) {
        Out& x = o[f];
        cudaMemcpy(x.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(x.lo, 0xff, (size_t) TM * LR * 4);
        cudaMemset(x.rs, 0xff, (size_t) TM * HC * 4);
        cudaMemset(x.io, 0xff, (size_t) TM * HC * 4);
        cudaMemset(x.mix, 0xff, (size_t) TM * N * 4);
        cudaMemset(x.logits, 0xff, (size_t) TM * NE * 4);
        cudaMemset(x.w, 0xff, (size_t) TM * KT * 4);
        cudaMemset(x.ids, 0xff, (size_t) TM * KT * 4);
    };
    auto fetch = [&](int f, int T) {
        Out& x = o[f];
        std::vector<float> v((size_t) T * (D + LR + HC + HC + N + NE + KT + KT));
        float* p = v.data();
        cudaMemcpy(p, x.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
        cudaMemcpy(p, x.lo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) T * LR;
        cudaMemcpy(p, x.rs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, x.io, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, x.mix, (size_t) T * N * 4, cudaMemcpyDeviceToHost); p += (size_t) T * N;
        cudaMemcpy(p, x.logits, (size_t) T * NE * 4, cudaMemcpyDeviceToHost); p += (size_t) T * NE;
        cudaMemcpy(p, x.w, (size_t) T * KT * 4, cudaMemcpyDeviceToHost); p += (size_t) T * KT;
        cudaMemcpy(p, x.ids, (size_t) T * KT * 4, cudaMemcpyDeviceToHost);
        return v;
    };
    static const char* const part_name[8] = {"R", "lo", "rs", "inject", "mixed", "logits", "weights", "ids"};
    auto part_of = [&](size_t i, int T) {
        const size_t sz[8] = {(size_t) T * D, (size_t) T * LR, (size_t) T * HC, (size_t) T * HC, (size_t) T * N,
                              (size_t) T * NE, (size_t) T * KT, (size_t) T * KT};
        for (int k = 0; k < 8; ++k) {
            if (i < sz[k]) return k;
            i -= sz[k];
        }
        return 7;
    };

    int failures = 0;
    // --tol (a path that is not bitwise): per output the max |diff| against the plain kernels (and relative to the
    // largest |value|), routed experts that differ, and - bit for bit - each token's outputs against the same token in a
    // window of t_hi tokens
    std::vector<std::vector<float>> by_t(t_hi + 1);
    for (int T = t_lo; T <= t_hi; ++T)
        for (int mode = 0; mode < 3; ++mode) {   // 0: attention-side read, 1: FFN read + router, 2: the head's read
            const bool inject = mode != 2, router = mode == 1;
            for (int apply = 0; apply < 2; ++apply) {
                std::vector<float> v[2];
                for (int f = 0; f < 2; ++f) {
                    reset(f);
                    run(f, T, apply, inject, router);
                    if (cudaStreamSynchronize(s) != cudaSuccess) {
                        std::printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                        return 1;
                    }
                    v[f] = fetch(f, T);
                }
                // repeated launches (the counters must come back to zero) give the same bits
                reset(1);
                for (int i = 0; i < 3; ++i) {
                    cudaMemcpyAsync(o[1].R, R.data(), (size_t) T * D * 4, cudaMemcpyHostToDevice, s);
                    run(1, T, apply, inject, router);
                }
                cudaStreamSynchronize(s);
                const std::vector<float> v2 = fetch(1, T);
                {
                    double mx[8] = {0}, ref[8] = {0}, sum[8] = {0};
                    size_t cnt[8] = {0};
                    for (size_t i = 0; i < v[0].size(); ++i) {
                        const int pt = part_of(i, T);
                        if ((!inject && pt == 3) || (!router && pt >= 5) || pt == 7) continue;
                        const double d = std::fabs((double) v[0][i] - (double) v[1][i]);
                        mx[pt] = std::max(mx[pt], d); sum[pt] += d; ++cnt[pt];
                        ref[pt] = std::max(ref[pt], (double) std::fabs(v[0][i]));
                    }
                    int idiff = 0;
                    if (router) {
                        const size_t base = v[0].size() - (size_t) T * KT;
                        for (int k = 0; k < T; ++k)
                            for (int j = 0; j < KT; ++j) {
                                bool found = false;
                                for (int j2 = 0; j2 < KT; ++j2)
                                    found |= v[0][base + k * KT + j] == v[1][base + k * KT + j2];
                                idiff += found ? 0 : 1;
                            }
                    }
                    bool bad = std::memcmp(v[1].data(), v2.data(), v[1].size() * 4) != 0 || idiff != 0;
                    for (int pt : {0, 1, 2, 3, 4, 5, 6})
                        if (cnt[pt] && ref[pt] > 0 && mx[pt] > kRelTol * ref[pt]) bad = true;
                    if (bad) ++failures;
                    if (!tol && !bad) {
                        if (mode == 1 && apply == 1) by_t[T] = v[1];
                        continue;
                    }
                    std::printf("T %d mode %d apply %d%s:", T, mode, apply, bad ? " FAIL" : "");
                    for (int pt : {0, 1, 4, 5, 6})
                        if (cnt[pt]) std::printf(" %s max %.2e (rel %.2e) mean %.2e |", part_name[pt], mx[pt],
                                                 ref[pt] > 0 ? mx[pt] / ref[pt] : 0.0, sum[pt] / cnt[pt]);
                    uint64_t hsh = 1469598103934665603ull;   // the fused read's bits, to compare runs of different gates
                    for (size_t i = 0; i < v[1].size(); ++i) {
                        uint32_t u;
                        std::memcpy(&u, &v[1][i], 4);
                        hsh = (hsh ^ u) * 1099511628211ull;
                    }
                    std::printf(" routed experts differing %d of %d | hash %016llx\n", idiff, router ? T * KT : 0,
                                (unsigned long long) hsh);
                    if (mode == 1 && apply == 1) by_t[T] = v[1];
                }
            }
        }
    if (!by_t[t_hi].empty()) {   // token k of a window of T against token k of the widest window
        int bad = 0;
        const int TH = t_hi;
        const size_t szh[8] = {(size_t) TH * D, (size_t) TH * LR, (size_t) TH * HC, (size_t) TH * HC, (size_t) TH * N,
                               (size_t) TH * NE, (size_t) TH * KT, (size_t) TH * KT};
        const int per[8] = {D, LR, HC, HC, N, NE, KT, KT};
        for (int T = t_lo; T < t_hi; ++T) {
            if (by_t[T].empty()) continue;
            size_t ot = 0, oh = 0;
            for (int pt = 0; pt < 8; ++pt) {
                for (size_t i = 0; i < (size_t) T * per[pt]; ++i)
                    if (std::memcmp(&by_t[T][ot + i], &by_t[t_hi][oh + i], 4) != 0) ++bad;
                ot += (size_t) T * per[pt];
                oh += szh[pt];
            }
        }
        std::printf("hc_fused_parity: %d values of a token differ between window sizes (must be 0)\n", bad);
        failures += bad ? 1 : 0;
        // the widest window again with its tokens in reverse order: token k's outputs at column TH - 1 - k
        std::vector<float> Rr((size_t) TH * D), bor((size_t) TH * N), injr((size_t) TH * HC);
        for (int k = 0; k < TH; ++k) {
            const int j = TH - 1 - k;
            std::memcpy(&Rr[(size_t) j * D], &R[(size_t) k * D], (size_t) D * 4);
            std::memcpy(&bor[(size_t) j * N], &bo[(size_t) k * N], (size_t) N * 4);
            std::memcpy(&injr[(size_t) j * HC], &inj[(size_t) k * HC], (size_t) HC * 4);
        }
        reset(1);
        cudaMemcpy(o[1].R, Rr.data(), Rr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbo, bor.data(), bor.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dinj, injr.data(), injr.size() * 4, cudaMemcpyHostToDevice);
        run(1, TH, true, true, true);
        cudaStreamSynchronize(s);
        const std::vector<float> vr = fetch(1, TH);
        cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
        int badc = 0;
        size_t off = 0;
        for (int pt = 0; pt < 8; ++pt) {
            for (int k = 0; k < TH; ++k)
                badc += std::memcmp(&vr[off + (size_t) (TH - 1 - k) * per[pt]], &by_t[TH][off + (size_t) k * per[pt]],
                                    (size_t) per[pt] * 4) != 0;
            off += szh[pt];
        }
        std::printf("hc_fused_parity: %d (token, output) pairs differ when the window's tokens are reversed (must be 0)\n", badc);
        failures += badc ? 1 : 0;
    }
    std::printf("hc_fused_parity: %d differing cases (T %d..%d, 3 modes, with and without the pending write)\n",
                failures, t_lo, t_hi);

    // a counting read (the MoE ring's block starts before it ends): the router items' and the top-10's counts land,
    // and every output - xq included - is the read's without them; xq is also checked against a CPU quantizer
    {
        const size_t qb = (size_t) TM * (N / 32) * 36;
        void* dxq = nullptr;
        void* dxq0 = nullptr;
        unsigned* dpc = nullptr;
        cudaMalloc(&dxq, qb);
        cudaMalloc(&dxq0, qb);
        cudaMalloc((void**) &dpc, 12);
        // hc_q8_1_store on the CPU: the butterfly sum of the 32 values, d = amax / 127, q = round(x / d)
        auto q8_ref = [&](const float* x, uint8_t* blk) {
            float v[32], amax = 0.f;
            for (int i = 0; i < 32; ++i) { v[i] = x[i]; amax = std::max(amax, std::fabs(x[i])); }
            for (int o = 16; o > 0; o >>= 1) {
                float t[32];
                for (int i = 0; i < 32; ++i) t[i] = v[i] + v[i ^ o];
                std::memcpy(v, t, sizeof v);
            }
            auto fin = [](float f) { return std::fabs(f) > 65504.f ? std::copysign(65504.f, f) : f; };
            const float d = fin(amax / 127.0f);
            const __half hd = __float2half(fin(d)), hs = __float2half(fin(v[0]));
            std::memcpy(blk, &hd, 2);
            std::memcpy(blk + 2, &hs, 2);
            for (int i = 0; i < 32; ++i) {
                float q = amax == 0.f ? 0.f : std::round(x[i] / d);
                q = q > 127.f ? 127.f : q < -127.f ? -127.f : q;
                blk[4 + i] = (uint8_t) (int8_t) q;
            }
        };
        int pbad = 0;
        for (int T = t_lo; T <= t_hi; ++T) {
            std::vector<std::vector<float>> v(2);
            std::vector<uint8_t> q[2];
            unsigned cnt[3] = {0, 0, 0};
            for (int counted = 0; counted < 2; ++counted) {
                void* const xqd = counted ? dxq : dxq0;
                cudaMemset(dpc, 0, 12);
                cudaMemset(xqd, 0xee, qb);
                reset(1);
                Out& x = o[1];
                K::HcFusedArgs a;
                a.T = T; a.R = x.R; a.R_out = x.R; a.apply = true; a.bo_prev = dbo; a.inj_prev = dinj;
                a.w_norm = dwn; a.w_down = dwd; a.w_up = dwu; a.w_inject = dwi;
                a.lo = x.lo; a.rs = x.rs; a.inject_out = x.io; a.mixed = x.mix; a.xn = x.xn;
                a.w_router = dwr; a.logits = x.logits; a.ids = x.ids; a.weights = x.w; a.xq = xqd;
                a.pub_ctr = counted ? dpc : nullptr;
                {
                    K::PdlScope ps(true);
                    K::hc_fused_read(a, s);
                }
                if (cudaStreamSynchronize(s) != cudaSuccess) {
                    std::printf("counting read failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                    return 1;
                }
                v[counted] = fetch(1, T);
                q[counted].resize((size_t) T * (N / 32) * 36);
                cudaMemcpy(q[counted].data(), xqd, q[counted].size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(cnt, dpc, 12, cudaMemcpyDeviceToHost);
            }
            std::vector<float> mix((size_t) T * N);
            cudaMemcpy(mix.data(), o[1].mix, mix.size() * 4, cudaMemcpyDeviceToHost);
            std::vector<uint8_t> qr(q[1].size());
            for (int b = 0; b < T * N / 32; ++b) q8_ref(mix.data() + (size_t) b * 32, qr.data() + (size_t) b * 36);
            const bool bad = std::memcmp(v[0].data(), v[1].data(), v[0].size() * 4) != 0 ||
                             std::memcmp(q[0].data(), q[1].data(), q[0].size()) != 0 ||
                             std::memcmp(qr.data(), q[1].data(), qr.size()) != 0 ||
                             cnt[0] != K::hc_fused_pub_arrivals() || cnt[1] != 1u || cnt[2] != 1u ||
                             (!by_t[T].empty() && std::memcmp(v[1].data(), by_t[T].data(), v[1].size() * 4) != 0);
            if (bad) std::printf("T %d: a counting read differs (counts %u of %u, %u, %u)\n", T, cnt[0], K::hc_fused_pub_arrivals(), cnt[1],
                                 cnt[2]);
            pbad += bad;
        }
        std::printf("hc_fused_parity: %d window sizes where a counting read's outputs, xq or counts differ (must be 0)\n",
                    pbad);
        failures += pbad ? 1 : 0;
        cudaFree(dxq);
        cudaFree(dxq0);
        cudaFree(dpc);
    }

    // a read that also gives the shared expert its gate and q8_1 rows: every other output is the plain read's, xq_sh is
    // native_quantize_q8_1's bytes, and the shared expert's projections from them (shared_expert_multi_rows, gate and
    // up on two streams) times gate_sh, one IEEE product per value, are shared_expert_multi's output bit for bit
    // (random Q6_K gate/up and Q8_0 down at the model's shapes, a random bf16 gate-input row)
    {
        const int FF = 640;
        std::mt19937 wr8(23);
        auto sane_half = [&]() {   // a normal fp16 in +-[2^-10, 2^-5)
            const uint32_t r = wr8();
            return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
        };
        auto blocks = [&](size_t n_blocks, int bytes, int scale_at) {
            std::vector<uint8_t> w(n_blocks * (size_t) bytes);
            for (auto& b : w) b = (uint8_t) (wr8() & 0xff);
            for (size_t k = 0; k < n_blocks; ++k) {
                const uint16_t h = sane_half();
                std::memcpy(&w[k * (size_t) bytes + (size_t) scale_at], &h, 2);
            }
            return w;
        };
        const std::vector<uint8_t> wg = blocks((size_t) FF * (N / 256), 210, 208), wu8 = blocks((size_t) FF * (N / 256), 210, 208),
                                   wdn = blocks((size_t) N * (FF / 32), 34, 0);
        std::vector<uint16_t> wgi(N);
        for (auto& x : wgi) x = bf16(0.05f * nd(rng));
        void *dwg = nullptr, *dwu8 = nullptr, *dwdn = nullptr, *dxq = nullptr, *dxqs = nullptr, *dxqn = nullptr;
        uint16_t* dwgi = nullptr;
        float *dgs = nullptr, *dga = nullptr, *dupb = nullptr, *dgl = nullptr, *dold = nullptr, *dnew = nullptr;
        const size_t qb = K::native_q8_1_bytes(N, TM);
        cudaMalloc(&dwg, wg.size());
        cudaMalloc(&dwu8, wu8.size());
        cudaMalloc(&dwdn, wdn.size());
        cudaMalloc((void**) &dwgi, (size_t) N * 2);
        cudaMalloc(&dxq, qb);
        cudaMalloc(&dxqs, qb);
        cudaMalloc(&dxqn, qb);
        cudaMalloc((void**) &dgs, TM * 4);
        cudaMalloc((void**) &dga, (size_t) TM * FF * 4);
        cudaMalloc((void**) &dupb, (size_t) TM * FF * 4);
        cudaMalloc((void**) &dgl, (TM + 4) * 4);
        cudaMalloc((void**) &dold, (size_t) TM * N * 4);
        cudaMalloc((void**) &dnew, (size_t) TM * N * 4);
        cudaMemcpy(dwg, wg.data(), wg.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dwu8, wu8.data(), wu8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dwdn, wdn.data(), wdn.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dwgi, wgi.data(), (size_t) N * 2, cudaMemcpyHostToDevice);
        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        cudaEvent_t evf, evj;
        cudaEventCreateWithFlags(&evf, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evj, cudaEventDisableTiming);
        int sbad = 0;
        for (int T = t_lo; T <= t_hi; ++T) {
            reset(1);
            cudaMemset(dxqs, 0xee, qb);
            cudaMemset(dgs, 0xee, TM * 4);
            Out& x = o[1];
            K::HcFusedArgs a;
            a.T = T; a.R = x.R; a.R_out = x.R; a.apply = true; a.bo_prev = dbo; a.inj_prev = dinj;
            a.w_norm = dwn; a.w_down = dwd; a.w_up = dwu; a.w_inject = dwi;
            a.lo = x.lo; a.rs = x.rs; a.inject_out = x.io; a.mixed = x.mix; a.xn = x.xn;
            a.w_router = dwr; a.logits = x.logits; a.ids = x.ids; a.weights = x.w; a.xq = dxq;
            a.w_gate_sh = dwgi; a.gate_sh = dgs; a.xq_sh = dxqs;
            {
                K::PdlScope ps(true);
                K::hc_fused_read(a, s);
            }
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                std::printf("shared read failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const std::vector<float> v = fetch(1, T);
            bool bad = !by_t[T].empty() && std::memcmp(v.data(), by_t[T].data(), v.size() * 4) != 0;
            // xq_sh against native_quantize_q8_1 of the read's mixed
            K::native_quantize_q8_1(x.mix, dxqn, N, T, s);
            cudaStreamSynchronize(s);
            std::vector<uint8_t> qs(K::native_q8_1_bytes(N, T)), qn(qs.size());
            cudaMemcpy(qs.data(), dxqs, qs.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(qn.data(), dxqn, qn.size(), cudaMemcpyDeviceToHost);
            const bool qbad = std::memcmp(qs.data(), qn.data(), qs.size()) != 0;
            // the shared expert both ways
            K::NativeSharedWeights nw;
            nw.gate_type = 14; nw.gate_data = dwg; nw.up_type = 14; nw.up_data = dwu8; nw.down_type = 8; nw.down_data = dwdn;
            nw.q8_1 = dxqn;
            K::shared_expert_multi(T, x.mix, nw, dwgi, dga, dupb, dgl, dold, N, FF, s);
            nw.q8_1 = dxqs;
            cudaEventRecord(evf, s);
            cudaStreamWaitEvent(s2, evf, 0);
            K::shared_expert_multi_rows(T, nw, dga, dupb, dnew, N, FF, s, s2, evj);
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                std::printf("shared expert failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            std::vector<float> yo((size_t) T * N), yn((size_t) T * N), gt(T);
            cudaMemcpy(yo.data(), dold, yo.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(yn.data(), dnew, yn.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gt.data(), dgs, (size_t) T * 4, cudaMemcpyDeviceToHost);
            int ybad = 0, nonfin = 0;
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < N; ++i) {
                    volatile float p = yn[(size_t) t * N + i] * gt[t];
                    const float pv = p;
                    nonfin += !std::isfinite(yo[(size_t) t * N + i]);
                    ybad += std::memcmp(&pv, &yo[(size_t) t * N + i], 4) != 0;
                }
            bad = bad || qbad || ybad != 0 || nonfin != 0;
            if (bad)
                std::printf("T %d: shared read differs (other outputs %s, xq_sh %s, %d of %d gated values, %d non-finite)\n",
                            T, !by_t[T].empty() && std::memcmp(v.data(), by_t[T].data(), v.size() * 4) != 0 ? "differ" : "same",
                            qbad ? "differs" : "same", ybad, T * N, nonfin);
            sbad += bad;
        }
        std::printf("hc_fused_parity: %d window sizes where the shared expert's gate, rows or the read's outputs differ "
                    "(must be 0)\n", sbad);
        failures += sbad ? 1 : 0;
        cudaEventDestroy(evf);
        cudaEventDestroy(evj);
        cudaStreamDestroy(s2);
        for (void* p : {dwg, dwu8, dwdn, dxq, dxqs, dxqn, (void*) dwgi, (void*) dgs, (void*) dga, (void*) dupb, (void*) dgl,
                        (void*) dold, (void*) dnew})
            cudaFree(p);
    }

    // --stress N: N reads of random window widths and token orders, 16 at a time with no host wait between them
    // (consecutive reads overlap through PDL); every output must equal the first read of the same (T, order)
    if (stress > 0) {
        constexpr int NS = 16;
        Out so[NS];
        for (auto& x : so) alloc_out(x);
        std::mt19937 kr(5);
        std::vector<std::vector<float>> first(8 * 8);
        long mism = 0, reads = 0;
        for (int base = 0; base < stress; base += NS) {
            int key[NS];
            for (int i = 0; i < NS; ++i) {
                const Out saved = o[1];
                o[1] = so[i];
                reset(1);
                o[1] = saved;
            }
            for (int i = 0; i < NS; ++i) {
                const int T = 1 + (int) (kr() % 8), seed = (int) (kr() % 8);
                key[i] = (T - 1) * 8 + seed;
                std::vector<float> r((size_t) T * D);
                for (int t = 0; t < T; ++t)
                    std::memcpy(&r[(size_t) t * D], &R[(size_t) ((seed + 3 * t) % TM) * D], (size_t) D * 4);
                const Out saved = o[1];
                o[1] = so[i];
                cudaMemcpyAsync(o[1].R, r.data(), r.size() * 4, cudaMemcpyHostToDevice, s);
                run(1, T, true, true, true);
                o[1] = saved;
            }
            cudaStreamSynchronize(s);
            for (int i = 0; i < NS; ++i) {
                const int T = key[i] / 8 + 1;
                const Out saved = o[1];
                o[1] = so[i];
                const std::vector<float> v = fetch(1, T);
                o[1] = saved;
                ++reads;
                if (first[key[i]].empty()) first[key[i]] = v;
                else if (std::memcmp(v.data(), first[key[i]].data(), v.size() * 4) != 0) ++mism;
            }
        }
        std::printf("hc_fused_parity: stress %ld reads, %ld differ from the first read of their (T, order) (must be 0)\n", reads, mism);
        failures += mism ? 1 : 0;
    }

    if (check_only) return failures ? 1 : 0;

    // time: hot (back to back) and cold (192 MB written before each read, then R, bo and w_norm rewritten as the
    // previous kernels leave them; the read alone timed)
    {
    for (int T = t_lo; T <= t_hi; ++T)
        for (int router = 0; router < 2; ++router) {
            const double bytes = 2.0 * LR * D * 2 + HC * D * 2 + (router ? (double) NE * N * 2 : 0.0);
            const double bytes_f = q8 ? (2.0 * LR * D + HC * D) * (1.0 + 2.0 / 32) + (router ? (double) NE * N * 2 : 0.0) : bytes;
            double hot[2], cold[2];
            for (int f = 0; f < 2; ++f) {
                for (int i = 0; i < 10; ++i) run(f, T, true, true, router);
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) run(f, T, true, true, router);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                hot[f] = 1e3 * ms / iters;
                double tot = 0;
                const int ci = iters / 4 > 10 ? iters / 4 : 10;
                for (int i = 0; i < ci; ++i) {
                    evict(f, i);
                    cudaEventRecord(e0, s);
                    run(f, T, true, true, router);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot += ms;
                }
                cold[f] = 1e3 * tot / ci;
                if (f == 1 && rot) {   // back to back over the weight sets: each read's matrices are out of L2
                    for (int i = 0; i < 2 * rot; ++i) { wcur = i % rot; run(1, T, true, true, router); }
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) { wcur = i % rot; run(1, T, true, true, router); }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&ms, e0, e1);
                    wcur = -1;
                    std::printf("T %d %-12s | rotating over %d weight sets: fused %6.1f us (%5.0f GB/s)\n", T,
                                router ? "read+router" : "read", rot, 1e3 * ms / iters, bytes_f / (1e3 * ms / iters) / 1e3);
                }
            }
            std::printf("T %d %-12s | hot: old %6.1f us  fused %6.1f us (%5.0f GB/s) | cold: old %6.1f us  fused %6.1f us "
                        "(%5.0f GB/s)\n",
                        T, router ? "read+router" : "read", hot[0], hot[1], bytes_f / hot[1] / 1e3, cold[0], cold[1],
                        bytes_f / cold[1] / 1e3);
        }
    }

    // per-phase times of single reads (globaltimer, relative to the first CTA's start)
    {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        unsigned long long* ddbg = nullptr;
        cudaMalloc((void**) &ddbg, (size_t) sms * 16 * 8);
        std::vector<unsigned long long> h((size_t) sms * 16);
        static const char* const ph[15] = {"start", "dep-wait", "input", "role-done", "rtr-in", "rtr-done", "top10", "exit",
                                           "ticket", "up-lo", "norm-wr/w-in", "norm-sig/dn-in", "t10-smax", "t10-cand", "t10-rank"};
        // --stamp-reps N: the median over N reads of each phase's min/max over CTAs
        const int reps = stamp_reps;
        for (int T = t_lo; T <= t_hi; ++T) {
            for (int rt = 0; rt < 2; ++rt)
            for (int cold = 0; cold < 2; ++cold) {
                if (rot && !cold) continue;
                if (!rot && rt == 0) continue;
                std::vector<double> acc[3][15][2];
                for (int rep = 0; rep < reps; ++rep) {
                    if (rot) wcur = (rep * 3) % rot;
                    for (int i = 0; i < 3; ++i) { run(1, T, true, true, rt != 0); if (rot) wcur = (wcur + 1) % rot; }
                    if (cold && !rot) evict(1, 1);
                    cudaMemsetAsync(ddbg, 0, (size_t) sms * 128, s);
                    K::hc_fused_set_debug(ddbg);
                    run(1, T, true, true, rt != 0);
                    K::hc_fused_set_debug(nullptr);
                    wcur = -1;
                    cudaStreamSynchronize(s);
                    cudaMemcpy(h.data(), ddbg, h.size() * 8, cudaMemcpyDeviceToHost);
                    unsigned long long t0 = ~0ull;
                    for (int b = 0; b < sms; ++b) if (h[b * 16] && h[b * 16] < t0) t0 = h[b * 16];
                    for (int grp = 0; grp < 3; ++grp)
                        for (int i = 0; i < 15; ++i) {
                            double mn = 1e30, mx = -1;
                            for (int b = 0; b < sms; ++b) {
                                if ((h[b * 16 + 15] >= 2 ? (int) h[b * 16 + 15] - 1 : 0) != grp || !h[b * 16 + i]) continue;
                                const double v = (double) (h[b * 16 + i] - t0) / 1e3;
                                mn = v < mn ? v : mn; mx = v > mx ? v : mx;
                            }
                            if (mx >= 0) { acc[grp][i][0].push_back(mn); acc[grp][i][1].push_back(mx); }
                        }
                }
                std::printf("T %d %s%s phases (us, median over %d reads of min/max over CTAs; CTAs with a down item | an up item "
                            "| the top-10 item):\n", T, rt ? "router " : "", rot ? "rotating" : cold ? "cold" : "hot", reps);
                for (int grp = 0; grp < 3; ++grp) {
                    bool any = false;
                    for (int i = 0; i < 15; ++i) any = any || !acc[grp][i][0].empty();
                    if (!any) continue;
                    std::printf("  %s:", grp == 2 ? "t10 " : grp ? "up  " : "down");
                    for (int i = 0; i < 15; ++i) {
                        if ((grp != 1 && i >= 10 && i <= 11) || acc[grp][i][0].empty()) continue;
                        double m[2];
                        for (int j = 0; j < 2; ++j) {
                            auto& w = acc[grp][i][j];
                            std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
                            m[j] = w[w.size() / 2];
                        }
                        std::printf(" %s %.1f/%.1f", ph[i], m[0], m[1]);
                    }
                    std::printf("\n");
                }
            }
        }
    }
    // co-residency: k SMs held by another stream's kernel until the read is done (as a spin kernel waiting on a host
    // that waits on the read would); the read must still finish
    {
        int* h_go = nullptr;
        int* h_started = nullptr;
        cudaHostAlloc((void**) &h_go, 64, cudaHostAllocMapped);
        cudaHostAlloc((void**) &h_started, 64, cudaHostAllocMapped);
        int *d_go, *d_started;
        cudaHostGetDevicePointer((void**) &d_go, h_go, 0);
        cudaHostGetDevicePointer((void**) &d_started, h_started, 0);
        cudaStream_t sb;
        cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);
        const std::string bs = block_list;
        size_t p0 = 0;
        int stuck = 0;
        while (p0 < bs.size()) {
            size_t p1 = bs.find(',', p0);
            if (p1 == std::string::npos) p1 = bs.size();
            int T = 2, k = 0;
            std::sscanf(bs.substr(p0, p1 - p0).c_str(), "%d:%d", &T, &k);
            p0 = p1 + 1;
            *(volatile int*) h_go = 0;
            *(volatile int*) h_started = 0;
            reset(1);
            K::hc_test_hold_sms(k, d_go, d_started, sb);
            auto t0 = std::chrono::steady_clock::now();
            while (*(volatile int*) h_started < k && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {}
            const int held = *(volatile int*) h_started;
            t0 = std::chrono::steady_clock::now();
            run(1, T, true, true, true);
            bool done = false;
            while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3))
                if (cudaStreamQuery(s) == cudaSuccess) { done = true; break; }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            *(volatile int*) h_go = 1;
            cudaDeviceSynchronize();
            stuck += done ? 0 : 1;
            std::printf("co-residency T %d, %d SMs held: %s (%.2f ms)\n", T, held, done ? "finished" : "STUCK until released", ms);
        }
        std::printf("hc_fused_parity: %d stuck reads\n", stuck);
        failures += stuck;
    }
    return failures ? 1 : 0;
}
