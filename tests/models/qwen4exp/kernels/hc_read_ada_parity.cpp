// tests/kernels/hc_read_ada_parity.cpp - the hc read's persistent kernel on compute capability 8.9 (hc_read_ada.cu)
// against the three kernels it replaces (hc8_gr_read_multi), bit for bit on every output, with real geometry and random
// inputs: T 1..8, with and without the pending write and the inject rows, R updated in place and apart, the device's
// grid and smaller ones.  Then the time per read in a graph of 48 reads, hot (one set of matrices) and cold (distinct
// sets over more than L2 holds), next to the three kernels; then the kernel's phase times.
//
//     build/hc_read_ada_parity [iters] [--bench-only]
#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"
#include "basalt/models/qwen4exp/kernels/hc_read_ada.hpp"
#include "basalt/kernels/wait_abort.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = basalt::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
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

struct Set {   // one read's matrices
    uint16_t *wd, *wu, *wi;
    float* wn;
};
Set make_set(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    std::vector<float> wn(D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
    Set s{upload(wd), upload(wu), upload(wi), upload(wn)};
    K::hc8_prepare(s.wd, LR, D);
    K::hc8_prepare(s.wu, D, LR);
    K::hc8_prepare(s.wi, HC, D);
    return s;
}

struct Bufs {
    float *R0, *R, *Rb, *bo, *ip0, *ip1, *lo, *rs, *inj, *mixed;
};
void fill_args(K::FusedGrArgs* fa, int T, const Set& w, const Bufs& b, bool apply, bool apart, bool inject, const float* ip, float* io) {
    for (int t = 0; t < T; ++t) {
        K::FusedGrArgs& a = fa[t];
        a = K::FusedGrArgs{};
        a.R = b.R + (size_t) t * D;
        a.R_out = (apart ? b.Rb : b.R) + (size_t) t * D;
        a.apply = apply;
        a.bo_prev = b.bo + (size_t) t * N;
        a.inj_prev = ip + (size_t) t * HC;
        a.w_norm = w.wn; a.w_down = w.wd; a.w_up = w.wu; a.w_inject = inject ? w.wi : nullptr;
        a.eps = 1e-6f;
        a.lo = b.lo + (size_t) t * LR;
        a.rs = b.rs + (size_t) t * HC;
        a.inject_out = io + (size_t) t * HC;
        a.mixed = b.mixed + (size_t) t * N;
    }
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 3;
    bool bench_only = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--bench-only") == 0) bench_only = true;
    int dev = 0, major = 0, minor = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    if (major != 8 || minor != 9) { std::printf("hc_read_ada_parity: not compute capability 8.9\n"); return 77; }
    K::hc_read_ada_enable(true);
    K::hc_read_ada_init();
    std::mt19937 rng(29);
    Bufs b{};
    auto zf = [&](size_t n) { return upload(std::vector<float>(n)); };
    b.R0 = upload(randn((size_t) TM * D, 1.f, rng));
    b.R = zf((size_t) TM * D);
    b.Rb = zf((size_t) TM * D);
    b.bo = upload(randn((size_t) TM * N, 0.5f, rng));
    b.ip0 = upload(randn((size_t) TM * HC, 1.f, rng));
    b.ip1 = zf((size_t) TM * HC);
    b.lo = zf((size_t) TM * LR);
    b.rs = zf((size_t) TM * HC);
    b.inj = zf((size_t) TM * HC);
    b.mixed = zf((size_t) TM * N);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const Set hot = make_set(rng);
    for (int T = 1; T <= TM; ++T) {
        int smem = 0;
        const int occ = K::hc_read_ada_occupancy(T, &smem);
        std::printf("T=%d: %d CTA(s) per SM at %d bytes of shared memory\n", T, occ, smem);
    }
    struct Out { const char* name; float* p; size_t per_row; };
    const Out outs[] = {{"R", b.R, (size_t) D}, {"Rb", b.Rb, (size_t) D}, {"lo", b.lo, (size_t) LR}, {"rs", b.rs, (size_t) HC},
                        {"inj", b.inj, (size_t) HC}, {"mixed", b.mixed, (size_t) N}};
    constexpr size_t kOuts = sizeof(outs) / sizeof(outs[0]);
    auto reset = [&] {
        CK(cudaMemcpy(b.R, b.R0, (size_t) TM * D * 4, cudaMemcpyDeviceToDevice));
        for (size_t k = 1; k < kOuts; ++k) CK(cudaMemset(outs[k].p, 0x7f, (size_t) TM * outs[k].per_row * 4));
    };
    auto snap = [&](int T) {
        std::vector<std::vector<uint8_t>> v;
        for (const Out& o : outs) {
            std::vector<uint8_t> h(o.per_row * 4 * T);
            CK(cudaMemcpy(h.data(), o.p, h.size(), cudaMemcpyDeviceToHost));
            v.push_back(std::move(h));
        }
        return v;
    };
    int bad = 0;
    if (!bench_only) {
        K::FusedGrArgs fa[TM];
        for (const int gr : {0, 64, 20}) {
            K::hc_read_ada_test_grid(gr);
            for (const bool apply : {true, false}) for (const bool apart : {false, true}) for (const bool inject : {true, false}) {
                if (apart && !apply) continue;
                size_t tot = 0;
                size_t perk[kOuts] = {};
                for (int T = 1; T <= TM; ++T) {
                    fill_args(fa, T, hot, b, apply, apart, inject, b.ip0, b.inj);
                    if (!K::hc_read_ada_supported(fa, T)) { std::printf("not supported: T=%d\n", T); ++bad; continue; }
                    reset();
                    if (!K::hc8_gr_read_multi(fa, T, N, HC, s)) { std::printf("no three-kernel read\n"); return 1; }
                    CK(cudaStreamSynchronize(s));
                    const auto ref = snap(T);
                    for (int it = 0; it < iters; ++it) {
                        reset();
                        K::hc_read_ada(fa, T, s);
                        CK(cudaStreamSynchronize(s));
                        const auto g = snap(T);
                        for (size_t k = 0; k < kOuts; ++k)
                            for (size_t j = 0; j < ref[k].size(); ++j) { const size_t df = ref[k][j] != g[k][j]; tot += df; perk[k] += df; }
                    }
                }
                std::printf("grid %3d apply %d R %s inject %d: T1-T%d x%d vs the three kernels: %zu differing bytes\n", gr, apply,
                            apart ? "apart   " : "in place", inject, TM, iters, tot);
                if (tot) {
                    std::printf("  by output:");
                    for (size_t k = 0; k < kOuts; ++k) std::printf(" %s %zu", outs[k].name, perk[k]);
                    std::printf("\n");
                }
                bad += (int) std::min<size_t>(tot, 1000000);
            }
        }
        K::hc_read_ada_test_grid(0);
        if (K::wait_abort_code() != 0) { std::printf("abort word %s\n", K::wait_abort_text(K::wait_abort_code()).c_str()); ++bad; }
    }

    // time per read in a graph of 48 reads
    std::vector<Set> cold;
    for (int i = 0; i < 24; ++i) cold.push_back(make_set(rng));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto graph_of = [&](bool ada, bool cold_run, int T, bool stamps, unsigned long long* sd, cudaGraph_t& g, cudaGraphExec_t& ge) {
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 48; ++i) {
            K::FusedGrArgs fa[TM];
            const Set& w = cold_run ? cold[(size_t) i % cold.size()] : hot;
            fill_args(fa, T, w, b, true, false, true, i & 1 ? b.ip1 : b.ip0, i & 1 ? b.ip0 : b.ip1);
            if (ada) {
                if (stamps) K::hc_read_ada_test_stamps(sd, i);
                K::hc_read_ada(fa, T, s);
            } else {
                K::hc8_gr_read_multi(fa, T, N, HC, s);
            }
        }
        CK(cudaStreamEndCapture(s, &g));
        K::hc_read_ada_test_stamps(nullptr, 0);
        CK(cudaGraphInstantiate(&ge, g, 0));
    };
    auto time_graph = [&](bool ada, bool cold_run, int T) {
        reset();
        CK(cudaMemset(b.ip1, 0, (size_t) TM * HC * 4));
        cudaGraph_t g; cudaGraphExec_t ge;
        graph_of(ada, cold_run, T, false, nullptr, g, ge);
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        const int reps = 20;
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        return 1e3 * ms / reps / 48;
    };
    for (const int T : {1, 2, 3, 4, 5, 6, 7, 8}) for (const int cold_run : {1, 0}) {
        // two rounds, modes rotated
        const double t0 = time_graph(false, cold_run, T), a0 = time_graph(true, cold_run, T);
        const double a1 = time_graph(true, cold_run, T), t1 = time_graph(false, cold_run, T);
        const double three = (t0 + t1) / 2, ada = (a0 + a1) / 2;
        std::printf("T=%d %s (us per read, graph of 48): three kernels %.2f, kernel %.2f (%.1f%%)\n", T, cold_run ? "cold" : "hot ",
                    three, ada, 100.0 * ada / three);
    }

    // the kernel's phases, in us after the first CTA's start; first / median / last over the CTAs, median over the reads
    {
        const char* names[K::kHcAdaStamps] = {"start", "tile", "R'", "down", "sums", "bar", "lo", "up", "epi"};
        const size_t cnt = (size_t) 48 * sms * K::kHcAdaStamps;
        unsigned long long* sd;
        CK(cudaMalloc(&sd, cnt * 8));
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        for (const int T : {1, 4, 8}) for (const int cold_run : {1, 0}) {
            reset();
            CK(cudaMemset(b.ip1, 0, (size_t) TM * HC * 4));
            CK(cudaMemset(sd, 0, cnt * 8));
            cudaGraph_t g; cudaGraphExec_t ge;
            graph_of(true, cold_run, T, true, sd, g, ge);
            for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            std::vector<unsigned long long> st(cnt);
            CK(cudaMemcpy(st.data(), sd, cnt * 8, cudaMemcpyDeviceToHost));
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
            std::printf("T=%d %s phases (us after the first CTA's start, first/median/last CTA):", T, cold_run ? "cold" : "hot ");
            for (int ph = 0; ph < K::kHcAdaStamps; ++ph) {
                std::vector<double> f, m, l;
                for (int r = 0; r < 48; ++r) {
                    auto at = [&](int c, int p) { return (double) st[((size_t) r * sms + c) * K::kHcAdaStamps + p]; };
                    double t0 = 1e30;
                    for (int c = 0; c < sms; ++c) t0 = std::min(t0, at(c, 0));
                    std::vector<double> v;
                    for (int c = 0; c < sms; ++c) v.push_back((at(c, ph) - t0) / 1e3);
                    std::sort(v.begin(), v.end());
                    f.push_back(v.front()); m.push_back(v[v.size() / 2]); l.push_back(v.back());
                }
                std::printf(" %s %.1f/%.1f/%.1f", names[ph], med(f), med(m), med(l));
            }
            std::printf("\n");
        }
        CK(cudaFree(sd));
    }
    if (K::wait_abort_code() != 0) { std::printf("abort word %s\n", K::wait_abort_text(K::wait_abort_code()).c_str()); ++bad; }
    std::printf(bad == 0 ? "hc_read_ada_parity: bitwise\n" : "hc_read_ada_parity: %d differing\n", bad);
    return bad != 0;
}
