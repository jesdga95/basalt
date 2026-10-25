// tests/kernels/pk_moe_bench.cpp - the MoE half of one decode layer (hc read 2 with the router, routed experts, shared
// expert, combine) as the engine launches it, against the persistent experts kernel (pk_moe.hpp): bitwise equality
// of the layer output and time per layer, over a pool of layers so no layer's weights are in L2 when it runs.
// Synthetic weights in the model's formats (random bytes are valid codes; only the fp16 block scales are set); all
// routed experts are VRAM-resident.
//
//     build/pk_moe_bench [--gu 18] [--dt 20] [--layers 6] [--tlo 1] [--thi 4] [--reps 40] [--corr 0.7] [--nowait]
//
// Per layer, reference: hc_fused_read (router, q8_1 rows) -> dec_moe_glue (gate/up, SwiGLU, down) with the shared
// expert on a side stream -> dec_moe_wait -> dec_moe_combine.  pk: hc_fused_read -> pk_moe_experts with the shared
// expert on the side stream and its end flagged to the experts kernel.
#include "basalt/core/diag.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/pk_moe.hpp"
#include "basalt/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace K = basalt::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes with a finite fp16 scale at the start of every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = K::iq_row_bytes(t, n), bs = K::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

uint8_t* upload(const std::vector<uint8_t>& v) {
    uint8_t* d = dalloc<uint8_t>(v.size());
    ck(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice), "upload");
    return d;
}
uint16_t* upload_bf16(size_t n, float sc, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = bf16(sc * nd(rng));
    uint16_t* d = dalloc<uint16_t>(n);
    ck(cudaMemcpy(d, v.data(), n * 2, cudaMemcpyHostToDevice), "upload");
    return d;
}

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, KT = 10, FF = 640, TM = 8;

struct Layer {
    uint16_t *wd, *wu, *wi, *wr, *gate_inp;
    float* wn;
    uint8_t *sg, *su, *sdn;   // shared expert gate / up / down (Q8_0)
    unsigned long long* ptab;
    uint8_t* arena;
};

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int gu = 18, dt = 20, NL = 6, tlo = 1, thi = 4, reps = 40, nblobs = 48;
    float corr = 0.7f;
    bool with_wait = true;
    int only = -1;
    bool trace = false;
    bool hot = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&](int& v) { if (i + 1 < argc) v = std::atoi(argv[++i]); };
        if (a == "--gu") num(gu);
        else if (a == "--dt") num(dt);
        else if (a == "--layers") num(NL);
        else if (a == "--tlo") num(tlo);
        else if (a == "--thi") num(thi);
        else if (a == "--reps") num(reps);
        else if (a == "--blobs") num(nblobs);
        else if (a == "--corr" && i + 1 < argc) corr = (float) std::atof(argv[++i]);
        else if (a == "--nowait") with_wait = false;
        else if (a == "--only") num(only);
        else if (a == "--trace") trace = true;
        else if (a == "--hot") hot = true;   // experts alias a few blobs: weights stay in L2 (compute-only timing)
        else if (a == "--exp" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const size_t eq = kv.find('=');
            basalt::core::diagnostics().exp[kv.substr(0, eq)] = std::atof(kv.c_str() + eq + 1);
        }
    }
    if (!K::hc_fused_supported()) { std::printf("pk_moe_bench: no fused hc read on this device\n"); return 1; }
    const K::NativeExpertLayout L = K::native_expert_layout(gu, dt, N, FF);
    if (!K::pk_moe_ok(L) || !K::dec_moe_glue_ok(L)) { std::printf("pk_moe_bench: formats %d/%d not covered\n", gu, dt); return 1; }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- weights: NL layers, each with its own hc matrices, router, shared expert and 512 experts
    const size_t slot = (L.bytes + 255) & ~(size_t) 255;
    std::vector<uint8_t*> blobs;
    for (int b = 0; b < nblobs; ++b) {
        auto blob = random_rows(gu, 2 * FF, N, rng);
        const auto down = random_rows(dt, N, FF, rng);
        blob.insert(blob.end(), down.begin(), down.end());
        if (blob.size() != L.bytes) { std::fprintf(stderr, "blob %zu != %zu\n", blob.size(), L.bytes); return 2; }
        blobs.push_back(upload(blob));
    }
    std::vector<Layer> lay(NL);
    for (int l = 0; l < NL; ++l) {
        Layer& y = lay[l];
        y.wd = upload_bf16((size_t) LR * D, 0.02f, rng);
        y.wu = upload_bf16((size_t) D * LR, 0.05f, rng);
        y.wi = upload_bf16((size_t) HC * D, 0.02f, rng);
        y.wr = upload_bf16((size_t) NE * N, 0.05f, rng);
        y.gate_inp = upload_bf16(N, 0.05f, rng);
        std::vector<float> wn(D);
        for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
        y.wn = dalloc<float>(D);
        ck(cudaMemcpy(y.wn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice), "wn");
        K::hc8_prepare(y.wd, LR, D);
        K::hc8_prepare(y.wu, D, LR);
        K::hc8_prepare(y.wi, HC, D);
        y.sg = upload(random_rows(8, FF, N, rng));
        y.su = upload(random_rows(8, FF, N, rng));
        y.sdn = upload(random_rows(8, N, FF, rng));
        y.arena = dalloc<uint8_t>(slot * NE);
        std::vector<unsigned long long> p(NE);
        for (int e = 0; e < NE; ++e) {
            uint8_t* d = y.arena + slot * (size_t) e;
            if (hot) { p[e] = (unsigned long long) blobs[e % nblobs]; continue; }
            ck(cudaMemcpy(d, blobs[(e * 7 + l * 13 + e / nblobs) % nblobs], L.bytes, cudaMemcpyDeviceToDevice), "blob");
            p[e] = (unsigned long long) d;
        }
        y.ptab = dalloc<unsigned long long>(NE);
        ck(cudaMemcpy(y.ptab, p.data(), NE * 8, cudaMemcpyHostToDevice), "ptab");
    }

    // ---- activations: tokens share a base row (corr scales the per-token part), so their routes overlap
    std::vector<float> Rh((size_t) TM * D), boh((size_t) TM * N), injh((size_t) TM * HC);
    {
        std::vector<float> base(D);
        for (auto& x : base) x = nd(rng);
        for (int t = 0; t < TM; ++t)
            for (int i = 0; i < D; ++i) Rh[(size_t) t * D + i] = base[i] + corr * nd(rng);
        for (auto& x : boh) x = 0.5f * nd(rng);
        for (auto& x : injh) x = nd(rng);
    }
    auto up_f = [&](const std::vector<float>& v) {
        float* d = dalloc<float>(v.size());
        ck(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "up");
        return d;
    };
    float *dR = up_f(Rh), *dbo = up_f(boh), *dinj = up_f(injh);
    float *R2 = dalloc<float>((size_t) TM * D), *xn = dalloc<float>((size_t) TM * D), *lo = dalloc<float>((size_t) TM * LR),
          *rs = dalloc<float>((size_t) TM * HC), *io = dalloc<float>((size_t) TM * HC), *mixed = dalloc<float>((size_t) TM * N),
          *logits = dalloc<float>((size_t) TM * NE), *wts = dalloc<float>((size_t) TM * KT);
    int32_t* ids = dalloc<int32_t>((size_t) TM * KT);
    void* xq = dalloc<uint8_t>((size_t) TM * (N / 32) * 36);
    const int cap = TM * KT;
    float* parts = dalloc<float>((size_t) cap * N);
    float* out_ref = dalloc<float>((size_t) TM * N);
    float* out_pk = dalloc<float>((size_t) TM * N);
    void* scratch = dalloc<uint8_t>(std::max(K::native_expert_scratch_bytes(K::kDmCap, FF), K::pk_moe_scratch_bytes(FF)) + 4096);
    K::DmPlan* plan = dalloc<K::DmPlan>(1);
    uint32_t *m_w = nullptr, *m_flag = nullptr;
    ck(cudaHostAlloc((void**) &m_w, 256, cudaHostAllocMapped), "hostalloc");
    ck(cudaHostAlloc((void**) &m_flag, 256, cudaHostAllocMapped), "hostalloc");
    std::memset(m_w, 0, 256); std::memset(m_flag, 0, 256);
    float* m_rows = dalloc<float>(1);
    unsigned int* sync = dalloc<unsigned int>(K::kPkSyncWords);
    // shared expert
    float *sh_gate = dalloc<float>((size_t) TM * FF), *sh_up = dalloc<float>((size_t) TM * FF), *sh_g = dalloc<float>(TM),
          *shared = dalloc<float>((size_t) TM * N);
    void* sh_xq = dalloc<uint8_t>((size_t) TM * (N / 32) * 36 + 4096);

    cudaStream_t cs, sh;
    int lo_p = 0, hi_p = 0;
    ck(cudaDeviceGetStreamPriorityRange(&lo_p, &hi_p), "prio");
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithPriority(&sh, cudaStreamNonBlocking, hi_p), "stream");
    cudaEvent_t ev_fork, ev_join, e0, e1;
    ck(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "event");
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");

    // the layer's hc read 2 with the router (q8_1 rows of `mixed` and top-10)
    auto hc = [&](int l, int T, bool first_plain) {
        K::HcFusedArgs h;
        h.T = T; h.R = dR; h.R_out = R2; h.apply = true; h.bo_prev = dbo; h.inj_prev = dinj;
        h.w_norm = lay[l].wn; h.w_down = lay[l].wd; h.w_up = lay[l].wu; h.w_inject = lay[l].wi;
        h.eps = 1e-6f; h.lo = lo; h.rs = rs; h.inject_out = io; h.mixed = mixed; h.xn = xn;
        h.w_router = lay[l].wr; h.logits = logits; h.ids = ids; h.weights = wts; h.xq = xq;
        K::PdlScope ps(true, first_plain);
        K::hc_fused_read(h, cs);
    };
    auto shared_chain = [&](int l, int T, bool flag) {
        ck(cudaEventRecord(ev_fork, cs), "record");
        ck(cudaStreamWaitEvent(sh, ev_fork, 0), "wait");
        K::NativeSharedWeights nsw;
        nsw.gate_type = 8; nsw.gate_data = lay[l].sg; nsw.up_type = 8; nsw.up_data = lay[l].su;
        nsw.down_type = 8; nsw.down_data = lay[l].sdn; nsw.q8_1 = sh_xq;
        K::PdlScope ps(false);
        K::shared_expert_multi(T, mixed, nsw, lay[l].gate_inp, sh_gate, sh_up, sh_g, shared, N, FF, sh);
        if (flag) K::pk_moe_shared_done(sync, sh);
    };
    auto join_side = [&]() {
        ck(cudaEventRecord(ev_join, sh), "record");
        ck(cudaStreamWaitEvent(cs, ev_join, 0), "wait");
    };
    auto ref = [&](int l, int T, bool first_plain) {
        hc(l, T, first_plain);
        shared_chain(l, T, false);
        {
            K::PdlScope ps(true);
            K::DmGlueArgs ga;
            ga.ids = ids; ga.ptab = lay[l].ptab; ga.n_tok = T; ga.k = KT; ga.n_expert = NE; ga.plan = plan; ga.xq = xq;
            K::dec_moe_glue(L, ga, scratch, parts, T * KT, cs);
        }
        ck(cudaEventRecord(ev_join, sh), "record");
        ck(cudaStreamWaitEvent(cs, ev_join, 0), "wait");
        if (with_wait) {
            K::PdlScope ps(false);
            K::dec_moe_wait(plan, m_w + 16, m_flag, (uint32_t) (l + 1), cs, nullptr, l, true);
        }
        K::dec_moe_combine(parts, m_rows, m_rows, plan, wts, shared, out_ref, N, KT, T, nullptr, nullptr, (uint32_t) (l + 1), cs,
                           with_wait);
    };
    unsigned long long* d_trace = dalloc<unsigned long long>(K::PK_T_SLOTS);
    bool tracing = false;
    auto pk = [&](int l, int T, bool first_plain) {
        hc(l, T, first_plain);
        shared_chain(l, T, true);
        K::PkMoeArgs a;
        a.ids = ids; a.weights = wts; a.ptab = lay[l].ptab; a.n_tok = T; a.k = KT; a.n_expert = NE; a.xq = xq;
        a.shared = shared; a.out = out_pk; a.parts = parts; a.sync = sync; a.shared_flag = true;
        if (tracing) a.trace = d_trace;
        K::PdlScope ps(true);
        K::pk_moe_experts(L, a, scratch, cs);
    };

    int fails = 0;
    std::vector<float> hr((size_t) TM * N), hp((size_t) TM * N);
    std::printf("pk_moe_bench: gate/up %d down %d, %d layers, expert blob %.2f MB, hc8, shared Q8_0, %s\n", gu, dt, NL,
                L.bytes / 1e6, with_wait ? "ref with dm_wait" : "ref without dm_wait");
    for (int T = tlo; T <= thi; ++T) {
        // ---- bitwise: every layer's output, ref vs pk
        size_t bad = 0, nonfinite = 0;
        double groups = 0;
        for (int l = 0; l < NL; ++l) {
            for (int v = 0; v < 2; ++v) {
                float* o = v == 0 ? out_ref : out_pk;
                ck(cudaMemset(o, 0xFF, (size_t) TM * N * 4), "memset");
                std::printf("  run %s layer %d\n", v == 0 ? "ref" : "pk", l);
                if (v == 0) ref(l, T, true); else pk(l, T, true);
                ck(cudaStreamSynchronize(cs), "sync");
                ck(cudaStreamSynchronize(sh), "sync");
                ck(cudaMemcpy(v == 0 ? hr.data() : hp.data(), o, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "copy");
            }
            for (size_t i = 0; i < (size_t) T * N; ++i) {
                uint32_t a, b;
                std::memcpy(&a, &hr[i], 4); std::memcpy(&b, &hp[i], 4);
                bad += a != b;
                nonfinite += !std::isfinite(hr[i]);
            }
            std::vector<int32_t> hid((size_t) T * KT);
            ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ids");
            groups += std::set<int32_t>(hid.begin(), hid.end()).size();
        }
        std::printf("T %d: %.1f distinct experts/layer; pk vs ref over %d layers: %s (%zu differ, %zu non-finite ref)\n", T,
                    groups / NL, NL, bad ? "FAIL" : "bitwise equal", bad, nonfinite);
        fails += bad != 0;

        if (trace) {
            // per-stage stamps of the persistent launch, microseconds after the first block left the PDL wait (the hc read's end)
            using namespace K;
            std::vector<std::vector<double>> col(PK_T_SLOTS);
            double distinct = 0;
            tracing = true;
            for (int i = 0; i < 6 + reps; ++i) {
                const int l = i % NL;
                std::vector<unsigned long long> init(PK_T_SLOTS, 0);
                for (int s2 : {PK_T_ENTRY_MIN, PK_T_WAIT_MIN, PK_T_PLAN_MIN, PK_T_GU_FIRST, PK_T_GU_END_MIN, PK_T_DOWN_FIRST, PK_T_BLOCK_END_MIN})
                    init[s2] = ~0ull;
                ck(cudaMemcpy(d_trace, init.data(), init.size() * 8, cudaMemcpyHostToDevice), "trace init");
                pk(l, T, true);
                ck(cudaStreamSynchronize(cs), "sync");
                ck(cudaStreamSynchronize(sh), "sync");
                std::vector<unsigned long long> h(PK_T_SLOTS);
                ck(cudaMemcpy(h.data(), d_trace, h.size() * 8, cudaMemcpyDeviceToHost), "trace");
                std::vector<int32_t> hid((size_t) T * KT);
                ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ids");
                if (i < 6) continue;
                distinct += std::set<int32_t>(hid.begin(), hid.end()).size();
                const double t0 = (double) h[PK_T_WAIT_MIN];
                for (int s2 = 0; s2 < PK_T_SLOTS; ++s2) {
                    const bool sum = s2 == PK_T_GU_BUSY || s2 == PK_T_DOWN_BUSY || s2 == PK_T_SWIGLU_WAIT;
                    const bool cnt = s2 == PK_T_GU_ITEMS || s2 == PK_T_DOWN_ITEMS;
                    col[s2].push_back(cnt ? (double) h[s2] : sum ? h[s2] * 1e-3 : ((double) h[s2] - t0) * 1e-3);
                }
            }
            tracing = false;
            auto md = [&](int s2) { auto v = col[s2]; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const double ne = distinct / reps;
            const double gub = ne * L.down_off, dnb = ne * (L.bytes - L.down_off);
            std::printf("  T %d trace: %.1f distinct experts, gate/up %.1f MB, down %.1f MB (us after the hc read's end, medians)\n", T, ne,
                        gub / 1e6, dnb / 1e6);
            std::printf("    first block in %.1f  last block in %.1f  plan %.1f..%.1f  gu first start %.1f  first gu end %.1f  last gu end %.1f\n",
                        md(PK_T_ENTRY_MIN), md(PK_T_ENTRY_MAX), md(PK_T_PLAN_MIN), md(PK_T_PLAN_MAX), md(PK_T_GU_FIRST), md(PK_T_GU_END_MIN), md(PK_T_GU_LAST));
            std::printf("    first down start %.1f  last down end %.1f  shared seen %.1f  combine end %.1f  first block done %.1f  last block done %.1f\n",
                        md(PK_T_DOWN_FIRST), md(PK_T_DOWN_LAST), md(PK_T_SHARED_SEEN), md(PK_T_COMBINE_END), md(PK_T_BLOCK_END_MIN), md(PK_T_BLOCK_END));
            const double gus = md(PK_T_GU_LAST) - md(PK_T_PLAN_MAX), dns = md(PK_T_DOWN_LAST) - md(PK_T_GU_LAST);
            std::printf("    gu items %.0f (mean %.2f us, busy %.0f block-us), down items %.0f (mean %.2f us, busy %.0f block-us), swiglu wait %.0f block-us\n",
                        md(PK_T_GU_ITEMS), md(PK_T_GU_BUSY) / md(PK_T_GU_ITEMS), md(PK_T_GU_BUSY), md(PK_T_DOWN_ITEMS),
                        md(PK_T_DOWN_BUSY) / md(PK_T_DOWN_ITEMS), md(PK_T_DOWN_BUSY), md(PK_T_SWIGLU_WAIT));
            std::printf("    gate/up phase (plan end -> last gu end) %.1f us = %.2f TB/s;  down phase (last gu end -> last down end) %.1f us = %.2f TB/s\n", gus,
                        gub / gus * 1e-6, dns, dnb / dns * 1e-6);
            continue;
        }

        // ---- time: NL layers per graph, ref and pk alternating
        cudaGraph_t g[2];
        cudaGraphExec_t ge[2];
        for (int v = 0; v < 2; ++v) {
            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture");
            for (int l = 0; l < NL; ++l) { if (v == 0) ref(l, T, l == 0); else pk(l, T, l == 0); }
            if (v == 1) join_side();
            ck(cudaStreamEndCapture(cs, &g[v]), "capture");
            ck(cudaGraphInstantiate(&ge[v], g[v], 0), "instantiate");
        }
        std::vector<float> t[2];
        for (int i = 0; i < 8 + reps; ++i)
            for (int v = 0; v < 2; ++v) {
                const int vv = (i & 1) ? 1 - v : v;
                if (only >= 0 && vv != only) continue;
                ck(cudaEventRecord(e0, cs), "record");
                ck(cudaGraphLaunch(ge[vv], cs), "launch");
                ck(cudaEventRecord(e1, cs), "record");
                ck(cudaEventSynchronize(e1), "bench");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                if (i >= 8) t[vv].push_back(1e3f * ms / NL);
            }
        std::printf("  T %d: ref %.1f us/layer   pk %.1f us/layer   (median of %d graphs of %d layers)\n", T, t[0].empty() ? 0.f : median(t[0]),
                    t[1].empty() ? 0.f : median(t[1]), reps, NL);
        for (int v = 0; v < 2; ++v) { cudaGraphExecDestroy(ge[v]); cudaGraphDestroy(g[v]); }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
