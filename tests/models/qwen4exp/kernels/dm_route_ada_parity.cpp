// tests/kernels/dm_route_ada_parity.cpp - dec_moe_route's two kernels for a card without peers on compute capability 8.9
// (dec_moe.cuh) against dm_route_kernel's modes 1 and 2, bit for bit on the plan, the q8_1 rows, the ring's header, its
// x and rows and both doorbells: T 1..8, entries from all CPU to all VRAM, repeated ids and ids past the table, with and
// without per-slot offsets and the ring's rows.  Then, in graphs of 48 layers at the engine's share of VRAM entries, the
// time per launch of each kernel and the time from the kernel before the ring to its doorbell.
//
//     build/dm_route_ada_parity [iters] [--bench-only]
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = basalt::kernels;

namespace {
constexpr int H = 2560, NE = 512, KK = 10, MT = 8, RINGS = 48;
constexpr size_t XQ_ROW = (size_t) (H / 32) * 36;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

template <typename V> V* dev_alloc(size_t n) {
    V* d;
    CK(cudaMalloc((void**) &d, n * sizeof(V) + 256));
    return d;
}
template <typename V> void mapped(size_t n, V** host, V** dev) {
    void* h = nullptr;
    void* d = nullptr;
    CK(cudaHostAlloc(&h, n * sizeof(V), cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&d, h, 0));
    *host = (V*) h;
    *dev = (V*) d;
}

// T tokens' routed ids: pct of the entries among the VRAM's experts (the table's first half), the rest the CPU's
void make_ids(std::vector<int32_t>& ids, int T, int pct, int flaws, std::mt19937& rng) {
    ids.assign(K::kDmCap, -1);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick;
        while ((int) pick.size() < KK) {
            const int e = (int) (rng() % 100) < pct ? (int) (rng() % (NE / 2)) : NE / 2 + (int) (rng() % (NE / 2));
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        if ((flaws & 1) && t % 2 == 0) pick[KK - 1] = pick[(size_t) (rng() % (KK - 1))];
        if ((flaws & 2) && t == T - 1) pick[KK - 2] = NE + 3;
        if ((flaws & 4) && t == 0) pick[KK - 3] = -2;
        for (int j = 0; j < KK; ++j) ids[(size_t) t * KK + j] = pick[j];
    }
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 3;
    bool bench_only = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--bench-only") == 0) bench_only = true;
    int dev = 0, major = 0, minor = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    if (major != 8 || minor != 9) { std::printf("dm_route_ada_parity: not compute capability 8.9\n"); return 77; }

    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) MT * H);
    for (auto& v : hx) v = nd(rng);
    float* d_x = dev_alloc<float>(hx.size());
    CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    int32_t* d_ids = dev_alloc<int32_t>((size_t) RINGS * K::kDmCap);
    int32_t* d_res = dev_alloc<int32_t>(NE);
    unsigned long long* d_off = dev_alloc<unsigned long long>(NE);
    K::DmPlan* d_plan = dev_alloc<K::DmPlan>(1);
    uint8_t* d_xq = dev_alloc<uint8_t>((size_t) MT * XQ_ROW);
    uint8_t* d_xq_pub = dev_alloc<uint8_t>((size_t) MT * XQ_ROW);
    unsigned* d_ctr = dev_alloc<unsigned>(16);
    CK(cudaMemset(d_ctr, 0, 16));
    {
        std::vector<int32_t> res(NE, -1);
        std::vector<unsigned long long> off(NE);
        for (int e = 0; e < NE / 2; ++e) res[(size_t) e] = (int32_t) (rng() % (NE / 2));
        for (auto& v : off) v = (unsigned long long) (rng() % 100000) * 4096;
        CK(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_off, off.data(), off.size() * 8, cudaMemcpyHostToDevice));
    }
    K::DmRing *h_ring, *m_ring;
    float *h_xm, *m_xm;
    uint8_t *h_xqm, *m_xqm;
    uint32_t *h_w, *m_w;
    mapped<K::DmRing>(RINGS, &h_ring, &m_ring);
    mapped<float>((size_t) RINGS * MT * H, &h_xm, &m_xm);
    mapped<uint8_t>((size_t) RINGS * MT * XQ_ROW, &h_xqm, &m_xqm);
    mapped<uint32_t>(64, &h_w, &m_w);
    // the staged share: every 7th expert has no host address; split tables 1: floor(m * 90 / 256), 2: every miss
    unsigned long long* d_src = dev_alloc<unsigned long long>(NE);
    int32_t* d_kt = dev_alloc<int32_t>((size_t) 2 * (K::kDmCap + 1));
    {
        std::vector<unsigned long long> src(NE);
        for (int e = 0; e < NE; ++e) src[(size_t) e] = e % 7 == 3 ? 0ull : 0x200000000000ull + (unsigned long long) e * (1ull << 20);
        CK(cudaMemcpy(d_src, src.data(), src.size() * 8, cudaMemcpyHostToDevice));
        std::vector<int32_t> kt((size_t) 2 * (K::kDmCap + 1), 0);
        for (int m = 0; m <= K::kDmCap; ++m) {
            kt[(size_t) m] = std::min(K::kDmStageMax, (m * 90) >> 8);
            kt[(size_t) (K::kDmCap + 1 + m)] = std::min(K::kDmStageMax, m);
        }
        CK(cudaMemcpy(d_kt, kt.data(), kt.size() * 4, cudaMemcpyHostToDevice));
    }

    auto args = [&](int r, int T, int mode, bool rows, bool offs, int stage = 0) {
        K::DmRouteArgs a;
        a.ids = d_ids + (size_t) r * K::kDmCap; a.x = d_x; a.n_tok = T; a.k = KK; a.n_embd = H; a.n_expert = NE; a.layer = r;
        a.res0 = d_res; a.cache_base = reinterpret_cast<const uint8_t*>(0x100000000000ull);
        a.slot_off = offs ? d_off : nullptr; a.blob = 1ll << 20;
        a.plan = d_plan; a.xq = d_xq; a.ring_m = m_ring + r; a.x_m = m_xm + (size_t) r * MT * H;
        a.xq_m = m_xqm + (size_t) r * MT * XQ_ROW; a.rows = rows; a.host_seq_m = m_w; a.peer_seq_m = m_w + 16;
        a.ring = (uint32_t) r + 1; a.ctr = d_ctr; a.mode = mode; a.xq_pub = d_xq_pub;
        if (stage > 0) {
            a.stage_src = d_src;
            a.stage_k = d_kt + (size_t) (stage - 1) * (K::kDmCap + 1);
        }
        return a;
    };
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    struct Snap { std::vector<uint8_t> plan, xq, ring, xm, xqm, w; };
    auto snap = [&](int T) {
        Snap v;
        v.plan.resize(sizeof(K::DmPlan));
        CK(cudaMemcpy(v.plan.data(), d_plan, sizeof(K::DmPlan), cudaMemcpyDeviceToHost));
        v.xq.resize((size_t) T * XQ_ROW);
        CK(cudaMemcpy(v.xq.data(), d_xq, v.xq.size(), cudaMemcpyDeviceToHost));
        v.ring.resize(sizeof(K::DmRing));
        std::memcpy(v.ring.data(), h_ring, sizeof(K::DmRing));
        v.xm.resize((size_t) T * H * 4);
        std::memcpy(v.xm.data(), h_xm, v.xm.size());
        v.xqm.resize((size_t) T * XQ_ROW);
        std::memcpy(v.xqm.data(), h_xqm, v.xqm.size());
        v.w.resize(64 * 4);
        std::memcpy(v.w.data(), h_w, v.w.size());
        return v;
    };
    auto reset = [&](int T) {
        CK(cudaMemset(d_plan, 0x5a, sizeof(K::DmPlan)));
        CK(cudaMemset(d_xq, 0x5a, (size_t) MT * XQ_ROW));
        std::memset((void*) h_ring, 0x5a, sizeof(K::DmRing));
        std::memset(h_xm, 0x5a, (size_t) MT * H * 4);
        std::memset(h_xqm, 0x5a, (size_t) MT * XQ_ROW);
        std::memset(h_w, 0x5a, 64 * 4);
        (void) T;
    };
    int bad = 0;
    if (!bench_only) {
        size_t tot = 0, cases = 0;
        size_t per[6] = {};
        const char* names[6] = {"plan", "xq", "ring", "x", "xq_m", "doorbells"};
        std::vector<int32_t> ids;
        size_t staged_cases = 0;
        for (const int stage : {0, 1, 2}) for (const bool rows : {false, true}) for (const bool offs : {false, true}) for (const int pct : {0, 50, 85, 100}) for (const int flaws : {0, 1, 2, 7}) {
            for (int it = 0; it < iters; ++it) for (int T = 1; T <= MT; ++T) {
                make_ids(ids, T, pct, flaws, rng);
                CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                Snap ref, got;
                K::dec_moe_route_ada_enable(false);
                reset(T);
                K::dec_moe_route(args(0, T, 2, rows, offs, stage), s);
                K::dec_moe_route(args(0, T, 1, rows, offs, stage), s);
                CK(cudaStreamSynchronize(s));
                ref = snap(T);
                if (stage > 0) {   // a case where the split moved entries off the CPU
                    const K::DmRing* rr = reinterpret_cast<const K::DmRing*>(ref.ring.data());
                    int nc = 0;
                    for (int j = 0; j < T * KK; ++j) nc += rr->kind[j] < 0;
                    staged_cases += rr->n_cpu == nc && nc < T * KK - rr->n_vram;
                }
                K::dec_moe_route_ada_enable(true, stage > 0);
                if (!K::dec_moe_route_ada(args(0, T, 1, rows, offs, stage)) || !K::dec_moe_route_ada(args(0, T, 2, rows, offs, stage))) {
                    std::printf("the two kernels do not take T=%d\n", T);
                    ++bad;
                }
                reset(T);
                K::dec_moe_route(args(0, T, 2, rows, offs, stage), s);
                K::dec_moe_route(args(0, T, 1, rows, offs, stage), s);
                CK(cudaStreamSynchronize(s));
                got = snap(T);
                const std::vector<uint8_t>* a[6] = {&ref.plan, &ref.xq, &ref.ring, &ref.xm, &ref.xqm, &ref.w};
                const std::vector<uint8_t>* b[6] = {&got.plan, &got.xq, &got.ring, &got.xm, &got.xqm, &got.w};
                for (int k = 0; k < 6; ++k)
                    for (size_t j = 0; j < a[k]->size(); ++j) {
                        const size_t df = (*a[k])[j] != (*b[k])[j];
                        tot += df;
                        per[k] += df;
                    }
                ++cases;
            }
        }
        std::printf("modes 1 and 2 vs dm_route_kernel: %zu cases (%zu with staged entries), %zu differing bytes\n", cases,
                    staged_cases, tot);
        if (staged_cases == 0) { std::printf("  no case staged an entry\n"); ++bad; }
        // without the staged switch a staged share keeps dm_route_kernel
        K::dec_moe_route_ada_enable(true, false);
        if (K::dec_moe_route_ada(args(0, 4, 1, false, false, 1)) || K::dec_moe_route_ada(args(0, 4, 2, false, false, 1))) {
            std::printf("a staged share took the two kernels without the switch\n");
            ++bad;
        }
        K::dec_moe_route_ada_enable(true);
        if (tot) {
            std::printf("  by output:");
            for (int k = 0; k < 6; ++k) std::printf(" %s %zu", names[k], per[k]);
            std::printf("\n");
        }
        bad += (int) std::min<size_t>(tot, 1000000);
    }

    // graphs of 48 layers, 85 % of the entries in VRAM
    unsigned long long* d_st = dev_alloc<unsigned long long>((size_t) RINGS * 2);
    auto layers_ids = [&](int T) {
        std::vector<int32_t> ids;
        for (int r = 0; r < RINGS; ++r) {
            make_ids(ids, T, 85, 0, rng);
            CK(cudaMemcpy(d_ids + (size_t) r * K::kDmCap, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        }
    };
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto time_launch = [&](int T, int mode, bool rows, bool fast) {
        K::dec_moe_route_ada_enable(fast);
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < RINGS; ++r) K::dec_moe_route(args(r, T, mode, rows, true), s);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
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
        return 1e3 * ms / reps / RINGS;
    };
    // from the stamp of the kernel before the ring to the ring's doorbell
    auto doorbell = [&](int T, bool rows, bool fast) {
        K::dec_moe_route_ada_enable(fast);
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < RINGS; ++r) {
            K::gpu_stamp(d_st, 2 * r, s);
            K::DmRouteArgs a = args(r, T, 2, rows, true);
            a.stats = d_st + 2 * r + 1;
            K::dec_moe_route(a, s);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        double sum = 0;
        int n = 0;
        std::vector<unsigned long long> st((size_t) RINGS * 2);
        for (int i = 0; i < 12; ++i) {
            CK(cudaGraphLaunch(ge, s));
            CK(cudaStreamSynchronize(s));
            if (i < 2) continue;
            CK(cudaMemcpy(st.data(), d_st, st.size() * 8, cudaMemcpyDeviceToHost));
            for (int r = 0; r < RINGS; ++r) { sum += (double) (st[(size_t) 2 * r + 1] - st[(size_t) 2 * r]); ++n; }
        }
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        return sum / n / 1e3;
    };
    for (const bool rows : {true, false}) for (const int T : {1, 2, 3, 4, 5, 6, 7, 8}) {
        layers_ids(T);
        // two rounds, modes rotated
        const double p = (time_launch(T, 1, rows, false) + time_launch(T, 1, rows, false)) / 2;
        const double pn = (time_launch(T, 1, rows, true) + time_launch(T, 1, rows, true)) / 2;
        const double r0 = time_launch(T, 2, rows, false), r1 = time_launch(T, 2, rows, true), r2 = time_launch(T, 2, rows, true), r3 = time_launch(T, 2, rows, false);
        const double d0 = doorbell(T, rows, false), d1 = doorbell(T, rows, true), d2 = doorbell(T, rows, true), d3 = doorbell(T, rows, false);
        std::printf("rows %d T=%d (us, graph of 48): chain %.2f -> %.2f; ring %.2f -> %.2f; to the doorbell %.2f -> %.2f\n", rows, T, p, pn,
                    (r0 + r3) / 2, (r1 + r2) / 2, (d0 + d3) / 2, (d1 + d2) / 2);
    }
    K::dec_moe_route_ada_enable(true);
    std::printf(bad == 0 ? "dm_route_ada_parity: bitwise\n" : "dm_route_ada_parity: %d differing\n", bad);
    return bad != 0;
}
