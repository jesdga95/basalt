// tests/kernels/dec_commit_parity.cpp - DEC_COMMIT's two launches against the commit graph's per-layer calls.
//
//     build/dec_commit_parity [--bench | --finisher [--bench] | --lazy [--bench]]
//
// GDN: gdn_commit_layers against gdn_conv_commit + gdn_step_norm_multi (state-only commit) per layer; QSA:
// native_qsa_indexer_commit_layers against the tail restore + n_tok native_qsa_indexer_append calls per layer.
// Every state, conv history and indexer buffer must match bit for bit for n_keep = 0..7 at the model's geometry.
// Inputs random normal.  --bench: both as CUDA graphs over a window's layers (36 GDN, 12 QSA), in microseconds; the
// grid-stride GDN form per block count, alone and beside a chain of small copies on another stream.
// --finisher: the commit with its items claimed from a counter (begin kernel, indexer + PLE commits, GDN kernel on one
// stream) against the per-layer calls, with the finisher (the first node of the next window) on another stream before,
// during and after it: what the finisher's stream sees when it returns - every GDN state, conv history, indexer buffer
// and PLE history - must match bit for bit, for n_keep 1..8, over 200 epochs of both counter sets, and with nothing
// pending it touches nothing.  With --bench: the finisher's cost with nothing pending and its drain of what is left.
// --lazy: gdn_commit_lazy against the commit kernel (bit for bit) and a CPU reference (within rounding), for n_keep 1..8:
// one launch owing every layer, one launch per layer with and without further items, a stop word turned mid-launch, two
// epochs in a row, nothing owed before the first epoch or after an epoch is done, and the read-ahead (waiting for the
// pool or the peers, never waiting, every entry on CUDA0).  With --bench: its throughput against the commit kernel, the
// cost of a launch with nothing owed and the read-ahead's rate.
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/rope_scaling.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/ngram.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S, MT = 8, ID = 128, R = 4, CELLS = 4096;
constexpr size_t GDN_FLOATS = (size_t) S * HV * S + (size_t) C * 3, TS = (size_t) (R - 1) * ID;
constexpr float EPS = 1e-6f;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

template<typename T> T* dev_rand(std::mt19937& rng, size_t n, float sd) {
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0.0f, sd);
    for (auto& v : h) v = nd(rng);
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
std::vector<uint8_t> host(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}

struct Gdn {
    float *state0, *state, *h, *qkv, *gate, *beta, *z, *y, *gamma;
    int nl;
};
Gdn make_gdn(std::mt19937& rng, int nl) {
    Gdn g;
    g.nl = nl;
    g.state0 = dev_rand<float>(rng, nl * GDN_FLOATS, 0.1f);
    CK(cudaMalloc(&g.state, nl * GDN_FLOATS * 4));
    g.h = dev_rand<float>(rng, (size_t) nl * MT * C, 0.1f);
    g.qkv = dev_rand<float>(rng, (size_t) nl * MT * C, 1.0f);
    g.gate = dev_rand<float>(rng, (size_t) nl * MT * HV, 0.3f);
    g.beta = dev_rand<float>(rng, (size_t) nl * MT * HV, 0.3f);
    g.z = dev_rand<float>(rng, (size_t) MT * HV * S, 1.0f);
    g.y = dev_rand<float>(rng, (size_t) MT * HV * S, 1.0f);
    g.gamma = dev_rand<float>(rng, S, 1.0f);
    return g;
}
void gdn_ref(const Gdn& g, const int32_t* n_keep, cudaStream_t s) {
    for (int l = 0; l < g.nl; ++l) {
        float* st = g.state + l * GDN_FLOATS;
        gdn_conv_commit(st + (size_t) S * HV * S, g.qkv + (size_t) l * MT * C, C, n_keep, s);
        gdn_step_norm_multi(st, g.h + (size_t) l * MT * C, C, g.gate + (size_t) l * MT * HV,
                            g.beta + (size_t) l * MT * HV, g.z, g.gamma, EPS, g.y, HK, HV, MT, n_keep, s, MT);
    }
}
void gdn_new(const Gdn& g, const int32_t* n_keep, cudaStream_t s, int blocks = 32) {
    GdnCommitLayers a;
    a.state = g.state; a.h = g.h; a.qkv = g.qkv; a.gate = g.gate; a.beta = g.beta;
    a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
    a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = g.nl; a.n_keep = n_keep;
    gdn_commit_layers(a, s, blocks);
}

struct Qsa {
    QsaCommitLayers a;
    std::vector<float*> bufs;   // per layer: tail, dead, pooled (the restore target and outputs)
    float *pooled0 = nullptr;
    size_t pooled_n = 0;
};
Qsa make_qsa(std::mt19937& rng, int nl, const int32_t* pos) {
    Qsa q;
    q.pooled_n = (size_t) (CELLS / R + 1) * ID;
    q.a.pos = pos; q.a.epsilon = EPS; q.a.n_layers = nl; q.a.n_tok = MT;
    for (int l = 0; l < nl; ++l) {
        QsaCommitLayer& L = q.a.layer[l];
        L.raw = dev_rand<float>(rng, (size_t) MT * ID, 1.0f);
        L.tail_snap = dev_rand<float>(rng, TS, 1.0f);
        L.gamma = dev_rand<float>(rng, ID, 1.0f);
        L.tail = dev_rand<float>(rng, TS, 1.0f);
        L.dead = dev_rand<float>(rng, ID, 1.0f);
        L.pooled = dev_rand<float>(rng, q.pooled_n, 1.0f);
        CK(cudaMalloc(&L.block_pos, 4)); CK(cudaMemset(L.block_pos, 0, 4));
        L.max_cells = CELLS;
    }
    return q;
}
void qsa_ref(const Qsa& q, cudaStream_t s) {
    const QsaShapes sh = qsa_real_shapes();
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        CK(cudaMemcpyAsync(L.tail, L.tail_snap, TS * 4, cudaMemcpyDeviceToDevice, s));
        const QsaIndexerBuffers b{L.tail, L.dead, L.pooled, L.block_pos};
        for (int t = 0; t < MT; ++t)
            native_qsa_indexer_append(L.raw + (size_t) t * ID, q.a.pos + t, 0, L.gamma, EPS, b, sh, L.max_cells,
                                      RopeScaling{}, s);
    }
}
// per layer: tail | dead | pooled | block_pos bytes
std::vector<uint8_t> qsa_snapshot(const Qsa& q) {
    std::vector<uint8_t> all;
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        for (const auto& v : {host(L.tail, TS * 4), host(L.dead, ID * 4), host(L.pooled, q.pooled_n * 4),
                              host(L.block_pos, 4)})
            all.insert(all.end(), v.begin(), v.end());
    }
    return all;
}
void qsa_restore(const Qsa& q, const std::vector<uint8_t>& snap) {
    size_t o = 0;
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        CK(cudaMemcpy(L.tail, snap.data() + o, TS * 4, cudaMemcpyHostToDevice)); o += TS * 4;
        CK(cudaMemcpy(L.dead, snap.data() + o, ID * 4, cudaMemcpyHostToDevice)); o += ID * 4;
        CK(cudaMemcpy(L.pooled, snap.data() + o, q.pooled_n * 4, cudaMemcpyHostToDevice)); o += q.pooled_n * 4;
        CK(cudaMemcpy(L.block_pos, snap.data() + o, 4, cudaMemcpyHostToDevice)); o += 4;
    }
}

// commit_[0] = n_keep, [1] = n_keep - 1, [2 + t] = p0 + t for t < n_keep else -1
void set_commit(int32_t* d, int n_keep, int p0) {
    int32_t h[2 + MT];
    h[0] = n_keep; h[1] = n_keep - 1;
    for (int t = 0; t < MT; ++t) h[2 + t] = t < n_keep ? p0 + t : -1;
    CK(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));
}

bool parity(cudaStream_t s) {
    std::mt19937 rng(7);
    int32_t* commit = nullptr;
    CK(cudaMalloc(&commit, (2 + MT) * 4));
    Gdn g = make_gdn(rng, 3);
    Qsa q = make_qsa(rng, 3, commit + 2);
    const auto q0 = qsa_snapshot(q);
    bool ok = true;
    for (int n_keep = 0; n_keep <= MT; ++n_keep)
        for (int p0 : {0, 1, 5, 6, 1001}) {
            set_commit(commit, n_keep, p0);
            CK(cudaMemcpy(g.state, g.state0, g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
            CK(cudaDeviceSynchronize());
            gdn_ref(g, commit, s);
            CK(cudaStreamSynchronize(s));
            const auto ra = host(g.state, g.nl * GDN_FLOATS * 4);
            for (int blocks : {1, 7, 24, 32, 170}) {
                CK(cudaMemcpy(g.state, g.state0, g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
                CK(cudaDeviceSynchronize());
                gdn_new(g, commit, s, blocks);
                CK(cudaStreamSynchronize(s));
                const auto rb = host(g.state, g.nl * GDN_FLOATS * 4);
                if (ra != rb) { std::printf("  GDN n_keep=%d p0=%d blocks=%d: state differs\n", n_keep, p0, blocks); ok = false; }
            }
            qsa_restore(q, q0);
            CK(cudaDeviceSynchronize());
            qsa_ref(q, s);
            CK(cudaStreamSynchronize(s));
            const auto qa = qsa_snapshot(q);
            qsa_restore(q, q0);
            CK(cudaDeviceSynchronize());
            native_qsa_indexer_commit_layers(q.a, qsa_real_shapes(), RopeScaling{}, s);
            CK(cudaStreamSynchronize(s));
            const auto qb = qsa_snapshot(q);
            if (qa != qb) { std::printf("  QSA n_keep=%d p0=%d: indexer buffers differ\n", n_keep, p0); ok = false; }
            if (qa == q0 && n_keep > 0) { std::printf("  QSA n_keep=%d p0=%d: no change (no power)\n", n_keep, p0); ok = false; }
        }
    std::printf("dec_commit parity (n_keep 0..7, GDN state + conv, QSA indexer): %s\n", ok ? "BITWISE" : "FAILED");
    return ok;
}

void bench(cudaStream_t s) {
    std::mt19937 rng(11);
    int32_t* commit = nullptr;
    CK(cudaMalloc(&commit, (2 + MT) * 4));
    Gdn g = make_gdn(rng, 36);
    Qsa q = make_qsa(rng, 12, commit + 2);
    for (int n_keep : {1, 2, 3, 5}) {
        set_commit(commit, n_keep, 100);
        float us[2];
        for (int path = 0; path < 2; ++path) {
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            if (path == 0) { gdn_ref(g, commit, s); qsa_ref(q, s); }
            else { gdn_new(g, commit, s); native_qsa_indexer_commit_layers(q.a, qsa_real_shapes(), RopeScaling{}, s); }
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
            constexpr int REPS = 50;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            us[path] = ms * 1000.0f / REPS;
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
        }
        const double gb = 2.0 * 36 * (double) S * HV * S * 4 / 1e9;   // the states read and written
        std::printf("bench n_keep=%d: per-layer calls %.1f us | two launches %.1f us (GDN states %.0f MB: %.0f GB/s)\n",
                    n_keep, us[0], us[1], gb * 1e3, gb / (us[1] * 1e-6));
        for (int blocks : {8, 16, 24, 32, 48, 85, 170, 340}) {   // the GDN part alone from persistent blocks
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) gdn_new(g, commit, s, blocks);
            constexpr int REPS = 20;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) gdn_new(g, commit, s, blocks);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double u = ms * 1000.0 / REPS;
            std::printf("  GDN blocks=%3d: %7.1f us (%.0f GB/s)\n", blocks, u, gb / (u * 1e-6));
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        }
    }
}

// The segment commit against one gdn_commit_layers launch per segment: the whole window's GDN layers (36) over S
// segments of one row each (the batch window), as CUDA graphs, in microseconds; bitwise equal states first.
void bench_seg(cudaStream_t s) {
    std::mt19937 rng(17);
    constexpr int NL = 36;
    Gdn g = make_gdn(rng, NL);
    const size_t layer_floats = GDN_FLOATS;
    for (int nseg : {1, 2, 5, 8}) {
        // each segment owns a copy of the states; the window's rows are shared (row k for segment k)
        std::vector<float*> st(nseg), st_ref(nseg);
        std::vector<int32_t*> keep(nseg);
        std::vector<SegRow> rows(nseg);
        for (int k = 0; k < nseg; ++k) {
            CK(cudaMalloc(&st[k], NL * layer_floats * 4)); CK(cudaMalloc(&st_ref[k], NL * layer_floats * 4));
            CK(cudaMemcpy(st[k], g.state0, NL * layer_floats * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(st_ref[k], g.state0, NL * layer_floats * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMalloc(&keep[k], 4));
            const int one = 1;
            CK(cudaMemcpy(keep[k], &one, 4, cudaMemcpyHostToDevice));
            rows[k] = SegRow{st[k], k, 0, 1, 1, k, 100};
        }
        SegRow* d_rows = nullptr;
        CK(cudaMalloc(&d_rows, nseg * sizeof(SegRow)));
        CK(cudaMemcpy(d_rows, rows.data(), nseg * sizeof(SegRow), cudaMemcpyHostToDevice));
        auto per_seg = [&] {
            for (int k = 0; k < nseg; ++k) {
                GdnCommitLayers a;
                a.state = st_ref[k]; a.h = g.h + (size_t) k * C; a.qkv = g.qkv + (size_t) k * C;
                a.gate = g.gate + (size_t) k * HV; a.beta = g.beta + (size_t) k * HV;
                a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
                a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = NL; a.n_keep = keep[k];
                gdn_commit_layers(a, s, 32);
            }
        };
        auto seg = [&] {
            GdnCommitSeg a;
            a.rows = d_rows; a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
            a.h = g.h; a.qkv = g.qkv; a.gate = g.gate; a.beta = g.beta;
            a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = NL; a.n_rows = nseg;
            gdn_commit_layers_seg(a, s, 32);
        };
        per_seg(); seg();
        CK(cudaStreamSynchronize(s));
        bool same = true;
        for (int k = 0; k < nseg; ++k) same = same && host(st[k], NL * layer_floats * 4) == host(st_ref[k], NL * layer_floats * 4);
        float us[2];
        for (int path = 0; path < 2; ++path) {
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            if (path == 0) per_seg(); else seg();
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
            constexpr int REPS = 50;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            us[path] = ms * 1000.0f / REPS;
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
        }
        const double gb = 2.0 * nseg * NL * (double) S * HV * S * 4 / 1e9;
        std::printf("bench seg S=%d x 1 row: per-segment launches %.1f us | one segment launch %.1f us (%.0f GB/s), states %s\n",
                    nseg, us[0], us[1], gb / (us[1] * 1e-6), same ? "bitwise" : "DIFFER");
        for (int k = 0; k < nseg; ++k) { CK(cudaFree(st[k])); CK(cudaFree(st_ref[k])); CK(cudaFree(keep[k])); }
        CK(cudaFree(d_rows));
    }
}

// A chain of small device copies on its own stream (a stand-in for the draft's short kernels), alone and with the
// GDN commit launched beside it: how long the chain takes and when the commit ends, both from the common start.
void contention(cudaStream_t s) {
    std::mt19937 rng(13);
    int32_t* commit = nullptr;
    CK(cudaMalloc(&commit, (2 + MT) * 4));
    set_commit(commit, 2, 100);
    Gdn g = make_gdn(rng, 36);
    cudaStream_t v;
    CK(cudaStreamCreateWithFlags(&v, cudaStreamNonBlocking));
    constexpr size_t CH = 4u << 20;
    constexpr int N = 40;
    char *src = nullptr, *dst = nullptr;
    CK(cudaMalloc(&src, CH * N)); CK(cudaMalloc(&dst, CH * N));
    cudaEvent_t e0, ev, ec;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&ev)); CK(cudaEventCreate(&ec));
    for (int blocks : {-1, 8, 16, 24, 32, 48, 85}) {   // -1: the chain alone
        double sv = 0, sc = 0;
        constexpr int REPS = 20;
        for (int r = 0; r < REPS + 2; ++r) {
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0, s));
            CK(cudaStreamWaitEvent(v, e0, 0));
            if (blocks >= 0) gdn_new(g, commit, s, blocks);
            CK(cudaEventRecord(ec, s));
            for (int i = 0; i < N; ++i) CK(cudaMemcpyAsync(dst + i * CH, src + i * CH, CH, cudaMemcpyDeviceToDevice, v));
            CK(cudaEventRecord(ev, v));
            CK(cudaDeviceSynchronize());
            float mv = 0, mc = 0;
            CK(cudaEventElapsedTime(&mv, e0, ev)); CK(cudaEventElapsedTime(&mc, e0, ec));
            if (r >= 2) { sv += mv; sc += mc; }
        }
        std::printf("  contention blocks=%3d: chain of %d x %zu MB copies %6.1f us, commit ends at %6.1f us\n", blocks,
                    N, CH >> 20, sv * 1000 / REPS, blocks >= 0 ? sc * 1000 / REPS : 0.0);
    }
}


// ---------------------------------------------------------------------------------------------------------------
// the finisher

// the engine launches the finisher with one block per SM
static int fin_blocks() {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms;
}
static const int FIN_BLOCKS = fin_blocks();
constexpr int64_t HS = (int64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM;

template<typename T> T* mapped(size_t n, T** dev) {
    void* h = nullptr;
    CK(cudaHostAlloc(&h, n * sizeof(T), cudaHostAllocMapped));
    std::memset(h, 0, n * sizeof(T));
    CK(cudaHostGetDevicePointer((void**) dev, h, 0));
    return (T*) h;
}

struct Rig {
    Gdn g;
    Qsa q;
    int32_t* commit = nullptr;
    CommitCtl* ctl = nullptr;
    uint32_t *h_begun = nullptr, *d_begun = nullptr;   // written by the begin kernel
    int32_t *h_epoch = nullptr, *d_epoch = nullptr;
    float *ple_snap = nullptr, *ple_hist = nullptr, *ple_snap_out = nullptr, *state_out = nullptr;
    char* q_out = nullptr;
    size_t q_bytes = 0;
    uint32_t epoch = 0;
    uint32_t *h_gate = nullptr, *d_gate = nullptr;   // raised by the host to let the finisher's stream go on
    uint32_t gate_n = 0;
    std::vector<uint8_t> q0;
    cudaStream_t sc, sf;
    cudaEvent_t ev;
};

GdnCommitLayers rig_args(const Rig& r, const int32_t* keep) {
    GdnCommitLayers a;
    a.state = r.g.state; a.h = r.g.h; a.qkv = r.g.qkv; a.gate = r.g.gate; a.beta = r.g.beta;
    a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
    a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = r.g.nl; a.n_keep = keep;
    return a;
}

// the commit graph's order: begin, indexer, PLE, then the GDN kernel
void rig_begin(Rig& r, uint32_t e) {
    r.h_epoch[0] = (int32_t) e;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    gdn_commit_begin(r.ctl, r.d_epoch, r.d_begun, r.sc);
    (void) cudaStreamQuery(r.sc);   // submit the launch now: the host polls mapped memory the kernel writes
}
void rig_rest(Rig& r) {
    native_qsa_indexer_commit_layers(r.q.a, qsa_real_shapes(), RopeScaling{}, r.sc);
    copy_indexed(r.ple_hist, r.ple_snap, HS, r.commit + 1, HS, r.sc);
    gdn_commit_layers_dyn(rig_args(r, r.commit), r.ctl, r.sc, 32);
}
void rig_commit(Rig& r, uint32_t e) { rig_begin(r, e); rig_rest(r); }
// the host's check before it launches a window: the begin kernel of the epoch has run
void wait_begun(const volatile uint32_t* begun, uint32_t e) {
    const auto t0 = std::chrono::steady_clock::now();
    while (*begun != e) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) { std::printf("begin kernel never ran\n"); std::exit(1); }
    }
}
void rig_finish(Rig& r) { gdn_commit_finish(rig_args(r, r.commit), r.ctl, r.sf, FIN_BLOCKS); }
// what the finisher's stream sees when the finisher has returned
void rig_snapshot(Rig& r) {
    CK(cudaMemcpyAsync(r.state_out, r.g.state, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice, r.sf));
    CK(cudaMemcpyAsync(r.ple_snap_out, r.ple_hist, HS * 4, cudaMemcpyDeviceToDevice, r.sf));
    size_t o = 0;
    for (int l = 0; l < r.q.a.n_layers; ++l) {
        const QsaCommitLayer& L = r.q.a.layer[l];
        CK(cudaMemcpyAsync(r.q_out + o, L.tail, TS * 4, cudaMemcpyDeviceToDevice, r.sf)); o += TS * 4;
        CK(cudaMemcpyAsync(r.q_out + o, L.dead, ID * 4, cudaMemcpyDeviceToDevice, r.sf)); o += ID * 4;
        CK(cudaMemcpyAsync(r.q_out + o, L.pooled, r.q.pooled_n * 4, cudaMemcpyDeviceToDevice, r.sf)); o += r.q.pooled_n * 4;
        CK(cudaMemcpyAsync(r.q_out + o, L.block_pos, 4, cudaMemcpyDeviceToDevice, r.sf)); o += 4;
    }
}

struct Ref {
    std::vector<uint8_t> state, q, ple;
};

void rig_init(Rig& r, std::mt19937& rng) {
    CK(cudaStreamCreateWithFlags(&r.sc, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&r.sf, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&r.ev, cudaEventDisableTiming));
    CK(cudaMalloc(&r.commit, (2 + MT) * 4));
    r.g = make_gdn(rng, 36);
    r.q = make_qsa(rng, 12, r.commit + 2);
    CK(cudaMalloc(&r.ctl, sizeof(CommitCtl)));
    CK(cudaMemset(r.ctl, 0, sizeof(CommitCtl)));
    r.h_begun = mapped<uint32_t>(32, &r.d_begun);
    r.h_epoch = mapped<int32_t>(32, &r.d_epoch);
    r.h_gate = mapped<uint32_t>(32, &r.d_gate);
    r.ple_snap = dev_rand<float>(rng, (size_t) MT * HS, 1.0f);
    CK(cudaMalloc(&r.ple_hist, HS * 4));
    CK(cudaMalloc(&r.ple_snap_out, HS * 4));
    CK(cudaMalloc(&r.state_out, r.g.nl * GDN_FLOATS * 4));
    r.q_bytes = (size_t) r.q.a.n_layers * ((TS + ID + r.q.pooled_n + 1) * 4);
    CK(cudaMalloc(&r.q_out, r.q_bytes));
    r.q0 = qsa_snapshot(r.q);
    // every kernel loaded before one waits on another that has not launched yet
    set_commit(r.commit, 1, 10);
    const uint32_t e = ++r.epoch;
    rig_commit(r, e);
    CK(cudaStreamSynchronize(r.sc));
    rig_finish(r);
    wait_flag_ge(r.d_gate, 0, r.sf);
    CK(cudaDeviceSynchronize());
}

bool finisher_tests(cudaStream_t) {
    std::mt19937 rng(23);
    Rig r;
    rig_init(r, rng);
    const std::vector<uint8_t> ple0 = host(r.ple_hist, HS * 4);

    // the references: today's per-layer calls, per n_keep
    Ref refs[MT + 1];
    for (int n_keep = 1; n_keep <= MT; ++n_keep) {
        set_commit(r.commit, n_keep, 37 + n_keep);
        CK(cudaMemcpy(r.g.state, r.g.state0, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
        qsa_restore(r.q, r.q0);
        CK(cudaDeviceSynchronize());
        gdn_ref(r.g, r.commit, r.sc);
        qsa_ref(r.q, r.sc);
        CK(cudaStreamSynchronize(r.sc));
        refs[n_keep].state = host(r.g.state, r.g.nl * GDN_FLOATS * 4);
        refs[n_keep].q = qsa_snapshot(r.q);
        refs[n_keep].ple = host(r.ple_snap + (size_t) (n_keep - 1) * HS, HS * 4);
    }
    bool ok = true;
    // variants: 0 the finisher launched before the commit's work (after its begin kernel), 1-3 launched a host delay
    // later (the commit is part way), 4 after the commit
    const int delays[3] = {40, 120, 300};   // microseconds
    auto run_one = [&](int n_keep, int variant) {
        set_commit(r.commit, n_keep, 37 + n_keep);
        CK(cudaMemcpy(r.g.state, r.g.state0, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
        qsa_restore(r.q, r.q0);
        CK(cudaMemcpy(r.ple_hist, ple0.data(), HS * 4, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        const uint32_t e = ++r.epoch;
        rig_begin(r, e);
        wait_begun(r.h_begun, e);
        if (variant == 0) { rig_finish(r); rig_snapshot(r); rig_rest(r); }
        else if (variant <= 3) {
            rig_rest(r);
            std::this_thread::sleep_for(std::chrono::microseconds(delays[variant - 1]));
            rig_finish(r);
            rig_snapshot(r);
        } else { rig_rest(r); CK(cudaStreamSynchronize(r.sc)); rig_finish(r); rig_snapshot(r); }
        CK(cudaStreamSynchronize(r.sf));
        CK(cudaStreamSynchronize(r.sc));
        const Ref& f = refs[n_keep];
        bool good = host(r.state_out, r.g.nl * GDN_FLOATS * 4) == f.state && host(r.q_out, r.q_bytes) == f.q &&
                    host(r.ple_snap_out, HS * 4) == f.ple;
        // and what is left after both have ended
        good = good && host(r.g.state, r.g.nl * GDN_FLOATS * 4) == f.state;
        if (!good) std::printf("  finisher n_keep=%d variant=%d epoch=%u: state differs\n", n_keep, variant, e);
        return good;
    };
    for (int n_keep = 1; n_keep <= MT; ++n_keep)
        for (int v = 0; v < 5; ++v) ok = run_one(n_keep, v) && ok;
    std::printf("finisher parity (n_keep 1..%d x before / during x3 / after, 36 GDN + 12 QSA layers): %s\n", MT,
                ok ? "BITWISE" : "FAILED");

    // 200 epochs: both counter sets alternate, the variant and n_keep vary
    bool ok200 = true;
    std::uniform_int_distribution<int> nk(1, MT), vr(0, 4);
    for (int k = 0; k < 200; ++k) ok200 = run_one(nk(rng), vr(rng)) && ok200;
    std::printf("finisher 200 epochs (both counter sets, mixed timing): %s\n", ok200 ? "BITWISE" : "FAILED");
    ok = ok && ok200;

    // nothing pending: the finisher touches nothing, whatever the counters hold
    {
        const std::vector<uint8_t> ctl0 = host(r.ctl, sizeof(CommitCtl));
        CK(cudaMemcpy(r.g.state, r.g.state0, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
        qsa_restore(r.q, r.q0);
        CK(cudaDeviceSynchronize());
        rig_finish(r);
        rig_snapshot(r);
        CK(cudaStreamSynchronize(r.sf));
        const bool same = host(r.state_out, r.g.nl * GDN_FLOATS * 4) == host(r.g.state0, r.g.nl * GDN_FLOATS * 4) &&
                          host(r.q_out, r.q_bytes) == r.q0 && host(r.ctl, sizeof(CommitCtl)) == ctl0;
        std::printf("finisher with nothing pending touches nothing: %s\n", same ? "OK" : "FAILED");
        ok = ok && same;
    }

    // the engine's pattern without a host sync: commit e after the finisher of e - 1, the state carried over; the GDN
    // state after 200 epochs against the per-layer calls applied in the same order
    {
        constexpr int N = 200;
        std::vector<int32_t*> cm(N + 1);
        std::vector<cudaEvent_t> fin(N + 1);
        int32_t* d_ep_arr = nullptr;
        int32_t* h_ep_arr = mapped<int32_t>(N + 1, &d_ep_arr);
        uint32_t* d_begun_arr = nullptr;
        uint32_t* h_begun_arr = mapped<uint32_t>(N + 1, &d_begun_arr);
        CK(cudaMemcpy(r.g.state, r.g.state0, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
        Gdn gref = r.g;
        CK(cudaMalloc(&gref.state, r.g.nl * GDN_FLOATS * 4));
        CK(cudaMemcpy(gref.state, r.g.state0, r.g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
        CK(cudaDeviceSynchronize());
        std::uniform_int_distribution<int> dl(0, 150);
        const uint32_t e0 = r.epoch;
        std::vector<int> keeps(N + 1);
        for (int k = 1; k <= N; ++k) {
            keeps[k] = nk(rng);
            CK(cudaMalloc(&cm[k], (2 + MT) * 4));
            set_commit(cm[k], keeps[k], 500 + k);
            CK(cudaEventCreateWithFlags(&fin[k], cudaEventDisableTiming));
            h_ep_arr[k] = (int32_t) (e0 + k);
        }
        CK(cudaDeviceSynchronize());
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (int k = 1; k <= N; ++k) {
            if (k > 1) CK(cudaStreamWaitEvent(r.sc, fin[k - 1], 0));
            gdn_commit_begin(r.ctl, d_ep_arr + k, d_begun_arr + k, r.sc);
            (void) cudaStreamQuery(r.sc);   // submit the launch now: the host polls mapped memory the kernel writes
            copy_indexed(r.ple_hist, r.ple_snap, HS, cm[k] + 1, HS, r.sc);
            gdn_commit_layers_dyn(rig_args(r, cm[k]), r.ctl, r.sc, 32);
            wait_begun(h_begun_arr + k, e0 + k);
            std::this_thread::sleep_for(std::chrono::microseconds(dl(rng)));   // against the commit's progress
            gdn_commit_finish(rig_args(r, cm[k]), r.ctl, r.sf, FIN_BLOCKS);
            CK(cudaEventRecord(fin[k], r.sf));
        }
        CK(cudaDeviceSynchronize());
        for (int k = 1; k <= N; ++k) {
            gdn_ref(gref, cm[k], r.sc);
            CK(cudaStreamSynchronize(r.sc));
        }
        const bool same = host(r.g.state, r.g.nl * GDN_FLOATS * 4) == host(gref.state, r.g.nl * GDN_FLOATS * 4);
        std::printf("finisher chain (200 epochs, no host sync, state carried): %s\n", same ? "BITWISE" : "FAILED");
        ok = ok && same;
    }
    return ok;
}

// us per node of a chain of `n` nodes captured from `fn`, over graph replays
template<typename F> double chain_us(cudaStream_t s, int n, F fn) {
    cudaGraph_t gr;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i) fn();
    CK(cudaStreamEndCapture(s, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int w = 0; w < 5; ++w) CK(cudaGraphLaunch(ge, s));
    constexpr int REPS = 50;
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
    return ms * 1000.0 / REPS / n;
}

void finisher_bench(cudaStream_t) {
    std::mt19937 rng(29);
    Rig r;
    rig_init(r, rng);
    unsigned long long* stamp = nullptr;
    CK(cudaMalloc(&stamp, 64 * 8));
    gpu_stamp(stamp, 0, r.sf);
    CK(cudaStreamSynchronize(r.sf));
    const GdnCommitLayers a = rig_args(r, r.commit);
    set_commit(r.commit, 3, 100);
    CK(cudaDeviceSynchronize());
    const double base = chain_us(r.sf, 50, [&] { gpu_stamp(stamp, 0, r.sf); });
    const double fm = chain_us(r.sf, 50, [&] { gdn_commit_finish(a, r.ctl, r.sf, FIN_BLOCKS); });
    std::printf("finisher, nothing pending, per node in a graph chain: %.2f us (a one-thread kernel: %.2f us)\n", fm, base);
    // the same node between two small kernels, against the two alone
    const double two = chain_us(r.sf, 50, [&] { gpu_stamp(stamp, 0, r.sf); gpu_stamp(stamp, 1, r.sf); });
    const double three = chain_us(r.sf, 50, [&] { gpu_stamp(stamp, 0, r.sf); gdn_commit_finish(a, r.ctl, r.sf, FIN_BLOCKS); gpu_stamp(stamp, 1, r.sf); });
    std::printf("  inserted between two one-thread kernels: +%.2f us per window\n", three - two);

    // the drain: the counters say `left` of the items are not claimed yet; time the finisher alone
    const int conv_blocks = (C + 127) / 128;
    const int per_layer = HV * (S / 32) + conv_blocks, total = per_layer * 36;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int n_keep : {2, 4}) {
        set_commit(r.commit, n_keep, 100);
        for (double left : {1.0, 0.75, 0.5, 0.25, 0.05}) {
            double sum = 0;
            constexpr int REPS = 20;
            for (int rep = 0; rep < REPS + 3; ++rep) {
                const uint32_t e = ++r.epoch;
                const uint32_t k0 = (uint32_t) (total * (1.0 - left));
                uint32_t w[10 * 32] = {};
                CK(cudaMemcpy(w, r.ctl, sizeof w, cudaMemcpyDeviceToHost));
                const int set = e & 1;
                w[(set * 4 + 0) * 32] = k0; w[(set * 4 + 1) * 32] = k0; w[(set * 4 + 2) * 32] = e; w[(set * 4 + 3) * 32] = e;
                w[8 * 32] = e; w[9 * 32] = e - 1;
                CK(cudaMemcpy(r.ctl, w, sizeof w, cudaMemcpyHostToDevice));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0, r.sf));
                gdn_commit_finish(a, r.ctl, r.sf, FIN_BLOCKS);
                CK(cudaEventRecord(e1, r.sf));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (rep >= 3) sum += ms * 1000.0;
            }
            const double us = sum / REPS;
            const double mb = left * 2.0 * 36 * (double) S * HV * S * 4 / 1e6;
            std::printf("  drain n_keep=%d, %3.0f%% left: %7.1f us  (%.0f MB of state traffic: %.0f GB/s)\n", n_keep,
                        left * 100, us, mb, mb * 1e-3 / (us * 1e-6));
        }
    }
    // the whole commit: the static 32-block kernel against the claimed one
    set_commit(r.commit, 3, 100);
    for (int pass = 0; pass < 2; ++pass) {
        double sum = 0;
        constexpr int REPS = 30;
        for (int rep = 0; rep < REPS + 3; ++rep) {
            const uint32_t e = ++r.epoch;
            r.h_epoch[0] = (int32_t) e;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            CK(cudaDeviceSynchronize());
            gdn_commit_begin(r.ctl, r.d_epoch, r.d_begun, r.sc);
            CK(cudaEventRecord(e0, r.sc));
            if (pass == 0) gdn_commit_layers(a, r.sc, 32); else gdn_commit_layers_dyn(a, r.ctl, r.sc, 32);
            CK(cudaEventRecord(e1, r.sc));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (rep >= 3) sum += ms * 1000.0;
        }
        std::printf("  GDN commit, 32 blocks, %s items: %.1f us\n", pass == 0 ? "static" : "claimed", sum / REPS);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// the lazy layers (gdn_commit_lazy)

// The GDN commit on the CPU (state rows r, head, column col; the conv history's last three inputs): the reference the
// GPU forms must meet within rounding (__expf).
void gdn_cpu(const std::vector<float>& st0, const std::vector<float>& h, const std::vector<float>& qkv,
             const std::vector<float>& gate, const std::vector<float>& beta, int nl, int n, std::vector<float>& out) {
    out = st0;
    const int qk = S * HK;
    for (int l = 0; l < nl; ++l) {
        float* st = out.data() + (size_t) l * GDN_FLOATS;
        const float* hl = h.data() + (size_t) l * MT * C;
        for (int head = 0; head < HV; ++head) {
            const int qh = head % HK;
            for (int col = 0; col < S; ++col) {
                float s[S];
                for (int r = 0; r < S; ++r) s[r] = st[((size_t) r * HV + head) * S + col];
                for (int t = 0; t < n; ++t) {
                    const float* ht = hl + (size_t) t * C;
                    const float g = std::exp(gate[((size_t) l * MT + t) * HV + head]);
                    const float b = beta[((size_t) l * MT + t) * HV + head];
                    double kv = 0;
                    for (int r = 0; r < S; ++r) kv += (double) s[r] * ht[qk + qh * S + r];
                    const float delta = (float) ((ht[2 * qk + head * S + col] - g * kv) * b);
                    for (int r = 0; r < S; ++r) s[r] = g * s[r] + ht[qk + qh * S + r] * delta;
                }
                for (int r = 0; r < S; ++r) st[((size_t) r * HV + head) * S + col] = s[r];
            }
        }
        float* hist = st + (size_t) S * HV * S;
        const float* ql = qkv.data() + (size_t) l * MT * C;
        for (int c = 0; c < C; ++c) {
            float seq[3];
            for (int j = 0; j < 3; ++j) {
                const int src = n + j;
                seq[j] = src < 3 ? hist[c * 3 + src] : ql[(size_t) (src - 3) * C + c];
            }
            for (int j = 0; j < 3; ++j) hist[c * 3 + j] = seq[j];
        }
    }
}

struct Lazy {
    Gdn g;
    int32_t* commit = nullptr;
    CommitCtl* ctl = nullptr;
    uint32_t *h_begun = nullptr, *d_begun = nullptr;
    int32_t *h_epoch = nullptr, *d_epoch = nullptr;
    uint32_t* stop = nullptr;   // the stand-in for the combine's word
    uint32_t epoch = 0;
};
GdnCommitLayers lazy_args(const Lazy& z) {
    GdnCommitLayers a;
    a.state = z.g.state; a.h = z.g.h; a.qkv = z.g.qkv; a.gate = z.g.gate; a.beta = z.g.beta;
    a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
    a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = z.g.nl; a.n_keep = z.commit;
    return a;
}
void lazy_init(Lazy& z, std::mt19937& rng, int nl) {
    z.g = make_gdn(rng, nl);
    CK(cudaMalloc(&z.commit, (2 + MT) * 4));
    CK(cudaMalloc(&z.ctl, sizeof(CommitCtl)));
    CK(cudaMemset(z.ctl, 0, sizeof(CommitCtl)));
    if (!gdn_commit_lazy_none(z.ctl)) { std::printf("gdn_commit_lazy_none failed\n"); std::exit(1); }
    z.h_begun = mapped<uint32_t>(1, &z.d_begun);
    z.h_epoch = mapped<int32_t>(1, &z.d_epoch);
    CK(cudaMalloc(&z.stop, 4));
    CK(cudaMemset(z.stop, 0, 4));
}
void lazy_begin(Lazy& z, cudaStream_t s) {
    z.h_epoch[0] = (int32_t) ++z.epoch;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    gdn_commit_begin(z.ctl, z.d_epoch, z.d_begun, s);
}
// the window's launches: one per layer, layer i's own items first; `go` bit i: launch i takes further items too
void lazy_pass(Lazy& z, cudaStream_t s, uint32_t go, int blocks) {
    for (int i = 0; i < z.g.nl; ++i) {
        const uint32_t v = (go >> i & 1u) ? 1u : 0u;   // *stop is 0: launch i goes on while its stop value is 1
        gdn_commit_lazy(lazy_args(z), z.ctl, i + 1, z.stop, v, s, blocks);
    }
}

bool lazy_tests(cudaStream_t s) {
    std::mt19937 rng(23);
    constexpr int NL = 5;
    Lazy z;
    lazy_init(z, rng, NL);
    const size_t bytes = (size_t) NL * GDN_FLOATS * 4;
    auto reset = [&] {
        CK(cudaMemcpy(z.g.state, z.g.state0, bytes, cudaMemcpyDeviceToDevice));
        CK(cudaDeviceSynchronize());
    };
    bool ok = true;
    // nothing owed before the first epoch, and nothing touched
    reset();
    set_commit(z.commit, 3, 100);
    gdn_commit_lazy(lazy_args(z), z.ctl, NL, nullptr, 0, s, 170);
    CK(cudaStreamSynchronize(s));
    if (host(z.g.state, bytes) != host(z.g.state0, bytes)) { std::printf("  lazy: a launch before any epoch changed the state\n"); ok = false; }
    std::vector<float> st0(NL * GDN_FLOATS), hh((size_t) NL * MT * C), qq((size_t) NL * MT * C), ga((size_t) NL * MT * HV),
        be((size_t) NL * MT * HV);
    CK(cudaMemcpy(st0.data(), z.g.state0, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hh.data(), z.g.h, hh.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qq.data(), z.g.qkv, qq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ga.data(), z.g.gate, ga.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(be.data(), z.g.beta, be.size() * 4, cudaMemcpyDeviceToHost));
    std::uniform_int_distribution<uint32_t> bits(0, (1u << NL) - 1);
    int cases = 0;
    double worst = 0;
    for (int n_keep = 1; n_keep <= MT; ++n_keep) {
        set_commit(z.commit, n_keep, 100);
        // the current commit kernel's result
        reset();
        gdn_new(z.g, z.commit, s, 32);
        CK(cudaStreamSynchronize(s));
        const auto ref = host(z.g.state, bytes);
        // the CPU's, within rounding
        {
            std::vector<float> cpu, gpu(NL * GDN_FLOATS);
            gdn_cpu(st0, hh, qq, ga, be, NL, n_keep, cpu);
            std::memcpy(gpu.data(), ref.data(), bytes);
            for (size_t i = 0; i < cpu.size(); ++i) {
                const double d = std::fabs((double) gpu[i] - cpu[i]) / std::max(1.0, std::fabs((double) cpu[i]));
                worst = std::max(worst, d);
            }
        }
        for (int blocks : {1, 7, 32, 170}) {
            // one launch owing every layer; then the window's form with each go pattern class
            for (int form = 0; form < 4; ++form) {
                reset();
                lazy_begin(z, s);
                if (form == 0) gdn_commit_lazy(lazy_args(z), z.ctl, NL, nullptr, 0, s, blocks);
                else lazy_pass(z, s, form == 1 ? 0u : form == 2 ? (1u << NL) - 1 : bits(rng), blocks);
                CK(cudaStreamSynchronize(s));
                const auto got = host(z.g.state, bytes);
                ++cases;
                if (got != ref) {
                    std::printf("  lazy n_keep=%d blocks=%d form %d: the state differs from the commit kernel's\n", n_keep, blocks, form);
                    ok = false;
                }
                // the epoch is done: a further launch owes nothing and touches nothing
                gdn_commit_lazy(lazy_args(z), z.ctl, NL, nullptr, 0, s, blocks);
                CK(cudaStreamSynchronize(s));
                if (host(z.g.state, bytes) != got) {
                    std::printf("  lazy n_keep=%d blocks=%d form %d: a launch after the epoch changed the state\n", n_keep, blocks, form);
                    ok = false;
                }
            }
        }
        // two epochs in a row: the second commits the result of the first
        reset();
        gdn_new(z.g, z.commit, s, 32);
        gdn_new(z.g, z.commit, s, 32);
        CK(cudaStreamSynchronize(s));
        const auto ref2 = host(z.g.state, bytes);
        reset();
        for (int e = 0; e < 2; ++e) {
            lazy_begin(z, s);
            lazy_pass(z, s, bits(rng), 170);
        }
        CK(cudaStreamSynchronize(s));
        ++cases;
        if (host(z.g.state, bytes) != ref2) { std::printf("  lazy n_keep=%d: two epochs differ from two commits\n", n_keep); ok = false; }
    }
    // the stop word turned while a launch takes further items (another stream lowers it after a spin)
    {
        set_commit(z.commit, 4, 100);
        reset();
        gdn_new(z.g, z.commit, s, 32);
        CK(cudaStreamSynchronize(s));
        const auto ref = host(z.g.state, bytes);
        cudaStream_t o;
        CK(cudaStreamCreateWithFlags(&o, cudaStreamNonBlocking));
        for (int rep = 0; rep < 20; ++rep) {
            reset();
            CK(cudaMemset(z.stop, 0, 4));
            lazy_begin(z, s);
            CK(cudaStreamSynchronize(s));
            gdn_commit_lazy(lazy_args(z), z.ctl, 1, z.stop, 1u, s, 170);   // goes on while *stop != 1
            std::this_thread::sleep_for(std::chrono::microseconds(rep * 3));
            const uint32_t one = 1;
            CK(cudaMemcpyAsync(z.stop, &one, 4, cudaMemcpyHostToDevice, o));
            for (int i = 1; i < NL; ++i) gdn_commit_lazy(lazy_args(z), z.ctl, i + 1, z.stop, 1u, s, 170);
            CK(cudaDeviceSynchronize());
            ++cases;
            if (host(z.g.state, bytes) != ref) { std::printf("  lazy stop turned at %d us: the state differs\n", rep * 3); ok = false; }
        }
        CK(cudaStreamDestroy(o));
    }
    // the read-ahead: weights read into L2 only while the stop word says the combine waits (its ring with the top bit),
    // only for a layer with entries off CUDA0, and the launch ends once the word turns; the state stays the commit's
    {
        set_commit(z.commit, 3, 100);
        reset();
        gdn_new(z.g, z.commit, s, 32);
        CK(cudaStreamSynchronize(s));
        const auto ref = host(z.g.state, bytes);
        constexpr size_t WB = (size_t) 48 << 20;
        char* wbuf = nullptr;
        CK(cudaMalloc(&wbuf, WB));
        CK(cudaMemset(wbuf, 1, WB));
        int32_t* ids = nullptr;
        unsigned long long* ptab = nullptr;
        CK(cudaMalloc(&ids, 80 * 4));
        CK(cudaMalloc(&ptab, 512 * 8));
        std::vector<int32_t> hid(80);
        for (int i = 0; i < 80; ++i) hid[i] = (i * 37) % 512;
        CK(cudaMemcpy(ids, hid.data(), 80 * 4, cudaMemcpyHostToDevice));
        cudaStream_t o;
        CK(cudaStreamCreateWithFlags(&o, cudaStreamNonBlocking));
        for (int form = 0; form < 4; ++form) {   // 0: waiting for the pool, then turned; 1: never waiting; 2: every entry
                                                 // on CUDA0; 3: waiting for the peers' rows, then turned
            std::vector<unsigned long long> hp(512, form == 2 ? 1ull : 0ull);
            CK(cudaMemcpy(ptab, hp.data(), 512 * 8, cudaMemcpyHostToDevice));
            GdnLazyPf pf;
            pf.p[0] = wbuf; pf.n[0] = WB / 2; pf.p[1] = wbuf + WB / 2; pf.n[1] = WB / 2;
            pf.ids = ids; pf.n_ids = 80; pf.n_expert = 512; pf.ptab = ptab;
            const uint32_t ring = 5;
            reset();
            const uint32_t w0 = form == 1 ? ring : form == 3 ? (ring | 0x40000000u) : (ring | 0x80000000u);
            CK(cudaMemcpy(z.stop, &w0, 4, cudaMemcpyHostToDevice));
            lazy_begin(z, s);
            for (int i = 0; i < NL; ++i) gdn_commit_lazy(lazy_args(z), z.ctl, i + 1, z.stop, ring, s, 170, &pf);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            CK(cudaMemcpyAsync(z.stop, &ring, 4, cudaMemcpyHostToDevice, o));
            CK(cudaDeviceSynchronize());
            ++cases;
            if (host(z.g.state, bytes) != ref) { std::printf("  lazy read-ahead form %d: the state differs\n", form); ok = false; }
        }
        CK(cudaStreamDestroy(o));
        CK(cudaFree(wbuf)); CK(cudaFree(ids)); CK(cudaFree(ptab));
    }
    std::printf("dec_commit lazy (n_keep 1..8, %d cases, launch forms, stop words, epochs, read-ahead): %s; GPU vs CPU max rel diff %.2e\n",
                cases, ok ? "BITWISE" : "FAILED", worst);
    if (worst > 1e-4) { std::printf("  lazy: the commit is off the CPU reference\n"); ok = false; }
    return ok;
}

// Microbench: the lazy launch over a window's 35 lazy layers owing everything, against the commit kernel at 32 and
// 170 blocks; and a launch with nothing owed (the window's idle launches).
void lazy_bench(cudaStream_t s) {
    std::mt19937 rng(29);
    Lazy z;
    lazy_init(z, rng, 35);
    set_commit(z.commit, 3, 100);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timed = [&](auto&& f, int reps) {
        double sum = 0;
        for (int r = 0; r < reps + 3; ++r) {
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0, s));
            f();
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 3) sum += ms * 1000.0;
        }
        return sum / reps;
    };
    const double gb = 2.0 * 35 * (double) S * HV * S * 4 / 1e9;
    const double a32 = timed([&] { gdn_new(z.g, z.commit, s, 32); }, 20);
    const double a170 = timed([&] { gdn_new(z.g, z.commit, s, 170); }, 20);
    const double lz = timed([&] { lazy_begin(z, s); gdn_commit_lazy(lazy_args(z), z.ctl, 35, nullptr, 0, s, 170); }, 20);
    const double idle = timed([&] { gdn_commit_lazy(lazy_args(z), z.ctl, 35, nullptr, 0, s, 170); }, 50);
    std::printf("bench lazy (35 layers, n_keep 3): commit kernel 32 blocks %.1f us (%.0f GB/s), 170 blocks %.1f us (%.0f GB/s); "
                "lazy 170 blocks %.1f us (%.0f GB/s); lazy with nothing owed %.2f us\n",
                a32, gb / (a32 * 1e-6), a170, gb / (a170 * 1e-6), lz, gb / (lz * 1e-6), idle);
    {   // the read-ahead of a GDN layer's projections (47.3 MB), the combine waiting throughout, nothing owed
        constexpr size_t WB = 47300000 & ~(size_t) 15;
        char* wbuf = nullptr;
        CK(cudaMalloc(&wbuf, WB));
        int32_t* ids = nullptr;
        unsigned long long* ptab = nullptr;
        CK(cudaMalloc(&ids, 20 * 4)); CK(cudaMemset(ids, 0, 20 * 4));
        CK(cudaMalloc(&ptab, 512 * 8)); CK(cudaMemset(ptab, 0, 512 * 8));
        GdnLazyPf pf;
        pf.p[0] = wbuf; pf.n[0] = WB; pf.ids = ids; pf.n_ids = 20; pf.n_expert = 512; pf.ptab = ptab;
        const uint32_t waiting = 5u | 0x80000000u;
        CK(cudaMemcpy(z.stop, &waiting, 4, cudaMemcpyHostToDevice));
        if (!gdn_commit_lazy_none(z.ctl)) { std::printf("gdn_commit_lazy_none failed\n"); std::exit(1); }   // nothing owed
        char* flush = nullptr;
        CK(cudaMalloc(&flush, (size_t) 256 << 20));
        const double ra = timed([&] {
            CK(cudaMemsetAsync(flush, 0, (size_t) 256 << 20, s));   // L2 cold
            CK(cudaEventRecord(e0, s));
            gdn_commit_lazy(lazy_args(z), z.ctl, 0, z.stop, 5u, s, 170, &pf);
        }, 20);
        std::printf("bench lazy read-ahead of %.1f MB (L2 cold, the combine waiting): %.1f us (%.0f GB/s incl. launch)\n",
                    WB / 1e6, ra, WB / 1e9 / (ra * 1e-6));
        CK(cudaFree(wbuf)); CK(cudaFree(ids)); CK(cudaFree(ptab)); CK(cudaFree(flush));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    bool ok = parity(s);
    if (mode == "--bench") { bench(s); bench_seg(s); contention(s); }
    if (mode == "--finisher") {
        ok = finisher_tests(s) && ok;
        if (argc > 2 && std::string(argv[2]) == "--bench") finisher_bench(s);
    }
    if (mode == "--lazy") {
        ok = lazy_tests(s) && ok;
        if (argc > 2 && std::string(argv[2]) == "--bench") lazy_bench(s);
    }
    return ok ? 0 : 1;
}
