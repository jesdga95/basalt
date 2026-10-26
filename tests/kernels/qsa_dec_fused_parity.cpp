// tests/kernels/qsa_dec_fused_parity.cpp - qsa_dec_kv_post / qsa_dec_q_post against the separate kernels the verify
// window ran (norm, RoPE, tail copy, INT8 appends with a host copy, indexer appends; q split, norms, RoPEs), bitwise,
// for windows of 1..8 rows at every block phase; then a microbench of both forms.
//   qsa_dec_fused_parity [iters]
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/rope_scaling.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
template <typename T> size_t ndiff(const T* a, const T* b, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += std::memcmp(a + i, b + i, sizeof(T)) != 0;
    return c;
}
template <typename T> void cpy(T* dst, const T* src, size_t n) { ck(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice), "cpy"); }

// one QSA layer's cache and indexer state, twice (reference, fused)
struct State {
    int8_t *kq, *vq, *hkq, *hvq;
    uint16_t *ks, *vs, *hks, *hvs;
    float *tail, *dead, *pooled, *snap;
    int32_t* bpos;
};
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const bool split = argc > 2 && std::atoi(argv[2]) != 0;   // kv_post as its two parts
    const k::QsaShapes s = k::qsa_real_shapes();
    const int NKV = (int) s.n_head_kv, HD = (int) s.head_dim, NH = (int) s.n_head, IQ = (int) s.idx_n_head,
              ID = (int) s.idx_dim, P = (int) s.page_size;
    const int max_cells = 4096, pages = max_cells / P, MT = 8;
    const size_t rows = (size_t) pages * NKV * P;
    const int TS = (int) (s.idx_block - 1) * ID;
    const k::RopeScaling& sc = k::rope_scaling();
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 7 + 3) % pages;
    for (int i = 5; i < pages; i += 61) table[i] = -1;   // streamed-out pages: host copy only
    int32_t* d_table = dalloc<int32_t>(pages);
    up(d_table, table);
    State st[2];
    for (State& x : st) {
        x.kq = dalloc<int8_t>(rows * HD); x.vq = dalloc<int8_t>(rows * HD);
        x.hkq = dalloc<int8_t>(rows * HD); x.hvq = dalloc<int8_t>(rows * HD);
        x.ks = dalloc<uint16_t>(rows * HD / 64); x.vs = dalloc<uint16_t>(rows * HD / 64);
        x.hks = dalloc<uint16_t>(rows * HD / 64); x.hvs = dalloc<uint16_t>(rows * HD / 64);
        x.tail = dalloc<float>(TS); x.dead = dalloc<float>(ID); x.pooled = dalloc<float>((size_t) (max_cells / 4 + 2) * ID);
        x.snap = dalloc<float>(TS); x.bpos = dalloc<int32_t>(1);
    }
    float *kcur0 = dalloc<float>((size_t) MT * NKV * HD), *kcur[2] = {dalloc<float>((size_t) MT * NKV * HD), dalloc<float>((size_t) MT * NKV * HD)};
    float* vcur = dalloc<float>((size_t) MT * NKV * HD);
    float* raw = dalloc<float>((size_t) MT * ID);
    float *knorm = dalloc<float>(HD), *iknorm = dalloc<float>(ID), *qnorm = dalloc<float>(HD), *iqnorm = dalloc<float>(ID);
    float* qfull = dalloc<float>((size_t) MT * NH * 2 * HD);
    float *qcur[2] = {dalloc<float>((size_t) MT * NH * HD), dalloc<float>((size_t) MT * NH * HD)};
    float *qidx0 = dalloc<float>((size_t) MT * IQ * ID), *qidx[2] = {dalloc<float>((size_t) MT * IQ * ID), dalloc<float>((size_t) MT * IQ * ID)};
    int32_t* steps = dalloc<int32_t>((size_t) MT * k::kStepCount);
    int32_t* pos = dalloc<int32_t>((size_t) MT * (NH + NKV + IQ));
    auto rnd = [&](float* d, size_t n, float scale, float off = 0.f) {
        std::vector<float> h(n);
        for (auto& x : h) x = nd(rng) * scale + off;
        up(d, h);
    };
    rnd(knorm, HD, 0.2f, 1.f); rnd(iknorm, ID, 0.2f, 1.f); rnd(qnorm, HD, 0.2f, 1.f); rnd(iqnorm, ID, 0.2f, 1.f);
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    auto run = [&](int which, int n, bool bench_only) {
        State& x = st[which];
        float* kc = kcur[which];
        const int32_t* pos_q = pos;
        const int32_t* pos_k = pos + MT * NH;
        const int32_t* pos_i = pos + MT * (NH + NKV);
        k::KvHostPools hp;
        hp.k_q = x.hkq; hp.v_q = x.hvq; hp.k_scale = x.hks; hp.v_scale = x.hvs;
        if (which == 0) {
            k::native_qsa_rms_norm_weighted(kc, knorm, kc, HD, n * NKV, 1e-6f, cs);
            k::native_rope_apply(kc, kc, n * NKV, HD, 64, sc, pos_k, cs);
            k::copy_from_mapped(x.snap, x.tail, TS, cs);
            for (int t = 0; t < n; ++t)
                k::kv_append_q8_step(x.kq, x.vq, x.ks, x.vs, d_table, steps + t * k::kStepCount, kc + t * NKV * HD,
                                     vcur + t * NKV * HD, s, cs, &hp);
            const k::QsaIndexerBuffers ib{x.tail, x.dead, x.pooled, x.bpos};
            for (int t = 0; t < n; ++t)
                k::native_qsa_indexer_append(raw + t * ID, steps + t * k::kStepCount + k::kStepPos, 0, iknorm, 1e-6f, ib,
                                             s, max_cells, sc, cs);
            ck(cudaMemcpy2DAsync(qcur[0], (size_t) HD * 4, qfull, (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) (n * NH),
                                 cudaMemcpyDeviceToDevice, cs), "split");
            k::native_qsa_rms_norm_weighted(qcur[0], qnorm, qcur[0], HD, n * NH, 1e-6f, cs);
            k::native_rope_apply(qcur[0], qcur[0], n * NH, HD, 64, sc, pos_q, cs);
            k::native_qsa_rms_norm_weighted(qidx[0], iqnorm, qidx[0], ID, n * IQ, 1e-6f, cs);
            k::native_rope_apply(qidx[0], qidx[0], n * IQ, ID, 64, sc, pos_i, cs);
        } else {
            k::QsaDecKvPost a;
            a.kcur = kc; a.vcur = vcur; a.pos_k = pos_k; a.k_norm = knorm; a.steps = steps;
            a.k_q = x.kq; a.v_q = x.vq; a.k_scale = x.ks; a.v_scale = x.vs; a.page_table = d_table; a.host = hp;
            a.idx_raw = raw; a.idx_k_norm = iknorm; a.tail = x.tail; a.dead = x.dead; a.pooled = x.pooled;
            a.block_pos = x.bpos; a.tail_snap = x.snap; a.max_cells = max_cells; a.eps = 1e-6f;
            if (split) {   // the indexer part, then the K/V part
                a.part = 2;
                k::qsa_dec_kv_post(a, n, s, sc, cs);
                a.part = 1;
            }
            k::qsa_dec_kv_post(a, n, s, sc, cs);
            k::QsaDecQPost b;
            b.q_full = qfull; b.qcur = qcur[1]; b.pos_q = pos_q; b.q_norm = qnorm; b.q_idx = qidx[1]; b.pos_i = pos_i;
            b.idx_q_norm = iqnorm; b.eps = 1e-6f;
            k::qsa_dec_q_post(b, n, s, sc, cs);
        }
        (void) bench_only;
    };
    // a sequence of windows of 1..8 rows from cell 0 (every block phase, the spare at cell 0)
    int64_t p = 0;
    int windows = 0;
    std::vector<int> ns = {1, 3, 2, 4, 1, 1, 5, 8, 7, 2, 6, 3, 1, 4, 8, 2, 2, 3};
    for (int rep = 0; rep < 40 && p + 8 < max_cells; ++rep) {
        for (int n : ns) {
            if (p + n >= max_cells) break;
            std::vector<int32_t> hs((size_t) MT * k::kStepCount), hp((size_t) MT * (NH + NKV + IQ));
            for (int t = 0; t < n; ++t) {
                k::qsa_step_fill(hs.data() + t * k::kStepCount, p + t, s);
                for (int h = 0; h < NH; ++h) hp[t * NH + h] = (int32_t) (p + t);
                for (int h = 0; h < NKV; ++h) hp[MT * NH + t * NKV + h] = (int32_t) (p + t);
                for (int h = 0; h < IQ; ++h) hp[MT * (NH + NKV) + t * IQ + h] = (int32_t) (p + t);
            }
            up(steps, hs);
            up(pos, hp);
            rnd(kcur0, (size_t) n * NKV * HD, 1.5f);
            rnd(vcur, (size_t) n * NKV * HD, (rep % 5 == 0) ? 1e-3f : 1.0f);
            rnd(raw, (size_t) n * ID, 2.0f);
            rnd(qfull, (size_t) n * NH * 2 * HD, 1.2f);
            rnd(qidx0, (size_t) n * IQ * ID, 1.2f);
            for (int w = 0; w < 2; ++w) {
                cpy(kcur[w], kcur0, (size_t) n * NKV * HD);
                cpy(qidx[w], qidx0, (size_t) n * IQ * ID);
                run(w, n, false);
            }
            ck(cudaStreamSynchronize(cs), "sync");
            ++windows;
            p += n;
        }
    }
    // compare the whole state after every window has run through both
    size_t bad = 0;
    auto cmp = [&](const char* what, size_t d) {
        if (d) std::printf("  %s: %zu differ\n", what, d);
        bad += d;
    };
    {
        auto a8 = down(st[0].kq, rows * HD), b8 = down(st[1].kq, rows * HD);
        cmp("k codes", ndiff(a8.data(), b8.data(), a8.size()));
        a8 = down(st[0].vq, rows * HD); b8 = down(st[1].vq, rows * HD);
        cmp("v codes", ndiff(a8.data(), b8.data(), a8.size()));
        a8 = down(st[0].hkq, rows * HD); b8 = down(st[1].hkq, rows * HD);
        cmp("host k codes", ndiff(a8.data(), b8.data(), a8.size()));
        a8 = down(st[0].hvq, rows * HD); b8 = down(st[1].hvq, rows * HD);
        cmp("host v codes", ndiff(a8.data(), b8.data(), a8.size()));
        auto a16 = down(st[0].ks, rows * HD / 64), b16 = down(st[1].ks, rows * HD / 64);
        cmp("k scales", ndiff(a16.data(), b16.data(), a16.size()));
        a16 = down(st[0].vs, rows * HD / 64); b16 = down(st[1].vs, rows * HD / 64);
        cmp("v scales", ndiff(a16.data(), b16.data(), a16.size()));
        a16 = down(st[0].hks, rows * HD / 64); b16 = down(st[1].hks, rows * HD / 64);
        cmp("host k scales", ndiff(a16.data(), b16.data(), a16.size()));
        auto af = down(st[0].pooled, (size_t) (max_cells / 4 + 2) * ID), bf = down(st[1].pooled, (size_t) (max_cells / 4 + 2) * ID);
        cmp("pooled", ndiff(af.data(), bf.data(), af.size()));
        af = down(st[0].tail, TS); bf = down(st[1].tail, TS);
        cmp("tail", ndiff(af.data(), bf.data(), af.size()));
        af = down(st[0].snap, TS); bf = down(st[1].snap, TS);
        cmp("tail snapshot", ndiff(af.data(), bf.data(), af.size()));
        af = down(st[0].dead, ID); bf = down(st[1].dead, ID);
        cmp("dead", ndiff(af.data(), bf.data(), af.size()));
        auto ai = down(st[0].bpos, 1), bi = down(st[1].bpos, 1);
        cmp("block_pos", ndiff(ai.data(), bi.data(), 1));
        af = down(kcur[0], (size_t) MT * NKV * HD); bf = down(kcur[1], (size_t) MT * NKV * HD);
        cmp("kcur (last window)", ndiff(af.data(), bf.data(), af.size()));
        af = down(qcur[0], (size_t) MT * NH * HD); bf = down(qcur[1], (size_t) MT * NH * HD);
        cmp("qcur (last window)", ndiff(af.data(), bf.data(), af.size()));
        af = down(qidx[0], (size_t) MT * IQ * ID); bf = down(qidx[1], (size_t) MT * IQ * ID);
        cmp("q_idx (last window)", ndiff(af.data(), bf.data(), af.size()));
    }
    if (bad) ++g_fail;
    std::printf("%d windows, %lld cells: %s\n", windows, (long long) p, bad ? "FAIL" : "bitwise");
    // microbench: a window of n rows at a block-completing position
    for (int n : {1, 2, 3, 4, 8}) {
        float ms[2] = {0, 0};
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int w = 0; w < 2; ++w) {
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, cs);
                for (int it = 0; it < iters; ++it) run(w, n, true);
                cudaEventRecord(e1, cs);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[w], e0, e1);
            }
        }
        const double bytes = (double) n * (NKV * HD * 4 * 3 + ID * 4 + NH * HD * 4 * 3 + IQ * ID * 4 * 2);
        std::printf("n %d: separate kernels %7.2f us | fused %6.2f us (%.1f GB/s) | %.2fx\n", n, 1e3 * ms[0] / iters,
                    1e3 * ms[1] / iters, bytes / (1e6 * ms[1] / iters), ms[0] / ms[1]);
    }
    std::printf("%s\n", g_fail ? "qsa_dec_fused_parity: FAIL" : "qsa_dec_fused_parity: OK (bitwise)");
    return g_fail ? 1 : 0;
}
