// tests/kernels/qsa_seg_parity.cpp - the segment variants of the QSA decode chain (qsa_seg.hpp) against the
// single-sequence kernels run per slot, bitwise: K/V post + indexer appends, query post, block scores, top-k, chunk
// attention + merge (+ gate and q8_1 rows), and the commit (tail restore + appends), for S in {1,2,5,8} segments of
// mixed row counts at different positions, segment order shuffled against slot order; INT8 and FP16 pools; then a
// microbench line per kernel (one segment call vs one call per segment).
//   qsa_seg_parity [iters]
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/kernels/rope_scaling.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <type_traits>
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
template <typename T> void cpy(T* dst, const T* src, size_t n) { ck(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice), "cpy"); }
template <typename T> size_t bad(const T* a, const T* b, size_t n) {
    const auto x = down(a, n), y = down(b, n);
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += std::memcmp(&x[i], &y[i], sizeof(T)) != 0;
    return c;
}
uint16_t f16(float x) { __half h = __float2half(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }

constexpr int NSLOT = 8, NL = 2, MAXROWS = k::kQsaSegMaxRows, MAX_CELLS = 6144;
const k::QsaShapes S_ = k::qsa_real_shapes();
const int NKV = (int) S_.n_head_kv, HD = (int) S_.head_dim, NH = (int) S_.n_head, IQ = (int) S_.idx_n_head,
          ID = (int) S_.idx_dim, P = (int) S_.page_size, TS = (int) (S_.idx_block - 1) * ID;
const int PAGES = MAX_CELLS / P, MAXB = MAX_CELLS / 4 + 2;
const size_t ROWS = (size_t) PAGES * NKV * P;

// one slot's state of one layer
struct Slot {
    int8_t *kq, *vq, *hkq, *hvq;
    uint16_t *ks, *vs, *hks, *hvs, *kp, *vp;
    int32_t* table;
    float *tail, *dead, *pooled, *snap;
    int32_t* bpos;
};
Slot make_slot(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.002f, 0.05f);
    Slot x;
    x.kq = dalloc<int8_t>(ROWS * HD); x.vq = dalloc<int8_t>(ROWS * HD);
    x.hkq = dalloc<int8_t>(ROWS * HD); x.hvq = dalloc<int8_t>(ROWS * HD);
    x.ks = dalloc<uint16_t>(ROWS * HD / 64); x.vs = dalloc<uint16_t>(ROWS * HD / 64);
    x.hks = dalloc<uint16_t>(ROWS * HD / 64); x.hvs = dalloc<uint16_t>(ROWS * HD / 64);
    x.kp = dalloc<uint16_t>(ROWS * HD); x.vp = dalloc<uint16_t>(ROWS * HD);
    x.table = dalloc<int32_t>(PAGES);
    x.tail = dalloc<float>(TS); x.dead = dalloc<float>(ID); x.pooled = dalloc<float>((size_t) MAXB * ID);
    x.snap = dalloc<float>(TS); x.bpos = dalloc<int32_t>(1);
    std::vector<int8_t> c(ROWS * HD);
    std::vector<uint16_t> sc(ROWS * HD / 64), hp(ROWS * HD);
    for (int which = 0; which < 2; ++which) {
        for (auto& v : c) v = (int8_t) ((int) (rng() % 255) - 127);
        for (auto& v : sc) v = f16(ud(rng));
        for (auto& v : hp) v = f16(nd(rng) * 0.7f);
        up(which ? x.vq : x.kq, c);
        up(which ? x.vs : x.ks, sc);
        up(which ? x.vp : x.kp, hp);
    }
    std::vector<int32_t> tb(PAGES);
    std::iota(tb.begin(), tb.end(), 0);
    std::shuffle(tb.begin(), tb.end(), rng);
    for (int i = 5; i < PAGES; i += 61) tb[i] = -1;   // streamed-out pages: host copy only
    up(x.table, tb);
    auto rf = [&](float* d, size_t n, float scale) {
        std::vector<float> h(n);
        for (auto& v : h) v = nd(rng) * scale;
        up(d, h);
    };
    rf(x.tail, TS, 1.f); rf(x.dead, ID, 1.f); rf(x.pooled, (size_t) MAXB * ID, 1.f);
    return x;
}
Slot clone(const Slot& a) {
    Slot b = a;
    auto cp = [&](auto*& dst, auto* src, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(dst)>>;
        dst = dalloc<T>(n);
        cpy(dst, src, n);
    };
    cp(b.kq, a.kq, ROWS * HD); cp(b.vq, a.vq, ROWS * HD); cp(b.hkq, a.hkq, ROWS * HD); cp(b.hvq, a.hvq, ROWS * HD);
    cp(b.ks, a.ks, ROWS * HD / 64); cp(b.vs, a.vs, ROWS * HD / 64); cp(b.hks, a.hks, ROWS * HD / 64);
    cp(b.hvs, a.hvs, ROWS * HD / 64); cp(b.kp, a.kp, ROWS * HD); cp(b.vp, a.vp, ROWS * HD);
    cp(b.table, a.table, PAGES); cp(b.tail, a.tail, TS); cp(b.dead, a.dead, ID); cp(b.pooled, a.pooled, (size_t) MAXB * ID);
    cp(b.snap, a.snap, TS); cp(b.bpos, a.bpos, 1);
    return b;
}
k::KvHostPools host_of(const Slot& x) {
    k::KvHostPools h;
    h.k_q = x.hkq; h.v_q = x.hvq; h.k_scale = x.hks; h.v_scale = x.hvs;
    return h;
}
k::QsaSegSlot seg_slot(const Slot& x) {
    k::QsaSegSlot t;
    t.k_pool = x.kp; t.v_pool = x.vp; t.k_q = x.kq; t.v_q = x.vq; t.k_scale = x.ks; t.v_scale = x.vs;
    t.page_table = x.table; t.host = host_of(x); t.tail = x.tail; t.dead = x.dead; t.pooled = x.pooled;
    t.tail_snap = x.snap; t.block_pos = x.bpos; t.max_cells = MAX_CELLS;
    return t;
}
k::QsaAttnPools pools_of(const Slot& x, bool int8) {
    k::QsaAttnPools a;
    a.page_table = x.table;
    if (int8) { a.k_q = x.kq; a.v_q = x.vq; a.k_scale = x.ks; a.v_scale = x.vs; }
    else { a.k_pool = x.kp; a.v_pool = x.vp; }
    return a;
}

// buffers of one variant (solo or segment): one set per layer where the chain writes it
struct Out {
    float *kcur, *qcur, *qidx, *scores, *attn, *gated, *scratch;
    int32_t* ids;
    uint8_t* xq;
};
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const k::RopeScaling& sc = k::rope_scaling();
    const int cap = (int) k::qsa_selection_width(k::kTopkMaxCells, S_);
    const size_t sf = k::qsa_decode_attn_scratch_floats(cap, S_);
    const int CB = 2 + MAXROWS;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");

    // slots: solo copy (A) and segment copy (B) of every slot and layer
    std::vector<Slot> A[NL], B[NL];
    for (int l = 0; l < NL; ++l)
        for (int i = 0; i < NSLOT; ++i) { A[l].push_back(make_slot(rng)); B[l].push_back(clone(A[l].back())); }
    std::vector<k::QsaSegSlot*> d_slots(NL);
    for (int l = 0; l < NL; ++l) {
        std::vector<k::QsaSegSlot> h;
        for (const Slot& x : B[l]) h.push_back(seg_slot(x));
        d_slots[l] = dalloc<k::QsaSegSlot>(NSLOT);
        up(d_slots[l], h);
    }
    k::QsaSeg *d_rows = dalloc<k::QsaSeg>(MAXROWS), *d_segs = dalloc<k::QsaSeg>(NSLOT);

    // inputs of a window, shared by both variants
    float *kcur0 = dalloc<float>((size_t) MAXROWS * NKV * HD), *vcur = dalloc<float>((size_t) MAXROWS * NKV * HD);
    float *raw = dalloc<float>((size_t) MAXROWS * ID), *qfull = dalloc<float>((size_t) MAXROWS * NH * 2 * HD);
    float* qidx0 = dalloc<float>((size_t) MAXROWS * IQ * ID);
    float *knorm = dalloc<float>(HD), *iknorm = dalloc<float>(ID), *qnorm = dalloc<float>(HD), *iqnorm = dalloc<float>(ID);
    int32_t* steps = dalloc<int32_t>((size_t) MAXROWS * k::kStepCount);
    int32_t* pos_q = dalloc<int32_t>((size_t) MAXROWS * NH);
    int32_t* pos_k = dalloc<int32_t>((size_t) MAXROWS * NKV);
    int32_t* pos_i = dalloc<int32_t>((size_t) MAXROWS * IQ);
    int32_t* cpos = dalloc<int32_t>((size_t) NSLOT * CB);
    auto rnd = [&](float* d, size_t n, float scale, float off = 0.f) {
        std::vector<float> h(n);
        for (auto& x : h) x = nd(rng) * scale + off;
        up(d, h);
    };
    rnd(knorm, HD, 0.2f, 1.f); rnd(iknorm, ID, 0.2f, 1.f); rnd(qnorm, HD, 0.2f, 1.f); rnd(iqnorm, ID, 0.2f, 1.f);
    const size_t q8row = (size_t) NH * HD / 32 * 36;   // q8_1 blocks of one row: 32 codes + half2
    const size_t q8b = q8row * MAXROWS;
    Out O[2];   // 0 solo, 1 segment
    for (Out& o : O) {
        o.kcur = dalloc<float>((size_t) MAXROWS * NKV * HD); o.qcur = dalloc<float>((size_t) MAXROWS * NH * HD);
        o.qidx = dalloc<float>((size_t) MAXROWS * IQ * ID); o.scores = dalloc<float>((size_t) MAXROWS * MAXB);
        o.attn = dalloc<float>((size_t) MAXROWS * NH * HD); o.gated = dalloc<float>((size_t) MAXROWS * NH * HD);
        o.scratch = dalloc<float>(sf * MAXROWS); o.ids = dalloc<int32_t>((size_t) MAXROWS * cap);
        o.xq = dalloc<uint8_t>(q8b);
    }

    struct Window {
        int S = 0, R = 0;
        int slot[NSLOT], first[NSLOT], len[NSLOT];
    };
    // the steps / positions of a window's rows from the slots' positions
    auto stage = [&](const Window& w, const std::vector<int>& pos0, bool commit_keep) {
        std::vector<int32_t> hs((size_t) MAXROWS * k::kStepCount), hq((size_t) MAXROWS * NH), hk((size_t) MAXROWS * NKV),
            hi((size_t) MAXROWS * IQ), hc((size_t) NSLOT * CB, -1);
        std::vector<k::QsaSeg> rows(MAXROWS), segs(NSLOT, k::QsaSeg{-1, 0, 0, 0});   // unused entries: negative slot
        for (int j = 0; j < w.S; ++j) {
            segs[j] = {w.slot[j], w.first[j], w.len[j], 0};
            const int keep = commit_keep ? (int) (rng() % (w.len[j] + 1)) : w.len[j];
            for (int t = 0; t < w.len[j]; ++t) {
                const int r = w.first[j] + t, p = pos0[w.slot[j]] + t;
                rows[r] = {w.slot[j], w.first[j], w.len[j], 0};
                k::qsa_step_fill(hs.data() + r * k::kStepCount, p, S_);
                for (int h = 0; h < NH; ++h) hq[r * NH + h] = p;
                for (int h = 0; h < NKV; ++h) hk[r * NKV + h] = p;
                for (int h = 0; h < IQ; ++h) hi[r * IQ + h] = p;
                hc[w.slot[j] * CB + t] = t < keep ? p : -1;
            }
        }
        up(steps, hs); up(pos_q, hq); up(pos_k, hk); up(pos_i, hi); up(cpos, hc);
        up(d_rows, rows); up(d_segs, segs);
        rnd(kcur0, (size_t) w.R * NKV * HD, 1.5f);
        rnd(vcur, (size_t) w.R * NKV * HD, 1.0f);
        rnd(raw, (size_t) w.R * ID, 2.0f);
        rnd(qfull, (size_t) w.R * NH * 2 * HD, 1.2f);
        rnd(qidx0, (size_t) w.R * IQ * ID, 1.2f);
        for (Out& o : O) {
            cpy(o.kcur, kcur0, (size_t) w.R * NKV * HD);
            cpy(o.qidx, qidx0, (size_t) w.R * IQ * ID);
        }
    };
    auto kv_args = [&](Out& o, bool snap, int f) {
        k::QsaDecKvPost a;
        a.kcur = o.kcur + (size_t) f * NKV * HD; a.vcur = vcur + (size_t) f * NKV * HD; a.pos_k = pos_k + f * NKV;
        a.k_norm = knorm; a.steps = steps + f * k::kStepCount; a.idx_raw = raw + (size_t) f * ID; a.idx_k_norm = iknorm;
        a.tail_snap = snap ? (float*) 1 : nullptr; a.max_cells = MAX_CELLS; a.eps = 1e-6f;
        return a;
    };
    auto q_args = [&](Out& o, int f) {
        k::QsaDecQPost b;
        b.q_full = qfull + (size_t) f * NH * 2 * HD; b.qcur = o.qcur + (size_t) f * NH * HD; b.pos_q = pos_q + f * NH;
        b.q_norm = qnorm; b.q_idx = o.qidx + (size_t) f * IQ * ID; b.pos_i = pos_i + f * IQ; b.idx_q_norm = iqnorm;
        b.eps = 1e-6f;
        return b;
    };
    // the solo chain for segment j of layer l (the kernels the verify window runs on one sequence's rows)
    auto solo_kv = [&](const Window& w, int j, int l, bool int8) {
        Slot& x = A[l][w.slot[j]];
        if (!int8) return;
        k::QsaDecKvPost a = kv_args(O[0], true, w.first[j]);
        a.k_q = x.kq; a.v_q = x.vq; a.k_scale = x.ks; a.v_scale = x.vs; a.page_table = x.table; a.host = host_of(x);
        a.tail = x.tail; a.dead = x.dead; a.pooled = x.pooled; a.block_pos = x.bpos; a.tail_snap = x.snap;
        k::qsa_dec_kv_post(a, w.len[j], S_, sc, cs);
    };
    auto solo_rest = [&](const Window& w, int j, int l, bool int8) {
        Slot& x = A[l][w.slot[j]];
        const int f = w.first[j], n = w.len[j];
        k::qsa_dec_q_post(q_args(O[0], f), n, S_, sc, cs);
        k::qsa_block_scores(x.pooled, x.dead, O[0].qidx + (size_t) f * IQ * ID, steps + f * k::kStepCount, n, MAXB, S_,
                            O[0].scores + (size_t) f * MAXB, cs);
        k::qsa_block_topk(O[0].scores + (size_t) f * MAXB, steps + f * k::kStepCount, n, MAXB, cap, S_,
                          O[0].ids + (size_t) f * cap, cs);
        k::qsa_dec_attn(O[0].qcur + (size_t) f * NH * HD, pools_of(x, int8), O[0].ids + (size_t) f * cap,
                        steps + f * k::kStepCount, cap, S_, O[0].scratch + sf * f, O[0].attn + (size_t) f * NH * HD, n, cs,
                        qfull + (size_t) f * NH * 2 * HD, O[0].gated + (size_t) f * NH * HD, O[0].xq + f * q8row);
    };
    auto seg_table = [&](const Window& w, int l) {
        k::QsaSegTable t;
        t.slots = d_slots[l]; t.rows = d_rows; t.segs = d_segs; t.n_rows = w.R; t.n_seg = std::min(w.R, NSLOT);   // as many entries as rows, the unused ones skipped
        return t;
    };
    auto seg_kv = [&](const Window& w, int l, bool int8) {
        if (!int8) return;
        k::qsa_dec_kv_post_seg(kv_args(O[1], true, 0), seg_table(w, l), w.R, S_, sc, cs);
    };
    auto seg_rest = [&](const Window& w, int l, bool int8) {
        const k::QsaSegTable t = seg_table(w, l);
        k::qsa_dec_q_post(q_args(O[1], 0), w.R, S_, sc, cs);
        k::qsa_block_scores_seg(t, O[1].qidx, steps, w.R, MAXB, S_, O[1].scores, cs);
        k::qsa_block_topk(O[1].scores, steps, w.R, MAXB, cap, S_, O[1].ids, cs);
        k::qsa_dec_attn_seg(O[1].qcur, t, int8, O[1].ids, steps, cap, S_, O[1].scratch, O[1].attn, w.R, cs, qfull,
                            O[1].gated, O[1].xq);
    };
    auto commit_solo = [&](const Window& w, int j) {
        for (int l = 0; l < NL; ++l) {
            Slot& x = A[l][w.slot[j]];
            k::QsaCommitLayers qa;
            qa.pos = cpos + w.slot[j] * CB; qa.epsilon = 1e-6f; qa.n_tok = w.len[j]; qa.n_layers = 1;
            k::QsaCommitLayer& q = qa.layer[0];
            q.raw = raw + (size_t) w.first[j] * ID; q.tail_snap = x.snap; q.gamma = iknorm; q.tail = x.tail;
            q.dead = x.dead; q.pooled = x.pooled; q.block_pos = x.bpos; q.max_cells = MAX_CELLS;
            k::native_qsa_indexer_commit_layers(qa, S_, sc, cs);
        }
    };
    auto commit_seg = [&](const Window& w) {
        k::QsaSegCommit c;
        for (int l = 0; l < NL; ++l) { c.layer[l].slots = d_slots[l]; c.layer[l].raw = raw; c.layer[l].gamma = iknorm; }
        c.segs = d_segs; c.pos = cpos; c.pos_stride = CB; c.epsilon = 1e-6f; c.n_layers = NL; c.n_seg = std::min(w.R, NSLOT);
        k::native_qsa_indexer_commit_seg(c, S_, sc, cs);
    };
    auto cmp = [&](const char* what, size_t d, size_t& tot) {
        if (d) std::printf("  %s: %zu differ\n", what, d);
        tot += d;
    };
    auto cmp_state = [&](size_t& tot, bool big) {
        for (int l = 0; l < NL; ++l)
            for (int i = 0; i < NSLOT; ++i) {
                const Slot &a = A[l][i], &b = B[l][i];
                cmp("tail", bad(a.tail, b.tail, TS), tot);
                cmp("snap", bad(a.snap, b.snap, TS), tot);
                cmp("dead", bad(a.dead, b.dead, ID), tot);
                cmp("block_pos", bad(a.bpos, b.bpos, 1), tot);
                cmp("pooled", bad(a.pooled, b.pooled, (size_t) MAXB * ID), tot);
                if (!big) continue;
                cmp("k codes", bad(a.kq, b.kq, ROWS * HD), tot);
                cmp("v codes", bad(a.vq, b.vq, ROWS * HD), tot);
                cmp("k scales", bad(a.ks, b.ks, ROWS * HD / 64), tot);
                cmp("v scales", bad(a.vs, b.vs, ROWS * HD / 64), tot);
                cmp("host k codes", bad(a.hkq, b.hkq, ROWS * HD), tot);
                cmp("host v codes", bad(a.hvq, b.hvq, ROWS * HD), tot);
                cmp("host k scales", bad(a.hks, b.hks, ROWS * HD / 64), tot);
                cmp("host v scales", bad(a.hvs, b.hvs, ROWS * HD / 64), tot);
            }
    };

    // ---- parity: windows of random segments
    auto parity = [&](const char* label, int nseg, const std::vector<int>& start, int windows, bool int8,
                      int max_rows) {
        size_t tot = 0;
        std::vector<int> pos0(NSLOT, 0);
        for (int i = 0; i < NSLOT; ++i) pos0[i] = start[i % start.size()];
        int rows_total = 0;
        for (int wi = 0; wi < windows; ++wi) {
            Window w;
            w.S = nseg;
            std::vector<int> order(NSLOT);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);
            int R = 0;
            for (int j = 0; j < nseg; ++j) {
                const int room = max_rows - R - (nseg - 1 - j);
                w.slot[j] = order[j];
                w.first[j] = R;
                w.len[j] = 1 + (int) (rng() % std::min(8, room));
                R += w.len[j];
            }
            w.R = R;
            rows_total += R;
            stage(w, pos0, true);
            for (int l = 0; l < NL; ++l) {
                for (Out& o : O) { cpy(o.kcur, kcur0, (size_t) R * NKV * HD); cpy(o.qidx, qidx0, (size_t) R * IQ * ID); }
                // the indexer appends of a layer-0 window are the commit's to redo: the snapshot keeps the pre-window tail
                for (int j = 0; j < w.S; ++j) { solo_kv(w, j, l, int8); solo_rest(w, j, l, int8); }
                seg_kv(w, l, int8);
                seg_rest(w, l, int8);
                ck(cudaStreamSynchronize(cs), "sync");
                size_t d = 0;
                cmp("kcur", bad(O[0].kcur, O[1].kcur, (size_t) R * NKV * HD), d);
                cmp("qcur", bad(O[0].qcur, O[1].qcur, (size_t) R * NH * HD), d);
                cmp("q_idx", bad(O[0].qidx, O[1].qidx, (size_t) R * IQ * ID), d);
                cmp("scores", bad(O[0].scores, O[1].scores, (size_t) R * MAXB), d);
                cmp("ids", bad(O[0].ids, O[1].ids, (size_t) R * cap), d);
                cmp("attn", bad(O[0].attn, O[1].attn, (size_t) R * NH * HD), d);
                cmp("gated", bad(O[0].gated, O[1].gated, (size_t) R * NH * HD), d);
                cmp("q8_1 rows", bad(O[0].xq, O[1].xq, (size_t) R * q8row), d);
                if (d) std::printf("  window %d layer %d (S %d, R %d)\n", wi, l, w.S, R);
                tot += d;
            }
            if (int8) {
                // commit the window's kept rows on a restored tail, then advance
                for (int j = 0; j < w.S; ++j) commit_solo(w, j);
                commit_seg(w);
                ck(cudaStreamSynchronize(cs), "sync");
            }
            cmp_state(tot, false);
            for (int j = 0; j < w.S; ++j) {
                pos0[w.slot[j]] += w.len[j];
                if (pos0[w.slot[j]] + 24 >= MAX_CELLS) pos0[w.slot[j]] = 3 + (int) (rng() % 40);
            }
        }
        cmp_state(tot, true);
        if (tot) ++g_fail;
        std::printf("%-30s S %d %s, %d windows, %d rows: %s\n", label, nseg, int8 ? "int8" : "fp16", windows, rows_total,
                    tot ? "FAIL" : "bitwise");
    };
    for (int mode = 1; mode >= 0; --mode) {
        parity("one sequence", 1, {1700}, 24, mode, 8);
        parity("two, one at cell 0", 2, {0, 3000}, 24, mode, 12);
        parity("five, around the width", 5, {5, 2040, 2049, 2300, 5000}, 24, mode, 14);
        parity("eight, mixed", 8, {0, 1, 2047, 2051, 3333, 4096, 5400, 100}, 24, mode, 16);
    }

    // ---- microbench: a fixed window of 5 segments (8 rows) and of 8 segments (8 rows), one call vs one per segment
    for (int nseg : {2, 5, 8}) {
        Window w;
        w.S = nseg;
        int R = 0;
        const int lens[8] = {2, 1, 3, 1, 1, 1, 1, 1};
        std::vector<int> pos0 = {1300, 2400, 5000, 3100, 900, 4700, 2051, 3800};
        for (int j = 0; j < nseg; ++j) { w.slot[j] = j; w.first[j] = R; w.len[j] = nseg == 2 ? 4 : lens[j]; R += w.len[j]; }
        w.R = R;
        stage(w, pos0, false);
        const k::QsaSegTable t = seg_table(w, 0);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        auto time = [&](auto&& fn) {
            fn();
            ck(cudaStreamSynchronize(cs), "warm");
            cudaEventRecord(e0, cs);
            for (int it = 0; it < iters; ++it) fn();
            cudaEventRecord(e1, cs);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double kv_s = time([&] { for (int j = 0; j < nseg; ++j) solo_kv(w, j, 0, true); });
        const double kv_g = time([&] { seg_kv(w, 0, true); });
        const double sc_s = time([&] {
            for (int j = 0; j < nseg; ++j)
                k::qsa_block_scores(A[0][j].pooled, A[0][j].dead, O[0].qidx + (size_t) w.first[j] * IQ * ID,
                                    steps + w.first[j] * k::kStepCount, w.len[j], MAXB, S_, O[0].scores + (size_t) w.first[j] * MAXB, cs);
        });
        const double sc_g = time([&] { k::qsa_block_scores_seg(t, O[1].qidx, steps, w.R, MAXB, S_, O[1].scores, cs); });
        const double at_s = time([&] {
            for (int j = 0; j < nseg; ++j) {
                const int f = w.first[j];
                k::qsa_dec_attn(O[0].qcur + (size_t) f * NH * HD, pools_of(A[0][j], true), O[0].ids + (size_t) f * cap,
                                steps + f * k::kStepCount, cap, S_, O[0].scratch + sf * f, O[0].attn + (size_t) f * NH * HD,
                                w.len[j], cs, qfull + (size_t) f * NH * 2 * HD, O[0].gated + (size_t) f * NH * HD,
                                O[0].xq + f * q8row);
            }
        });
        const double at_g = time([&] {
            k::qsa_dec_attn_seg(O[1].qcur, t, true, O[1].ids, steps, cap, S_, O[1].scratch, O[1].attn, w.R, cs, qfull,
                                O[1].gated, O[1].xq);
        });
        const double cm_s = time([&] { for (int j = 0; j < nseg; ++j) commit_solo(w, j); });
        const double cm_g = time([&] { commit_seg(w); });
        std::printf("S %d (%d rows): kv_post %6.1f -> %6.1f us | scores %6.1f -> %6.1f | attn+merge %6.1f -> %6.1f | commit "
                    "(%d layers) %6.1f -> %6.1f   (per-segment launches -> one segment launch)\n",
                    nseg, R, kv_s, kv_g, sc_s, sc_g, at_s, at_g, NL, cm_s, cm_g);
    }
    std::printf("%s\n", g_fail ? "qsa_seg_parity: FAIL" : "qsa_seg_parity: OK (bitwise)");
    return g_fail ? 1 : 0;
}
