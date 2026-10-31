// tests/kernels/kv_stream_parity.cpp - KV streaming (kv_stream.hpp) against a fully resident pool (GPU, no model).
//
// Decodes a synthetic sequence twice: into a fully resident pool with the identity page table, and into a streamed
// layer (host copy + a small VRAM slot pool that must evict constantly). Every few cells a batch of 1-8 queries
// with selections shaped like the indexer's (whole 4-cell blocks, a partial threshold block, the tail block, half of
// them recent) is attended in both; `kv_stream_resolve` runs before the streamed attention. Checks:
//   1. the attention outputs are BITWISE equal (the streamed readers see exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other);
//   3. no call overflowed, and the hit/miss counters add up;
//   4. a ring (the MTP drafter's layout) restored from the host copy reads the same values as the resident pool;
//   5. that ring saved back (kv_ring_save) over a second host copy gives the first's bytes.
// INT8, FP16 and Q4_0 pools. Then the batch windows' form, kv_stream_resolve_seg (run_segments).
//
//     build/kv_stream_parity [--bench]
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_seg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = basalt::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {   // pinned, mapped; returns the device pointer
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

struct Pools {   // one K/V pool set of `pages` pages
    k::KvHostPools p;   // reused as a plain pointer bundle
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool host) {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size;
        if (fmt == k::kKvQ4) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvInt8) {
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.v_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
        } else {
            p.k_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
            p.v_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
};

void append(const Pools& pl, const int32_t* table, const int32_t* step, const float* kc, const float* vc,
            const k::QsaShapes& s, int fmt, const k::KvHostPools* host) {
    if (fmt == k::kKvQ4) k::kv_append_q4_step(pl.p.k_q4, pl.p.v_q4, table, step, kc, vc, s, nullptr, host);
    else if (fmt == k::kKvInt8) k::kv_append_q8_step(pl.p.k_q, pl.p.v_q, pl.p.k_scale, pl.p.v_scale, table, step, kc, vc, s, nullptr, host);
    else k::kv_append_step(pl.p.k_pool, pl.p.v_pool, table, step, kc, vc, s, nullptr, host);
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, the tail block's cells, and one partial block.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

const uint8_t* host_of(const void* dev) {   // a mapped host pool's host address
    cudaPointerAttributes a{};
    ck(cudaPointerGetAttributes(&a, dev), "pointer attributes");
    if (a.hostPointer == nullptr) { std::fprintf(stderr, "no host address\n"); std::exit(2); }
    return (const uint8_t*) a.hostPointer;
}

struct HostArray { const uint8_t* x; const uint8_t* y; size_t len; };   // an array of two host copies, bytes per block
std::vector<HostArray> host_arrays(const Pools& x, const Pools& y, int fmt, const k::QsaShapes& s) {
    const size_t rows = (size_t) (s.n_head_kv * s.page_size);
    if (fmt == k::kKvQ4) {
        const size_t n = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
        return {{host_of(x.p.k_q4), host_of(y.p.k_q4), n}, {host_of(x.p.v_q4), host_of(y.p.v_q4), n}};
    }
    if (fmt == k::kKvInt8) {
        const size_t c = rows * (size_t) s.head_dim, sc = rows * (size_t) (s.head_dim / k::KV_Q8_GROUP) * 2;
        return {{host_of(x.p.k_q), host_of(y.p.k_q), c}, {host_of(x.p.v_q), host_of(y.p.v_q), c},
                {host_of(x.p.k_scale), host_of(y.p.k_scale), sc}, {host_of(x.p.v_scale), host_of(y.p.v_scale), sc}};
    }
    const size_t n = rows * (size_t) s.head_dim * 2;
    return {{host_of(x.p.k_pool), host_of(y.p.k_pool), n}, {host_of(x.p.v_pool), host_of(y.p.v_pool), n}};
}

bool run(int fmt) {
    const char* name = fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 40000, n_blocks = (N + 3) / 4, n_slots = 8 * 516 + 700;   // must evict: slots < blocks
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), NQ = 8;
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(7 + 4 * fmt);
    std::normal_distribution<float> nd(0.f, 1.f);

    Pools ref, slots, host;
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
    host.alloc(n_blocks, s, fmt, true);
    int32_t* ident = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
    }
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(n_blocks);
    m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
    m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    k::kv_stream_reset(m, nullptr);

    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(NQ * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(NQ * cap);
    float* q = dalloc<float>(NQ * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(NQ * scr);
    float* out_ref = dalloc<float>(NQ * NH * D);
    float* out_str = dalloc<float>(NQ * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(NQ * NH * D), a((size_t) NQ * NH * D), b2((size_t) NQ * NH * D);
    std::vector<int32_t> pt(n_blocks), sb(n_slots);

    int batches = 0, bad = 0;
    for (int64_t pos = 0; pos < N; ++pos) {
        for (auto& x : hk) x = nd(rng) * ((pos % 17 == 0) ? 30.f : 1.f);
        for (auto& x : hv) x = nd(rng);
        const int32_t st[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice), "step");
        append(ref, ident, step, kc, vc, s, fmt, nullptr);
        append(slots, m.page_table, step, kc, vc, s, fmt, &host.p);
        if (pos % 131 != 130 && pos != N - 1) continue;
        // a batch: n_q queries at the last n_q positions (as a verify window), each with its own selection
        const int n_q = 1 + (int) (rng() % NQ);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = std::max<int64_t>(0, pos - (n_q - 1 - t)), n_kv = p + 1;
            const int64_t width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            if ((int64_t) sel.size() != width) { std::fprintf(stderr, "selection size %zu != %lld\n", sel.size(), (long long) width); return false; }
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv; hst[t * 4 + 2] = (int32_t) (n_kv / 4);
            hst[t * 4 + 3] = (int32_t) width;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, n_q, nullptr);
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids, steps, n_q, cap, s, nullptr);
        k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "batch");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) n_q * NH * D * 4) != 0) {
            if (bad++ < 5) std::fprintf(stderr, "  %s pos %lld n_q %d: streamed attention differs\n", name, (long long) pos, n_q);
        }
        // the map inverts itself
        ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        ck(cudaMemcpy(sb.data(), m.slot_block, n_slots * 4, cudaMemcpyDeviceToHost), "sb");
        int64_t resident = 0;
        for (int64_t bk = 0; bk < n_blocks; ++bk) {
            if (pt[bk] < -1 || pt[bk] >= n_slots || (pt[bk] >= 0 && sb[pt[bk]] != bk)) {
                if (bad++ < 5) std::fprintf(stderr, "  map broken at block %lld (table %d)\n", (long long) bk, pt[bk]);
                break;
            }
            resident += pt[bk] >= 0;
        }
        for (int64_t sl = 0; sl < n_slots; ++sl)
            if (sb[sl] >= 0 && pt[sb[sl]] != sl) { if (bad++ < 5) std::fprintf(stderr, "  slot %lld not in the table\n", (long long) sl); break; }
        ++batches;
    }
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %s: %d batches, %llu block lookups, %llu misses (%.1f%% hit), overflow %d, %d failures\n",
                name, batches, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow, bad);
    if (c.overflow || c.calls != (uint64_t) batches || c.misses == 0 || c.misses >= c.lookups) ++bad;

    // 4. a ring over the same host copy: blocks [b1 - R, b1) restored, read through `b % R`
    {
        const int64_t R = 1500, b1 = n_blocks, b0 = b1 - R;
        Pools ring;
        ring.alloc(R, s, fmt, false);
        int32_t* rt = dalloc<int32_t>(n_blocks);
        k::kv_ring_table(rt, n_blocks, R, nullptr);
        k::kv_ring_restore(ring.attn(rt), host.p, fmt, b0, b1, R, s, nullptr);
        const int64_t width = cap;
        std::vector<int32_t> hids((size_t) cap), hst = {(int32_t) (N - 1), (int32_t) N, (int32_t) (N / 4), (int32_t) width};
        for (int64_t i = 0; i < width; ++i) hids[i] = (int32_t) (N - width + i);   // the window's last cells
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), 16, cudaMemcpyHostToDevice), "steps");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, 1, nullptr);
        k::qsa_decode_attn_batch(q, ring.attn(rt), ids, steps, cap, s, scratch, out_str, 1, nullptr);
        ck(cudaDeviceSynchronize(), "ring");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "b");
        const bool ok = std::memcmp(a.data(), b2.data(), (size_t) NH * D * 4) == 0;
        std::printf("  %s ring restore: %s\n", name, ok ? "identical" : "DIFFERS");
        if (!ok) ++bad;
        // 5. the ring saved back over a second host copy: blocks [b0, b1) byte-equal to the first's (on the CPU)
        Pools back;
        back.alloc(n_blocks, s, fmt, true);
        k::kv_ring_save(ring.attn(rt), back.p, fmt, b0, b1, R, s, nullptr);
        ck(cudaDeviceSynchronize(), "ring save");
        bool same = true;
        for (const auto& [x, y, len] : host_arrays(host, back, fmt, s))
            same = same && std::memcmp(x + (size_t) b0 * len, y + (size_t) b0 * len, (size_t) (b1 - b0) * len) == 0;
        std::printf("  %s ring save: %s\n", name, same ? "identical" : "DIFFERS");
        if (!same) ++bad;
    }
    return bad == 0;
}

// The batch windows' form, kv_stream_resolve_seg (INT8). NS slots decode their own sequences, the last one resident
// (a null map, which the resolve skips); each streamed slot has its own map, pool and host copy, and a twin map and
// pool that the solo kv_stream_resolve drives over that slot's rows alone. A window is a random set of slots in
// shuffled order, 1-4 rows each, with unused table entries. After its one segment resolve:
//   1. the segment attention over the streamed slots is BITWISE what it is over the resident pools;
//   2. every streamed slot's map is bitwise its twin's: the solo kernel's, so a slot's residency does not depend on
//      the window's other segments, its width or the slot's place in it;
//   3. on the CPU: every block a segment selected is resident and its page holds the host copy's bytes, and every
//      map inverts itself;
//   4. no overflow, a slot's resolve calls equal the windows it was in, and the pools evict.
// --bench: the segment resolve against one solo resolve per slot, in steady state and cold (every block missing).
bool run_segments(bool bench) {
    const int fmt = k::kKvInt8, NS = 4, RESIDENT = NS - 1;
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 24000, n_blocks = (N + 3) / 4, n_slots = 8 * 516 + 700;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), MAXR = k::kQsaSegMaxRows;
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    const size_t codes = (size_t) (H * s.page_size * D), scales = (size_t) (H * s.page_size * (D / k::KV_Q8_GROUP) * 2);
    const size_t run_len[4] = {codes, codes, scales, scales};
    std::mt19937 rng(91);
    std::normal_distribution<float> nd(0.f, 1.f);

    int32_t* ident = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
    }
    auto new_map = [&] {
        k::KvStreamMap m;
        m.page_table = dalloc<int32_t>(n_blocks);
        m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
        m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
        m.ctl = dalloc<int32_t>(k::kKvCtlInts);
        m.n_blocks = n_blocks; m.n_slots = n_slots;
        k::kv_stream_reset(m, nullptr);
        return m;
    };
    auto seg_slot = [](const Pools& p, int32_t* table) {
        k::QsaSegSlot e;
        e.k_q = p.p.k_q; e.v_q = p.p.v_q; e.k_scale = p.p.k_scale; e.v_scale = p.p.v_scale; e.page_table = table;
        return e;
    };
    std::vector<Pools> ref(NS), pool(NS), twin(NS), host(NS);
    std::vector<k::KvStreamMap> maps(NS), tmaps(NS);
    std::vector<k::QsaSegSlot> h_str(NS), h_res(NS);   // the streamed table and the all-resident one
    for (int i = 0; i < NS; ++i) {
        ref[i].alloc(n_blocks, s, fmt, false);
        h_res[i] = seg_slot(ref[i], ident);
        if (i == RESIDENT) { h_str[i] = h_res[i]; continue; }
        pool[i].alloc(n_slots, s, fmt, false);
        twin[i].alloc(n_slots, s, fmt, false);
        host[i].alloc(n_blocks, s, fmt, true);
        maps[i] = new_map();
        tmaps[i] = new_map();
        h_str[i] = seg_slot(pool[i], maps[i].page_table);
        h_str[i].host = host[i].p;
        h_str[i].map = maps[i];
    }
    std::vector<std::array<const uint8_t*, 4>> hh(NS);   // the host copies' arrays, read by the CPU
    for (int i = 0; i < NS; ++i)
        if (i != RESIDENT) hh[i] = {host_of(host[i].p.k_q), host_of(host[i].p.v_q), host_of(host[i].p.k_scale),
                                    host_of(host[i].p.v_scale)};
    k::QsaSegSlot* d_str = dalloc<k::QsaSegSlot>(NS);
    k::QsaSegSlot* d_res = dalloc<k::QsaSegSlot>(NS);
    ck(cudaMemcpy(d_str, h_str.data(), NS * sizeof(k::QsaSegSlot), cudaMemcpyHostToDevice), "slots");
    ck(cudaMemcpy(d_res, h_res.data(), NS * sizeof(k::QsaSegSlot), cudaMemcpyHostToDevice), "slots");
    k::QsaSeg* d_rows = dalloc<k::QsaSeg>(MAXR);
    k::QsaSeg* d_segs = dalloc<k::QsaSeg>(MAXR);

    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(MAXR * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(MAXR * cap);
    float* q = dalloc<float>(MAXR * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(MAXR * scr);
    float* out_ref = dalloc<float>(MAXR * NH * D);
    float* out_str = dalloc<float>(MAXR * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(MAXR * NH * D), a((size_t) MAXR * NH * D), b2((size_t) MAXR * NH * D);
    std::vector<int32_t> pt(n_blocks), sb(n_slots), tpt(n_blocks), tsb(n_slots);
    std::vector<int32_t> mx(n_slots), tmx(n_slots), ctl(k::kKvCtlInts), tctl(k::kKvCtlInts);
    std::vector<std::vector<uint8_t>> dl(4);
    for (int i = 0; i < 4; ++i) dl[i].resize((size_t) n_slots * run_len[i]);
    std::vector<int> windows_in(NS, 0);
    std::vector<k::QsaSeg> segs, rows;
    std::vector<int32_t> hids, hst;
    int R = 0, nseg = 0;
    // a window at `pos`: a random set of slots in shuffled order, each 1-4 rows at its last positions; unused entries -1
    auto make_window = [&](int64_t pos, int want, int len_fixed) {
        std::vector<int> order(NS);
        for (int i = 0; i < NS; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        nseg = want;
        segs.assign((size_t) MAXR, k::QsaSeg{-1, 0, 0, 0});
        rows.assign((size_t) MAXR, k::QsaSeg{-1, 0, 0, 0});
        hids.assign((size_t) MAXR * cap, 0);
        hst.assign((size_t) MAXR * 4, 0);
        R = 0;
        for (int j = 0; j < nseg; ++j) {
            const int len = len_fixed > 0 ? len_fixed : 1 + (int) (rng() % 4);
            segs[j] = {order[j], R, len, 0};
            for (int t = 0; t < len; ++t) {
                rows[R + t] = segs[j];
                const int64_t p = std::max<int64_t>(0, pos - (len - 1 - t)), n_kv = p + 1;
                const int64_t width = std::min<int64_t>(n_kv, cap);
                const std::vector<int32_t> sel = selection(n_kv, width, rng);
                std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) (R + t) * cap);
                int32_t* st = hst.data() + (R + t) * 4;
                st[0] = (int32_t) p; st[1] = (int32_t) n_kv; st[2] = (int32_t) (n_kv / 4); st[3] = (int32_t) width;
            }
            R += len;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        ck(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(k::QsaSeg), cudaMemcpyHostToDevice), "rows");
        ck(cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(k::QsaSeg), cudaMemcpyHostToDevice), "segs");
    };
    auto table = [&](const k::QsaSegSlot* d) {   // as many entries as rows, as the engine stages them
        k::QsaSegTable t;
        t.slots = d; t.rows = d_rows; t.segs = d_segs; t.n_rows = R; t.n_seg = R;
        return t;
    };
    auto solo = [&](int j) {   // segment j through the current kernel, on its slot's twin
        const k::QsaSeg& g = segs[(size_t) j];
        if (g.slot == RESIDENT) return;
        k::kv_stream_resolve(tmaps[g.slot], twin[g.slot].attn(tmaps[g.slot].page_table), host[g.slot].p, fmt,
                             ids + (size_t) g.first * cap, steps + (size_t) g.first * 4, g.len, cap, s, nullptr);
    };

    int windows = 0, bad = 0;
    int64_t checked_blocks = 0;
    for (int64_t pos = 0; pos < N; ++pos) {
        for (int i = 0; i < NS; ++i) {   // every slot advances one cell, its own values
            for (auto& x : hk) x = nd(rng) * ((pos % 17 == i) ? 30.f : 1.f);
            for (auto& x : hv) x = nd(rng);
            const int32_t st[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
            ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
            ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
            ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice), "step");
            append(ref[i], ident, step, kc, vc, s, fmt, nullptr);
            if (i == RESIDENT) continue;
            append(pool[i], maps[i].page_table, step, kc, vc, s, fmt, &host[i].p);
            append(twin[i], tmaps[i].page_table, step, kc, vc, s, fmt, nullptr);
        }
        if (pos % 131 != 130 && pos != N - 1) continue;
        make_window(pos, 1 + (int) (rng() % NS), 0);
        for (int j = 0; j < nseg; ++j) ++windows_in[segs[(size_t) j].slot];
        k::kv_stream_resolve_seg(table(d_str), ids, steps, cap, s, nullptr);
        for (int j = 0; j < nseg; ++j) solo(j);
        k::qsa_dec_attn_seg(q, table(d_res), true, ids, steps, cap, s, scratch, out_ref, R, nullptr);
        k::qsa_dec_attn_seg(q, table(d_str), true, ids, steps, cap, s, scratch, out_str, R, nullptr);
        ck(cudaDeviceSynchronize(), "window");
        // 1. the attention
        ck(cudaMemcpy(a.data(), out_ref, (size_t) R * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) R * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) R * NH * D * 4) != 0)
            if (bad++ < 5) std::fprintf(stderr, "  segments pos %lld (%d segments, %d rows): streamed attention differs\n",
                                        (long long) pos, nseg, R);
        for (int j = 0; j < nseg; ++j) {
            const int i = segs[(size_t) j].slot;
            if (i == RESIDENT) continue;
            const k::KvStreamMap& m = maps[i];
            const k::KvStreamMap& tm = tmaps[i];
            // 2. the map against the solo kernel's
            auto same = [&](const int32_t* x, const int32_t* y, int64_t n, std::vector<int32_t>& hx, std::vector<int32_t>& hy) {
                hx.resize((size_t) n);
                hy.resize((size_t) n);
                ck(cudaMemcpy(hx.data(), x, (size_t) n * 4, cudaMemcpyDeviceToHost), "map");
                ck(cudaMemcpy(hy.data(), y, (size_t) n * 4, cudaMemcpyDeviceToHost), "twin map");
                return hx == hy;
            };
            const bool eq = same(m.page_table, tm.page_table, n_blocks, pt, tpt) & same(m.slot_block, tm.slot_block, n_slots, sb, tsb) &
                            same(m.slot_stamp, tm.slot_stamp, n_slots, mx, tmx) & same(m.slot_ref, tm.slot_ref, n_slots, mx, tmx) &
                            same(m.ctl, tm.ctl, k::kKvCtlInts, ctl, tctl);
            if (!eq && bad++ < 5)
                std::fprintf(stderr, "  segments pos %lld slot %d: the map differs from the solo resolve's\n", (long long) pos, i);
            // 3. on the CPU: the selected blocks resident with the host copy's bytes, and the map inverts itself
            const void* pa[4] = {pool[i].p.k_q, pool[i].p.v_q, pool[i].p.k_scale, pool[i].p.v_scale};
            for (int r = 0; r < 4; ++r) ck(cudaMemcpy(dl[r].data(), pa[r], dl[r].size(), cudaMemcpyDeviceToHost), "pool");
            std::set<int32_t> blocks;
            const k::QsaSeg& g = segs[(size_t) j];
            for (int t = g.first; t < g.first + g.len; ++t)
                for (int32_t c = 0; c < hst[(size_t) t * 4 + 3]; ++c) blocks.insert(hids[(size_t) t * cap + c] / (int32_t) s.page_size);
            for (int32_t bk : blocks) {
                const int32_t pg = pt[(size_t) bk];
                bool ok = pg >= 0 && pg < n_slots;
                for (int r = 0; ok && r < 4; ++r)
                    ok = std::memcmp(dl[r].data() + (size_t) pg * run_len[r], hh[i][r] + (size_t) bk * run_len[r],
                                     run_len[r]) == 0;
                if (!ok) {
                    if (bad++ < 5) std::fprintf(stderr, "  segments pos %lld slot %d: block %d not resident as its host copy\n",
                                                (long long) pos, i, bk);
                    break;
                }
            }
            checked_blocks += (int64_t) blocks.size();
            for (int64_t bk = 0; bk < n_blocks; ++bk)
                if (pt[bk] < -1 || pt[bk] >= n_slots || (pt[bk] >= 0 && sb[pt[bk]] != bk)) {
                    if (bad++ < 5) std::fprintf(stderr, "  slot %d: map broken at block %lld\n", i, (long long) bk);
                    break;
                }
            for (int64_t sl = 0; sl < n_slots; ++sl)
                if (sb[sl] >= 0 && pt[sb[sl]] != sl) {
                    if (bad++ < 5) std::fprintf(stderr, "  slot %d: pool page %lld not in the table\n", i, (long long) sl);
                    break;
                }
        }
        ++windows;
    }
    // 4. the counters
    for (int i = 0; i < NS; ++i) {
        if (i == RESIDENT) continue;
        const k::KvStreamCounters c = k::kv_stream_counters(maps[i]);
        std::printf("  segments int8, slot %d: %llu calls in %d windows, %llu block lookups, %llu misses (%.1f%% hit), "
                    "overflow %d\n", i, (unsigned long long) c.calls, windows_in[i], (unsigned long long) c.lookups,
                    (unsigned long long) c.misses,
                    c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow);
        if (c.overflow || c.calls != (uint64_t) windows_in[i] || c.misses == 0 || c.misses >= c.lookups) ++bad;
    }
    std::printf("  segments int8: %d windows, the attention bitwise the resident pools', every map bitwise the solo "
                "resolve's, %lld selected blocks byte-equal to their host copies; %d failures\n", windows,
                (long long) checked_blocks, bad);

    if (bench && bad == 0) {   // every streamed slot in one window, 4 rows each, at the sequence's end
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        constexpr int kIters = 50;
        const uint64_t blk = k::kv_block_bytes(s, fmt);
        for (const bool cold : {false, true}) {
            double us_seg = 0, us_solo = 0;
            uint64_t misses = 0;
            for (int it = 0; it < kIters; ++it) {
                make_window(N - 1, NS, 4);
                for (int pass = 0; pass < 2; ++pass) {   // 0: the segment resolve, 1: one solo resolve per slot
                    if (cold)
                        for (int i = 0; i < NS; ++i)
                            if (i != RESIDENT) k::kv_stream_reset(pass == 0 ? maps[i] : tmaps[i], nullptr);
                    uint64_t before = 0;
                    for (int i = 0; i < NS; ++i)
                        if (i != RESIDENT) before += k::kv_stream_counters(pass == 0 ? maps[i] : tmaps[i]).misses;
                    ck(cudaEventRecord(e0, nullptr), "record");
                    if (pass == 0) k::kv_stream_resolve_seg(table(d_str), ids, steps, cap, s, nullptr);
                    else for (int j = 0; j < nseg; ++j) solo(j);
                    ck(cudaEventRecord(e1, nullptr), "record");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms = 0;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                    (pass == 0 ? us_seg : us_solo) += ms * 1000.0 / kIters;
                    uint64_t after = 0;
                    for (int i = 0; i < NS; ++i)
                        if (i != RESIDENT) after += k::kv_stream_counters(pass == 0 ? maps[i] : tmaps[i]).misses;
                    if (pass == 0) misses += after - before;
                }
            }
            const double mb = (double) misses / kIters * (double) blk / 1e6;
            std::printf("bench segments %s (%d streamed slots x 4 rows, %lld-cell selections): one segment resolve "
                        "%.1f us (%.2f MB of misses per call: %.1f GB/s) | one solo resolve per slot %.1f us\n",
                        cold ? "cold, every block missing" : "steady", NS - 1, (long long) cap, us_seg, mb,
                        us_seg > 0 ? mb / 1e3 / (us_seg * 1e-6) : 0.0, us_solo);
        }
    }
    return bad == 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::printf("kv_stream_parity: streamed vs resident KV, bitwise\n");
    const bool a = run(k::kKvInt8), b = run(k::kKvF16), c = run(k::kKvQ4), d = run_segments(bench);
    if (!a || !b || !c || !d) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
