// tests/core/kv_mode_switch_test.cpp - one elastic QSA state switched between whole-in-VRAM and streamed in place
// (qsa_state_set_mode, kv_host_pool.hpp), against a plain whole-in-VRAM state fed the same cells (GPU, no model).
//
// A synthetic sequence is appended to both.  The elastic state starts in mode 0, switches to mode 1 at a depth (a host
// pool for the request's cells), moves to a larger host pool, switches back to mode 0, and streams once more.  Every
// few cells a batch of 1-8 queries is attended in both (kv_stream_resolve before the streamed attention).  Checks:
//   1. the attention outputs are BITWISE equal in every phase;
//   2. while streamed, the K/V arrays stay at the slot pool's pages (the depth follows only in the pooled keys);
//   3. after the switch back, the VRAM K/V of every live cell equals the reference's;
//   4. a conversation snapshot taken in mode 1 equals one taken in mode 0 at the same depth, and restores into mode 1.
// INT8, FP16 and Q4_0, with the elastic floor at the slot pool and below it.  Exits 77 without a CUDA device or virtual
// memory management.
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/kv_arena.hpp"
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace k = basalt::kernels;
namespace c = basalt::core;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// as kv_stream_parity: ascending cells, whole blocks, the tail block, one partial block
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t i = 0; i < n_kv; ++i) ids.push_back((int32_t) i);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    const int64_t partial = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks)
        for (int64_t i = 0; i < (b == partial ? part : 4); ++i) ids.push_back((int32_t) (b * 4 + i));
    for (int64_t i = n_bid * 4; i < n_kv; ++i) ids.push_back((int32_t) i);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void append(const c::QsaState& st, const int32_t* step, const float* kc, const float* vc, const k::QsaShapes& s) {
    const k::KvHostPools* host = st.host.present() ? &st.host : nullptr;
    if (st.kv_q4) k::kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step, kc, vc, s, nullptr, host);
    else if (st.kv_int8) k::kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step, kc, vc, s, nullptr, host);
    else k::kv_append_step(st.k_pool, st.v_pool, st.page_table, step, kc, vc, s, nullptr, host);
}

bool same_device(const void* a, const void* b, size_t n) {
    std::vector<uint8_t> x(n), y(n);
    ck(cudaMemcpy(x.data(), a, n, cudaMemcpyDefault), "cmp a");
    ck(cudaMemcpy(y.data(), b, n, cudaMemcpyDefault), "cmp b");
    return std::memcmp(x.data(), y.data(), n) == 0;
}

int run(int fmt, int64_t floor) {
    const char* name = fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    std::printf("  %s, elastic floor %lld cells:\n", name, (long long) floor);
    c::qsa_set_kv_int8(fmt == k::kKvInt8);
    c::qsa_set_kv_q4(fmt == k::kKvQ4);
    const c::ModelGeometry g{};
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t max_cells = 65536, slot = 32768, N = 60000, switch_at = 20000, grow_at = 36000, back_at = 48000;
    // the plain reference: whole in VRAM, identity table
    c::qsa_set_kv_elastic(0);
    c::qsa_set_kv_host(0);
    c::QsaState ref;
    void* rb = dalloc<uint8_t>(c::qsa_state_bytes(g, max_cells));
    if (c::qsa_state_init(g, max_cells, rb, ref) == 0) { std::fprintf(stderr, "reference init failed\n"); return 1; }
    // the switchable state: elastic, a slot pool of `slot` cells
    c::qsa_set_kv_elastic(floor);
    c::qsa_set_kv_host(slot);
    c::QsaState st;
    void* sb = dalloc<uint8_t>(c::qsa_state_bytes(g, max_cells));
    const uint64_t used = c::qsa_state_init(g, max_cells, sb, st);
    c::qsa_set_kv_elastic(0);
    c::qsa_set_kv_host(0);
    if (used == 0 || st.kv_arena == nullptr || st.host_slots != slot / s.page_size) {
        std::fprintf(stderr, "%s: switchable init failed (arena %p, host slots %lld)\n", name, (void*) st.kv_arena,
                     (long long) st.host_slots);
        return 1;
    }
    c::SessionState ss;   // a one-layer session over the state, for the host pool
    ss.qsa_states = &st;
    ss.qsa_ord0 = 0;
    ss.qsa_alloc = 1;
    c::qsa_state_zero(ref, g, nullptr);
    c::qsa_state_zero(st, g, nullptr);

    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), NQ = 8;
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(11 + 5 * fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(NQ * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(NQ * cap);
    float* q = dalloc<float>(NQ * NH * D);
    float* scratch = dalloc<float>(NQ * k::qsa_decode_attn_scratch_floats(cap, s));
    float* out_ref = dalloc<float>(NQ * NH * D);
    float* out_st = dalloc<float>(NQ * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(NQ * NH * D), a((size_t) NQ * NH * D), b((size_t) NQ * NH * D);
    c::KvHostPool pool, pool2;
    std::string err;
    int bad = 0, batches = 0;
    k::KvStreamCounters cn{};   // the streamed phase's, read before the switch back frees the residency map
    const uint64_t page_kv = k::kv_block_bytes(s, fmt);

    for (int64_t pos = 0; pos < N; ++pos) {
        if (pos == switch_at) {   // 0 -> 1: the slot pages backed, a host pool for the request's cells, the VRAM K/V past
                                  // the slots given back
            if (!st.kv_arena->map(slot, err) || !pool.reserve(ss, g, grow_at + 64, err) ||
                !c::session_kv_set_mode(ss, g, 1, &pool, pos, nullptr, err) || !st.kv_arena->unmap(slot, err)) {
                std::fprintf(stderr, "%s: switch to 1 at %lld: %s\n", name, (long long) pos, err.c_str());
                return 1;
            }
            std::printf("  %s: streamed at %lld cells, %.1f MiB host, %.1f MiB VRAM arena\n", name, (long long) pos,
                        (double) pool.bytes() / 1048576.0, (double) st.kv_arena->mapped_bytes() / 1048576.0);
            if (st.host_map == nullptr || st.map.slot_block == nullptr) {
                std::fprintf(stderr, "  %s: streamed without a residency map\n", name);
                ++bad;
            }
        }
        // a parked conversation deeper than the pool held: valid for the state (admission grows the pool before the
        // restore), restored only once the pool holds it
        c::ConversationKv deep;
        const int64_t deep_upto = pos == grow_at ? pool.cells() + 1024 : 0;
        bool deep_early = false;
        if (pos == grow_at) {
            std::string verr, rerr;
            if (!st.kv_arena->map(deep_upto + 4096, err) || !c::conversation_kv_save(deep, ref, g, deep_upto, true, err)) {
                std::fprintf(stderr, "%s: deep snapshot: %s\n", name, err.c_str());
                return 1;
            }
            const bool valid = c::conversation_kv_validate(deep, st, g, deep_upto, true, verr);
            deep_early = c::conversation_kv_restore(deep, st, g, deep_upto, true, rerr);
            std::printf("  %s: a %lld-cell snapshot beside a %lld-cell pool: validate %s, restore %s\n", name,
                        (long long) deep_upto, (long long) pool.cells(), valid ? "ok" : verr.c_str(),
                        deep_early ? "ran (wrong)" : rerr.c_str());
            if (!valid || deep_early || rerr.find("host pool") == std::string::npos) ++bad;
        }
        if (pos == grow_at) {   // 1 -> 1: a larger pool, the cells moved over in host memory
            if (!pool2.reserve(ss, g, N + 64, err) || !pool2.copy_from(pool, pos, err) ||
                !c::session_kv_set_mode(ss, g, 1, &pool2, pos, nullptr, err)) {
                std::fprintf(stderr, "%s: pool move at %lld: %s\n", name, (long long) pos, err.c_str());
                return 1;
            }
            pool.release();
            // the pool now holds the deep snapshot (its cells past `pos` are rewritten as the run goes on)
            if (!c::conversation_kv_restore(deep, st, g, deep_upto, true, err)) {
                std::fprintf(stderr, "  %s: deep snapshot restore after the pool grew: %s\n", name, err.c_str());
                ++bad;
            }
        }
        if (pos == back_at) {   // 1 -> 0: uncap, back the depth, copy the cells in
            cn = k::kv_stream_counters(st.map);
            c::session_kv_cap(ss, false);
            if (!st.kv_arena->map(pos + 4096, err) || !c::session_kv_set_mode(ss, g, 0, nullptr, pos, nullptr, err)) {
                std::fprintf(stderr, "%s: switch to 0 at %lld: %s\n", name, (long long) pos, err.c_str());
                return 1;
            }
            // every live cell's VRAM K/V equals the reference's (whole pages, arrays in carve order)
            const k::QsaAttnPools x = c::qsa_attn_pools(st), y = c::qsa_attn_pools(ref);
            const size_t live = (size_t) (pos / s.page_size);
            const size_t rows = live * s.n_head_kv * s.page_size;
            bool eq = true;
            if (fmt == k::kKvQ4) {
                const size_t n = rows * k::kv_q4_bytes_per_head((int) D);
                eq = same_device(x.k_q4, y.k_q4, n) && same_device(x.v_q4, y.v_q4, n);
            } else if (fmt == k::kKvInt8) {
                eq = same_device(x.k_q, y.k_q, rows * D) && same_device(x.v_q, y.v_q, rows * D) &&
                     same_device(x.k_scale, y.k_scale, rows * (D / 64) * 2) && same_device(x.v_scale, y.v_scale, rows * (D / 64) * 2);
            } else {
                eq = same_device(x.k_pool, y.k_pool, rows * D * 2) && same_device(x.v_pool, y.v_pool, rows * D * 2);
            }
            std::printf("  %s: back in VRAM at %lld cells: K/V %s\n", name, (long long) pos, eq ? "identical" : "DIFFERS");
            if (!eq) ++bad;
            if (st.host_map != nullptr || st.map.slot_block != nullptr) {
                std::fprintf(stderr, "  %s: the residency map outlived the switch back\n", name);
                ++bad;
            }
            pool2.release();
        }
        // the arena follows the depth as the engine's kv_fit does (in mode 1 only the pooled keys and rope grow)
        if (st.kv_arena->cells() < pos + 1 && !st.kv_arena->map(pos + 4096, err)) {
            std::fprintf(stderr, "%s: map at %lld: %s\n", name, (long long) pos, err.c_str());
            return 1;
        }
        if (st.kv_mode == 1) {
            // 2. streamed: the K/V arrays hold the slot pool's pages; only the pooled keys and the rope rows follow
            const uint64_t depth = (uint64_t) st.kv_arena->cells();
            const uint64_t bound = (uint64_t) st.host_slots * page_kv + (depth / s.idx_block + 2) * s.idx_dim * 4 +
                                   depth * (s.n_rot / 2) * 4 * 2 + 8 * c::vmm::granularity();
            if (st.kv_arena->mapped_bytes() > bound && bad++ < 5)
                std::fprintf(stderr, "  %s: streamed arena %.1f MiB at %lld cells, more than %.1f\n", name,
                             (double) st.kv_arena->mapped_bytes() / 1048576.0, (long long) pos, (double) bound / 1048576.0);
        }
        for (auto& x : hk) x = nd(rng) * ((pos % 17 == 0) ? 30.f : 1.f);
        for (auto& x : hv) x = nd(rng);
        const int32_t sv[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(step, sv, sizeof(sv), cudaMemcpyHostToDevice), "step");
        append(ref, step, kc, vc, s);
        append(st, step, kc, vc, s);
        if (pos % 211 != 210 && pos != N - 1 && pos != switch_at && pos != back_at) continue;
        const int n_q = 1 + (int) (rng() % NQ);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = std::max<int64_t>(0, pos - (n_q - 1 - t)), n_kv = p + 1;
            const int64_t width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv; hst[t * 4 + 2] = (int32_t) (n_kv / 4);
            hst[t * 4 + 3] = (int32_t) width;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        k::qsa_decode_attn_batch(q, c::qsa_attn_pools(ref), ids, steps, cap, s, scratch, out_ref, n_q, nullptr);
        c::qsa_kv_resolve(st, g, ids, steps, n_q, cap, nullptr);   // a no-op in mode 0
        k::qsa_decode_attn_batch(q, c::qsa_attn_pools(st), ids, steps, cap, s, scratch, out_st, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "batch");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), out_st, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b.data(), (size_t) n_q * NH * D * 4) != 0 && bad++ < 5)
            std::fprintf(stderr, "  %s pos %lld mode %d n_q %d: attention differs\n", name, (long long) pos, st.kv_mode, n_q);
        ++batches;
        // 4. at the first streamed batch: a snapshot in mode 1 equals the mode-0 reference's, and restores
        if (pos == switch_at) {
            c::ConversationKv x, y;
            if (!c::conversation_kv_save(x, st, g, pos + 1, true, err) ||
                !c::conversation_kv_save(y, ref, g, pos + 1, true, err)) {
                std::fprintf(stderr, "%s: snapshot: %s\n", name, err.c_str());
                return 1;
            }
            const bool same = x.k == y.k && x.v == y.v && x.k_scale == y.k_scale && x.v_scale == y.v_scale &&
                              x.pooled == y.pooled && x.format == y.format && x.cells == y.cells;
            const bool back = c::conversation_kv_restore(y, st, g, pos + 1, true, err);
            std::printf("  %s: snapshot mode 1 vs mode 0 %s, restore into mode 1 %s\n", name,
                        same ? "identical" : "DIFFERS", back ? "ok" : err.c_str());
            if (!same || !back) ++bad;
        }
    }
    std::printf("  %s: %d batches, %llu lookups, %llu misses, overflow %d, %d failures\n", name, batches,
                (unsigned long long) cn.lookups, (unsigned long long) cn.misses, (int) cn.overflow, bad);
    if (cn.overflow || cn.misses == 0) ++bad;
    // the state zeroes in either mode (a streamed one only its slot pages)
    std::string ignored;
    if (!pool.reserve(ss, g, 4096, err) || !c::session_kv_set_mode(ss, g, 1, &pool, 0, nullptr, err)) {
        std::fprintf(stderr, "%s: final switch: %s\n", name, err.c_str());
        return 1;
    }
    c::qsa_state_zero(st, g, nullptr);
    ck(cudaDeviceSynchronize(), "zero streamed");
    (void) c::session_kv_set_mode(ss, g, 0, nullptr, 0, nullptr, ignored);
    pool.release();
    c::qsa_state_release(st);
    return bad;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::puts("no CUDA device: skipped"); return 77; }
    if (!c::vmm::supported()) { std::puts("no virtual memory management: skipped"); return 77; }
    std::printf("kv_mode_switch_test: one elastic state, mode 0 -> 1 -> 1 (larger pool) -> 0 -> 1, vs a resident state\n");
    int bad = 0;
    for (int64_t floor : {32768, 8192})
        for (int fmt : {k::kKvInt8, k::kKvF16, k::kKvQ4}) bad += run(fmt, floor);
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
