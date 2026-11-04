// tests/core/prompt_split_state_test.cpp - the prompt-time layer split's state moves, at the model's geometry and for
// every KV format: a layer range's state from a whole session into a prompt session carved for that range
// (session_copy_state through StagedCopy), the range's new cells back, and the range into a whole-model checkpoint
// (conversation_checkpoint_save_layers). Every byte is checked against the source, the rest against what was there.
// With KV streaming the whole session's cells are its host copy.
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {
int g_fail = 0;
#define CHECK(c)                                                                          \
    do {                                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

std::mt19937_64 g_rng(7);

void fill(void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    for (uint8_t& b : h) b = (uint8_t) g_rng();
    cudaMemcpy(d, h.data(), bytes, cudaMemcpyDefault);
}
std::vector<uint8_t> get(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    cudaMemcpy(h.data(), d, bytes, cudaMemcpyDefault);
    return h;
}

// one QSA array: its bytes (a streamed state's cells: its host copy) and its bytes per KV page (0: not paged)
struct Arr { const void* p; uint64_t page, bytes; };

std::vector<Arr> arrays(const QsaState& st, const ModelGeometry& g) {
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    const uint64_t pr = (uint64_t) s.n_head_kv * s.page_size, n = (uint64_t) st.n_pages;
    const uint64_t code = pr * s.head_dim, scale = pr * (s.head_dim / basalt::kernels::KV_Q8_GROUP) * 2;
    const uint64_t q4 = pr * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    const bool h = st.kv_mode == 1;
    std::vector<Arr> v;
    if (st.kv_hybrid) v = {{st.k_q, code, 0}, {st.k_scale, scale, 0}, {st.v_q4, q4, 0}};
    else if (st.kv_q4) v = {{h ? st.host.k_q4 : st.k_q4, q4, 0}, {h ? st.host.v_q4 : st.v_q4, q4, 0}};
    else if (st.kv_int8)
        v = {{h ? st.host.k_q : st.k_q, code, 0}, {h ? st.host.v_q : st.v_q, code, 0},
             {h ? st.host.k_scale : st.k_scale, scale, 0}, {h ? st.host.v_scale : st.v_scale, scale, 0}};
    else v = {{h ? st.host.k_pool : st.k_pool, code * 2, 0}, {h ? st.host.v_pool : st.v_pool, code * 2, 0}};
    for (Arr& a : v) a.bytes = a.page * n;
    const uint64_t row = (uint64_t) g.idx_key_dim * 4;
    v.push_back({st.idx_pooled, row, row * (uint64_t) st.idx_pooled_rows});   // "page": one pooled row
    v.push_back({st.idx_tail, 0, (uint64_t) (s.idx_block - 1) * row});
    v.push_back({st.idx_dead, 0, row});
    v.push_back({st.idx_block_pos, 0, 4});
    return v;
}

void fill_state(const SessionState& s, const ModelGeometry& g, uint64_t gdn_row) {
    fill(s.gdn_state, (size_t) (s.gdn_alloc * (int64_t) gdn_row));
    for (int64_t j = 0; j < s.qsa_alloc; ++j)
        for (const Arr& a : arrays(s.qsa_states[s.qsa_ord0 + j], g)) fill(const_cast<void*>(a.p), a.bytes);
}

// [lo, hi) of the bytes of `got` equal `want`'s, the rest `before`'s
bool range_is(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::vector<uint8_t>& before,
              uint64_t lo, uint64_t hi) {
    for (uint64_t i = 0; i < got.size(); ++i)
        if (got[i] != (i >= lo && i < hi ? want[i] : before[i])) return false;
    return true;
}

void run_format(const char* name, bool int8, bool q4, bool hybrid, int dst_dev, bool streamed = false) {
    qsa_set_kv_int8(int8);
    qsa_set_kv_q4(q4);
    qsa_set_kv_hybrid(hybrid);
    const ModelGeometry g;
    // cell bounds mid-page and mid-block; a streamed session needs more cells than it keeps in VRAM
    const int64_t K = 10, cells = streamed ? qsa_kv_resident_min() + 3000 : 2600, lb = 37, a0 = 1001, c1 = 2302;
    const int64_t I = g.qsa_interval, g0 = lb - lb / I, q0 = lb / I;
    std::string err;
    ConversationStateSizes z;
    CHECK(conversation_state_sizes(g, z, err));
    const uint64_t gdn_row = z.gdn / (uint64_t) g.n_gdn_layers();
    // the whole session on device 0, the prompt session of layers [lb, n_layers) on `dst_dev`
    SessionState ss, ps;
    void* sb = nullptr;
    void* pb = nullptr;
    cudaSetDevice(0);
    qsa_set_kv_resident(streamed ? qsa_kv_resident_min() : 0);
    const uint64_t sbytes = session_bytes(g, cells, K);
    CHECK(cudaMalloc(&sb, sbytes) == cudaSuccess && session_init(g, cells, K, sb, ss) > 0);
    qsa_set_kv_resident(0);
    CHECK(ss.qsa_states[ss.qsa_primary()].kv_mode == (streamed ? 1 : 0));
    cudaSetDevice(dst_dev);
    const uint64_t pbytes = prompt_session_bytes(g, cells, lb, g.n_layers);
    CHECK(cudaMalloc(&pb, pbytes) == cudaSuccess);
    prompt_session_carve(g, cells, K, pb, ps, lb, g.n_layers);
    CHECK(ps.gdn_ord0 == g0 && ps.gdn_alloc == g.n_gdn_layers() - g0 && ps.qsa_ord0 == q0 &&
          ps.qsa_alloc == g.n_qsa_layers() - q0);
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    prompt_session_clear(ps, g, st);
    cudaStreamSynchronize(st);
    cudaSetDevice(0);
    fill_state(ss, g, gdn_row);
    StagedCopy in, back;
    CHECK(in.init(0, dst_dev, 1u << 20, err) && back.init(dst_dev, 0, 3u << 18, err));   // pieces off every array size
    const DeviceCopy copy_in = [&](void* d, const void* s, uint64_t n) { return in.copy(d, s, n, err); };
    const DeviceCopy copy_back = [&](void* d, const void* s, uint64_t n) { return back.copy(d, s, n, err); };
    // ---- in: the range's state and its cells [0, a0)
    CHECK(session_copy_state(ps, ss, g, lb, g.n_layers, 0, a0, copy_in, err) && in.wait(err));
    CHECK(get(ps.gdn_state, ps.gdn_alloc * gdn_row) ==
          get((const uint8_t*) ss.gdn_state + g0 * gdn_row, ps.gdn_alloc * gdn_row));
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    for (int64_t q = q0; q < g.n_qsa_layers(); ++q) {
        const std::vector<Arr> from = arrays(ss.qsa_states[q], g), to = arrays(ps.qsa_states[q], g);
        CHECK(from.size() == to.size());
        for (size_t i = 0; i < from.size() && i < to.size(); ++i) {
            // the pages (or pooled rows) the cells reach, every byte of the unpaged arrays
            uint64_t n = to[i].bytes;
            if (to[i].page > 0)
                n = i + 4 == to.size() ? std::min<uint64_t>(to[i].bytes, (uint64_t) (a0 / s.idx_block + 2) * to[i].page)
                                       : (uint64_t) ((a0 + s.page_size - 1) / s.page_size) * to[i].page;
            CHECK(get(to[i].p, n) == get(from[i].p, n));
        }
    }
    // ---- the stage reads its cells [a0, c1); then they go back, and nothing else of the whole session moves
    std::vector<std::vector<uint8_t>> before;
    cudaSetDevice(dst_dev);
    fill_state(ps, g, gdn_row);
    cudaSetDevice(0);
    for (int64_t q = 0; q < g.n_qsa_layers(); ++q)
        for (const Arr& a : arrays(ss.qsa_states[q], g)) before.push_back(get(a.p, a.bytes));
    const std::vector<uint8_t> gdn_before = get(ss.gdn_state, ss.gdn_alloc * gdn_row), ple_before = get(ss.ple_hist, z.ple);
    CHECK(session_copy_state(ss, ps, g, lb, g.n_layers, a0, c1, copy_back, err) && back.wait(err));
    CHECK(get(ss.ple_hist, z.ple) == ple_before);   // the range does not reach layer 1: the PLE window stays
    std::vector<uint8_t> gdn_want = gdn_before;
    {
        const std::vector<uint8_t> p = get(ps.gdn_state, ps.gdn_alloc * gdn_row);
        std::copy(p.begin(), p.end(), gdn_want.begin() + (std::ptrdiff_t) (g0 * gdn_row));
    }
    CHECK(get(ss.gdn_state, ss.gdn_alloc * gdn_row) == gdn_want);
    size_t k = 0;
    for (int64_t q = 0; q < g.n_qsa_layers(); ++q) {
        const std::vector<Arr> to = arrays(ss.qsa_states[q], g);
        for (size_t i = 0; i < to.size(); ++i, ++k) {
            const std::vector<uint8_t> got = get(to[i].p, to[i].bytes);
            if (q < q0) {
                CHECK(got == before[k]);
                continue;
            }
            const std::vector<uint8_t> want = get(arrays(ps.qsa_states[q], g)[i].p, to[i].bytes);
            uint64_t lo = 0, hi = to[i].bytes;
            if (to[i].page > 0 && i + 4 == to.size()) {   // the pooled rows the cells reach and the spare one after
                lo = (uint64_t) (a0 / s.idx_block) * to[i].page;
                hi = std::min<uint64_t>(to[i].bytes, (uint64_t) (c1 / s.idx_block + 2) * to[i].page);
            } else if (to[i].page > 0) {
                lo = (uint64_t) (a0 / s.page_size) * to[i].page;
                hi = (uint64_t) ((c1 + s.page_size - 1) / s.page_size) * to[i].page;
            }
            CHECK(range_is(got, want, before[k], lo, hi));
        }
    }
    // ---- a whole-model checkpoint with the range's running state from the prompt session
    ConversationCheckpoint c;
    c.ids.assign((size_t) c1, 1);
    CHECK(conversation_checkpoint_save(c, ss, g, err));
    fill_state(ss, g, gdn_row);   // the whole session moves on; the checkpoint keeps the range's state from `ps`
    ConversationCheckpoint w = c;
    CHECK(conversation_checkpoint_save_layers(w, ps, g, lb, g.n_layers, err));
    std::vector<uint8_t> want(c.gdn.begin(), c.gdn.end());
    {
        const std::vector<uint8_t> p = get(ps.gdn_state, ps.gdn_alloc * gdn_row);
        std::copy(p.begin(), p.end(), want.begin() + (std::ptrdiff_t) (g0 * gdn_row));
    }
    CHECK(std::vector<uint8_t>(w.gdn.begin(), w.gdn.end()) == want);
    CHECK(std::vector<uint8_t>(w.ple.begin(), w.ple.end()) == std::vector<uint8_t>(c.ple.begin(), c.ple.end()));
    for (int64_t q = 0; q < g.n_qsa_layers(); ++q) {
        auto slice = [q](const CheckpointBytes& b, uint64_t each) {
            return std::vector<uint8_t>(b.begin() + (std::ptrdiff_t) (q * each), b.begin() + (std::ptrdiff_t) ((q + 1) * each));
        };
        if (q < q0) {
            CHECK(slice(w.tails, z.tail) == slice(c.tails, z.tail) && slice(w.dead, z.dead) == slice(c.dead, z.dead) &&
                  slice(w.block_pos, z.block_pos) == slice(c.block_pos, z.block_pos));
        } else {   // (arrays() ends with the tail, the spare key and the block position)
            const std::vector<Arr> a = arrays(ps.qsa_states[q], g);
            const size_t n = a.size();
            CHECK(slice(w.tails, z.tail) == get(a[n - 3].p, z.tail) && slice(w.dead, z.dead) == get(a[n - 2].p, z.dead) &&
                  slice(w.block_pos, z.block_pos) == get(a[n - 1].p, z.block_pos));
        }
    }
    // a range the session does not hold is refused
    CHECK(!conversation_checkpoint_save_layers(w, ps, g, lb - 4, g.n_layers, err));
    CHECK(!session_copy_state(ss, ps, g, lb - 4, g.n_layers, a0, c1, copy_back, err));
    std::printf("prompt_split_state_test: %s KV%s, tier card CUDA%d: %s\n", name, streamed ? " streamed" : "", dst_dev,
                g_fail ? "FAILED" : "ok");
    cudaSetDevice(dst_dev);
    cudaStreamDestroy(st);
    cudaFree(pb);
    delete[] ps.qsa_states;
    cudaSetDevice(0);
    session_release(ss);
    cudaFree(sb);
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("prompt_split_state_test: no GPU, skipped\n");
        return 77;
    }
    for (int dev : {0, n > 1 ? 1 : 0}) {
        run_format("fp16", false, false, false, dev);
        run_format("int8", true, false, false, dev);
        run_format("q4", false, true, false, dev);
        run_format("hybrid K8V4", false, false, true, dev);
        run_format("fp16", false, false, false, dev, true);
        run_format("int8", true, false, false, dev, true);
        run_format("q4", false, true, false, dev, true);
        if (n < 2) break;
    }
    if (g_fail) { std::printf("prompt_split_state_test: %d failures\n", g_fail); return 1; }
    std::printf("prompt_split_state_test: ok\n");
    return 0;
}
