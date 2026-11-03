// src/dense/stage.cpp - a second card that holds a dense model's last layers and reads them for the main card.  As the
// prompt split it reads a long prompt's chunks through its layers while the main card reads the next chunk's first layers.
//
// Chunk i: the main card reads layers [0, l0) and hands the BF16 residual to the second card through page-locked memory;
// the second card reads layers [l0, n) of chunk i while the main card reads layers [0, l0) of chunk i + 1, and sends its
// attention layers' cells of the chunk back as it goes.  The main card keeps the last chunks: once the second card's
// recurrent state is back it reads them whole, so the session ends as after a read on the main card alone and decoding is
// untouched.  The stage reads layers [lo, n), its cut set at start from the cards' shares (model/multi_gpu.hpp stage_cuts), and
// the main card keeps the last chunk.  The second card runs the main card's kernels on the same inputs (its int8 attention planned as the main
// card's), so its cells and state are the main card's to the bit.  The block drafter's features are made on the main card
// only (a product's rounding may follow the card's SM count): for the chunks whose rows its context ring keeps, the
// second card sends back the rows of its target layers and the main card projects them and makes their cells after the
// last chunk, in order.
// Without GPU-to-GPU access every copy goes card -> page-locked RAM -> card, each on a copy stream beside the reads.
#include "internal.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/log.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace basalt::dense {

// A chunk's way across: the main card's residual to page-locked memory (on m_out) and on to the stage's card (on l_in),
// kSlots chunks in flight; the rows of the block drafter's target layers on the stage's card come back the same way.
class Stage {
public:
    static constexpr int kSlots = 2;             // hand-off buffers in turn
    int dev = -1, main_dev = 0, lo = 0;          // the stage's card, the main card, the first layer the stage holds
    int64_t max_context = 0;
    ModelPtr model;                              // on dev: layers [lo, n) and a chunk's scratch
    Arena arena;                                 // on dev: the landing residuals (and the stage's session)
    uint16_t* lin[kSlots] = {};                  // on dev: a chunk's residual as it lands
    void* main_mem = nullptr;                    // on the main card: the residuals going out, the drafter features kept
    uint16_t* mout[kSlots] = {};
    std::vector<float*> mtail;                   // the main card's features of the ring's chunks, the last chunk's last
    cudaStream_t m_out = nullptr, l_in = nullptr;   // main card: residuals out; stage card: residuals in
    uint8_t* hand[kSlots] = {};                  // page-locked: a chunk's residual (BF16)
    std::vector<uint8_t*> tap_rows;              // page-locked: [tail chunk][target layer on the card] BF16 rows
    int n_taps = 0;                              // the block drafter's target layers at or past lo
    int32_t* pos3 = nullptr;                     // page-locked: two bounds pairs, then [max_context][3] positions
    core::StagedCopy state_in;                   // a read's starting state, main card to dev
    // main card: m_done (chunk's residual copied out on the main stream), handed (it is in RAM)
    // stage card: lin_ready (landed), lin_free (taken by the read), l_done (read), taps_ready
    cudaEvent_t m_done[kSlots] = {}, handed[kSlots] = {};
    cudaEvent_t lin_ready[kSlots] = {}, lin_free[kSlots] = {}, l_done[kSlots] = {}, taps_ready = nullptr;
    std::vector<cudaEvent_t> stage_ev;           // each stage chunk's read on the stage's card, begin and end
    // the prompt split's: its session (every GDN layer's state, the cells of attention layers >= lo), the cells and state going
    // back
    SessionState st;
    cudaStream_t l_out = nullptr, m_back = nullptr;   // stage card: cells and state out; main card: them in
    uint8_t* kv[kSlots] = {};                    // page-locked: a chunk's cells of the stage's attention layers
    uint8_t* state = nullptr;                    // page-locked: the stage's GDN states
    size_t kv_bytes = 0, state_bytes = 0;
    // main card: kv_free, back; stage card: kv_ready, state_ready
    cudaEvent_t kv_free[kSlots] = {}, back = nullptr, kv_ready[kSlots] = {}, state_ready = nullptr;
    ~Stage();

    /// The stage on CUDA device `d` with layers [l, n) of `w` (their weights already on `d`), its chunk scratch, the landing
    /// buffers and the main card's outgoing ones; called with the main card current.
    /// `split`: the stage keeps its layers' cells and runs every window too (its model sized for decode windows of the main
    /// card's context and sessions), and its landing buffers are its own.
    bool open(Model& main, int d, int l, const Weights& w, bool split, std::string& err);
    /// Chunk slot k's residual (m.cs.x after the main card's layers, on `st`'s stream) on its way to the stage's card;
    /// `tail` (null: none) takes the chunk's drafter features too.  `again`: the slot held an earlier chunk.
    void send(Model& m, SessionState& st, int k, bool again, float* tail);
    /// The stage's card reads layers [l0, n) of chunk slot k as it lands, on `ps`'s stream, between stage_ev[2i] and
    /// stage_ev[2i + 1]; with `tail` >= 0 the rows of the drafter's target layers `tap_layers` (those at or past l0) go to
    /// tap_rows[tail].
    void read(SessionState& ps, int i, int k, bool again, int64_t p, const int32_t* p3, const int32_t* bnd, int l0,
              const std::vector<int>& tap_layers, int tail, bool read_last, int T = kChunk, uint8_t* last = nullptr);
    /// The layer split's: its decode hand-off (both models point at it).
    SplitHand split_hand;
    uint8_t* hand_mem = nullptr;                 // page-locked, mapped: split_hand's buffers
};

void StageDeleter::operator()(Stage* p) const { delete p; }

Stage::~Stage() {
    {
        const core::OnDevice on(dev);
        cudaDeviceSynchronize();
        for (cudaEvent_t* set : {lin_ready, lin_free, l_done, kv_ready})
            for (int k = 0; k < kSlots; ++k) if (set[k]) cudaEventDestroy(set[k]);
        for (cudaEvent_t e : {state_ready, taps_ready}) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : stage_ev) if (e) cudaEventDestroy(e);
        for (cudaStream_t s : {st.stream, l_in, l_out}) if (s) cudaStreamDestroy(s);
        model.reset();
        arena.reserve(0);
    }
    {
        const core::OnDevice on(main_dev);
        for (cudaStream_t s : {m_out, m_back}) if (s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); }
        for (cudaEvent_t* set : {m_done, handed, kv_free})
            for (int k = 0; k < kSlots; ++k) if (set[k]) cudaEventDestroy(set[k]);
        if (back) cudaEventDestroy(back);
        if (main_mem) cudaFree(main_mem);
    }
    for (uint8_t* p : hand) if (p) cudaFreeHost(p);
    for (uint8_t* p : kv) if (p) cudaFreeHost(p);
    for (uint8_t* p : tap_rows) if (p) cudaFreeHost(p);
    if (state) cudaFreeHost(state);
    if (pos3) cudaFreeHost(pos3);
    if (hand_mem) cudaFreeHost(hand_mem);
}

void Stage::send(Model& m, SessionState& st, int k, bool again, float* tail) {
    const size_t bytes = (size_t) kChunk * m.g.n_embd * 2;
    const cudaStream_t ms = st.stream;
    if (again) cudaStreamWaitEvent(ms, handed[k], 0);
    cudaMemcpyAsync(mout[k], m.cs.x, bytes, cudaMemcpyDeviceToDevice, ms);
    if (tail) cudaMemcpyAsync(tail, m.cs.dacc, bytes * 2, cudaMemcpyDeviceToDevice, ms);
    cudaEventRecord(m_done[k], ms);
    cudaStreamWaitEvent(m_out, m_done[k], 0);
    if (again) cudaStreamWaitEvent(m_out, lin_ready[k], 0);
    cudaMemcpyAsync(hand[k], mout[k], bytes, cudaMemcpyDeviceToHost, m_out);
    cudaEventRecord(handed[k], m_out);
}

void Stage::read(SessionState& ps, int i, int k, bool again, int64_t p, const int32_t* p3, const int32_t* bnd, int l0,
                 const std::vector<int>& tap_layers, int tail, bool read_last, int T, uint8_t* last) {
    const core::OnDevice on(dev);
    Model& lm = *model;
    const int n = lm.g.n_layers;
    const size_t bytes = (size_t) T * lm.g.n_embd * 2;
    const cudaStream_t ls = ps.stream;
    // lands beside the read of the chunk before
    cudaStreamWaitEvent(l_in, handed[k], 0);
    if (again) cudaStreamWaitEvent(l_in, lin_free[k], 0);
    cudaMemcpyAsync(lin[k], hand[k], bytes, cudaMemcpyHostToDevice, l_in);
    cudaEventRecord(lin_ready[k], l_in);
    // the read of its last layers; the rows of the drafter's target layers among them out for the ring's chunks
    cudaStreamWaitEvent(ls, lin_ready[k], 0);
    cudaMemcpyAsync(lm.cs.x, lin[k], bytes, cudaMemcpyDeviceToDevice, ls);
    cudaEventRecord(lin_free[k], ls);
    chunk_start(lm, ps, nullptr, nullptr, T, p3, bnd);
    cudaEventRecord(stage_ev[2 * (size_t) i], ls);
    int at = l0;
    if (tail >= 0) {
        for (size_t q = 0; q < tap_layers.size(); ++q) {
            const int tl = tap_layers[q];
            chunk_layers(lm, ps, T, p, at, tl + 1, nullptr, read_last);
            cudaMemcpyAsync(tap_rows[(size_t) tail * (size_t) n_taps + q], lm.cs.x, bytes, cudaMemcpyDeviceToHost, ls);
            at = tl + 1;
        }
        cudaEventRecord(taps_ready, ls);
    }
    chunk_layers(lm, ps, T, p, at, n, nullptr, read_last);
    if (last) cudaMemcpyAsync(last, lm.cs.x, (size_t) T * lm.g.n_embd * 2, cudaMemcpyDeviceToHost, ls);
    cudaEventRecord(stage_ev[2 * (size_t) i + 1], ls);
    cudaEventRecord(l_done[k], ls);
}

namespace {

// a layer's cells, K and V: fp16 rows, or int8 codes with an fp16 scale per 64 values
size_t cell_bytes(const Geometry& g, bool int8) {
    const size_t row = (size_t) g.n_kv * g.head_dim;
    return int8 ? 2 * (row + row / 64 * 2) : 2 * row * 2;
}

// the attention layers and GDN layers before layer l
int count_kind(const Geometry& g, int l, Mixer k) { return (int) std::count(g.kinds.begin(), g.kinds.begin() + l, k); }

constexpr int kAhead = 2;                   // chunks the host queues ahead of the stage's finished ones: a stop waits for at most these
constexpr size_t kStatePiece = 8u << 20;   // the pieces a read's starting state crosses over in

// page-locked host memory (portable: both cards copy to and from it)
bool pin(uint8_t*& at, size_t bytes) {
    void* v = nullptr;
    if (bytes > 0 && cudaHostAlloc(&v, bytes, cudaHostAllocPortable) != cudaSuccess) return false;
    at = (uint8_t*) v;
    return true;
}

struct Region {
    void* main;
    void* stage;
    size_t bytes;
};
// cells [c0, c0 + n) of attention layers ai0.. of both sessions, as copy regions
std::vector<Region> cell_regions(const Model& m, const SessionState& main, const SessionState& stage, int ai0, int64_t c0, int64_t n) {
    std::vector<Region> r;
    const size_t row = (size_t) m.g.n_kv * m.g.head_dim;
    for (int ai = ai0; ai < m.g.attn_layers(); ++ai) {
        if (m.o.kv_int8) {
            const kernels::KvInt8 &a = main.kq[(size_t) ai], &b = stage.kq[(size_t) ai];
            r.push_back({a.k + c0 * row, b.k + c0 * row, (size_t) n * row});
            r.push_back({a.v + c0 * row, b.v + c0 * row, (size_t) n * row});
            r.push_back({a.ks + c0 * row / 64, b.ks + c0 * row / 64, (size_t) n * row / 64 * 2});
            r.push_back({a.vs + c0 * row / 64, b.vs + c0 * row / 64, (size_t) n * row / 64 * 2});
        } else {
            r.push_back({main.kc[(size_t) ai] + c0 * row, stage.kc[(size_t) ai] + c0 * row, (size_t) n * row * 2});
            r.push_back({main.vc[(size_t) ai] + c0 * row, stage.vc[(size_t) ai] + c0 * row, (size_t) n * row * 2});
        }
    }
    return r;
}

// the stage's chunks whose drafter features the context ring still holds at the end of a read: at most this many
int ring_tails(const BlockDrafter* bd) {
    if (!bd || !bd->present) return 0;
    const int64_t cap = (int64_t) bd->window + 256;   // a session's ring (SessionState::dctx_cap)
    return (int) ((cap + kChunk - 1) / kChunk) + 1;
}

// the cells of attention layers ai0.. for `cells` positions and of the layers before for `low` (0: none), from `a`
bool take_cells(Arena& a, SessionState& s, const Geometry& g, bool int8, int ai0, int64_t cells, int64_t low = 0) {
    for (int ai = 0; ai < g.attn_layers(); ++ai) {
        kernels::KvInt8 q8;
        uint16_t *k = nullptr, *v = nullptr;
        const size_t n = (size_t) (ai >= ai0 ? cells : low) * g.n_kv * g.head_dim;
        if (n > 0) {
            if (int8) {
                q8.k = a.take<int8_t>(n);
                q8.v = a.take<int8_t>(n);
                q8.ks = a.take<uint16_t>(n / 64);
                q8.vs = a.take<uint16_t>(n / 64);
                if (!q8.vs) return false;
            } else {
                k = a.take<uint16_t>(n);
                v = a.take<uint16_t>(n);
                if (!v) return false;
            }
        }
        s.kc.push_back(k);
        s.vc.push_back(v);
        s.kq.push_back(q8);
        s.far_kv.push_back({});   // all of its cells are on the card
    }
    return true;
}

}  // namespace

bool Stage::open(Model& main, int d, int l, const Weights& w, bool split, std::string& err) {
    cudaGetDevice(&main_dev);
    dev = d;
    lo = l;
    max_context = main.o.max_context;
    const Geometry& g = main.g;
    const size_t T = kChunk, H = (size_t) g.n_embd;
    const bool block = main.o.drafter == Drafter::Block;
    const BlockDrafter& bd = main.w.dflash;
    if (block) n_taps = (int) std::count_if(bd.target_layers.begin(), bd.target_layers.end(), [&](int t) { return t >= l; });
    const int n_tail = ring_tails(block ? &bd : nullptr);
    {
        const core::OnDevice on(dev);
        Options o = main.o;
        if (split) {
            o.layer_begin = l;
            o.layer_end = 0;
        } else {
            o.max_context = kChunk;   // its decode scratch is never used: the stage reads prompt chunks only
            o.sessions = 1;
            o.spec = 0;
        }
        o.draft = false;
        o.drafter = Drafter::None;
        o.vision = false;
        o.mmproj = {};
        Weights lw = w;
        lw.head = main.w.head;   // (the main card's: never read here; its type keeps MMQ from loading a kernel for it)
        lw.mtp = {};
        lw.dflash = {};
        model = make_model(g, lw, o, err);
        if (!model) return false;
        if (!model->ensure_chunk(err)) return false;
        if (split) {   // the landing residuals
            if (!arena.reserve(kSlots * T * H * 2 + 4096)) {
                err = "out of device memory for its landing buffers";
                return false;
            }
            for (uint16_t*& b : lin) b = arena.take<uint16_t>(T * H);
        }
        bool ok = true;
        for (cudaStream_t* s : {&l_in}) ok = ok && cudaStreamCreateWithFlags(s, cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* set : {lin_ready, lin_free, l_done})
            for (int k = 0; k < kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&set[k], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&taps_ready, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            err = "cannot create its streams and events";
            return false;
        }
    }
    {
        const core::OnDevice on(main_dev);
        const size_t bytes = stage_main_bytes(g, block ? bd : BlockDrafter{});
        if (cudaMalloc(&main_mem, bytes) != cudaSuccess) {
            err = "out of device memory for its buffers on the main card";
            return false;
        }
        uint8_t* at = (uint8_t*) main_mem;
        for (uint16_t*& b : mout) { b = (uint16_t*) at; at += T * H * 2; }
        for (int j = 0; j < (n_tail ? n_tail + 1 : 0); ++j) { mtail.push_back((float*) at); at += T * H * 4; }
        bool ok = cudaStreamCreateWithFlags(&m_out, cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* set : {m_done, handed})
            for (int k = 0; k < kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&set[k], cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            err = "cannot create its streams and events";
            return false;
        }
    }
    uint8_t* positions = nullptr;
    bool ok = pin(positions, ((size_t) max_context * 3 + 4) * sizeof(int32_t));
    pos3 = (int32_t*) positions;
    for (int k = 0; k < kSlots; ++k) ok = ok && pin(hand[k], T * H * 2);
    tap_rows.assign((size_t) n_tail * n_taps, nullptr);
    for (uint8_t*& t : tap_rows) ok = ok && pin(t, T * H * 2);
    if (!ok) {
        err = "out of page-locked memory for its buffers";
        return false;
    }
    if (!state_in.init(main_dev, dev, kStatePiece, err)) return false;
    {   // a chunk's events
        const core::OnDevice on(dev);
        for (int j = 0; j < 2; ++j) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) {
                err = "cannot create CUDA events";
                return false;
            }
            stage_ev.push_back(e);
        }
    }
    // the second card's int8 prompt attention splits its keys as the main card's does, so its rows round the same
    kernels::gqa_prompt_plan_device(main_dev);
    return true;
}

StagePtr make_prompt_split(Model& main, int dev, int l_min, const Weights& w, std::string& err) {
    StagePtr p(new Stage());
    Stage& L = *p;
    const Geometry& g = main.g;
    const size_t T = kChunk, H = (size_t) g.n_embd;
    const auto fail = [&](const std::string& what) {
        err = "CUDA" + std::to_string(dev) + ": " + what;
        return nullptr;
    };
    if (!L.open(main, dev, l_min, w, false, err)) return fail(err);
    {   // its session: every GDN layer's state, the cells of its attention layers, and its streams for the cells going back
        const core::OnDevice on(dev);
        SessionState& s = L.st;
        const size_t C = (size_t) g.conv_channels(), HV = (size_t) g.gdn_v_heads;
        s.gdn_floats = (size_t) g.gdn_state * HV * g.gdn_state + C * 3;
        const int ai0 = count_kind(g, l_min, Mixer::Attention);
        const size_t need = (size_t) g.gdn_layers() * s.gdn_floats * 4 +
                            (size_t) (g.attn_layers() - ai0) * (size_t) L.max_context * cell_bytes(g, main.o.kv_int8) +
                            Stage::kSlots * T * H * 2 + (size_t) g.attn_layers() * 4 * 256 + 4096;
        if (!L.arena.reserve(need)) return fail(std::to_string((unsigned long long) (need >> 20)) + " MiB of cells do not fit");
        s.gdn = s.gdn_first = L.arena.take<float>((size_t) g.gdn_layers() * s.gdn_floats);
        if (!take_cells(L.arena, s, g, main.o.kv_int8, ai0, L.max_context)) return fail("its cells do not fit their arena");
        for (uint16_t*& b : L.lin) b = L.arena.take<uint16_t>(T * H);
        if (!L.lin[Stage::kSlots - 1]) return fail("its buffers do not fit their arena");
        bool ok = true;
        for (cudaStream_t* st : {&s.stream, &L.l_out}) ok = ok && cudaStreamCreateWithFlags(st, cudaStreamNonBlocking) == cudaSuccess;
        for (int k = 0; k < Stage::kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&L.kv_ready[k], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&L.state_ready, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) return fail("cannot create its streams and events");
    }
    {
        const core::OnDevice on(L.main_dev);
        bool ok = cudaStreamCreateWithFlags(&L.m_back, cudaStreamNonBlocking) == cudaSuccess;
        for (int k = 0; k < Stage::kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&L.kv_free[k], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&L.back, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) return fail("cannot create its streams and events");
    }
    L.kv_bytes = (size_t) (g.attn_layers() - count_kind(g, l_min, Mixer::Attention)) * T * cell_bytes(g, main.o.kv_int8);
    L.state_bytes = (size_t) (g.gdn_layers() - count_kind(g, l_min, Mixer::Gdn)) * L.st.gdn_floats * 4;
    bool ok = pin(L.state, L.state_bytes);
    for (int k = 0; k < Stage::kSlots; ++k) ok = ok && pin(L.kv[k], L.kv_bytes);
    if (!ok) {
        err = "out of page-locked memory for its hand-off";
        return nullptr;
    }
    const size_t pinned = Stage::kSlots * (T * H * 2 + L.kv_bytes) + L.tap_rows.size() * T * H * 2 + L.state_bytes +
                          core::StagedCopy::kBufs * kStatePiece;
    basalt::log::info("basalt dense", "prompt split: CUDA%d reads layers %d-%d of prompts over %d tokens; %.2f GiB of page-locked "
                      "hand-off buffers", dev, l_min, g.n_layers - 1, 2 * kChunk, (double) pinned / 1073741824.0);
    return p;
}

uint64_t stage_main_bytes(const Geometry& g, const BlockDrafter& drafter) {
    const int n_tail = ring_tails(&drafter);   // and one more: the main card's first chunk's, kept aside
    const uint64_t T = kChunk, H = (uint64_t) g.n_embd;
    return Stage::kSlots * T * H * 2 + (uint64_t) (n_tail ? n_tail + 1 : 0) * T * H * 4;
}

uint64_t prompt_split_main_bytes(const Geometry& g, const BlockDrafter& drafter, int64_t max_context) {
    return max_context <= 2 * (int64_t) kChunk ? 0 : stage_main_bytes(g, drafter);
}

bool plan_prompt_split(Model& m, int dev, int l_min, const Weights& w, std::string& err) {
    if (m.o.max_context <= 2 * (int64_t) kChunk) {
        basalt::log::info("basalt dense", "prompt split: off (a context of %lld holds no prompt over %d tokens)", (long long) m.o.max_context,
                          2 * kChunk);
        return true;
    }
    m.prompt_split = make_prompt_split(m, dev, l_min, w, err);
    return m.prompt_split != nullptr;
}

bool warm_prompt_split(Model& m, std::string& err) {
    if (!m.prompt_split) return true;
    Stage& L = *m.prompt_split;
    Model& lm = *L.model;
    const Geometry& g = m.g;
    const size_t T = kChunk, H = (size_t) g.n_embd;
    const core::OnDevice on(L.dev);
    std::memset(L.pos3, 0, ((size_t) kChunk * 3 + 4) * sizeof(int32_t));
    L.pos3[1] = kChunk;
    const cudaStream_t ls = L.st.stream;
    const int gi0 = count_kind(g, L.lo, Mixer::Gdn);
    cudaMemsetAsync(L.lin[0], 0, T * H * 2, ls);
    cudaMemcpyAsync(lm.cs.x, L.lin[0], T * H * 2, cudaMemcpyDeviceToDevice, ls);
    chunk_start(lm, L.st, nullptr, nullptr, kChunk, L.pos3 + 4, L.pos3);
    chunk_layers(lm, L.st, kChunk, 0, L.lo, g.n_layers, nullptr);
    cudaMemsetAsync(L.st.gdn + (size_t) gi0 * L.st.gdn_floats, 0, (size_t) (g.gdn_layers() - gi0) * L.st.gdn_floats * 4, ls);
    if (cudaStreamSynchronize(ls) != cudaSuccess) {
        err = "CUDA" + std::to_string(L.dev) + ": warming the prompt layers failed: " + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool prompt_split_read(Stage& L, Model& m, SessionState& st, const int32_t* tokens, int n, int64_t pos,
                      const std::function<void(int)>& progress, const std::function<bool()>& stop, std::string& err) {
    const Geometry& g = m.g;
    const int T = kChunk, H = g.n_embd;
    constexpr int S = Stage::kSlots;
    if (n <= 2 * T || st.mtp_kc != nullptr || pos + n > L.max_context) return false;
    const int l0 = L.lo;
    if (!m.ensure_chunk(err)) return false;
    const bool block = m.o.drafter == Drafter::Block;
    const BlockDrafter& bd = m.w.dflash;
    const BlockDrafter* mtaps = block ? &bd : nullptr;
    // the drafter's target layers on the card (their rows come back for the ring's chunks), and whether it reads the last
    std::vector<int> ltaps;
    for (size_t j = 0; block && j < bd.target_layers.size(); ++j)
        if (bd.target_layers[j] >= l0) ltaps.push_back((int) j);
    const bool read_last = block && !ltaps.empty() && bd.target_layers[(size_t) ltaps.back()] == g.n_layers - 1;
    std::vector<int> tap_layers;
    for (int j : ltaps) tap_layers.push_back(bd.target_layers[(size_t) j]);
    const int64_t cap = block ? st.dctx_cap : 0, b0 = pos + n;
    const int chunks = (n + T - 1) / T;
    const int t_last = n - (chunks - 1) * T;
    // The chunks the stage reads: the main card keeps the last one, and with it the one before when the last is a part of a chunk, so
    // that the first layers of a whole chunk fill the time the stage takes to finish its last chunk.
    const int lc = t_last == T ? chunks - 1 : chunks - 2;
    const int ai0 = count_kind(g, l0, Mixer::Attention), gi0 = count_kind(g, l0, Mixer::Gdn);
    const size_t gdn_off = (size_t) gi0 * st.gdn_floats, gdn_bytes = (size_t) (g.gdn_layers() - gi0) * st.gdn_floats * 4;
    // the bounds of a whole chunk and of the last one, then the positions, page-locked so that every upload is queued
    int32_t* bnd = L.pos3;
    int32_t* pos3 = L.pos3 + 4;
    bnd[0] = 0;
    bnd[1] = T;
    bnd[2] = 0;
    bnd[3] = t_last;
    for (int t = 0; t < n; ++t) pos3[t * 3] = pos3[t * 3 + 1] = pos3[t * 3 + 2] = (int32_t) (pos + t + st.rope_delta);
    const cudaStream_t ms = st.stream, ls = L.st.stream;
    if (cudaStreamSynchronize(ms) != cudaSuccess) {
        err = m.o.arch + ": " + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    {   // the stage's layers start from the session's state: empty at the start, else copied from the main card
        const core::OnDevice on(L.dev);
        if (pos == 0) {
            cudaMemsetAsync(L.st.gdn + gdn_off, 0, gdn_bytes, ls);
        } else {
            bool ok = L.state_in.copy(L.st.gdn + gdn_off, st.gdn + gdn_off, gdn_bytes, err);
            for (const Region& r : cell_regions(m, st, L.st, ai0, 0, pos)) ok = ok && L.state_in.copy(r.stage, r.main, r.bytes, err);
            if (!ok || !L.state_in.wait(err)) return false;
        }
    }
    {
        const core::OnDevice on(L.dev);
        while (L.stage_ev.size() < 2 * (size_t) lc) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) {
                err = "cannot create CUDA events";
                return false;
            }
            L.stage_ev.push_back(e);
        }
    }
    std::vector<int64_t> tail_pos;   // the first cells of the stage's chunks the ring keeps rows of
    // Chunks are queued no more than kAhead past the stage's finished ones; each finished one is reported, and `stop` asked.
    cudaError_t e = cudaSuccess;
    bool stopped = false;
    int reaped = 0;   // stage chunks seen finished
    const auto reap = [&](int upto) {
        for (; reaped < upto && e == cudaSuccess; ++reaped) {
            {
                const core::OnDevice on(L.dev);
                e = cudaEventSynchronize(L.stage_ev[2 * (size_t) reaped + 1]);
            }
            if (e == cudaSuccess && progress) progress((reaped + 1) * T);
        }
    };
    for (int i = 0; i < lc; ++i) {
        if (i >= kAhead) {
            reap(i - kAhead + 1);
            if (e != cudaSuccess) break;
            if (stop && stop()) {
                stopped = true;
                break;
            }
        }
        const int64_t p = pos + (int64_t) i * T;
        const int k = i % S;
        const bool again = i >= S;   // the slot's buffers held chunk i - S
        const bool tail = block && p + T > b0 - cap && tail_pos.size() + 1 < L.mtail.size();
        const int32_t* p3 = pos3 + (size_t) i * T * 3;
        // main card: the chunk's first layers, its residual out to page-locked memory beside the next chunk
        chunk_start(m, st, tokens + (size_t) i * T, nullptr, T, p3, bnd);
        chunk_layers(m, st, T, p, 0, l0, mtaps);
        L.send(m, st, k, again, tail ? L.mtail[tail_pos.size()] : nullptr);
        // the stage reads its last layers beside the main card's next chunk
        L.read(L.st, i, k, again, p, p3, bnd, l0, tap_layers, tail ? (int) tail_pos.size() : -1, read_last);
        std::vector<Region> cells = cell_regions(m, st, L.st, ai0, p, T);
        {   // its cells out beside the next chunk's read
            const core::OnDevice on(L.dev);
            cudaStreamWaitEvent(L.l_out, L.l_done[k], 0);
            if (again) cudaStreamWaitEvent(L.l_out, L.kv_free[k], 0);
            size_t off = 0;
            for (const Region& r : cells) {
                cudaMemcpyAsync(L.kv[k] + off, r.stage, r.bytes, cudaMemcpyDeviceToHost, L.l_out);
                off += r.bytes;
            }
            cudaEventRecord(L.kv_ready[k], L.l_out);
        }
        cudaStreamWaitEvent(L.m_back, L.kv_ready[k], 0);
        size_t off = 0;
        for (const Region& r : cells) {
            cudaMemcpyAsync(r.main, L.kv[k] + off, r.bytes, cudaMemcpyHostToDevice, L.m_back);
            off += r.bytes;
        }
        cudaEventRecord(L.kv_free[k], L.m_back);
        if (tail) tail_pos.push_back(p);
    }
    if (stopped || e != cudaSuccess) {   // what is queued runs out, so that no copy is left in flight: the session holds part of the prompt
        for (int d : {L.main_dev, L.dev}) {
            const core::OnDevice on(d);
            const cudaError_t s = cudaDeviceSynchronize();
            if (e == cudaSuccess) e = s;
        }
        if (e != cudaSuccess) {
            err = m.o.arch + ": the prompt split failed: " + cudaGetErrorString(e);
            return false;
        }
        basalt::log::info("basalt dense", "prompt split: stopped, %d of %d tokens read", reaped * T, n);
        return true;
    }
    {   // the stage's recurrent state after its last chunk goes back
        const core::OnDevice on(L.dev);
        cudaStreamWaitEvent(L.l_out, L.l_done[(lc - 1) % S], 0);
        cudaMemcpyAsync(L.state, L.st.gdn + gdn_off, gdn_bytes, cudaMemcpyDeviceToHost, L.l_out);
        cudaEventRecord(L.state_ready, L.l_out);
    }
    cudaStreamWaitEvent(L.m_back, L.state_ready, 0);
    cudaMemcpyAsync(st.gdn + gdn_off, L.state, gdn_bytes, cudaMemcpyHostToDevice, L.m_back);
    cudaEventRecord(L.back, L.m_back);
    // the main card's chunks: the first one's first layers beside the stage's last chunk, the rest once the state is back,
    // then the others whole
    const int64_t pf = pos + (int64_t) lc * T;
    const int tf = lc == chunks - 1 ? t_last : T;
    const int32_t* bf = tf == T ? bnd : bnd + 2;
    chunk_start(m, st, tokens + (size_t) lc * T, nullptr, tf, pos3 + (size_t) lc * T * 3, bf);
    chunk_layers(m, st, tf, pf, 0, l0, mtaps);
    cudaStreamWaitEvent(ms, L.back, 0);
    chunk_layers(m, st, tf, pf, l0, g.n_layers, mtaps);
    // the drafter's features of the stage's chunks its ring keeps: the main card's own taps, then the projections of the
    // rows of the target layers the card read, as a read on the main card makes them; their cells in order, before the
    // main card's chunks' (chunk_finish), so every slot of the ring ends as a read on the main card leaves it
    if (!tail_pos.empty()) {
        float* kept = L.mtail.back();
        cudaMemcpyAsync(kept, m.cs.dacc, (size_t) tf * H * 4, cudaMemcpyDeviceToDevice, ms);
        cudaStreamWaitEvent(ms, L.taps_ready, 0);
        cudaMemcpyAsync(m.cs.bounds, bnd, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, ms);   // whole chunks' rows
        for (size_t j = 0; j < tail_pos.size(); ++j) {
            cudaMemcpyAsync(m.cs.dacc, L.mtail[j], (size_t) T * H * 4, cudaMemcpyDeviceToDevice, ms);
            for (size_t q = 0; q < ltaps.size(); ++q) {
                cudaMemcpyAsync(m.cs.x, L.tap_rows[j * (size_t) L.n_taps + q], (size_t) T * H * 2, cudaMemcpyHostToDevice, ms);
                chunk_tap(m, st, T, bd, (size_t) ltaps[q]);
            }
            block_ctx_chunk(m, st, m.cs.dacc, T, tail_pos[j]);
        }
        cudaMemcpyAsync(m.cs.dacc, kept, (size_t) tf * H * 4, cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(m.cs.bounds, bf, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, ms);
    }
    if (!chunk_finish(m, st, tf, pf, err)) return false;
    // the stage's last chunks' reports, then the main card's whole chunks, one at a time: a stop is met between them
    reap(lc);
    for (int c = lc; c < chunks && e == cudaSuccess; ++c) {
        const int tc = c == chunks - 1 ? t_last : T;
        if (c > lc) {
            const int64_t pc = pos + (int64_t) c * T;
            chunk_start(m, st, tokens + (size_t) c * T, nullptr, tc, pos3 + (size_t) c * T * 3, tc == T ? bnd : bnd + 2);
            chunk_layers(m, st, tc, pc, 0, g.n_layers, mtaps);
            if (!chunk_finish(m, st, tc, pc, err)) return false;
        }
        e = cudaStreamSynchronize(ms);
        if (e != cudaSuccess) break;
        if (progress) progress(c * T + tc);
        if (c + 1 < chunks && stop && stop()) {
            stopped = true;
            break;
        }
    }
    if (e == cudaSuccess) {
        const core::OnDevice on(L.dev);
        e = cudaDeviceSynchronize();
    }
    if (e != cudaSuccess) {
        err = m.o.arch + ": the prompt split failed: " + cudaGetErrorString(e);
        return false;
    }
    if (stopped) {
        basalt::log::info("basalt dense", "prompt split: stopped after %d of %d tokens", (int) std::min<int64_t>(n, (int64_t) (reaped + 1) * T), n);
        return true;
    }
    if (progress) progress(n);
    basalt::log::info("basalt dense", "prompt split: %d tokens, layers %d-%d of chunks 1-%d on CUDA%d", n, l0, g.n_layers - 1, lc,
                      L.dev);
    return true;
}

void PeerStateDeleter::operator()(SessionState* s) const {
    const core::OnDevice on(s->dev);
    free_state(*s);
    delete s;
}

bool attach_split(Model& m, int dev, int cut, const Weights& w, std::string& err) {
    const Geometry& g = m.g;
    const size_t H = (size_t) g.n_embd, slot = (size_t) kBatch * H * 2;
    int main_dev = 0;
    cudaGetDevice(&main_dev);
    // the second card's window products plan as the main card's (before its model sizes their workspace), so its rows round
    // as the main card's
    kernels::wgemm::plan_device(main_dev);
    StagePtr p(new Stage());
    Stage& L = *p;
    const auto fail = [&](const std::string& what) {
        err = m.o.arch + ": the layer split on CUDA" + std::to_string(dev) + ": " + what;
        return false;
    };
    if (!L.open(m, dev, cut, w, true, err)) return fail(err);
    SplitHand& h = L.split_hand;
    const BlockDrafter& bd = m.w.dflash;
    if (m.o.drafter == Drafter::Block)
        for (size_t j = 0; j < bd.target_layers.size(); ++j)
            if (bd.target_layers[j] >= cut) {
                if (h.taps.empty()) h.first_tap = j;
                h.taps.push_back(bd.target_layers[j]);
            }
    const size_t bytes = slot * (2 + h.taps.size()) + 256;
    void* dptr = nullptr;
    if (cudaHostAlloc((void**) &L.hand_mem, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&dptr, L.hand_mem, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        L.hand_mem = nullptr;
        return fail("out of page-locked memory for its hand-off");
    }
    std::memset(L.hand_mem, 0, bytes);
    uint8_t* at = (uint8_t*) dptr;
    h.flag = (uint32_t*) at;
    h.fwd = (uint16_t*) (at + 256);
    h.ret = (uint16_t*) (at + 256 + slot);
    m.hand = &h;
    L.model->hand = &h;
    m.split_model = L.model.get();
    m.split_dev = dev;
    basalt::log::info("basalt dense", "layer split: CUDA%d runs layers %d-%d of every window and prompt chunk; its products planned as "
                      "CUDA%d's", dev, cut, g.n_layers - 1, main_dev);
    m.split = std::move(p);
    return true;
}

bool split_session(Model& m, SessionState& st, std::string& err) {
    st.peer.reset(new SessionState());
    st.peer->dev = m.split_dev;
    const core::OnDevice on(m.split_dev);
    return init_state(*m.split_model, *st.peer, err);
}

bool split_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos, const int32_t* pos3,
                 std::string& err) {
    Stage& L = *m.split;
    const Geometry& g = m.g;
    if (!m.ensure_chunk(err)) return false;
    const bool block = m.o.drafter == Drafter::Block, mtp = st.mtp_kc != nullptr;
    const BlockDrafter& bd = m.w.dflash;
    const SplitHand& h = L.split_hand;
    const size_t bytes = (size_t) T * g.n_embd * 2;
    // the bounds and positions page-locked (the stage's uploads are queued)
    int32_t* bnd = L.pos3;
    bnd[0] = 0;
    bnd[1] = T;
    std::memcpy(L.pos3 + 4, pos3, (size_t) T * 3 * sizeof(int32_t));
    const cudaStream_t ms = st.stream;
    // the main card's layers, the residual across, the second card's layers; the rows of the drafter's target layers there (and
    // with an MTP drafter the last layer's) come back
    chunk_start(m, st, tokens, rows, T, L.pos3 + 4, bnd);
    chunk_layers(m, st, T, pos, 0, L.lo, block ? &bd : nullptr);
    L.send(m, st, 0, false, nullptr);
    const bool taps = block && !h.taps.empty();
    const bool read_last = mtp || (taps && h.taps.back() == g.n_layers - 1);
    L.read(*st.peer, 0, 0, false, pos, L.pos3 + 4, bnd, L.lo, h.taps, taps ? 0 : -1, read_last, T, mtp ? L.hand[1] : nullptr);
    cudaStreamWaitEvent(ms, L.l_done[0], 0);
    for (size_t q = 0; taps && q < h.taps.size(); ++q) {   // their features after the main card's, in order
        cudaMemcpyAsync(m.cs.x, L.tap_rows[q], bytes, cudaMemcpyHostToDevice, ms);
        chunk_tap(m, st, T, bd, h.first_tap + q);
    }
    if (mtp) cudaMemcpyAsync(m.cs.x, L.hand[1], bytes, cudaMemcpyHostToDevice, ms);
    return chunk_finish(m, st, T, pos, err);
}

bool split_read(Model& m, SessionState& st, const int32_t* tokens, int n, int64_t pos, const std::function<void(int)>& progress,
                const std::function<bool()>& stop, std::string& err) {
    Stage& L = *m.split;
    const Geometry& g = m.g;
    const int T = kChunk, H = g.n_embd;
    constexpr int S = Stage::kSlots;
    if (!m.ensure_chunk(err)) return false;
    const bool block = m.o.drafter == Drafter::Block;
    const BlockDrafter& bd = m.w.dflash;
    const SplitHand& h = L.split_hand;
    const bool taps = block && !h.taps.empty();
    const bool read_last = taps && h.taps.back() == g.n_layers - 1;
    const int64_t cap = block ? st.dctx_cap : 0, b0 = pos + n;
    const int chunks = (n + T - 1) / T;
    const int t_last = n - (chunks - 1) * T;
    // the bounds of a whole chunk and of the last one, then the positions, page-locked so that every upload is queued
    int32_t* bnd = L.pos3;
    int32_t* pos3 = L.pos3 + 4;
    bnd[0] = 0;
    bnd[1] = T;
    bnd[2] = 0;
    bnd[3] = t_last;
    for (int t = 0; t < n; ++t) pos3[t * 3] = pos3[t * 3 + 1] = pos3[t * 3 + 2] = (int32_t) (pos + t + st.rope_delta);
    const cudaStream_t ms = st.stream;
    {
        const core::OnDevice on(L.dev);
        while (L.stage_ev.size() < 2 * (size_t) chunks) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) {
                err = "cannot create CUDA events";
                return false;
            }
            L.stage_ev.push_back(e);
        }
    }
    // the chunks whose drafter features the context ring keeps: the main card's part kept aside, the second card's rows back
    std::vector<int64_t> tail_pos;
    std::vector<int> tail_rows;
    const size_t n_tail = L.n_taps ? L.tap_rows.size() / (size_t) L.n_taps : 0;
    cudaError_t e = cudaSuccess;
    bool stopped = false;
    int reaped = 0;   // chunks seen finished on the second card
    const auto reap = [&](int upto) {
        for (; reaped < upto && e == cudaSuccess; ++reaped) {
            {
                const core::OnDevice on(L.dev);
                e = cudaEventSynchronize(L.stage_ev[2 * (size_t) reaped + 1]);
            }
            if (e == cudaSuccess && progress) progress(std::min(n, (reaped + 1) * T));
        }
    };
    for (int i = 0; i < chunks; ++i) {
        if (i >= kAhead) {
            reap(i - kAhead + 1);
            if (e != cudaSuccess) break;
            if (stop && stop()) {
                stopped = true;
                break;
            }
        }
        const int64_t p = pos + (int64_t) i * T;
        const int k = i % S, t = i == chunks - 1 ? t_last : T;
        const bool again = i >= S;
        const int32_t* bc = t == T ? bnd : bnd + 2;
        const bool tail = taps && p + t > b0 - cap && tail_pos.size() < n_tail && tail_pos.size() < L.mtail.size();
        const int32_t* p3 = pos3 + (size_t) i * T * 3;
        // the main card's layers, its residual on to the second card, whose layers run beside the main card's next chunk
        chunk_start(m, st, tokens + (size_t) i * T, nullptr, t, p3, bc);
        chunk_layers(m, st, t, p, 0, L.lo, block ? &bd : nullptr);
        L.send(m, st, k, again, tail ? L.mtail[tail_pos.size()] : nullptr);
        L.read(*st.peer, i, k, again, p, p3, bc, L.lo, h.taps, tail ? (int) tail_pos.size() : -1, read_last, t);
        if (tail) {
            tail_pos.push_back(p);
            tail_rows.push_back(t);
        }
    }
    if (e == cudaSuccess && !stopped) {
        // the ring's chunks' features in order: the main card's part, then the projections of the second card's rows, as a
        // read on the main card makes them; their cells, so the ring ends as a read on one card leaves it
        cudaStreamWaitEvent(ms, L.l_done[(chunks - 1) % S], 0);
        for (size_t j = 0; j < tail_pos.size(); ++j) {
            const int t = tail_rows[j];
            cudaMemcpyAsync(m.cs.bounds, t == T ? bnd : bnd + 2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, ms);
            cudaMemcpyAsync(m.cs.dacc, L.mtail[j], (size_t) t * H * 4, cudaMemcpyDeviceToDevice, ms);
            for (size_t q = 0; q < h.taps.size(); ++q) {
                cudaMemcpyAsync(m.cs.x, L.tap_rows[j * (size_t) L.n_taps + q], (size_t) t * H * 2, cudaMemcpyHostToDevice, ms);
                chunk_tap(m, st, t, bd, h.first_tap + q);
            }
            block_ctx_chunk(m, st, m.cs.dacc, t, tail_pos[j]);
        }
    }
    // what is queued runs out, so that no copy is left in flight (after a stop the session holds part of the prompt)
    for (int d : {L.main_dev, L.dev}) {
        const core::OnDevice on(d);
        const cudaError_t s = cudaDeviceSynchronize();
        if (e == cudaSuccess) e = s;
    }
    if (e != cudaSuccess) {
        err = m.o.arch + ": the layer split's prompt read failed: " + cudaGetErrorString(e);
        return false;
    }
    if (stopped) {
        basalt::log::info("basalt dense", "layer split: stopped, %d of %d tokens read", std::min(n, reaped * T), n);
        return true;
    }
    reap(chunks);
    return true;
}

}  // namespace basalt::dense
