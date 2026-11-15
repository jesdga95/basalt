// src/dense/prompt_line.cpp - the prompt line: a second card holds the last layers' weights and reads a long prompt's
// chunks through them while the main card reads the next chunk's first layers.
//
// Chunk i: the main card reads layers [0, l0) and hands the BF16 residual to the second card through page-locked memory;
// the second card reads layers [l0, n) of chunk i while the main card reads layers [0, l0) of chunk i + 1, and sends its
// attention layers' cells of the chunk back as it goes.  The main card keeps the last chunks: once the second card's
// recurrent state is back it reads them whole, so the session ends as after a read on the main card alone and decoding is
// untouched.  The split (l0, and how many chunks the main card keeps) is priced per read from both cards' rates, timed at
// start.  The second card runs the main card's kernels on the same inputs (its int8 attention planned as the main
// card's), so its cells and state are the main card's to the bit.  The block drafter's features are made on the main card
// only (a product's rounding may follow the card's SM count): for the chunks whose rows its context ring keeps, the
// second card sends back the rows of its target layers and the main card projects them and makes their cells after the
// last chunk, in order.
// Without GPU-to-GPU access every copy goes card -> page-locked RAM -> card, each on a copy stream beside the reads.
#include "internal.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/log.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace basalt::dense {

class PromptLine {
public:
    static constexpr int kSlots = 2;             // hand-off buffers in turn
    int dev = -1, main_dev = 0, l_min = 0;
    int64_t max_context = 0;
    ModelPtr model;                              // on dev: layers [l_min, n) and a chunk's scratch
    SessionState st;                             // on dev: every GDN layer's state, the cells of attention layers >= l_min
    Arena arena;                                 // on dev: st's buffers and the landing residuals
    uint16_t* lin[kSlots] = {};                  // on dev: a chunk's residual as it lands
    void* main_mem = nullptr;                    // on the main card: the residuals going out, the drafter features kept
    uint16_t* mout[kSlots] = {};
    std::vector<float*> mtail;                   // the main card's features of the ring's chunks, the last chunk's last
    cudaStream_t m_out = nullptr, m_back = nullptr;   // main card: residuals out, the second card's cells and state in
    cudaStream_t l_in = nullptr, l_out = nullptr;     // second card: residuals in, cells and state out
    uint8_t* hand[kSlots] = {};                  // page-locked: a chunk's residual (BF16)
    uint8_t* kv[kSlots] = {};                    // page-locked: a chunk's cells of the line's attention layers
    std::vector<uint8_t*> tap_rows;              // page-locked: [tail chunk][target layer on the card] BF16 rows
    int n_taps = 0;                              // the block drafter's target layers at or past l_min
    uint8_t* state = nullptr;                    // page-locked: the line's GDN states
    int32_t* pos3 = nullptr;                     // page-locked: two bounds pairs, then [max_context][3] positions
    size_t kv_bytes = 0, state_bytes = 0;
    // each card's prompt rates, timed at start over the card's layers: ms per row per layer, and per row per visible cell
    // per attention layer; a chunk's residual crossing over (ms per row), the state's return (ms)
    struct Rate {
        double row = 0, cell = 0;
    } main_rate, line_rate;
    double hand_row_ms = 0, state_ms = 0;
    // main card: m_done (chunk's residual copied out on the main stream), handed (it is in RAM), kv_free, back
    // second card: lin_ready (landed), lin_free (taken by the read), l_done (read), kv_ready, state_ready, taps_ready
    cudaEvent_t m_done[kSlots] = {}, handed[kSlots] = {}, kv_free[kSlots] = {}, back = nullptr;
    std::vector<cudaEvent_t> main_ev, line_ev;   // timing: each line chunk's reads on either card, begin and end
    cudaEvent_t lin_ready[kSlots] = {}, lin_free[kSlots] = {}, l_done[kSlots] = {}, kv_ready[kSlots] = {}, state_ready = nullptr,
                taps_ready = nullptr;
    ~PromptLine();
};

void PromptLineDeleter::operator()(PromptLine* p) const { delete p; }

PromptLine::~PromptLine() {
    {
        const core::OnDevice on(dev);
        cudaDeviceSynchronize();
        for (cudaEvent_t* set : {lin_ready, lin_free, l_done, kv_ready})
            for (int k = 0; k < kSlots; ++k) if (set[k]) cudaEventDestroy(set[k]);
        for (cudaEvent_t e : {state_ready, taps_ready}) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : line_ev) if (e) cudaEventDestroy(e);
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
        for (cudaEvent_t e : main_ev) if (e) cudaEventDestroy(e);
        if (main_mem) cudaFree(main_mem);
    }
    for (uint8_t* p : hand) if (p) cudaFreeHost(p);
    for (uint8_t* p : kv) if (p) cudaFreeHost(p);
    for (uint8_t* p : tap_rows) if (p) cudaFreeHost(p);
    if (state) cudaFreeHost(state);
    if (pos3) cudaFreeHost(pos3);
}

namespace {

// a layer's cells, K and V: fp16 rows, or int8 codes with an fp16 scale per 64 values
size_t cell_bytes(const Geometry& g, bool int8) {
    const size_t row = (size_t) g.n_kv * g.head_dim;
    return int8 ? 2 * (row + row / 64 * 2) : 2 * row * 2;
}

// the attention layers and GDN layers before layer l
int count_kind(const Geometry& g, int l, Mixer k) { return (int) std::count(g.kinds.begin(), g.kinds.begin() + l, k); }

constexpr double kPriceMargin = 0.95;   // a split must price this far below the main card alone
constexpr int kRefitChunks = 6;             // line chunks a read needs for its timings to refit both rates
constexpr int kAhead = 2;                   // chunks the host queues ahead of the line's finished ones: a stop waits for at most these
constexpr int kCalRuns = 4;                 // timed runs per card and position at start (the later half counts)
constexpr int64_t kCalFar = 11 * kChunk;   // the furthest position timed at start

struct Region {
    void* main;
    void* line;
    size_t bytes;
};
// cells [c0, c0 + n) of attention layers ai0.. of both sessions, as copy regions
std::vector<Region> cell_regions(const Model& m, const SessionState& main, const SessionState& line, int ai0, int64_t c0, int64_t n) {
    std::vector<Region> r;
    const size_t row = (size_t) m.g.n_kv * m.g.head_dim;
    for (int ai = ai0; ai < m.g.attn_layers(); ++ai) {
        if (m.o.kv_int8) {
            const kernels::KvInt8 &a = main.kq[(size_t) ai], &b = line.kq[(size_t) ai];
            r.push_back({a.k + c0 * row, b.k + c0 * row, (size_t) n * row});
            r.push_back({a.v + c0 * row, b.v + c0 * row, (size_t) n * row});
            r.push_back({a.ks + c0 * row / 64, b.ks + c0 * row / 64, (size_t) n * row / 64 * 2});
            r.push_back({a.vs + c0 * row / 64, b.vs + c0 * row / 64, (size_t) n * row / 64 * 2});
        } else {
            r.push_back({main.kc[(size_t) ai] + c0 * row, line.kc[(size_t) ai] + c0 * row, (size_t) n * row * 2});
            r.push_back({main.vc[(size_t) ai] + c0 * row, line.vc[(size_t) ai] + c0 * row, (size_t) n * row * 2});
        }
    }
    return r;
}

// the line's chunks whose drafter features the context ring still holds at the end of a read: at most this many
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

// Timed runs of one chunk of rows through layers [l0, n) of a card's model at a position (its cells below it hold whatever
// they hold), each between its own events.
struct Runs {
    Model* md = nullptr;
    SessionState* s = nullptr;
    const uint16_t* x = nullptr;   // the rows (BF16, on the card)
    int dev = 0, reps = 0;
    std::vector<cudaEvent_t> ev;
    ~Runs() {
        const core::OnDevice on(dev);
        for (cudaEvent_t e : ev) if (e) cudaEventDestroy(e);
    }
    bool queue(int l0, int64_t p, const int32_t* bnd, const int32_t* pos3) {
        const core::OnDevice on(dev);
        if (ev.empty()) {
            ev.assign(2 * (size_t) reps, nullptr);
            for (cudaEvent_t& e : ev)
                if (cudaEventCreate(&e) != cudaSuccess) return false;
        }
        for (int r = 0; r < reps; ++r) {
            cudaMemcpyAsync(md->cs.x, x, (size_t) kChunk * md->g.n_embd * 2, cudaMemcpyDeviceToDevice, s->stream);
            chunk_start(*md, *s, nullptr, nullptr, kChunk, pos3, bnd);
            cudaEventRecord(ev[2 * (size_t) r], s->stream);
            chunk_layers(*md, *s, kChunk, p, l0, md->g.n_layers, nullptr);
            cudaEventRecord(ev[2 * (size_t) r + 1], s->stream);
        }
        return true;
    }
    // the mean of the later half of the runs (the card settled into its sustained clocks), -1 on failure
    float ms() {
        const core::OnDevice on(dev);
        if (cudaStreamSynchronize(s->stream) != cudaSuccess) return -1;
        float sum = 0;
        int n = 0;
        for (int r = reps / 2; r < reps; ++r) {
            float t = 0;
            if (cudaEventElapsedTime(&t, ev[2 * (size_t) r], ev[2 * (size_t) r + 1]) != cudaSuccess) return -1;
            sum += t;
            ++n;
        }
        return n ? sum / n : -1;
    }
};

// a card's rates from a chunk timed at position 0 (t0) and at p (t1) over nl layers, na of them attention layers
PromptLine::Rate fit_rate(float t0, float t1, int64_t p, int nl, int na) {
    PromptLine::Rate r;
    const double T = kChunk;
    r.cell = p > 0 && na > 0 ? std::max(0.0, (double) (t1 - t0)) / ((double) na * T * (double) p) : 0.0;
    r.row = std::max(0.0, (double) t0 - r.cell * na * T * T / 2) / (nl * T);
    return r;
}

// ms of layers [l0, l1) of a chunk of t rows at position p at rate r
double layers_ms(const PromptLine::Rate& r, const Geometry& g, int l0, int l1, int64_t p, int t) {
    const int na = count_kind(g, l1, Mixer::Attention) - count_kind(g, l0, Mixer::Attention);
    return r.row * t * (l1 - l0) + r.cell * t * ((double) p + t / 2.0) * na;
}

struct Plan {
    int l0 = 0, keep = 0;
    double ms = 0, alone_ms = 0;
};
// The split of an n-token read at pos that the rates price lowest: the line reads layers [l0, n) of every chunk but the
// last `keep` ones, which the main card reads whole once the line's state is back; keep == 0 when no split is cheaper
// than the main card alone by a clear margin.
Plan price(const PromptLine& L, const Geometry& g, int64_t pos, int n) {
    const int T = kChunk, chunks = (n + T - 1) / T;
    const auto rows = [&](int c) { return c == chunks - 1 ? n - c * T : T; };
    Plan best;
    for (int c = 0; c < chunks; ++c) best.alone_ms += layers_ms(L.main_rate, g, 0, g.n_layers, pos + (int64_t) c * T, rows(c));
    best.ms = best.alone_ms;
    for (int l0 = L.l_min; l0 < g.n_layers; ++l0)
        for (int keep = 1; keep < chunks && keep <= 3; ++keep) {
            double main_t = 0, line_t = 0;
            const int lc = chunks - keep;
            for (int c = 0; c < lc; ++c) {
                const int64_t p = pos + (int64_t) c * T;
                main_t += layers_ms(L.main_rate, g, 0, l0, p, T);
                line_t = std::max(main_t + L.hand_row_ms * T, line_t) + layers_ms(L.line_rate, g, l0, g.n_layers, p, T);
            }
            const int64_t pf = pos + (int64_t) lc * T;
            main_t += layers_ms(L.main_rate, g, 0, l0, pf, rows(lc));
            double t = std::max(main_t, line_t + L.state_ms) + layers_ms(L.main_rate, g, l0, g.n_layers, pf, rows(lc));
            for (int c = lc + 1; c < chunks; ++c) t += layers_ms(L.main_rate, g, 0, g.n_layers, pos + (int64_t) c * T, rows(c));
            if (t < best.ms) best = {l0, keep, t, best.alone_ms};
        }
    if (best.keep > 0 && best.ms > kPriceMargin * best.alone_ms) best.keep = 0;
    return best;
}

// A card's rates refit from the reads of the last line: segment i of `n` was `ms[i]` over layers [l0, l1) of a whole chunk
// at pos + i chunks.  With a spread of positions both rates are fitted, else the per-row one alone; what the cards met on
// real rows is what the next split is priced with.
void refit(PromptLine::Rate& r, const Geometry& g, int l0, int l1, int64_t pos, const std::vector<float>& ms) {
    const int n = (int) ms.size(), T = kChunk;
    const int na = count_kind(g, l1, Mixer::Attention) - count_kind(g, l0, Mixer::Attention);
    if (n == 0) return;
    double sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0, priced = 0, took = 0;
    for (int i = 0; i < n; ++i) {
        const double x = (double) T * (l1 - l0), y = (double) T * ((double) (pos + (int64_t) i * T) + T / 2.0) * na, z = ms[(size_t) i];
        sxx += x * x, sxy += x * y, syy += y * y, sxz += x * z, syz += y * z;
        priced += layers_ms(r, g, l0, l1, pos + (int64_t) i * T, T);
        took += z;
    }
    const double det = sxx * syy - sxy * sxy;
    if (n >= kRefitChunks && na > 0 && det > 1e-9 * sxx * syy) {
        const double row = (sxz * syy - syz * sxy) / det, cell = (sxx * syz - sxy * sxz) / det;
        if (row > 0 && cell >= 0) {
            r.row = row;
            r.cell = cell;
            return;
        }
    }
    if (priced > 0) r.row *= took / priced;
}

}  // namespace

// Each card's rates: a chunk of random rows (a card's speed may follow what its products multiply) through its layers (the
// main card's all, the line's its own) at position 0 and further on (where attention reads more cells), both cards at
// once, timed after a first run that loads the kernels; the main card's in a session of its own for the time.  The rows cross over as a chunk's residual does, which
// times the hand-off.
static bool calibrate(PromptLine& L, Model& main, std::string& err) {
    const Geometry& g = main.g;
    const size_t T = kChunk, H = (size_t) g.n_embd;
    const int64_t far_pos = std::max<int64_t>(0, std::min<int64_t>(L.max_context - kChunk, kCalFar)) / kChunk * kChunk;
    const int nl = g.n_layers - L.l_min, na = g.attn_layers() - count_kind(g, L.l_min, Mixer::Attention);
    std::memset(L.pos3, 0, ((size_t) kChunk * 3 + 4) * sizeof(int32_t));
    L.pos3[1] = kChunk;
    const int32_t *bnd = L.pos3, *pos3 = L.pos3 + 4;
    {   // the rows: BF16 of about unit spread, on both cards
        uint16_t* h = (uint16_t*) L.hand[0];
        uint32_t x = 0x9e3779b9u;
        for (size_t i = 0; i < T * H; ++i) {
            float v = -2.0f;
            for (int k = 0; k < 4; ++k) {
                x = x * 1664525u + 1013904223u;
                v += (float) (x >> 8) * (1.0f / 16777216.0f);
            }
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof bits);
            h[i] = (uint16_t) (bits >> 16);
        }
    }
    const auto t_hand = std::chrono::steady_clock::now();
    {
        const core::OnDevice on(L.dev);
        if (cudaMemcpy(L.lin[0], L.hand[0], T * H * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "a copy to the card failed";
            return false;
        }
    }
    {
        const core::OnDevice on(L.main_dev);
        if (cudaMemcpy(L.mout[0], L.lin[0], T * H * 2, cudaMemcpyDefault) != cudaSuccess) {   // (staged when no peer access)
            err = "a copy to the main card failed";
            return false;
        }
    }
    const double hand_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hand).count() * 2 / 3;   // three hops timed, a hand-off takes two
    // both cards at once, as a read runs them: the main card's whole stack and the line's layers at position 0, then the
    // line's layers further on beside the main card's same layers at 0 and further on (attention's part is their difference)
    const core::OnDevice on_main(L.main_dev);
    if (!main.ensure_chunk(err)) return false;
    Arena a;
    SessionState s;
    s.gdn_floats = L.st.gdn_floats;
    const int ai0 = count_kind(g, L.l_min, Mixer::Attention);
    const size_t need = (size_t) g.gdn_layers() * s.gdn_floats * 4 + (size_t) na * (size_t) (far_pos + kChunk) * cell_bytes(g, main.o.kv_int8) +
                        (size_t) ai0 * kChunk * cell_bytes(g, main.o.kv_int8) + (size_t) g.attn_layers() * 4 * 256 + 4096;
    bool ok = far_pos > 0 && a.reserve(need) && cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) == cudaSuccess;
    s.gdn = ok ? a.take<float>((size_t) g.gdn_layers() * s.gdn_floats) : nullptr;
    ok = ok && s.gdn && take_cells(a, s, g, main.o.kv_int8, ai0, far_pos + kChunk, kChunk);
    // the cells the runs further on read hold random values too: what attention multiplies moves a card's speed as well
    const auto fill = [&](const SessionState& ss, cudaStream_t st) {
        for (const Region& r : cell_regions(main, ss, ss, ai0, 0, far_pos)) {
            for (size_t at = 0; at < r.bytes; at += T * H * 2)
                if (cudaMemcpyAsync((uint8_t*) r.line + at, L.hand[0], std::min(T * H * 2, r.bytes - at), cudaMemcpyHostToDevice, st) !=
                    cudaSuccess)
                    return false;
        }
        return cudaStreamSynchronize(st) == cudaSuccess;
    };
    ok = ok && fill(s, s.stream);
    if (ok) {
        const core::OnDevice on(L.dev);
        ok = fill(L.st, L.st.stream);
    }
    const auto runs = [&](Runs& r, bool line, int reps) {
        r.md = line ? L.model.get() : &main;
        r.s = line ? &L.st : &s;
        r.x = line ? L.lin[0] : L.mout[0];
        r.dev = line ? L.dev : L.main_dev;
        r.reps = reps;
    };
    float main_all = -1, main_0 = -1, main_far = -1, line_0 = -1, line_far = -1;
    if (ok) {
        Runs mw, lw;   // a first run of each loads the kernels
        runs(mw, false, 1);
        runs(lw, true, 1);
        ok = mw.queue(0, 0, bnd, pos3) && lw.queue(L.l_min, 0, bnd, pos3) && mw.ms() >= 0 && lw.ms() >= 0;
    }
    if (ok) {
        Runs ma, la;
        runs(ma, false, kCalRuns);
        runs(la, true, kCalRuns);
        ok = la.queue(L.l_min, 0, bnd, pos3) && ma.queue(0, 0, bnd, pos3);
        main_all = ok ? ma.ms() : -1;
        line_0 = ok ? la.ms() : -1;
    }
    if (ok) {
        Runs m0, mf, lf;
        runs(m0, false, kCalRuns);
        runs(mf, false, kCalRuns);
        runs(lf, true, kCalRuns);
        ok = lf.queue(L.l_min, far_pos, bnd, pos3) && m0.queue(L.l_min, 0, bnd, pos3) && mf.queue(L.l_min, far_pos, bnd, pos3);
        main_0 = ok ? m0.ms() : -1;
        main_far = ok ? mf.ms() : -1;
        line_far = ok ? lf.ms() : -1;
    }
    if (s.stream) cudaStreamDestroy(s.stream);
    {
        const core::OnDevice on(L.dev);
        cudaMemsetAsync(L.st.gdn, 0, (size_t) g.gdn_layers() * L.st.gdn_floats * 4, L.st.stream);
        ok = ok && cudaStreamSynchronize(L.st.stream) == cudaSuccess;
    }
    ok = ok && main_all >= 0 && main_0 >= 0 && main_far >= 0 && line_0 >= 0 && line_far >= 0;
    if (!ok) {
        err = far_pos > 0 ? std::string("timing the cards' layers failed: ") + cudaGetErrorString(cudaGetLastError())
                      : "the context is too short for a split";
        return false;
    }
    L.line_rate = fit_rate(line_0, line_far, far_pos, nl, na);
    L.main_rate.cell = fit_rate(main_0, main_far, far_pos, nl, na).cell;
    L.main_rate.row = std::max(0.0, (double) main_all - L.main_rate.cell * g.attn_layers() * (double) T * T / 2) /
                      ((double) g.n_layers * T);
    L.hand_row_ms = hand_ms / (double) T;
    L.state_ms = hand_ms * (double) L.state_bytes / (double) (T * H * 2);
    basalt::log::info("basalt dense", "prompt line: ms per 1K rows and layer: CUDA%d %.2f, CUDA%d %.2f; per 1K rows and 1K cells of "
                      "attention: %.3f, %.3f; a chunk's hand-off %.1f ms", L.main_dev, L.main_rate.row * 1000, L.dev,
                      L.line_rate.row * 1000, L.main_rate.cell * 1e6, L.line_rate.cell * 1e6, hand_ms);
    return true;
}

PromptLinePtr make_prompt_line(Model& main, int dev, int l_min, const Weights& w, std::string& err) {
    PromptLinePtr p(new PromptLine());
    PromptLine& L = *p;
    cudaGetDevice(&L.main_dev);
    L.dev = dev;
    L.l_min = l_min;
    L.max_context = main.o.max_context;
    const Geometry& g = main.g;
    const size_t T = kChunk, H = (size_t) g.n_embd;
    const bool block = main.o.drafter == Drafter::Block;
    const BlockDrafter& bd = main.w.dflash;
    if (block) L.n_taps = (int) std::count_if(bd.target_layers.begin(), bd.target_layers.end(), [&](int t) { return t >= l_min; });
    const int n_tail = ring_tails(block ? &bd : nullptr);
    const auto fail = [&](const std::string& what) {
        err = main.o.arch + ": the prompt line on CUDA" + std::to_string(dev) + ": " + what;
        return nullptr;
    };
    {
        const core::OnDevice on(dev);
        Options o = main.o;
        o.max_context = kChunk;   // its decode scratch is never used: the line reads prompt chunks only
        o.sessions = 1;
        o.spec = 0;
        o.draft = false;
        o.drafter = Drafter::None;
        o.vision = false;
        o.mmproj = {};
        Weights lw = w;
        lw.head = main.w.head;   // (the main card's: never read here; its type keeps MMQ from loading a kernel for it)
        lw.mtp = {};
        lw.dflash = {};
        L.model = make_model(g, lw, o, err);
        if (!L.model) return nullptr;
        if (!L.model->ensure_chunk(err)) return nullptr;
        SessionState& s = L.st;
        const size_t C = (size_t) g.conv_channels(), HV = (size_t) g.gdn_v_heads;
        s.gdn_floats = (size_t) g.gdn_state * HV * g.gdn_state + C * 3;
        const int ai0 = count_kind(g, l_min, Mixer::Attention);
        const size_t need = (size_t) g.gdn_layers() * s.gdn_floats * 4 +
                            (size_t) (g.attn_layers() - ai0) * (size_t) L.max_context * cell_bytes(g, main.o.kv_int8) +
                            PromptLine::kSlots * T * H * 2 + (size_t) g.attn_layers() * 4 * 256 + 4096;
        if (!L.arena.reserve(need)) return fail(std::to_string((unsigned long long) (need >> 20)) + " MiB of cells do not fit");
        s.gdn = s.gdn_first = L.arena.take<float>((size_t) g.gdn_layers() * s.gdn_floats);
        if (!take_cells(L.arena, s, g, main.o.kv_int8, ai0, L.max_context)) return fail("its cells do not fit their arena");
        for (uint16_t*& b : L.lin) b = L.arena.take<uint16_t>(T * H);
        if (!L.lin[PromptLine::kSlots - 1]) return fail("its buffers do not fit their arena");
        bool ok = true;
        for (cudaStream_t* st : {&s.stream, &L.l_in, &L.l_out}) ok = ok && cudaStreamCreateWithFlags(st, cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* set : {L.lin_ready, L.lin_free, L.l_done, L.kv_ready})
            for (int k = 0; k < PromptLine::kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&set[k], cudaEventDisableTiming) == cudaSuccess;
        for (cudaEvent_t* e : {&L.state_ready, &L.taps_ready}) ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) return fail("cannot create its streams and events");
    }
    {
        const core::OnDevice on(L.main_dev);
        const size_t bytes = prompt_line_main_bytes(g, block ? bd : BlockDrafter{}, L.max_context);
        if (cudaMalloc(&L.main_mem, bytes) != cudaSuccess) {
            err = main.o.arch + ": out of device memory for the prompt line's buffers";
            return nullptr;
        }
        uint8_t* at = (uint8_t*) L.main_mem;
        for (uint16_t*& b : L.mout) { b = (uint16_t*) at; at += T * H * 2; }
        for (int j = 0; j < (n_tail ? n_tail + 1 : 0); ++j) { L.mtail.push_back((float*) at); at += T * H * 4; }
        bool ok = true;
        for (cudaStream_t* st : {&L.m_out, &L.m_back}) ok = ok && cudaStreamCreateWithFlags(st, cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* set : {L.m_done, L.handed, L.kv_free})
            for (int k = 0; k < PromptLine::kSlots; ++k) ok = ok && cudaEventCreateWithFlags(&set[k], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&L.back, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) return fail("cannot create its streams and events");
    }
    L.kv_bytes = (size_t) (g.attn_layers() - count_kind(g, l_min, Mixer::Attention)) * T * cell_bytes(g, main.o.kv_int8);
    L.state_bytes = (size_t) (g.gdn_layers() - count_kind(g, l_min, Mixer::Gdn)) * L.st.gdn_floats * 4;
    const auto pin = [&](uint8_t*& at, size_t bytes) {
        void* v = nullptr;
        if (bytes > 0 && cudaHostAlloc(&v, bytes, cudaHostAllocPortable) != cudaSuccess) return false;
        at = (uint8_t*) v;
        return true;
    };
    uint8_t* positions = nullptr;
    bool ok = pin(L.state, L.state_bytes) && pin(positions, ((size_t) L.max_context * 3 + 4) * sizeof(int32_t));
    L.pos3 = (int32_t*) positions;
    for (int k = 0; k < PromptLine::kSlots; ++k) ok = ok && pin(L.hand[k], T * H * 2) && pin(L.kv[k], L.kv_bytes);
    L.tap_rows.assign((size_t) n_tail * L.n_taps, nullptr);
    for (uint8_t*& t : L.tap_rows) ok = ok && pin(t, T * H * 2);
    if (!ok) {
        err = main.o.arch + ": out of page-locked memory for the prompt line";
        return nullptr;
    }
    // the second card's int8 prompt attention splits its keys as the main card's does, so its rows round the same
    kernels::gqa_prompt_plan_device(L.main_dev);
    if (!calibrate(L, main, err)) return fail(err);
    const size_t pinned = PromptLine::kSlots * (T * H * 2 + L.kv_bytes) + L.tap_rows.size() * T * H * 2 + L.state_bytes;
    basalt::log::info("basalt dense", "prompt line: CUDA%d reads layers %d-%d of prompts over %d tokens; %.2f GiB of page-locked "
                      "hand-off buffers", dev, l_min, g.n_layers - 1, 2 * kChunk, (double) pinned / 1073741824.0);
    return p;
}

uint64_t prompt_line_main_bytes(const Geometry& g, const BlockDrafter& drafter, int64_t max_context) {
    if (max_context <= 2 * (int64_t) kChunk) return 0;
    const int n_tail = ring_tails(&drafter);   // and one more: the main card's first chunk's, kept aside
    const uint64_t T = kChunk, H = (uint64_t) g.n_embd;
    return PromptLine::kSlots * T * H * 2 + (uint64_t) (n_tail ? n_tail + 1 : 0) * T * H * 4;
}

bool attach_prompt_line(Model& m, int dev, int l_min, const Weights& w, std::string& err) {
    if (m.o.max_context <= 2 * (int64_t) kChunk) {
        basalt::log::info("basalt dense", "prompt line: off, a context of %lld holds no prompt over %d tokens", (long long) m.o.max_context,
                          2 * kChunk);
        return true;
    }
    m.line = make_prompt_line(m, dev, l_min, w, err);
    return m.line != nullptr;
}

bool prompt_line_read(PromptLine& L, Model& m, SessionState& st, const int32_t* tokens, int n, int64_t pos,
                      const std::function<void(int)>& progress, const std::function<bool()>& stop, std::string& err) {
    const Geometry& g = m.g;
    const int T = kChunk, H = g.n_embd;
    constexpr int S = PromptLine::kSlots;
    if (n <= 2 * T || st.mtp_kc != nullptr || pos + n > L.max_context) return false;
    const Plan plan = price(L, g, pos, n);
    if (plan.keep == 0) {
        basalt::log::info("basalt dense", "prompt line: %d tokens read on CUDA%d (%.0f ms priced alone, %.0f ms across)", n, L.main_dev,
                          plan.alone_ms, plan.ms);
        return false;
    }
    const int l0 = plan.l0;
    Model& lm = *L.model;
    if (!m.ensure_chunk(err)) return false;
    const bool block = m.o.drafter == Drafter::Block;
    const BlockDrafter& bd = m.w.dflash;
    const BlockDrafter* mtaps = block ? &bd : nullptr;
    // the drafter's target layers on the card (their rows come back for the ring's chunks), and whether it reads the last
    std::vector<int> ltaps;
    for (size_t j = 0; block && j < bd.target_layers.size(); ++j)
        if (bd.target_layers[j] >= l0) ltaps.push_back((int) j);
    const bool read_last = block && !ltaps.empty() && bd.target_layers[(size_t) ltaps.back()] == g.n_layers - 1;
    const int64_t cap = block ? st.dctx_cap : 0, b0 = pos + n;
    const int chunks = (n + T - 1) / T;
    const int t_last = n - (chunks - 1) * T;
    const int lc = chunks - plan.keep;   // the chunks the line reads
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
    {   // the line's layers start from the session's state: empty at the start, else copied from the main card
        const core::OnDevice on(L.dev);
        if (pos == 0) {
            cudaMemsetAsync(L.st.gdn + gdn_off, 0, gdn_bytes, ls);
        } else {
            cudaMemcpyPeerAsync(L.st.gdn + gdn_off, L.dev, st.gdn + gdn_off, L.main_dev, gdn_bytes, ls);
            for (const Region& r : cell_regions(m, st, L.st, ai0, 0, pos)) cudaMemcpyPeerAsync(r.line, L.dev, r.main, L.main_dev, r.bytes, ls);
        }
    }
    for (auto [evs, dev] : {std::pair{&L.main_ev, L.main_dev}, std::pair{&L.line_ev, L.dev}}) {
        const core::OnDevice on(dev);
        while (evs->size() < 2 * (size_t) lc) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) {
                err = "cannot create CUDA events";
                return false;
            }
            evs->push_back(e);
        }
    }
    std::vector<int64_t> tail_pos;   // the first cells of the line's chunks the ring keeps rows of
    // Chunks are queued no more than kAhead past the line's finished ones; each finished one is reported, and `stop` asked.
    cudaError_t e = cudaSuccess;
    bool stopped = false;
    int reaped = 0;   // line chunks seen finished
    const auto reap = [&](int upto) {
        for (; reaped < upto && e == cudaSuccess; ++reaped) {
            {
                const core::OnDevice on(L.dev);
                e = cudaEventSynchronize(L.line_ev[2 * (size_t) reaped + 1]);
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
        cudaEventRecord(L.main_ev[2 * (size_t) i], ms);
        chunk_layers(m, st, T, p, 0, l0, mtaps);
        cudaEventRecord(L.main_ev[2 * (size_t) i + 1], ms);
        if (again) cudaStreamWaitEvent(ms, L.handed[k], 0);
        cudaMemcpyAsync(L.mout[k], m.cs.x, (size_t) T * H * 2, cudaMemcpyDeviceToDevice, ms);
        if (tail) cudaMemcpyAsync(L.mtail[tail_pos.size()], m.cs.dacc, (size_t) T * H * 4, cudaMemcpyDeviceToDevice, ms);
        cudaEventRecord(L.m_done[k], ms);
        cudaStreamWaitEvent(L.m_out, L.m_done[k], 0);
        if (again) cudaStreamWaitEvent(L.m_out, L.lin_ready[k], 0);
        cudaMemcpyAsync(L.hand[k], L.mout[k], (size_t) T * H * 2, cudaMemcpyDeviceToHost, L.m_out);
        cudaEventRecord(L.handed[k], L.m_out);
        std::vector<Region> cells = cell_regions(m, st, L.st, ai0, p, T);
        {
            const core::OnDevice on(L.dev);
            // lands beside the read of the chunk before
            cudaStreamWaitEvent(L.l_in, L.handed[k], 0);
            if (again) cudaStreamWaitEvent(L.l_in, L.lin_free[k], 0);
            cudaMemcpyAsync(L.lin[k], L.hand[k], (size_t) T * H * 2, cudaMemcpyHostToDevice, L.l_in);
            cudaEventRecord(L.lin_ready[k], L.l_in);
            // the read of its last layers; the rows of the drafter's target layers among them out for the ring's chunks
            cudaStreamWaitEvent(ls, L.lin_ready[k], 0);
            cudaMemcpyAsync(lm.cs.x, L.lin[k], (size_t) T * H * 2, cudaMemcpyDeviceToDevice, ls);
            cudaEventRecord(L.lin_free[k], ls);
            chunk_start(lm, L.st, nullptr, nullptr, T, p3, bnd);
            cudaEventRecord(L.line_ev[2 * (size_t) i], ls);
            int at = l0;
            if (tail) {
                for (size_t q = 0; q < ltaps.size(); ++q) {
                    const int tl = bd.target_layers[(size_t) ltaps[q]];
                    chunk_layers(lm, L.st, T, p, at, tl + 1, nullptr, read_last);
                    cudaMemcpyAsync(L.tap_rows[tail_pos.size() * (size_t) L.n_taps + q], lm.cs.x, (size_t) T * H * 2,
                                    cudaMemcpyDeviceToHost, ls);
                    at = tl + 1;
                }
                cudaEventRecord(L.taps_ready, ls);
            }
            chunk_layers(lm, L.st, T, p, at, g.n_layers, nullptr, read_last);
            cudaEventRecord(L.line_ev[2 * (size_t) i + 1], ls);
            cudaEventRecord(L.l_done[k], ls);
            // its cells out beside the next chunk's read
            cudaStreamWaitEvent(L.l_out, L.l_done[k], 0);
            if (again) cudaStreamWaitEvent(L.l_out, L.kv_free[k], 0);
            size_t off = 0;
            for (const Region& r : cells) {
                cudaMemcpyAsync(L.kv[k] + off, r.line, r.bytes, cudaMemcpyDeviceToHost, L.l_out);
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
            err = m.o.arch + ": the prompt line failed: " + cudaGetErrorString(e);
            return false;
        }
        basalt::log::info("basalt dense", "prompt line: stopped, %d of %d tokens read", reaped * T, n);
        return true;
    }
    {   // the line's recurrent state after its last chunk goes back
        const core::OnDevice on(L.dev);
        cudaStreamWaitEvent(L.l_out, L.l_done[(lc - 1) % S], 0);
        cudaMemcpyAsync(L.state, L.st.gdn + gdn_off, gdn_bytes, cudaMemcpyDeviceToHost, L.l_out);
        cudaEventRecord(L.state_ready, L.l_out);
    }
    cudaStreamWaitEvent(L.m_back, L.state_ready, 0);
    cudaMemcpyAsync(st.gdn + gdn_off, L.state, gdn_bytes, cudaMemcpyHostToDevice, L.m_back);
    cudaEventRecord(L.back, L.m_back);
    // the main card's chunks: the first one's first layers beside the line's last chunk, the rest once the state is back,
    // then the others whole
    const int64_t pf = pos + (int64_t) lc * T;
    const int tf = lc == chunks - 1 ? t_last : T;
    const int32_t* bf = tf == T ? bnd : bnd + 2;
    chunk_start(m, st, tokens + (size_t) lc * T, nullptr, tf, pos3 + (size_t) lc * T * 3, bf);
    chunk_layers(m, st, tf, pf, 0, l0, mtaps);
    cudaStreamWaitEvent(ms, L.back, 0);
    chunk_layers(m, st, tf, pf, l0, g.n_layers, mtaps);
    // the drafter's features of the line's chunks its ring keeps: the main card's own taps, then the projections of the
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
    // the line's last chunks' reports, then the main card's whole chunks, one at a time: a stop is met between them
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
        err = m.o.arch + ": the prompt line failed: " + cudaGetErrorString(e);
        return false;
    }
    if (stopped) {
        basalt::log::info("basalt dense", "prompt line: stopped after %d of %d tokens", (int) std::min<int64_t>(n, (int64_t) (reaped + 1) * T), n);
        return true;
    }
    // the cards' rates as this read met them, for the next split
    std::vector<float> main_ms((size_t) lc), line_ms((size_t) lc);
    bool timed = true;
    for (int i = 0; i < lc; ++i) {
        timed = timed && cudaEventElapsedTime(&main_ms[(size_t) i], L.main_ev[2 * (size_t) i], L.main_ev[2 * (size_t) i + 1]) == cudaSuccess;
        const core::OnDevice on(L.dev);
        timed = timed && cudaEventElapsedTime(&line_ms[(size_t) i], L.line_ev[2 * (size_t) i], L.line_ev[2 * (size_t) i + 1]) == cudaSuccess;
    }
    if (timed) {
        refit(L.main_rate, g, 0, l0, pos, main_ms);
        refit(L.line_rate, g, l0, g.n_layers, pos, line_ms);
    }
    if (progress) progress(n);
    basalt::log::info("basalt dense", "prompt line: %d tokens, layers %d-%d of chunks 1-%d on CUDA%d; priced %.0f ms, %.0f ms alone", n,
                      l0, g.n_layers - 1, lc, L.dev, plan.ms, plan.alone_ms);
    return true;
}

}  // namespace basalt::dense
