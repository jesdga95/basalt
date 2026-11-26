// src/models/inkling2/session.cpp - one conversation of Inkling 2: its caches (a window ring per sliding-window visit, a
// position per cell for the full-attention visits, the short convolutions' rings) and the forward of a decode window,
// replayed from a CUDA graph per window shape.
//
// A token visits 48 layer applications: the prelude (layers 0..5), the loop's core (6..17) three times, the coda
// (18..23). Each visit has its own caches. A layer's inputs are block-attention mixes of the finished blocks and the
// partial sum; in the loop's later passes they also take a message read from the previous pass through gated controllers.
// A row's result does not depend on the other rows of its call: the products take 8 rows at most per launch with the
// single-row reduction order, the attention and the glue run a block per row (or row and head), the experts per (row,
// slot) entry. The positions and tokens are read on the card, and the attention's launch bound (a power of two of keys)
// only sizes its grid. With a spill, the experts in RAM run on the CPU beside the card's (cpu_experts.hpp): the layers that
// have any are part of a window's shape, and the window's captured graph queues their CPU work too.
//
// The layer split (model.hpp card_of): the second card runs its layers' visits on its own scratch and streams, and a window
// crosses between the cards where walk.hpp's split_walk says, the row buffers the next card reads going through page-locked
// RAM (consumer cards have no peer access). The whole window, both cards' launches and the crossings, is one graph.
#include "basalt/core/device_copy.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/elastic_kv.hpp"
#include "basalt/core/graph.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/program/expert_spill.hpp"
#include "models/inkling2/caches.hpp"
#include "models/inkling2/cpu_experts.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/expert_map.hpp"
#include "models/inkling2/model.hpp"
#include "models/inkling2/prompt.hpp"
#include "models/inkling2/walk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWindowRows = kDecodeRows; // rows of one conversation's window (also its ring margin and logits buffer)
constexpr int kMaxRows = kBatchRows;     // widest batched forward call across conversations
constexpr int kChunkRead = 16;           // a longer read runs on the prompt path, a shorter one as decode windows
// with experts in RAM, a read of up to this many tokens runs as decode windows: the prompt path's fixed cost (staging the
// spilled layers' experts on the card) is more than eight of them
constexpr int kSpillWindowRead = 64;
constexpr int kTypeBF16 = 30;
constexpr int kOutRows = 3072;           // the attention output's width (heads x head width)
constexpr int kStage = 64;               // windows whose tokens and positions are in flight from the pinned ring at once
constexpr int kSegWords = (sizeof(bk::WindowSegs) + 15) / 4;   // device WindowSegs words staged alongside tokens & positions

int call_rows_of(const InklingModel& m) {
    return m.sessions() > 1 ? std::min(kMaxRows, m.sessions() * kWindowRows) : kWindowRows;
}

// The forward scratch: the rows of one call through the layers and the head. Every session of the model runs on the same
// buffers (Shared below), one call at a time: each call syncs its stream before it returns, and a batch window syncs every
// session's stream before it starts.
struct Scratch {
    float* role_[kBufs] = {};
    float* emb_ = nullptr;
    float *mix_ = nullptr, *msg_ = nullptr, *ctrl_scr_ = nullptr, *nrm_ = nullptr, *y_ = nullptr;
    float *q_ = nullptr, *r_ = nullptr, *sink_ = nullptr, *o_ = nullptr, *resid_ = nullptr, *fin_ = nullptr;
    void* att_scr_ = nullptr;
    void* xq_[3] = {};                       // the products' q8_1 inputs of the hidden width
    void* xo_ = nullptr;                     // the output projection's
    float *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    unsigned long long* ex_ptr_ = nullptr;
    int32_t *ex_start_ = nullptr, *ex_ng_ = nullptr, *ex_dst_ = nullptr, *ex_tok_ = nullptr;
    void* ex_scr_ = nullptr;
    float* ex_rows_ = nullptr;
    float* nrm1_ = nullptr;                     // [visit][kGuessRows][E] a guessed window's MLP inputs, a visit each
    int32_t* gs_ids_ = nullptr;                 // [visit][kGuessExperts] the guessed experts
    float *gs_log_ = nullptr, *gs_w_ = nullptr; // the guess's router scratch
    uint32_t* pf_sink_ = nullptr;               // the prefetch's keep-alive word
    float* lp_ = nullptr;                       // the log probabilities' scratch
    float* blog_ = nullptr;                     // a batch window's logits, every row, before each session takes its own
    int32_t *io_d_ = nullptr, *tok_d_ = nullptr, *pos_d_ = nullptr;   // a call's tokens, positions, and segment table
    const bk::WindowSegs* at_d_ = nullptr;
    static constexpr int kGuessExperts = 8;     // the experts a window's guess requests a visit (rows x kGuessExperts / rows)
    static constexpr int kGuessRows = 2;        // the widest window guessed (wider ones keep the read-ahead)
};

// What the sessions of one model share: the forward scratch (with the layer split, the second card's too), the prompt path,
// the drafter's tap, the CPU experts and the batch windows' graphs. Made by the first session; the model keeps a weak
// reference (InklingModel::shared()).
class Session;

struct Shared {
    Scratch x, x1;                          // the first card's scratch, the second card's (the layer split)
    int max_rows = kWindowRows;
    uint64_t logits_gen = 0;             // widest forward call this model's sessions run
    float* hand = nullptr;                  // the layer split: page-locked slots of a call's rows crossing between the cards,
    int hand_slots = 0, tap_slot0 = 0;      // the crossings' first, then one per drafter tap
    std::unique_ptr<PromptRunner> prompt;   // PROMPT HOOK
    std::unique_ptr<CpuExperts> cpu;        // with a spill: the RAM experts' rows
    std::unique_ptr<DsparkBatch> ds_batch;  // DSPARK HOOK: multi-session batched commit & draft
    float* tap = nullptr;                   // DSPARK HOOK: the drafter's features of a call's rows (dspark_tap_bytes)
    model::BatchShared batch;
    std::vector<Session*> heat_sessions;    // with copies: the sessions whose kept rows rank them (a batch counts on its leader)
    std::vector<std::pair<int, void*>> bufs;   // (device, allocation)
    ~Shared() {
        ds_batch.reset();
        prompt.reset();
        cpu.reset();
        int dev = 0;
        cudaGetDevice(&dev);
        for (auto [d, p] : bufs) {
            cudaSetDevice(d);
            cudaFree(p);
        }
        cudaSetDevice(dev);
        if (hand) cudaFreeHost(hand);
    }
    bool init(InklingModel& m, cudaStream_t stream, int max_pulls, std::string& err) {
        max_rows = call_rows_of(m);
        if (!alloc_scratch(m, x, m.device(), true, err)) return false;
        if (m.cards() > 1) {
            cudaSetDevice(m.card_device(1));
            const bool ok = alloc_scratch(m, x1, m.card_device(1), false, err);
            cudaSetDevice(m.device());
            if (!ok) return false;
            tap_slot0 = max_pulls;
            hand_slots = max_pulls + kDsTargets;
            if (cudaHostAlloc((void**) &hand, (size_t) hand_slots * max_rows * m.geo().n_embd * 4, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot pin the layer split's hand-off rows";
                return false;
            }
        }
        if (!(cpu = CpuExperts::make(m, err)) && !err.empty()) return false;
        if (m.sessions() > 1 && m.dspark()) {
            ds_batch = std::make_unique<DsparkBatch>();
            if (!ds_batch->init(m.dspark(), m.sessions(), max_rows, std::max(max_rows, kPromptChunk), tap, m.context(), err))
                return false;
        }
        prompt = make_prompt(m, stream, err);   // PROMPT HOOK
        return prompt != nullptr || err.empty();
    }
    // One card's scratch on the current device; the first card's also the drafter's tap.
    bool alloc_scratch(InklingModel& m, Scratch& x, int dev, bool first, std::string& err) {
        const Geometry& g = m.geo();
        bool ok = true;
        const auto alloc = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
                ok = false;
                return nullptr;
            }
            bufs.emplace_back(dev, p);
            return p;
        };
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4); };
        const int64_t E = g.n_embd, R = max_rows;
        const int64_t ctx = m.context();
        // the row buffers of one call, by their roles in the walk
        for (float*& b : x.role_) b = f(R * E);
        x.emb_ = f(R * E);
        x.mix_ = f(3 * R * E);
        x.msg_ = f((int64_t) (g.loop_end - g.loop_start) * kStreams * R * E);
        x.ctrl_scr_ = f((int64_t) ctrl_scratch_floats(g.gate_rank));
        x.nrm_ = f(R * E);
        x.y_ = f(R * kProj);
        x.q_ = f(R * kHeads * kHeadDim); x.r_ = f(R * kHeads * kRel); x.sink_ = f(R * kHeads);
        x.o_ = f(R * E); x.resid_ = f(R * E); x.fin_ = f(R * E);
        x.att_scr_ = alloc(attention_decode_scratch(max_rows, std::max<int64_t>(ctx, g.window)));
        for (void*& q : x.xq_) q = alloc(bk::native_q8_1_bytes((int) E, max_rows));
        x.xo_ = alloc(bk::native_q8_1_bytes(kOutRows, max_rows));
        x.rlog_ = f(R * g.n_expert); x.rw_ = f(R * g.n_used); x.ids_ = (int32_t*) alloc((size_t) R * g.n_used * 4);
        const int64_t pairs = R * g.n_used;
        x.ex_ptr_ = (unsigned long long*) alloc((size_t) pairs * 8);
        x.ex_start_ = (int32_t*) alloc((size_t) (pairs + 2) * 4);
        x.ex_ng_ = (int32_t*) alloc(16);
        x.ex_dst_ = (int32_t*) alloc((size_t) pairs * 4);
        x.ex_tok_ = (int32_t*) alloc((size_t) pairs * 4);
        x.ex_scr_ = alloc(bk::native_expert_scratch_bytes(pairs, g.n_ff_exp));
        x.ex_rows_ = f(pairs * E);
        x.nrm1_ = f((int64_t) g.visits.size() * Scratch::kGuessRows * E);
        x.gs_ids_ = (int32_t*) alloc((size_t) g.visits.size() * Scratch::kGuessExperts * 4);
        x.gs_log_ = f((int64_t) Scratch::kGuessRows * g.n_expert);
        x.gs_w_ = f(Scratch::kGuessExperts);
        x.pf_sink_ = (uint32_t*) alloc(4);
        x.lp_ = f((int64_t) bk::row_logprobs_scratch_floats(max_rows, g.n_vocab) + R * (bk::kLogprobStride + 1));
        x.blog_ = f(R * (int64_t) g.n_vocab);
        x.io_d_ = (int32_t*) alloc((size_t) (2 * R + kSegWords) * 4);
        x.tok_d_ = x.io_d_;
        x.pos_d_ = x.io_d_ + R;
        x.at_d_ = reinterpret_cast<const bk::WindowSegs*>(x.io_d_ + 2 * R);
        if (first && m.dspark()) tap = (float*) alloc(dspark_tap_bytes(std::max(max_rows, kPromptChunk)));   // DSPARK HOOK
        if (!ok) {
            cudaGetLastError();
            err = "out of device memory for the sessions' scratch";
            return false;
        }
        return true;
    }
};

// One card's side of a session's forward: its streams and events and its running counts of the experts' lookups. The first
// card's are the session's own members (stream_, ...); the layer split's second card has its own.
struct Lane {
    int device = 0;
    const Scratch* x = nullptr;
    cudaStream_t stream = nullptr, ra_st = nullptr, pf_st = nullptr;
    cudaEvent_t ra_ev = nullptr, ra_join = nullptr, gs_ev = nullptr, pf_ev = nullptr, pf_join = nullptr;
    cudaEvent_t hand = nullptr;                 // behind the card's half of a crossing
    uint32_t* ex_tally = nullptr;               // [layer][expert] lookups of the card's layers
    unsigned long long* rec_blobs = nullptr;    // [visit][kAheadExperts] the experts each visit took in the last window
    int32_t* rec_n = nullptr;                   // [visit] how many
    bool ra_on = false, pf_on = false;
    int32_t* win_ids = nullptr;                 // [visit][kMaxRows][n_used] the last window's routed ids (with copies)
};

class Session final : public model::BatchSession, private Scratch {
public:
    explicit Session(InklingModel& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        if (sh_) std::erase(sh_->heat_sessions, this);
        if (stream_) cudaStreamSynchronize(stream_);
        if (lane_[1].stream) cudaStreamSynchronize(lane_[1].stream);
        if (!m_.wedged() && m_.has_spill() && ex_tally_) {   // how decode's lookups fell between the card and RAM
            std::vector<uint32_t> t;
            if (read_tally(t)) {
                uint64_t loan = 0;
                const CardShare sh = lookup_share(t, loan);
                const uint64_t all = sh.all, card = sh.card;
                if (m_.peer_device() >= 0)
                    basalt::log::info("basalt inkling2", "experts: %llu decode lookups, %.2f%% on the card, %llu on GPU %d, %llu on the CPU",
                                      (unsigned long long) all, all ? 100.0 * (double) card / (double) all : 100.0,
                                      (unsigned long long) sh.peer, m_.peer_device(), (unsigned long long) (all - card - sh.peer));
                else
                    basalt::log::info("basalt inkling2", "experts: %llu decode lookups, %.2f%% on the card, %llu on the CPU",
                                      (unsigned long long) all, all ? 100.0 * (double) card / (double) all : 100.0,
                                      (unsigned long long) (all - card));
                if (loan > 0)
                    basalt::log::info("basalt inkling2", "experts: %llu of the card's decode lookups went to loaned copies",
                                      (unsigned long long) loan);
            }
        }
        for (auto [d, p] : bufs_) {
            cudaSetDevice(d);
            cudaFree(p);
        }
        for (void* p : pinned_) cudaFreeHost(p);
        if (Lane& k = lane_[1]; k.stream) {
            cudaSetDevice(k.device);
            for (cudaEvent_t e : {k.ra_ev, k.ra_join, k.gs_ev, k.pf_ev, k.pf_join, k.hand, io_ev1_})
                if (e) cudaEventDestroy(e);
            for (cudaStream_t st : {k.ra_st, k.pf_st, k.stream})
                if (st) cudaStreamDestroy(st);
        }
        cudaSetDevice(m_.device());
        for (cudaEvent_t e : {hand0_, io_ev0_})
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : stage_ev_)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : {ra_ev_, ra_join_, gs_ev_, pf_ev_, pf_join_})
            if (e) cudaEventDestroy(e);
        if (ra_st_) cudaStreamDestroy(ra_st_);
        if (pf_st_) cudaStreamDestroy(pf_st_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&ra_st_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&pf_st_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ra_ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ra_join_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&gs_ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&pf_ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&pf_join_, cudaEventDisableTiming) != cudaSuccess) {
            err = "cannot create a stream";
            return false;
        }
        if (!walk(g_, walk_, err)) return false;
        {   // the runs of visits on each card (one run, the whole walk, on one card)
            std::vector<int> card_of((size_t) g_.n_layers);
            for (int l = 0; l < g_.n_layers; ++l) card_of[(size_t) l] = m_.card_of(l);
            segs_ = split_walk(g_, walk_, card_of);
        }
        if (m_.cards() > 1) {
            Lane& k = lane_[1];
            k.device = m_.card_device(1);
            cudaSetDevice(k.device);
            bool sok = cudaStreamCreateWithFlags(&k.stream, cudaStreamNonBlocking) == cudaSuccess &&
                       cudaStreamCreateWithFlags(&k.ra_st, cudaStreamNonBlocking) == cudaSuccess &&
                       cudaStreamCreateWithFlags(&k.pf_st, cudaStreamNonBlocking) == cudaSuccess;
            for (cudaEvent_t* e : {&k.ra_ev, &k.ra_join, &k.gs_ev, &k.pf_ev, &k.pf_join, &k.hand, &io_ev1_})
                sok = sok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
            cudaSetDevice(m_.device());
            for (cudaEvent_t* e : {&hand0_, &io_ev0_}) sok = sok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
            if (!sok) {
                err = "cannot create the second card's streams";
                return false;
            }
        }
        bool ok = true;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        const int64_t E = g_.n_embd;
        const int call_rows = call_rows_of(m_);
        const size_t row_bytes = (size_t) kv_cache_row_bytes(m_.cache_fp8(), m_.cache_int8());
        const int64_t ctx = m_.context();
        {
            auto align256 = [](size_t b) { return (std::max<size_t>(b, 16) + 255) & ~(size_t) 255; };
            size_t need[2] = {0, 0};
            for (const Visit& v : g_.visits) {
                const int c = m_.card_of(v.layer);
                const bool ring = g_.swa[(size_t) v.layer];
                const int64_t cap = ring ? (int64_t) g_.window + kWindowRows : ctx;
                if (ring || !m_.elastic()) need[c] += 2 * align256((size_t) kKvHeads * (size_t) cap * row_bytes);
                need[c] += 2 * align256((size_t) kConvRing * kKvHeads * kHeadDim * 4);
                need[c] += 2 * align256((size_t) kConvRing * (size_t) E * 4);
            }
            const size_t tally_b = align256((size_t) expert_pairs() * 4) +
                                   align256(g_.visits.size() * kAheadExperts * 8) +
                                   align256(g_.visits.size() * 4);
            need[0] += tally_b + align256(4 * 32);
            if (m_.copies())
                need[0] += align256((size_t) expert_pairs() * 4) +
                           align256(g_.visits.size() * (size_t) call_rows * g_.n_used * 4) +
                           align256(g_.visits.size() * 4);
            if (lane_[1].stream) need[1] += tally_b;
            for (int c = 0; c < 2; ++c) {
                if (!need[c]) continue;
                const int dev = c == 0 ? m_.device() : lane_[1].device;
                cudaSetDevice(dev);
                void* slab = nullptr;
                if (cudaMalloc(&slab, need[c]) != cudaSuccess) { ok = false; break; }
                bufs_.emplace_back(dev, slab);
                arena_dev_[c] = dev;
                arena_ptr_[c] = static_cast<uint8_t*>(slab);
                arena_rem_[c] = need[c];
            }
            cudaSetDevice(m_.device());
        }
        int max_slot = 0;
        for (const Visit& v : g_.visits) max_slot = std::max(max_slot, v.slot);
        vc_.assign((size_t) max_slot + 1, VisitCache{});
        slot_card_.assign(vc_.size(), 0);
        for (const Visit& v : g_.visits) {
            VisitCache& c = vc_[(size_t) v.slot];
            slot_card_[(size_t) v.slot] = m_.card_of(v.layer);
            cudaSetDevice(m_.card_device(m_.card_of(v.layer)));   // a visit's caches on its layer's card
            c.ring = g_.swa[(size_t) v.layer];
            c.cap = c.ring ? (int64_t) g_.window + kWindowRows : ctx;
            const size_t bytes = (size_t) kKvHeads * (size_t) c.cap * row_bytes;
            if (!c.ring && m_.elastic()) {   // the elastic KV: the range reserved, its first kElasticKvBase cells backed
                for (int j = 0; j < 2; ++j) {
                    auto a = std::make_unique<core::ElasticArray>();
                    if (ok && (!a->open(bytes, err) || !a->ensure(cell_bytes() * (uint64_t) core::kElasticKvBase, 0, err))) ok = false;
                    (j ? c.v : c.k) = a->data();
                    elastic_.push_back(std::move(a));
                }
            } else {
                c.k = alloc(bytes, ok);
                c.v = alloc(bytes, ok);
            }
            c.k_ring = f((int64_t) kConvRing * kKvHeads * kHeadDim);
            c.v_ring = f((int64_t) kConvRing * kKvHeads * kHeadDim);
            c.a_ring = f((int64_t) kConvRing * E);
            c.m_ring = f((int64_t) kConvRing * E);
        }
        if (Lane& k = lane_[1]; k.stream) {   // the second card's running counts
            cudaSetDevice(k.device);
            k.ex_tally = (uint32_t*) alloc((size_t) expert_pairs() * 4, ok);
            k.rec_blobs = (unsigned long long*) alloc((size_t) g_.visits.size() * kAheadExperts * 8, ok);
            k.rec_n = (int32_t*) alloc((size_t) g_.visits.size() * 4, ok);
            if (ok) cudaMemset(k.ex_tally, 0, (size_t) expert_pairs() * 4);
            if (ok) cudaMemset(k.rec_n, 0, (size_t) g_.visits.size() * 4);
            for (const VisitCache& c : vc_)
                if (ok && slot_card_[(size_t) (&c - vc_.data())] == 1) {
                    cudaMemset(c.k_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4);
                    cudaMemset(c.v_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4);
                    cudaMemset(c.a_ring, 0, (size_t) kConvRing * E * 4);
                    cudaMemset(c.m_ring, 0, (size_t) kConvRing * E * 4);
                }
            cudaDeviceSynchronize();
        }
        cudaSetDevice(m_.device());
        ex_tally_ = (uint32_t*) alloc((size_t) expert_pairs() * 4, ok);
        if (m_.copies()) {   // the kept rows' lookups that rank the copies (decode_kept)
            heat_tally_ = (uint32_t*) alloc((size_t) expert_pairs() * 4, ok);
            win_ids_ = (int32_t*) alloc(g_.visits.size() * (size_t) call_rows * g_.n_used * 4, ok);
            visit_layer_ = (int32_t*) alloc(g_.visits.size() * 4, ok);
            std::vector<int32_t> layer;
            for (const Visit& v : g_.visits) layer.push_back(v.layer);
            if (ok && basalt::core::device_copy(visit_layer_, layer.data(), layer.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
        }
        rec_blobs_ = (unsigned long long*) alloc((size_t) g_.visits.size() * kAheadExperts * 8, ok);
        rec_n_ = (int32_t*) alloc((size_t) g_.visits.size() * 4, ok);
        pick_ = (int32_t*) alloc(4 * 32, ok);
        if (ok && cudaMallocHost((void**) &pick_h_, 4 * 32) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(pick_h_);
        if (ok && cudaMallocHost((void**) &io_h_, (size_t) kStage * (2 * call_rows + kSegWords) * 4) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(io_h_);
        for (cudaEvent_t& e : stage_ev_)
            if (ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) ok = false;
        arena_rem_[0] = arena_rem_[1] = 0;
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        // the scratch every session of the model shares: the first session makes it
        sh_ = std::static_pointer_cast<Shared>(m_.shared().lock());
        if (!sh_) {
            size_t pulls = 0;   // the most row buffers one crossing takes
            for (const Segment& sg : segs_) pulls = std::max(pulls, sg.pull.size());
            sh_ = std::make_shared<Shared>();
            if (!sh_->init(m_, stream_, (int) pulls, err)) return false;
            m_.shared() = sh_;
        }
        static_cast<Scratch&>(*this) = sh_->x;
        lane_[0] = Lane{m_.device(), &sh_->x, stream_, ra_st_, pf_st_, ra_ev_, ra_join_, gs_ev_, pf_ev_, pf_join_, hand0_, ex_tally_,
                        rec_blobs_, rec_n_};
        lane_[0].win_ids = win_ids_;
        lane_[1].x = &sh_->x1;
        for (const VisitCache& c : vc_) {
            if (slot_card_[(size_t) (&c - vc_.data())] != 0) continue;
            cudaMemsetAsync(c.k_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.v_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.a_ring, 0, (size_t) kConvRing * E * 4, stream_);
            cudaMemsetAsync(c.m_ring, 0, (size_t) kConvRing * E * 4, stream_);
        }
        if (ok) cudaMemsetAsync(ex_tally_, 0, (size_t) expert_pairs() * 4, stream_);
        if (ok && heat_tally_) cudaMemsetAsync(heat_tally_, 0, (size_t) expert_pairs() * 4, stream_);
        if (ok) cudaMemsetAsync(rec_n_, 0, (size_t) g_.visits.size() * 4, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the session stream failed during initialization";
            return false;
        }
        if (!dspark_.init(m_.dspark(), g_, m_.context(), std::max(call_rows, kPromptChunk), sh_->tap, m_.flags().diag_dump,   // DSPARK HOOK
                          elastic_.empty() ? 0 : core::kElasticKvBase, err))
            return false;
        count_backed();
        if (heat_tally_) sh_->heat_sessions.push_back(this);
        return true;
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "inkling2: an empty window"};
        if (stale_) return stale();
        if (m_.wedged()) return wedged();
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the window runs past the context"};
        std::string err;
        const int64_t loaned_before = m_.loaned();
        if (heat_tally_) settle_window(r.pos);
        if (!ensure_positions(r.pos + r.n_tokens, err)) return grow_failed(err);
        if (heat_tally_) {   // the placement changes due at this position, before any row at or past it runs
            if (rebalances() && next_rebalance_ >= 0 && r.pos >= next_rebalance_) rebalance_copies(r.pos, false);
            if (m_.loan_pending() && r.pos >= m_.loan_publish_at() && !publish_loan(r.pos)) return wedged();
            if (m_.wedged()) return wedged();
        }
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        hist_.resize((size_t) r.pos);
        for (int t0 = 0; t0 < r.n_tokens; t0 += kWindowRows) {
            const int n = std::min(kWindowRows, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (t0 > 0) fold_rows(win_rows_);   // a window wider than one call: the rows before its last call are prompt rows
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, head, err)) {
                if (!m_.wedged() && m_.loaned() != loaned_before) {
                    m_.wedge();
                    basalt::log::error("basalt inkling2", "a window failed after publishing prompt loan copies; restart the engine");
                }
                return {model::Status::Error, "inkling2: " + err};
            }
            if (heat_tally_) {
                win_pos_ = r.pos;
                win_off_ = t0;
                win_rows_ = n;
            }
            lookups_ += (int64_t) n * g_.n_used * (int64_t) g_.visits.size();
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            head_rows_ = head;
            logits_ = head > 0 ? blog_ : nullptr;
            if (head > 0) logits_gen_ = ++sh_->logits_gen;
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "inkling2: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits)
                cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        if (const model::Result done = end_call("the window failed"); !done.ok()) return done;
        if (heat_tally_ && !m_.submit_loan(r.pos + kLoanLagRows)) return wedged();
        return {};
    }

    // The rows the last window keeps rank the copies; rejected drafts never count, so the ranking follows the token stream.
    model::Result decode_kept(model::ModelSession* const* sessions, const int* kept, int n) override {
        for (int i = 0; i < n; ++i) {
            Session& s = *static_cast<Session*>(sessions[i]);
            if (s.win_rows_ > 0) s.fold_rows(std::clamp(kept[i] - s.win_off_, 0, s.win_rows_));
            for (const BatchSeg& b : batch_segs_)   // a batch's: on this session, its leader
                if (b.s == &s) fold_rows(std::clamp(kept[i], 0, b.rows), nullptr, b.row0);
        }
        batch_segs_.clear();
        return {};
    }

    // A window never crosses a position where the placement changes (a rebalance, the loan's publication, a cache grow that
    // can send copies back), so every row runs under the placement its position fixes, whatever the windows' widths.
    int window_limit(int64_t pos) const override {
        if (!heat_tally_ || sh_.use_count() > 1) return 0;
        int64_t room = INT64_MAX;
        const auto bound = [&](int64_t at) {
            if (at > pos) room = std::min(room, at - pos);
        };
        if (rebalances() && next_rebalance_ >= 0) bound(next_rebalance_after(pos));
        if (m_.loan_pending()) bound(m_.loan_publish_at());
        if (!elastic_.empty() && m_.promoted() > 0 && backed_ < m_.context())
            bound(backed_ > pos ? backed_ : (pos / kElasticKvStep + 1) * kElasticKvStep);
        return room == INT64_MAX ? 0 : (int) std::min<int64_t>(room, kWindowRows);
    }

    // ---- batch windows (docs/design/batching.md): several sessions' windows in one pass through the weights, on this
    // session's scratch and stream. Each conversation keeps its caches and rings (the kernels take them per segment), its
    // drafter's context, its logits and its draws; a row gives the bits it gives in its own window.
    int batch_rows() const override { return sh_->max_rows; }

protected:
    // the hooks of the shared driver (model/batch_driver.hpp)
    const char* batch_name() const override { return "inkling2"; }
    int64_t batch_context() const override { return m_.context(); }
    model::BatchShared& batch_shared() override { return sh_->batch; }
    cudaStream_t batch_stream() const override { return stream_; }
    bool batch_ready() const override { return !m_.wedged(); }
    bool batch_key_per_seg() const override { return false; }
    bool batch_reserve(const model::DecodeRequest& r, std::string& err) override {
        if (stale_) {
            err = stale().message;
            return false;
        }
        return ensure_positions(r.pos + r.n_tokens, err);
    }
    bool batch_stage(const model::BatchView& v, model::GraphLru<core::CapturedGraph>::Key& key, std::string& err) override {
        if (!admit(err)) return false;
        Admitted held{*this};   // passed to the driver (batch_end) once the stage succeeds
        if (!drop_stale_read_ahead(err)) return false;
        const bk::WindowSegs wsegs = v.w.device_segs();
        int max_seg_rows = 1;
        for (int i = 0; i < wsegs.n; ++i) max_seg_rows = std::max(max_seg_rows, wsegs.row0[i + 1] - wsegs.row0[i]);
        key.push_back((uint64_t) stage(v.w.tokens.data(), nullptr, v.w.rows, 0, v.w.positions.data(), &wsegs));
        key.push_back((uint64_t) (max_seg_rows * (kHeads / kKvHeads) > 16));
        key.push_back(cpu_ = cpu_layers());
        if (cpu_) sh_->cpu->wake();
        if (!check(err)) return false;
        held.passed = true;
        return true;
    }
    bool batch_end(std::optional<cudaError_t> synced) override {
        if (synced) return complete(*synced);
        release();
        return false;
    }
    bool batch_forward(const model::BatchView& v, std::string&) override {
        ++sh_->logits_gen;
        view_ = &v;
        body(v.w.rows, v.w.rows, full_keys_of(v.w.max_pos, v.w.rows));
        view_ = nullptr;
        return true;
    }
    void batch_take(const model::BatchView& v, int i) override {
        const model::DecodeRequest& r = v.reqs[i];
        const model::BatchWindow::Seg& sg = v.w.segs[(size_t) i];
        const int h = v.head[i];
        const int64_t V = g_.n_vocab;
        hist_.resize((size_t) r.pos);
        hist_.insert(hist_.end(), r.tokens, r.tokens + r.n_tokens);
        dspark_.note(r.pos, r.n_tokens);   // DSPARK HOOK
        lookups_ += (int64_t) r.n_tokens * g_.n_used * (int64_t) g_.visits.size();
        if (heat_tally_) {   // this session's rows of the batch, in its leader's routed ids, wait for decode_kept
            Session& lead = *static_cast<Session*>(v.ss[0]);
            if (i == 0) lead.batch_segs_.clear();
            lead.batch_segs_.push_back({this, sg.row0, sg.rows});
        }
        head_rows_ = h;
        if (h == 0) { logits_ = nullptr; return; }
        logits_ = blog_ + (size_t) (sg.row0 + sg.rows - h) * V;
        logits_gen_ = sh_->logits_gen;
        if (r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (h - 1) * V, (size_t) V * 4, cudaMemcpyDeviceToHost, v.stream);
    }
    const int32_t* batch_draw(const model::BatchView& v, int i, std::string& err) override {
        return draw(v.reqs[i], 0, v.head[i], err, v.stream);
    }

public:
    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (stale_ && r.pos > 0) return stale();
        if (m_.wedged()) return wedged();
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the prompt runs past the context"};
        const bool staged = sh_->prompt && r.n_tokens > (m_.has_spill() ? kSpillWindowRead : kChunkRead);
        if (heat_tally_) settle_window(r.pos);
        if (!restart_positions(r.pos == 0 ? r.n_tokens : r.pos + std::max(r.n_tokens, r.total_tokens), r.pos)) return wedged();
        if (std::string e; !ensure_positions(r.pos + r.n_tokens, e)) return grow_failed(e);
        // a request's start: the copies follow the lookups so far (a staged read takes the loan's memory back first)
        if (heat_tally_ && !staged) rebalance_copies(r.pos + r.n_tokens, true);
        if (m_.wedged()) return wedged();
        hist_.resize((size_t) r.pos);
        stale_ = true;   // until the rows are in (an error or an exception leaves it set)
        if (staged) {   // PROMPT HOOK; a failed read's rows are no prefix to go on from
            PromptRunner::Prepare prepare;
            if (heat_tally_)
                prepare = [&] {
                    if (!recall_loan(r.pos)) return false;
                    rebalance_copies(r.pos + r.n_tokens, true);
                    return !m_.wedged();
                };
            const model::Result res = sh_->prompt->read(r, hist_, vc_, dspark_, stream_, prepare);
            stale_ = !res.ok();
            if (!res.ok()) hist_.resize((size_t) r.pos);
            return res;
        }
        std::string err;
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        for (int t0 = 0; t0 < r.n_tokens; t0 += kWindowRows) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kWindowRows, r.n_tokens - t0);
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, 0, err)) return {model::Status::Error, "inkling2: " + err};
            if (heat_tally_) fold_rows(n);
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            if (r.progress && ((t0 / kWindowRows) % 64 == 63 || t0 + n == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        const model::Result res = end_call("the prompt failed");
        if (res.ok()) stale_ = false;
        return res;
    }

    // ---- [vision] images (vision.hpp, vision_encoder.hpp): a picture's rows stand in for its pad tokens' embeddings, taken
    // as they are (no embedding norm); the history holds the pad token at their positions.
    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        if (m_.wedged()) return wedged();
        ImageEncoder& enc = m_.encoder();
        if (!enc.staged()) return {model::Status::NotImplemented, "inkling2: started without --vision"};
        std::string err;
        if (!feed_heat()) return wedged();   // the encoder's room sends the lowest-ranked copies back
        if (!enc.encode(data, bytes, rows, n_rows, grid, err)) return {model::Status::Error, "inkling2: " + err};
        if (n_rows <= 0 || rows.size() != (size_t) n_rows * (size_t) g_.n_embd)
            return {model::Status::Error, "inkling2: the projector's rows are not the model's width"};
        return {};
    }
    void images_done() override {
        for (Session* o : sh_->heat_sessions) o->settle_hits();   // the refill places copies again
        m_.encoder().release();
    }
    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid) override {
        if (stale_ && pos > 0) return stale();
        if (m_.wedged()) return wedged();
        if (pos + n_rows > m_.context()) return {model::Status::Refused, "inkling2: the picture runs past the context"};
        if (heat_tally_) settle_window(pos);
        if (std::string e; !ensure_positions(pos + n_rows, e)) return grow_failed(e);
        hist_.resize((size_t) pos);
        stale_ = true;
        if (sh_->prompt && n_rows > kChunkRead) {
            PromptRunner::Prepare prepare;
            if (heat_tally_) prepare = [&] { return recall_loan(pos); };
            const model::Result res = sh_->prompt->read_rows(rows, n_rows, pos, g_.image_token, hist_, vc_, dspark_, stream_, prepare);
            stale_ = !res.ok();
            if (!res.ok()) hist_.resize((size_t) pos);
            return res;
        }
        std::string err;
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        for (int at = 0; at < n_rows; at += kWindowRows) {
            const int n = std::min(kWindowRows, n_rows - at);
            if (!forward(nullptr, rows + (size_t) at * g_.n_embd, n, pos + at, 0, err)) return {model::Status::Error, "inkling2: " + err};
            if (heat_tally_) fold_rows(n);
            for (int i = 0; i < n; ++i) hist_.push_back(g_.image_token);
        }
        const model::Result res = end_call("the image rows failed");
        if (res.ok()) stale_ = false;
        return res;
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "inkling2: " + e};
        return {};
    }

    // a draw of one row again, with the tokens the mask bans
    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (!logits_ || logits_gen_ != sh_->logits_gen || row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "inkling2: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                          history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (!logits_ || logits_gen_ != sh_->logits_gen || row0 < 0 || n < 1 || row0 + n > head_rows_) return {model::Status::Error, "inkling2: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2: the log probabilities failed"};
        return {};
    }

    // SLEEP / WAKE: the card's copies of the RAM experts go back to RAM (decode reads them there) and return at the wake; the
    // driver moves the rest. A model that kept every expert on the card has nothing to lend: ok, no slots, SLEEP goes on.
    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        if (m_.wedged()) return wedged();
        if (!m_.copies()) return {};
        if (cudaStreamSynchronize(stream_) != cudaSuccess || !feed_heat()) return wedged();
        if (low && !recall_loan(-1)) return wedged();
        if (!m_.lend(low))
            return {model::Status::Error, low ? "inkling2: the expert copies could not go back to RAM"
                                              : "inkling2: the expert copies could not come back to the card"};
        expert_slots = expert_pairs() - m_.spill().spilled + m_.promoted();
        return {};
    }

    // VRAM: the copies resize to leave `mib` free (< 0: the start's --vram-reserve-mib). A card that holds every expert has
    // no copies to resize: refused unless it already leaves that much.
    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        if (m_.wedged()) return wedged();
        if (cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2: the session stream failed before resizing VRAM"};
        if (!feed_heat()) return wedged();
        const bool fits = m_.resize_reserve((uint64_t) (mib < 0 ? std::max(m_.flags().vram_reserve_mib, 0) : mib) << 20);
        expert_slots = m_.copies() ? expert_pairs() - m_.spill().spilled + m_.promoted() : -1;
        if (!fits) return {model::Status::Error, "inkling2: the card cannot leave " + std::to_string(mib) + " MiB free: the experts it holds are not copies, or the copies could not go back"};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (!logits_ || logits_gen_ != sh_->logits_gen || row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: no logits for that row"};
        if (basalt::core::device_copy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "inkling2: copying the logits failed"};
        return {};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    // DSPARK HOOK: the drafter's windows (the anchor and up to seven drafts) and blocks
    int max_window() const override { return dspark_.on() ? std::clamp(m_.flags().spec, 1, kWindowRows) : 1; }
    bool window_cold() const override { return window_cold_; }
    int prompt_fold_window() const override { return kWindowRows; }
    bool copy_drafts() const override { return max_window() > 1; }   // prompt lookup: any window's rows verify alike
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (m_.wedged()) return wedged();
        return dspark_.draft_begin(anchor, pos, n, stream_);
    }
    model::Result draft_batch(model::ModelSession* const* peers, const int32_t* anchors, const int64_t* pos, const int* n,
                              int count) override {
        if (m_.wedged()) return wedged();
        if (!sh_->ds_batch || !sh_->ds_batch->on() || count <= 1 || count > bk::kWindowSegsMax)
            return ModelSession::draft_batch(peers, anchors, pos, n, count);
        DsparkHook* hooks[bk::kWindowSegsMax] = {};
        for (int i = 0; i < count; ++i) {
            Session* s = dynamic_cast<Session*>(peers[i]);
            if (!s || s->sh_ != sh_) return ModelSession::draft_batch(peers, anchors, pos, n, count);
            hooks[i] = &s->dspark_;
        }
        return sh_->ds_batch->draft(hooks, anchors, pos, n, count, stream_);
    }
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override {
        if (m_.wedged()) return wedged();
        return dspark_.draft_end(ids, conf, max_n, n, need, stream_);
    }

    // ---- conversations: the full-attention visits' cells by position (save_cells / restore_cells), the window rings and
    // the convolutions' rings by checkpoint (running_state). A window ring holds window + kWindowRows positions and a
    // convolution ring kConvRing, so a checkpoint at `pos` is intact while the session has not run more than kWindowRows
    // rows past it.
    size_t row_bytes() const { return (size_t) kv_cache_row_bytes(m_.cache_fp8(), m_.cache_int8()); }
    static model::Result stale() {
        return {model::Status::Error, "inkling2: a failed read left the conversation's state unknown: start it over or restore a checkpoint"};
    }
    uint64_t cell_bytes() const { return (uint64_t) kKvHeads * row_bytes(); }

    static model::Result wedged() { return {model::Status::Error, "inkling2: the model is wedged; restart the engine"}; }

    // False once a cache's backing could not be confirmed (a clear or a release): the model stops then.
    bool caches_usable() {
        bool ok = dspark_.usable();   // DSPARK HOOK
        for (const auto& a : elastic_) ok = ok && a->usable();
        if (!ok && !m_.wedged()) {
            m_.wedge();
            basalt::log::error("basalt inkling2", "a cache's device memory could not be confirmed: every later request is refused");
        }
        return ok;
    }

    // The positions every full-attention cache backs: the visits' and the drafter's full layers'.
    void count_backed() {
        backed_ = m_.context();
        for (const auto& a : elastic_) backed_ = std::min(backed_, (int64_t) (a->backed() / cell_bytes()));
        if (dspark_.on() && !elastic_.empty()) backed_ = std::min(backed_, std::max<int64_t>(0, dspark_.backed_cells() - kDsBlock));
    }

    // The elastic KV: before a run reaches `positions`, the full visits' caches and the drafter's full layers back them (in
    // steps of kElasticKvStep cells), and when the card has no room the RAM experts' card copies go back to RAM first, the
    // lowest ranked first. False when even without any copy there is no room.
    static constexpr int64_t kElasticKvStep = 8192;
    static constexpr uint64_t kElasticKeep = 512ull << 20;   // free bytes a grow leaves (the driver, the graphs)
    // The copies' placement changes at positions the token stream fixes (window_limit): a rebalance every kRebalanceRows
    // positions of a turn, the prompt loan kLoanLagRows positions after the turn's first window (its uploads overlap the
    // windows before).
    static constexpr int kRebalanceSwaps = 64;
    static constexpr int64_t kRebalanceRows = 256;
    static constexpr int64_t kLoanLagRows = 32;

    // The decode read-ahead's saved expert addresses (rec_blobs_, written by the window before) are dropped once a copy has
    // moved or gone since: their memory may hold another expert or be unmapped. The epoch counts as seen once the drop is
    // queued.
    bool drop_stale_read_ahead(std::string& err) {
        const uint64_t epoch = m_.placement_epoch();
        if (seen_epoch_ == epoch) return true;
        if (cudaMemsetAsync(rec_n_, 0, (size_t) g_.visits.size() * 4, stream_) != cudaSuccess) {
            err = "inkling2: the read-ahead records could not be dropped";
            return false;
        }
        seen_epoch_ = epoch;
        return true;
    }
    // A refused grow: request-local while the model goes on (no room even with every copy back), the end of the engine when
    // it stopped (a shortage of counted memory, an unconfirmed step).
    model::Result grow_failed(const std::string& err) const {
        return {m_.wedged() ? model::Status::Error : model::Status::Refused, "inkling2: " + err};
    }
    bool ensure_positions(int64_t positions, std::string& err) {
        if (elastic_.empty() || positions <= backed_) return true;
        const int64_t target = std::min<int64_t>(m_.context(), (positions + kElasticKvStep - 1) / kElasticKvStep * kElasticKvStep);
        const uint64_t bytes = cell_bytes() * (uint64_t) target;
        uint64_t grow = grow_bytes(target);
        const uint64_t keep_free = m_.held_free(kElasticKeep);
        const int64_t at = backed_;
        // the copies go back until the grow leaves the reserve free by the counted room (the caches take the reserve itself
        // only once no copy is left: the reserve is the copies' target, not a floor the caches stop at)
        const int64_t before = m_.card_copies();
        const auto make_room = [&] {   // false: a demote failed (the model is wedged then)
            if (give_back(grow + keep_free, [&] { return m_.room(); }, [&] { return m_.promoted(); },
                          [&](uint64_t b, uint64_t& freed) { return feed_heat() && m_.demote(b, freed); }))
                return true;
            err = "the caches cannot grow: the expert copies could not go back to RAM";
            return false;
        };
        const auto log_grow = [&](bool ok) {
            if (m_.card_copies() != before || m_.wedged())
                m_.log_copy_change(((ok ? "cache grow to " : "cache grow failed for ") + std::to_string(target) + " positions at " +
                                    std::to_string(at)).c_str(), before, grow);
        };
        if (!make_room()) {
            log_grow(false);
            return false;
        }
        uint64_t backing0 = backing_bytes();
        bool out_of_memory = false;   // the failure was the device lacking the memory, nothing left half done
        const auto map = [&] {
            for (auto& a : elastic_)
                if (!a->ensure(bytes, 0, err)) {
                    out_of_memory = a->out_of_memory();
                    return false;
                }
            if (dspark_.ensure(target, err)) return true;   // DSPARK HOOK
            out_of_memory = dspark_.out_of_memory();
            return false;
        };
        bool mapped = map();
        if (!mapped && out_of_memory && caches_usable()) {
            // The card has less free memory than placement counted (another program, or graphs and modules past the spare
            // took it). As without the count: read the card again, send copies back for the grow and try once more. From
            // here placement depends on that reading, which the line says.
            basalt::log::warn("basalt inkling2", "elastic spill: placement changed by a memory shortage at position %lld (%s)",
                              (long long) at, err.c_str());
            m_.sync_room();
            backing0 = backing_bytes();   // the reading includes what the first try mapped
            grow = grow_bytes(target);    // and the retry needs only what it left
            if (make_room()) {
                err.clear();
                mapped = map();
            }
        }
        m_.account((int64_t) backing_bytes() - (int64_t) backing0);
        log_grow(mapped);
        count_backed();
        if (!mapped) {   // no room even with every copy back: this request is refused, as when the context does not fit
            caches_usable();
            err = "the caches cannot grow to " + std::to_string(target) + " positions: " + err;
            return false;
        }
        if (positions <= backed_) return true;
        err = "the caches do not back the requested positions";
        return false;
    }
    // The device bytes the elastic caches (the visits' and the drafter's) back now.
    uint64_t backing_bytes() const {
        uint64_t n = dspark_.backing_bytes();   // DSPARK HOOK
        for (const auto& a : elastic_) n += a->backed();
        return n;
    }
    // The device bytes the caches map to back `positions` positions (ensure_positions), the allocation rounding included.
    uint64_t grow_bytes(int64_t positions) const {
        uint64_t n = dspark_.grow_bytes(positions);   // DSPARK HOOK
        for (const auto& a : elastic_) n += a->growth(cell_bytes() * (uint64_t) positions);
        return n;
    }

    // ---- the copies' ranking: the lookups of the rows the windows keep (decode_kept), and of prompt rows read through the
    // decode path, a per-visit kernel each (heat_fold), summed over the sessions at each placement change.
    void fold_rows(int rows, cudaStream_t st = nullptr, int row0 = 0) {
        win_rows_ = 0;
        if (rows <= 0) return;
        heat_fold(win_ids_ + (size_t) row0 * g_.n_used, visit_layer_, (int) g_.visits.size(), sh_->max_rows, rows, g_.n_used,
                  g_.n_expert, heat_tally_, st ? st : stream_);
        heat_rows_ += rows;
    }
    // A window no decode_kept settled: its rows before `next`, the position the session goes on from, were kept.
    void settle_window(int64_t next) {
        if (win_rows_ > 0) fold_rows((int) std::clamp<int64_t>(next - (win_pos_ + win_off_), 0, win_rows_));
    }
    bool feed_heat() {
        if (m_.wedged()) return false;
        if (!heat_tally_) return true;
        const auto& owners = sh_->heat_sessions;
        for (Session* o : owners)
            if (!o->settle_hits()) return false;
        std::vector<std::vector<uint32_t>> samples(owners.size());
        for (size_t i = 0; i < owners.size(); ++i) {
            Session& o = *owners[i];
            samples[i].resize((size_t) expert_pairs());
            if (cudaStreamSynchronize(o.stream_) != cudaSuccess ||
                basalt::core::device_copy(samples[i].data(), o.heat_tally_, samples[i].size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
                m_.wedge();   // no ranking from a partial sample
                basalt::log::error("basalt inkling2", "expert lookup counters could not be read; restart the engine");
                return false;
            }
        }
        std::vector<uint32_t> delta((size_t) expert_pairs(), 0u);
        int64_t rows = 0;
        for (size_t i = 0; i < owners.size(); ++i) {
            Session& o = *owners[i];
            o.heat_last_.resize(delta.size(), 0u);
            for (size_t pair = 0; pair < delta.size(); ++pair) delta[pair] += samples[i][pair] - o.heat_last_[pair];   // wraps
            o.heat_last_ = std::move(samples[i]);
            rows += o.heat_rows_;
            o.heat_rows_ = 0;
        }
        if (rows > 0) m_.learn(delta, rows);
        return true;
    }
    // One session alone with its copies: they follow its lookups between windows too.
    bool rebalances() const { return m_.elastic() && !m_.lent() && m_.promoted() > 0 && sh_.use_count() == 1; }
    int64_t next_rebalance_after(int64_t pos) const {
        int64_t at = next_rebalance_;
        if (at <= pos) at += ((pos - at) / kRebalanceRows + 1) * kRebalanceRows;
        return at;
    }
    // Swaps copies for hotter RAM experts at position `at`: a request's start (the next one kRebalanceRows later), or a decode
    // window at a scheduled position.
    void rebalance_copies(int64_t at, bool request) {
        if (m_.wedged()) return;
        next_rebalance_ = request ? at + kRebalanceRows : next_rebalance_after(at);
        if (!rebalances() || !feed_heat()) return;
        const int64_t before = m_.card_copies();
        const int n = m_.rebalance(kRebalanceSwaps);   // paid copies only; the loan stays
        if (n > 0 || m_.wedged()) m_.log_copy_change(("rebalance at " + std::to_string(at) + " (" + std::to_string(n) + " swaps)").c_str(), before, 0, true);
    }
    bool publish_loan(int64_t at) { return settle_hits() && m_.publish_loan(stream_, at); }
    // `at`: the prompt's position (-1: a sleep)
    bool recall_loan(int64_t at) {
        if (m_.wedged()) return false;
        if (m_.loaned() > 0 && !settle_hits()) return false;
        return m_.recall_loan(stream_, at < 0 ? std::string("prompt loan recall at sleep") : "prompt loan recall at " + std::to_string(at));
    }
    Served served(int l, int e) const {
        if (m_.on_card(l, e)) return Served::Card;
        if (m_.on_loan(l, e)) return Served::Loan;
        return m_.peer_held(l, e) ? Served::Peer : Served::Ram;
    }
    // Banks the lookups so far where their experts were: runs before every placement change, so a lookup counts where it
    // was served, not where its expert is later.
    bool settle_hits() {
        if (!m_.has_spill() || !ex_tally_) return true;
        std::vector<uint32_t> t;
        bool ok = true;
        for (const Lane& k : lane_)
            if (k.stream && cudaStreamSynchronize(k.stream) != cudaSuccess) ok = false;
        if (!ok || !read_tally(t)) {
            m_.wedge();
            basalt::log::error("basalt inkling2", "expert lookup counters could not be read; restart the engine");
            return false;
        }
        hits_.settle(t, g_.n_expert, [&](int l, int e) { return served(l, e); });
        return true;
    }
    // How the lookups fell between the card and RAM; `loan`: the card's that went to loaned copies.
    CardShare lookup_share(const std::vector<uint32_t>& t, uint64_t& loan) const {
        return hits_.read(t, g_.n_expert, [&](int l, int e) { return served(l, e); }, loan);
    }
    // A conversation starts over (a prompt at position 0, `at`), or resumes at `at` needing at most half the positions its
    // caches back (resume_trims): the caches give back the cells past what the prompt needs (never below the base), and the
    // RAM experts come back to the card while it has room. Any other resume leaves both as they are.
    bool restart_positions(int64_t positions, int64_t at) {
        if (elastic_.empty()) return true;
        const int64_t keep = kept_positions(positions, kElasticKvStep, core::kElasticKvBase, m_.context());
        if (at > 0 && !resume_trims(keep, backed_)) return true;
        if (!feed_heat()) return false;
        if (keep < backed_) {
            const uint64_t backing0 = backing_bytes();
            for (auto& a : elastic_) a->trim(cell_bytes() * (uint64_t) keep);
            dspark_.trim(keep);
            m_.account((int64_t) backing_bytes() - (int64_t) backing0);
            count_backed();
            if (!caches_usable()) return false;
        }
        if (m_.promoted() < m_.spill().spilled && !m_.lent()) {
            const int64_t before = m_.card_copies();
            const uint64_t grow = grow_bytes(keep);
            m_.promote(m_.held_free(kElasticKeep) + grow);
            m_.note_shortage("position " + std::to_string(at));
            m_.log_copy_change(at == 0 ? "new conversation" : ("resume at " + std::to_string(at)).c_str(), before, grow);
        }
        return !m_.wedged();
    }
    size_t kv_bytes_of(const VisitCache& c) const { return (size_t) kKvHeads * (size_t) c.cap * row_bytes(); }
    size_t conv_ring_bytes() const { return (size_t) kConvRing * (2 * kKvHeads * kHeadDim + 2 * (size_t) g_.n_embd) * 4; }

    size_t running_bytes() const override {
        size_t n = 0;
        for (const VisitCache& c : vc_) n += (c.ring ? 2 * kv_bytes_of(c) : 0) + conv_ring_bytes();
        return n + dspark_.running_bytes();   // DSPARK HOOK
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && stale_) return stale();
        if (save && (int64_t) hist_.size() - pos > kWindowRows)
            return {model::Status::Error, "inkling2: the rings have moved on more than " + std::to_string(kWindowRows) + " positions past the checkpoint"};
        if (save && pos > (int64_t) hist_.size()) return {model::Status::Error, "inkling2: a checkpoint past the session's position"};
        const size_t kv_ring = (size_t) kConvRing * kKvHeads * kHeadDim * 4, e_ring = (size_t) kConvRing * g_.n_embd * 4;
        for (size_t i = 0; i < vc_.size(); ++i) {   // the second card's visits go with its stream
            const VisitCache& c = vc_[i];
            std::vector<core::DeviceRegion>& to = slot_card_[i] ? st.peer_regions : st.regions;
            if (c.ring) {
                to.push_back({c.k, kv_bytes_of(c)});
                to.push_back({c.v, kv_bytes_of(c)});
            }
            to.push_back({c.k_ring, kv_ring});
            to.push_back({c.v_ring, kv_ring});
            to.push_back({c.a_ring, e_ring});
            to.push_back({c.m_ring, e_ring});
        }
        dspark_.running_regions(st);   // DSPARK HOOK
        st.stream = stream_;
        if (lane_[1].stream) {
            st.peer_stream = lane_[1].stream;
            st.peer_device = lane_[1].device;
        }
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        stale_ = false;   // the rings are the checkpoint's again
        return {};
    }

    int cell_format() const { return m_.cache_int8() ? 4 : m_.cache_fp8() ? 1 : 3; }   // 3: fp16, 1: E4M3, 4: scaled INT8

    size_t cells_bytes(int64_t positions) const override {
        size_t n = 0;
        for (const VisitCache& c : vc_)
            if (!c.ring) n += 2 * (size_t) positions * cell_bytes();
        return n + dspark_.cells_bytes(positions);   // DSPARK HOOK
    }

    // A full-attention visit's cache is cell-major ([cell][head][dim]): its first `positions` cells are one run.
    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (stale_) return stale();
        if (m_.wedged()) return wedged();
        if (positions < 0 || positions > backed_) return {model::Status::Error, "inkling2: positions past the backed caches"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t run = (size_t) positions * cell_bytes();
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = 1;
            k.cells = positions;
            k.heads = kKvHeads;
            k.head_dim = kHeadDim;
            k.k.resize_for_overwrite(run);
            k.v.resize_for_overwrite(run);
            const auto down = [](const void* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return basalt::core::device_copy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
                });
            };
            if (!down(c.k, k.k) || !down(c.v, k.v)) return {model::Status::Error, "inkling2: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        if (!dspark_.save_cells(kv, positions, stream_)) return {model::Status::Error, "inkling2: reading the drafter's cache back failed"};   // DSPARK HOOK
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (m_.wedged()) return wedged();
        if (positions < 0 || positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
        if (std::string e; !ensure_positions(positions, e)) return grow_failed(e);
        cudaStreamSynchronize(stream_);
        const size_t run = (size_t) positions * cell_bytes();
        size_t i = 0;
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].cells != positions || kv[i].heads != kKvHeads ||
                kv[i].k.size() < run || kv[i].v.size() < run)
                return {model::Status::Error, "inkling2: a parked conversation has other caches"};
            const auto up = [&](void* dev, const core::ConversationBuffer& from) {
                return from.visit(0, run, [&](const uint8_t* p, size_t n, size_t at) {
                    return basalt::core::device_copy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
                });
            };
            if (!up(c.k, kv[i].k) || !up(c.v, kv[i].v)) return {model::Status::Error, "inkling2: writing a cache back failed"};
            ++i;
        }
        if (dspark_.on() && !dspark_.restore_cells(kv, i, positions, stream_))   // DSPARK HOOK
            return {model::Status::Error, "inkling2: a parked conversation has no drafter cells"};
        return {};
    }

    int64_t kv_bytes() const override {
        int64_t n = 0;
        for (const VisitCache& c : vc_) n += 2 * (int64_t) kv_bytes_of(c);
        return n;
    }

    // ---- the console's experts: a lookup is a hit when its expert is on a GPU (the main card or the peer). With every expert
    // resident that is each window's rows times the experts a row takes at each of its visits; with a spill the card's running
    // counts say which experts the lookups went to, and those whose experts are in RAM are no hits
    ExpertCounts expert_counts() const override {
        if (!m_.has_spill() || !ex_tally_) return {lookups_, lookups_, 0, 0};
        std::vector<uint32_t> t;
        if (cudaStreamSynchronize(stream_) != cudaSuccess || !read_tally(t)) return {lookups_, lookups_, 0, 0};
        uint64_t loan = 0;
        const CardShare sh = lookup_share(t, loan);
        return {(int64_t) (sh.card + sh.peer), (int64_t) sh.all, 0, 0};
    }
    std::vector<CardExperts> experts_by_card() const override {
        std::vector<CardExperts> out;   // each card its layers' (the layer split spills none)
        for (int c = 0; c < m_.cards(); ++c) {
            CardExperts ce{m_.card_device(c), 0, 0};
            for (int l = 0; l < g_.n_layers; ++l)
                if (m_.card_of(l) == c) {
                    ce.slots += g_.n_expert - m_.layer(l).n_spill;
                    ce.bytes += (uint64_t) (g_.n_expert - m_.layer(l).n_spill) * m_.layer(l).expert_stride;
                }
            out.push_back(ce);
        }
        return out;
    }
    bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) override {
        if (cudaStreamSynchronize(stream_) != cudaSuccess || !read_tally(tally_h_)) return false;
        map_.take(tally_h_, total, homes, routed);
        return true;
    }
    std::string info_words() const override {
        const std::string n = std::to_string(expert_pairs() - m_.spill().spilled);
        uint64_t ram = 0;
        for (int l = 0; l < g_.n_layers; ++l) ram += (uint64_t) m_.layer(l).n_spill * m_.layer(l).expert_stride;
        return "expert_slots=" + n + " expert_slots_primary=" + n + " ram_experts_mib=" + std::to_string(ram >> 20) + (m_.cache_int8() ? " kv=int8 " : m_.cache_fp8() ? " kv=fp8 " : " kv=fp16 ") +
               program::spill_words({0, expert_pairs(), 0});
    }

private:
    int64_t expert_pairs() const { return (int64_t) g_.n_layers * g_.n_expert; }

    // On the current device.
    void* alloc(size_t bytes, bool& ok) {
        if (!ok) return nullptr;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { ok = false; return nullptr; }
        const size_t aligned = (std::max<size_t>(bytes, 16) + 255) & ~(size_t) 255;
        for (int c = 0; c < 2; ++c) {
            if (arena_rem_[c] >= aligned && arena_dev_[c] == dev) {
                void* p = arena_ptr_[c];
                arena_ptr_[c] += aligned;
                arena_rem_[c] -= aligned;
                return p;
            }
        }
        void* p = nullptr;
        if (cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs_.emplace_back(dev, p);
        return p;
    }

    // Every card's running counts of the experts' lookups, summed (a card counts its own layers').
    bool read_tally(std::vector<uint32_t>& t) const {
        t.assign((size_t) expert_pairs(), 0u);
        std::vector<uint32_t> c(t.size());
        for (const Lane& k : lane_) {
            if (!k.ex_tally) continue;
            if (basalt::core::device_copy(c.data(), k.ex_tally, c.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (size_t i = 0; i < t.size(); ++i) t[i] += c[i];
        }
        return true;
    }

    // Layer `l` (visit `v`) over `rows` rows: `hid` its input, `blocks` the finished blocks before it, `msgs` (or null)
    // the previous pass's messages [kStreams][rows][E]; `full_keys` bounds a full-attention visit's keys. Writes its
    // output to `out` (not `hid`), and to `tap` too when set.
    void run_layer(const Visit& v, const float* hid, const float* const* blocks, int n_blocks, const float* msgs, int rows,
                   int64_t full_keys, float* out, float* tap) {
        const int l = v.layer;
        const int card = m_.card_of(l);   // the layer split: the layer's card, its scratch and streams
        Lane& k = lane_[card];
        const Scratch& x = *k.x;
        const LayerWeights& w = m_.layer(l);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        VisitCache& c = vc_[(size_t) v.slot];
        const float* norms[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
        const int n_in = full ? 3 : 1;   // the attention's distinct inputs before the controllers
        // the attention's inputs: the mixes (layer 0's input as it is), normed into q8_1; in the loop's later passes each
        // first takes its message through its controller
        MixNorm mn;
        for (int i = 0; i < n_blocks; ++i) mn.st.s[mn.st.n++] = blocks[i];
        mn.st.s[mn.st.n++] = hid;
        mn.queries = w.attn_query;
        mn.eps = g_.rms_eps;
        mn.kv_pos = view_ ? nullptr : x.pos_d_;   // the attention's older cells into L2 while this mix and the projections run
        mn.kv_k = view_ ? nullptr : c.k;          // (one conversation's window only)
        mn.kv_v = c.v;
        mn.kv_cap = c.cap;
        mn.kv_window = c.ring ? g_.window : 0;
        mn.kv_cell = (int) cell_bytes();
        mn.ring_k = c.k_ring;
        mn.ring_v = c.v_ring;
        mn.ring_a = c.a_ring;
        if (!msgs) {
            for (int j = 0; j < n_in; ++j) {
                mn.norm[j] = norms[j];
                mn.q8[j] = x.xq_[j];
            }
            mix_norm(mn, n_in, rows, k.stream);
        } else {
            mn.mix_out = x.mix_;
            mn.out_stride = (int64_t) RE;
            mix_norm(mn, n_in, rows, k.stream);
            CtrlArgs ca;
            for (int s = 0; s < 3; ++s) {
                ca.dest[s] = x.mix_ + (full ? s : 0) * RE;
                ca.msg[s] = msgs + s * RE;
                ca.down[s] = w.ctrl_down[s];
                ca.up[s] = w.ctrl_gf[s];
                ca.norm[s] = norms[s];
                ca.q8[s] = x.xq_[s];
            }
            ca.scratch = x.ctrl_scr_;
            ca.rank = g_.gate_rank;
            ca.eps = g_.rms_eps;
            controllers(ca, 3, rows, k.stream);
        }
        // the projections: one product over every row of the fused matrix when the three inputs are one
        AttnPrep ap;
        if (!full && !msgs) {
            bk::native_q8_planes_mmvq(w.proj, x.xq_[0], x.y_, E, kProj, rows, k.stream);
            ap.yq = x.y_;
            ap.yk = x.y_ + kProjQ;
            ap.yv = x.y_ + kProjQ + kProjKV;
            ap.ldq = ap.ldk = ap.ldv = kProj;
        } else {
            float* yk = x.y_ + (size_t) rows * kProjQ;
            float* yv = yk + (size_t) rows * kProjKV;
            // the three products in one launch, each row range with its own input (the same bits as three calls)
            bk::Q80Segments sg;
            sg.n = 3;
            sg.row0[0] = 0, sg.row0[1] = kRowK, sg.row0[2] = kRowV;
            sg.n_out[0] = kProjQ, sg.n_out[1] = sg.n_out[2] = kProjKV;
            sg.x[0] = x.xq_[0], sg.x[1] = x.xq_[1], sg.x[2] = x.xq_[2];
            sg.y[0] = x.y_, sg.y[1] = yk, sg.y[2] = yv;
            bk::native_q8_planes_mmvq_segments(w.proj, kProj, sg, E, rows, k.stream);
            ap.yq = x.y_;
            ap.yk = yk;
            ap.yv = yv;
            ap.ldq = kProjQ;
            ap.ldk = ap.ldv = kProjKV;
        }
        // a window of up to kGuessRows rows: the experts guessed for this visit beside the visit before's router (each row's
        // kGuessExperts / rows best) into L2 on the other side stream, while the attention and the glue before the experts
        // run (it starts once the projections are done)
        const int vi = (int) (&v - g_.visits.data());
        const bool guess = rows <= kGuessRows;
        const bool has_prev_guess = guess && vi > 0 && m_.card_of(g_.visits[(size_t) vi - 1].layer) == card;
        // On deep full-attention visits (>= 16K keys = 32..535 MB KV), prefetching before attention_window both contends
        // with attn_mma_kernel on SMs/DRAM and gets evicted from L2 by the KV sweep; deferring past attention_window lands
        // the guessed experts into L2 during o_proj + post_attn + router right before MoE reads them.
        const bool defer_guess = full && full_keys >= 16384;
        const auto launch_guess_prefetch = [&]() {
            const bk::NativeExpertLayout Lx = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
            cudaEventRecord(k.pf_ev, k.stream);
            cudaStreamWaitEvent(k.pf_st, k.pf_ev, 0);
            cudaStreamWaitEvent(k.pf_st, k.gs_ev, 0);
            expert_prefetch(x.gs_ids_ + (size_t) vi * kGuessExperts, kGuessExperts / rows * rows, w.blob_table,
                            (uint64_t) Lx.down_off & ~(uint64_t) 15, x.pf_sink_, k.pf_st);
            k.pf_on = true;
        };
        if (has_prev_guess && !defer_guess) launch_guess_prefetch();
        // convolutions, clips, caches (the attention prepares them)
        ap.pos = x.pos_d_;
        ap.full = full;
        ap.k_conv = w.k_conv;
        ap.v_conv = w.v_conv;
        ap.q_scale = w.q_clip;
        ap.r_scale = w.r_clip;
        ap.qk_clip = g_.qk_clip;
        ap.rel_clip = g_.rel_clip;
        ap.eps = g_.rms_eps;
        ap.log_floor = g_.log_floor;
        ap.log_alpha = g_.log_alpha;
        ap.k_ring = c.k_ring;
        ap.v_ring = c.v_ring;
        ap.k_cache = c.k;
        ap.v_cache = c.v;
        ap.cap = c.cap;
        ap.fp8 = m_.cache_fp8();
        ap.int8 = m_.cache_int8();
        ap.q = x.q_;
        ap.r = x.r_;
        ap.sink = x.sink_;
        // the attention, gated by its row's gate projection and rounded for the output projection
        AttnArgs aa;
        aa.q = x.q_;
        aa.r = x.r_;
        aa.sink = x.sink_;
        aa.rel_table = w.rel_table;
        aa.pos = x.pos_d_;
        aa.k_cache = c.k;
        aa.v_cache = c.v;
        aa.cap = c.cap;
        aa.fp8 = m_.cache_fp8();
        aa.int8 = m_.cache_int8();
        aa.window = c.ring ? g_.window : 0;
        aa.scale = 1.0f / std::sqrt((float) kHeadDim);
        aa.gate = ap.yq + kOffGate;
        aa.gate_ld = ap.ldq;
        aa.q8 = x.xo_;
        const Segments seg = segments(v.slot, card);   // a batch window's: each conversation's caches of this visit
        ap.seg = aa.seg = seg;
        // The previous graph bucket bounds the first consecutive row from below.
        if (seg.at.n == 0 && seg.at_d == nullptr && full_keys > 1024) {
            int64_t previous = full_keys / 2;
            if (rows == 1 && m_.cache_int8() && full_keys == 2 * kMmaDeepMinKeys)
                previous = kMmaDeepMinKeys - 1;
            aa.min_full_keys = previous + 2 - rows;
        }
        attention_window(ap, aa, rows, c.ring ? (int64_t) g_.window : full_keys, x.att_scr_, k.stream);
        if (has_prev_guess && defer_guess) launch_guess_prefetch();
        bk::native_q8_planes_mmvq(w.out, x.xo_, x.o_, kOutRows, E, rows, k.stream);
        // a wider window: the experts this visit took in the window before into L2 on the side stream, while the glue
        // before the experts runs (it starts once the output projection is done)
        if (!guess && rows <= kDecodeRows) {
            const bk::NativeExpertLayout Lx = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
            cudaEventRecord(k.ra_ev, k.stream);
            cudaStreamWaitEvent(k.ra_st, k.ra_ev, 0);
            expert_read_ahead(k.rec_blobs + (size_t) vi * kAheadExperts, k.rec_n + vi, (uint64_t) Lx.down_off & ~(uint64_t) 15, k.ra_st);
            k.ra_on = true;
        }
        // the attention's convolution and the residual (a block's first layer starts a new sum), then the MLP's input
        PostAttn pa;
        pa.o = x.o_;
        pa.pos = x.pos_d_;
        pa.taps = w.attn_post_conv;
        pa.ring = c.a_ring;
        pa.seg = seg;
        pa.hid = hid;
        pa.boundary = g_.boundary(l);
        pa.resid = x.resid_;
        for (int i = 0; i < n_blocks; ++i) pa.blocks[pa.n_blocks++] = blocks[i];
        pa.query = w.ffn_query;
        pa.norm = w.ffn_norm;
        pa.eps = g_.rms_eps;
        pa.q8 = x.xq_[0];
        float* nrm = guess ? x.nrm1_ + (size_t) vi * kGuessRows * E : x.nrm_;   // a visit's own rows: the next visit's guess reads them later
        pa.norm_out = nrm;
        if (msgs) pa.mix_out = x.mix_;
        post_attn(pa, rows, k.stream);
        if (msgs) {
            CtrlArgs ca;
            ca.dest[0] = x.mix_;
            ca.msg[0] = msgs + 3 * RE;
            ca.down[0] = w.ctrl_down[3];
            ca.up[0] = w.ctrl_gf[3];
            ca.norm[0] = w.ffn_norm;
            ca.q8[0] = x.xq_[0];
            ca.norm_out = nrm;
            ca.scratch = x.ctrl_scr_;
            ca.rank = g_.gate_rank;
            ca.eps = g_.rms_eps;
            controllers(ca, 1, rows, k.stream);
        }
        // the experts
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        const bool guess_next = guess && vi + 1 < (int) g_.visits.size() && m_.card_of(g_.visits[(size_t) vi + 1].layer) == card;
        if (guess_next) cudaEventRecord(k.ra_ev, k.stream);   // the MLP input is in
        // the routed ids: a visit's own slot when the session ranks its copies by kept rows (decode_kept), else the scratch
        int32_t* const ids = k.win_ids ? k.win_ids + (size_t) vi * sh_->max_rows * g_.n_used : x.ids_;
        if (g_.n_expert == 128 && rows <= kDecodeRows) {
            bk::router_logits(nrm, w.router, rows, ra, x.rlog_, k.stream);
        } else {
            bk::router(nrm, w.router, w.router_bias, rows, ra, x.rlog_, ids, x.rw_, k.stream);
        }
        if (guess_next) {   // the next visit's experts: its router on this visit's MLP input, beside this visit's router
            const LayerWeights& nw = m_.layer(g_.visits[(size_t) vi + 1].layer);
            bk::RouterArgs ga = ra;
            ga.k = kGuessExperts / rows;
            cudaStreamWaitEvent(k.ra_st, k.ra_ev, 0);
            bk::router(nrm, nw.router, nw.router_bias, rows, ga, x.gs_log_, x.gs_ids_ + (size_t) (vi + 1) * kGuessExperts, x.gs_w_, k.ra_st);
            cudaEventRecord(k.gs_ev, k.ra_st);
            k.ra_on = true;
        }
        if (g_.n_expert == 128 && rows <= kDecodeRows) {
            router_select_plan(x.rlog_, w.router_bias, rows, g_.n_used, g_.n_expert, g_.expert_scale, ids, x.rw_,
                               w.blob_table, x.ex_ptr_, x.ex_start_, x.ex_ng_, x.ex_dst_, x.ex_tok_, k.stream,
                               k.ex_tally + (size_t) l * g_.n_expert, k.rec_blobs + (size_t) vi * kAheadExperts, k.rec_n + vi);
        } else {
            moe_plan(ids, rows, g_.n_used, g_.n_expert, w.blob_table, x.ex_ptr_, x.ex_start_, x.ex_ng_, x.ex_dst_, x.ex_tok_, k.stream,
                     k.ex_tally + (size_t) l * g_.n_expert, k.rec_blobs + (size_t) vi * kAheadExperts, k.rec_n + vi);
        }
        const bool on_cpu = cpu_ >> l & 1;   // the RAM experts' pairs on the CPU, beside the card's
        if (on_cpu) {   // open until finish has queued the join: a throw between leaves the fork outside the call's sync
            fork_open_ = true;
            sh_->cpu->start(l, rows, nrm, ids, k.stream);
        }
        bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
        L.gate_softcap = g_.gate_softcap;
        L.up_softcap = g_.up_softcap;
        const int pairs = rows * g_.n_used;
        bk::native_expert_grouped(L, x.ex_ptr_, x.ex_start_, x.ex_ng_, x.ex_dst_, x.ex_tok_, std::min(pairs, g_.n_expert), pairs, x.xq_[0], x.ex_scr_,
                                  x.ex_rows_, k.stream);
        if (on_cpu) {
            sh_->cpu->finish(l, rows, ids, x.ex_rows_, k.ex_tally + (size_t) l * g_.n_expert, k.stream);
            fork_open_ = false;
        }
        MoeFinish mf;
        mf.rows = x.ex_rows_;
        mf.w = x.rw_;
        mf.k = g_.n_used;
        mf.pos = x.pos_d_;
        mf.taps = w.ffn_post_conv;
        mf.ring = c.m_ring;
        mf.seg = seg;
        mf.resid = x.resid_;
        mf.out = out;
        mf.tap = tap;
        moe_finish(mf, rows, k.stream);
    }

    float* buf(Buf b) const { return role_[(int) b]; }

    // The kernels' segment table of visit cache `slot` in the batch window in flight (none outside one).
    Segments segments(int slot, int card) const {
        Segments s;
        if (!view_) return s;
        s.at_d = lane_[card].x->at_d_;
        s.n_segs = view_->n;
        s.max_seg_rows = 1;
        for (int i = 0; i < view_->n; ++i) {
            s.max_seg_rows = std::max(s.max_seg_rows, view_->w.segs[(size_t) i].rows);
            const VisitCache& c = static_cast<const Session*>(view_->ss[i])->vc_[(size_t) slot];
            s.c[i] = {c.k, c.v, c.k_ring, c.v_ring, c.a_ring, c.m_ring};
        }
        return s;
    }

    // The forward of `rows` rows at positions pos0.. (tokens, or host embedding rows when `emb` is set, which must stay
    // as they are until the window's sync); then the head over the last `head` rows (0: none).
    bool forward(const int32_t* tokens, const float* emb, int rows, int64_t pos0, int head, std::string& err) {
        if (sh_->cpu && sh_->cpu->stopped()) {
            err = "the CPU experts were stopped";
            return false;
        }
        cpu_ = cpu_layers();
        if (cpu_) sh_->cpu->wake();
        if (!layers(rows, head, stage(tokens, emb, rows, pos0, nullptr), err) || !check(err)) return false;
        dspark_.note(pos0, rows);   // DSPARK HOOK: the host's record of the rows the drafter took
        return true;
    }

    // Separate the scalar and MMA singleton buckets at the INT8 cutoff.
    int64_t full_keys_of(int64_t last, int rows) const {
        int64_t full_keys = 1024;
        while (full_keys < last + 1) full_keys <<= 1;
        if (rows == 1 && m_.cache_int8() && full_keys == kMmaDeepMinKeys)
            return last + 1 < kMmaDeepMinKeys ? kMmaDeepMinKeys - 1 : 2 * kMmaDeepMinKeys;
        return full_keys;
    }

    // A call's inputs to the card: its tokens and positions (pos0.., or every row's own in `positions`), its first rows;
    // returns the full visits' launch bound.
    int64_t stage(const int32_t* tokens, const float* emb, int rows, int64_t pos0, const int32_t* positions,
                  const bk::WindowSegs* wsegs = nullptr) {
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const int mr = sh_->max_rows;
        const int stride = 2 * mr + kSegWords;
        // the tokens, positions, and segment table through a ring of pinned slots
        const int slot = (int) (stage_n_++ % kStage);
        cudaEventSynchronize(stage_ev_[slot]);
        int32_t* io = io_h_ + (size_t) slot * stride;
        int64_t last = 0;
        for (int i = 0; i < rows; ++i) {
            io[i] = tokens ? tokens[i] : 0;
            io[mr + i] = positions ? positions[i] : (int32_t) (pos0 + i);
            last = std::max<int64_t>(last, io[mr + i]);
        }
        bk::WindowSegs empty_segs{};
        std::memcpy(io + 2 * mr, wsegs ? wsegs : &empty_segs, sizeof(bk::WindowSegs));
        cudaMemcpyAsync(io_d_, io, (size_t) stride * 4, cudaMemcpyHostToDevice, stream_);
        if (Lane& k = lane_[1]; k.stream) {   // the positions to the second card too, after its last window, before this one
            cudaEventRecord(io_ev0_, stream_);
            cudaSetDevice(k.device);
            cudaStreamWaitEvent(k.stream, io_ev0_, 0);
            cudaMemcpyAsync(k.x->io_d_, io, (size_t) stride * 4, cudaMemcpyHostToDevice, k.stream);
            cudaEventRecord(io_ev1_, k.stream);
            cudaSetDevice(m_.device());
            cudaStreamWaitEvent(stream_, io_ev1_, 0);
        }
        cudaEventRecord(stage_ev_[slot], stream_);
        float* x0 = buf(Buf::X0);
        if (emb) {
            cudaMemcpyAsync(x0, emb, RE * 4, cudaMemcpyHostToDevice, stream_);
        } else {
            bk::iq_embed_rows(kTypeBF16, m_.embedding(), (size_t) E * 2, tok_d_, rows, E, emb_, stream_);
            bk::rms_norm_rows(emb_, m_.embed_norm(), rows, E, g_.rms_eps, x0, stream_);
        }
        return full_keys_of(last, rows);
    }

    // The layers with experts on the CPU (bit l: layer l) for the next window: part of its shape.
    uint64_t cpu_layers() const { return sh_->cpu ? m_.cpu_layers() : 0; }

    // The layers and the head of a window of `rows` rows: replayed from its shape's graph, captured on the shape's second
    // window (its first runs direct, outside a capture).
    bool layers(int rows, int head, int64_t full_keys, std::string& err) {
        if (!drop_stale_read_ahead(err)) return false;
        const Shape key{cpu_, ((uint64_t) full_keys << 16) | ((uint64_t) head << 8) | (uint64_t) rows};
        auto it = graphs_.find(key);
        if (it != graphs_.end()) {
            window_cold_ = false;
            return it->second.launch(stream_, err);
        }
        window_cold_ = true;
        if (direct_.insert(key).second || graphless_.count(key)) {
            body(rows, head, full_keys);
            return true;
        }
        core::CapturedGraph g;
        if (!g.begin(stream_, err)) return false;
        try {
            body(rows, head, full_keys);
        } catch (...) {   // the stream leaves the capture before the call's cleanup syncs it
            cudaGraph_t dropped = nullptr;
            cudaStreamEndCapture(stream_, &dropped);
            if (dropped) cudaGraphDestroy(dropped);
            cudaGetLastError();
            throw;
        }
        if (!g.end(stream_, err)) {
            // the graph's own device memory can be what the card is short of when the experts' copies fill it to its spare:
            // the window runs direct and its shape keeps running without a graph. No copy moves: which shape meets the
            // shortage first depends on the windows' widths, and the placement must not. Any other failure fails the window.
            if (g.status() != cudaErrorMemoryAllocation) return false;
            (void) cudaGetLastError();
            err.clear();
            if (graphless_.empty())
                basalt::log::warn("basalt inkling2", "a window's graph found no device memory: windows of its shape run without one");
            graphless_.insert(key);
            body(rows, head, full_keys);
            return true;
        }
        return graphs_.emplace(key, std::move(g)).first->second.launch(stream_, err);
    }

    // Launches only: the visits, then the head. Each kernel may start while the one before it finishes (it waits for it
    // before reading what it wrote), so a product streams its first weights during the glue before it.
    void body(int rows, int head, int64_t full_keys) {
        for (Lane& k : lane_) k.ra_on = k.pf_on = false;
        visits_and_head(rows, head, full_keys);
        join_sides(lane_[0]);
    }

    // A card's side streams join its window stream.
    static void join_sides(Lane& k) {
        if (k.ra_on) {
            cudaEventRecord(k.ra_join, k.ra_st);
            cudaStreamWaitEvent(k.stream, k.ra_join, 0);
        }
        if (k.pf_on) {
            cudaEventRecord(k.pf_join, k.pf_st);
            cudaStreamWaitEvent(k.stream, k.pf_join, 0);
        }
        k.ra_on = k.pf_on = false;
    }

    // The layer split: the window crosses from card `from` to segment `sg`'s card, which waits for it and takes the row
    // buffers it reads that were last written on `from` (and, coming back to the first card, the drafter taps the second
    // card's visits wrote), through the page-locked slots.
    void cross(int from, const Segment& sg, int rows) {
        Lane& a = lane_[from];
        Lane& b = lane_[sg.card];
        const size_t n = (size_t) rows * g_.n_embd, slot = (size_t) sh_->max_rows * g_.n_embd;
        join_sides(a);
        cudaSetDevice(a.device);
        for (size_t i = 0; i < sg.pull.size(); ++i)
            cudaMemcpyAsync(sh_->hand + i * slot, a.x->role_[(int) sg.pull[i]], n * 4, cudaMemcpyDeviceToHost, a.stream);
        cudaEventRecord(a.hand, a.stream);
        cudaSetDevice(b.device);
        cudaStreamWaitEvent(b.stream, a.hand, 0);
        for (size_t i = 0; i < sg.pull.size(); ++i)
            cudaMemcpyAsync(b.x->role_[(int) sg.pull[i]], sh_->hand + i * slot, n * 4, cudaMemcpyHostToDevice, b.stream);
    }

    // DSPARK HOOK: a second-card visit's input rows to the drafter's tap on the first card (through a page-locked slot of
    // its own beyond the crossings').
    void tap_across(int t, const float* rows_dev, int rows) {
        Lane& a = lane_[1];
        const size_t n = (size_t) rows * g_.n_embd;
        float* slot = sh_->hand + (size_t) (sh_->tap_slot0 + t) * sh_->max_rows * g_.n_embd;
        cudaMemcpyAsync(slot, rows_dev, n * 4, cudaMemcpyDeviceToHost, a.stream);
        cudaEventRecord(a.hand, a.stream);
        cudaSetDevice(m_.device());
        cudaStreamWaitEvent(stream_, a.hand, 0);
        cudaMemcpyAsync(dspark_.tap(t), slot, n * 4, cudaMemcpyHostToDevice, stream_);
        cudaSetDevice(a.device);
    }

    void visits_and_head(int rows, int head, int64_t full_keys) {
        bk::PdlScope pdl(true);
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const float* blocks[kMaxMix];
        bool tapped_in = false;   // the visit's input rows were written to its drafter tap by the visit before
        for (size_t gi = 0; gi < segs_.size(); ++gi) {   // one segment on one card; the layer split's crossings between
            const Segment& sg = segs_[gi];
            if (gi > 0) cross(segs_[gi - 1].card, sg, rows);
            const Lane& k = lane_[sg.card];
            const Scratch& x = *k.x;
            auto kb = [&](Buf b) { return x.role_[(int) b]; };
            for (size_t si = sg.s0; si < sg.s1; ++si) {
                const Step& s = walk_.steps[si];
                const Visit& v = g_.visits[(size_t) s.visit];
                const float* msgs = nullptr;
                if (s.msgs) {
                    const int core = g_.loop_end - g_.loop_start;
                    if (sg.mix_msgs[si - sg.s0]) {   // a pass's messages, every core layer's four streams, from its bank
                        MixNorm mm;
                        mm.st.s[mm.st.n++] = kb(s.bank[0]);
                        mm.st.s[mm.st.n++] = kb(s.bank[1]);
                        mm.queries = m_.msg_queries(sg.card);
                        mm.mix_out = x.msg_;
                        mm.out_stride = (int64_t) RE;
                        mix_norm(mm, core * kStreams, rows, k.stream);
                    }
                    msgs = x.msg_ + (size_t) (v.layer - g_.loop_start) * kStreams * RE;
                }
                for (int i = 0; i < s.n_blocks; ++i) blocks[i] = kb(s.blocks[i]);
                if (int t = dspark_.slot_of_visit(s.visit); t >= 0 && !tapped_in) {   // DSPARK HOOK: the visit's input rows
                    if (sg.card == 0) cudaMemcpyAsync(dspark_.tap(t), kb(s.hid), RE * 4, cudaMemcpyDeviceToDevice, stream_);
                    else tap_across(t, kb(s.hid), rows);
                }
                // the next visit's tap, when its input is this visit's output on the first card, is written by this visit's
                // last kernel
                float* tap = nullptr;
                if (si + 1 < sg.s1 && sg.card == 0) {
                    const Step& nx = walk_.steps[si + 1];
                    if (int t = dspark_.slot_of_visit(nx.visit); t >= 0 && nx.hid == s.out) tap = dspark_.tap(t);
                }
                tapped_in = tap != nullptr;
                run_layer(v, kb(s.hid), blocks, s.n_blocks, msgs, rows, full_keys, kb(s.out), tap);
            }
        }
        const bool tapped = dspark_.on();   // DSPARK HOOK: the drafter reads every row's final normed state
        if (head <= 0 && !tapped) return;
        // the final normed state over the last `head` rows (every row for the drafter), then the head over the last `head`
        const int first = tapped ? 0 : rows - head, n_fin = rows - first;
        MixNorm hn;
        for (int i = 0; i < walk_.n_blocks; ++i) hn.st.s[hn.st.n++] = buf(walk_.blocks[i]) + (size_t) first * E;
        hn.st.s[hn.st.n++] = buf(walk_.final_hid) + (size_t) first * E;
        hn.queries = m_.output_query();
        hn.norm[0] = m_.output_norm();
        hn.q8[0] = xq_[0];
        hn.norm_out = tapped ? dspark_.tap(kDsTargets - 1) : fin_;
        hn.eps = g_.rms_eps;
        mix_norm(hn, 1, n_fin, stream_);
        if (tapped) {   // DSPARK HOOK: fork onto the side stream when the LM head also runs, overlapping both inside the graph
            cudaStream_t cst = stream_;
            if (head > 0) {
                Lane& k0 = lane_[0];
                join_sides(k0);
                cudaEventRecord(k0.ra_ev, stream_);
                cudaStreamWaitEvent(k0.ra_st, k0.ra_ev, 0);
                k0.ra_on = true;
                cst = k0.ra_st;
            }
            if (!view_) {
                dspark_.commit(pos_d_, rows, cst);
            } else if (view_->n > 1 && sh_->ds_batch && sh_->ds_batch->on()) {
                DsparkHook* hooks[bk::kWindowSegsMax] = {};
                for (int i = 0; i < view_->n; ++i) hooks[i] = &static_cast<Session*>(view_->ss[i])->dspark_;
                sh_->ds_batch->commit(hooks, view_->n, at_d_, pos_d_, rows, cst);
            } else {
                for (int i = 0; i < view_->n; ++i) {
                    const model::BatchWindow::Seg& sg = view_->w.segs[(size_t) i];
                    static_cast<Session*>(view_->ss[i])->dspark_.commit(pos_d_ + sg.row0, sg.rows, cst, sg.row0);
                }
            }
        }
        if (head <= 0) return;
        const size_t q8_row = (size_t) E / 32 * 36;   // the head's rows' q8_1 blocks follow the rows before them
        float* lg = blog_;                            // solo and batch both write into shared blog_
        bk::native_q8_planes_mmvq(m_.head(), (const uint8_t*) xq_[0] + (size_t) (rows - head - first) * q8_row, lg, E, g_.n_vocab, head,
                             stream_, g_.logit_softcap);
    }

    // Admission to the CPU experts' windows (CpuExperts::enter): held from before a call's first window until the sync
    // that ends them, so an abort waits for the calls in flight, and a call it interrupted discards its results.
    bool admit(std::string& err) {
        if (!sh_->cpu) return true;
        if (!sh_->cpu->enter()) {
            err = "the CPU experts were stopped";
            return false;
        }
        admitted_ = true;
        return true;
    }
    // After the sync of the window stream that ends a call's windows (`synced`: its status): gives admission up. False:
    // a sync failed or the abort came first, and the call's results are discarded.
    bool complete(cudaError_t synced) {
        if (!admitted_) return synced == cudaSuccess;
        admitted_ = false;
        if (synced == cudaSuccess && lane_[1].stream) synced = cudaStreamSynchronize(lane_[1].stream);
        const bool joined = !fork_open_;
        fork_open_ = false;
        return sh_->cpu->leave(synced, joined) && synced == cudaSuccess && joined;
    }
    // An exit before that sync: the window stream is synchronized first.
    void release() {
        if (admitted_) complete(cudaStreamSynchronize(stream_));
    }
    // The end of a call that took admission: its closing sync, then the abort's verdict on its results.
    model::Result end_call(const char* failed) {
        const cudaError_t synced = cudaStreamSynchronize(stream_);
        const bool kept = complete(synced);
        if (synced != cudaSuccess) return {model::Status::Error, std::string("inkling2: ") + failed};
        if (!kept) return {model::Status::Error, "inkling2: the CPU experts were stopped"};
        return {};
    }
    // Gives admission up on every return of a call that took it, unless passed on.
    struct Admitted {
        Session& s;
        bool passed = false;
        ~Admitted() {
            if (!passed) s.release();
        }
    };

    bool check(std::string& err) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            bool ok = true;
            dev = (float*) alloc(bytes, ok);
            if (!ok) {
                cudaGetLastError();
                err = "out of device memory for a logit bias";
                return false;
            }
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            err = "the logit bias upload failed";
            return false;
        }
        return true;
    }

    // The draws of the `rows` rows of logits_, synced, into picks_.
    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        const int32_t* p = draw(r, t0, rows, err, stream_);
        if (p == nullptr) return false;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        picks_.assign(p, p + rows);
        return true;
    }

    // The draws of the `rows` rows of logits_, queued on `st`; the host array they land in, read once `st` has run.
    const int32_t* draw(const model::DecodeRequest& r, int t0, int rows, std::string& err, cudaStream_t st) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {   // the penalties' history of each row of the window
            const size_t n = (size_t) kWindowRows * history_len_;
            if (d_hist_len_ < n) {
                bool ok = true;
                d_hist_ = (int32_t*) alloc(n * 4, ok);
                if (!ok) {
                    cudaGetLastError();
                    err = "out of device memory for the sampling history";
                    return nullptr;
                }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history + (size_t) t0 * history_len_, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, st);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, st);
        if (cudaMemcpyAsync(pick_h_, pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            err = "the pick failed";
            return nullptr;
        }
        return pick_h_;
    }

    InklingModel& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    std::vector<std::pair<int, void*>> bufs_;   // (device, allocation)
    std::vector<void*> pinned_;
    Lane lane_[2];                           // each card's side of the forward (the first card's: the members above)
    std::vector<Segment> segs_;              // the walk's runs of visits on each card (walk.hpp split_walk)
    std::vector<int> slot_card_;             // [cache slot] the card of its visits
    cudaEvent_t hand0_ = nullptr, io_ev0_ = nullptr, io_ev1_ = nullptr;   // the first card's crossing; the second card's positions
    std::vector<VisitCache> vc_;
    std::shared_ptr<Shared> sh_;             // the scratch, the prompt path and the batch graphs every session shares
    DsparkHook dspark_;                      // DSPARK HOOK
    Walk walk_;
    // a window's shape: the layers with experts on the CPU; rows, head rows and the full visits' launch bound
    using Shape = std::pair<uint64_t, uint64_t>;
    std::map<Shape, core::CapturedGraph> graphs_;      // by window shape
    std::set<Shape> direct_;                           // the shapes whose first window has run
    std::set<Shape> graphless_;                        // the shapes whose graph found no device memory: always direct
    uint64_t cpu_ = 0;                                 // the window in flight's layers with experts on the CPU
    bool admitted_ = false;
    bool window_cold_ = false;                            // holds admission to the CPU experts' windows (admit)
    bool fork_open_ = false;                           // a start whose finish has not queued its join
    const model::BatchView* view_ = nullptr;           // the batch window in flight, while batch_forward runs it
    float* logits_ = nullptr;                // slice in blog_ of the last window's head rows
    uint64_t logits_gen_ = 0;
    int arena_dev_[2] = {-1, -1};
    uint8_t* arena_ptr_[2] = {nullptr, nullptr};
    size_t arena_rem_[2] = {0, 0};
    uint32_t* ex_tally_ = nullptr;           // [layer][expert] the card's running count of each expert's lookups
    unsigned long long* rec_blobs_ = nullptr;   // [visit][kAheadExperts] the experts each visit took in the last window
    int32_t* rec_n_ = nullptr;                  // [visit] how many
    uint64_t seen_epoch_ = 0;                   // the model's placement epoch rec_blobs_ was written in
    cudaStream_t ra_st_ = nullptr;              // the side stream (the read-ahead, the guesses), its fork and join events
    cudaEvent_t ra_ev_ = nullptr, ra_join_ = nullptr, gs_ev_ = nullptr;   // gs_ev_: a guess is in
    cudaStream_t pf_st_ = nullptr;              // the guesses' prefetch stream, its fork and join events
    cudaEvent_t pf_ev_ = nullptr, pf_join_ = nullptr;
    std::vector<uint32_t> tally_h_;
    // the copies' ranking (with copies): the kept rows' lookups, the last window's routed ids by visit and its rows not
    // folded yet (from window position win_pos_ + win_off_), the counts and rows the model has not learned yet
    uint32_t* heat_tally_ = nullptr;
    int32_t* win_ids_ = nullptr;
    int32_t* visit_layer_ = nullptr;
    int64_t win_pos_ = 0;
    int win_off_ = 0, win_rows_ = 0;
    std::vector<uint32_t> heat_last_;
    int64_t heat_rows_ = 0;
    struct BatchSeg {
        Session* s;
        int row0, rows;
    };
    std::vector<BatchSeg> batch_segs_;          // a batch window's sessions and their rows, as its leader, until decode_kept
    int64_t next_rebalance_ = -1;               // the position of the next rebalance between windows (-1: before a request)
    SettledShare hits_;                         // the lookups by where they were served (settle_hits)
    ResidentExpertMap map_;
    int64_t lookups_ = 0;                    // decode's routed lookups so far
    // the elastic KV (model.hpp elastic()): the full visits' caches back their first backed_ cells
    std::vector<std::unique_ptr<core::ElasticArray>> elastic_;
    int64_t backed_ = 0;
    int32_t *pick_ = nullptr, *pick_h_ = nullptr, *io_h_ = nullptr;
    cudaEvent_t stage_ev_[kStage] = {};
    uint64_t stage_n_ = 0;
    std::vector<int32_t> picks_;
    std::vector<int32_t> hist_;
    bool stale_ = false;   // a read failed past its start: only a new conversation (position 0) or a restored checkpoint goes on
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
    bool bias_on_ = false;
    int32_t* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bk::SamplerParams sampling_;
    int head_rows_ = 0;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::inkling2
