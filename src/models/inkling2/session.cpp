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

constexpr int kMaxRows = kDecodeRows;    // rows of one forward call (a decode window, a step of a prompt)
constexpr int kTypeBF16 = 30;
constexpr int kOutRows = 3072;           // the attention output's width (heads x head width)
constexpr int kStage = 64;               // windows whose tokens and positions are in flight from the pinned ring at once

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
    int32_t *io_d_ = nullptr, *tok_d_ = nullptr, *pos_d_ = nullptr;   // a call's tokens, then its positions
    static constexpr int kGuessExperts = 8;     // the experts a window's guess requests a visit (rows x kGuessExperts / rows)
    static constexpr int kGuessRows = 2;        // the widest window guessed (wider ones keep the read-ahead)
};

// What the sessions of one model share: the forward scratch (with the layer split, the second card's too), the prompt path,
// the drafter's tap, the CPU experts and the batch windows' graphs. Made by the first session; the model keeps a weak
// reference (InklingModel::shared()).
struct Shared {
    Scratch x, x1;                          // the first card's scratch, the second card's (the layer split)
    float* hand = nullptr;                  // the layer split: page-locked slots of a call's rows crossing between the cards,
    int hand_slots = 0, tap_slot0 = 0;      // the crossings' first, then one per drafter tap
    std::unique_ptr<PromptRunner> prompt;   // PROMPT HOOK
    std::unique_ptr<CpuExperts> cpu;        // with a spill: the RAM experts' rows
    float* tap = nullptr;                   // DSPARK HOOK: the drafter's features of a call's rows (dspark_tap_bytes)
    model::BatchShared batch;
    std::vector<std::pair<int, void*>> bufs;   // (device, allocation)
    ~Shared() {
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
        if (!alloc_scratch(m, x, m.device(), true, err)) return false;
        if (m.cards() > 1) {
            cudaSetDevice(m.card_device(1));
            const bool ok = alloc_scratch(m, x1, m.card_device(1), false, err);
            cudaSetDevice(m.device());
            if (!ok) return false;
            tap_slot0 = max_pulls;
            hand_slots = max_pulls + kDsTargets;
            if (cudaHostAlloc((void**) &hand, (size_t) hand_slots * kMaxRows * m.geo().n_embd * 4, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                err = "cannot pin the layer split's hand-off rows";
                return false;
            }
        }
        if (!(cpu = CpuExperts::make(m, err)) && !err.empty()) return false;
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
        const int64_t E = g.n_embd, R = kMaxRows;
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
        x.att_scr_ = alloc(attention_decode_scratch(kMaxRows, std::max<int64_t>(ctx, g.window)));
        for (void*& q : x.xq_) q = alloc(bk::native_q8_1_bytes((int) E, kMaxRows));
        x.xo_ = alloc(bk::native_q8_1_bytes(kOutRows, kMaxRows));
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
        x.lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxRows, g.n_vocab) + R * (bk::kLogprobStride + 1));
        x.blog_ = f(R * (int64_t) g.n_vocab);
        x.io_d_ = (int32_t*) alloc((size_t) 2 * R * 4);
        x.tok_d_ = x.io_d_;
        x.pos_d_ = x.io_d_ + R;
        if (first && m.dspark()) tap = (float*) alloc(dspark_tap_bytes(std::max(kMaxRows, kPromptChunk)));   // DSPARK HOOK
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
};

class Session final : public model::BatchSession, private Scratch {
public:
    explicit Session(InklingModel& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        if (stream_) cudaStreamSynchronize(stream_);
        if (lane_[1].stream) cudaStreamSynchronize(lane_[1].stream);
        if (m_.spill().any() && ex_tally_) {   // how decode's lookups fell between the card and RAM
            std::vector<uint32_t> t;
            if (read_tally(t)) {
                const CardShare sh = card_share(t, g_.n_expert, [&](int l, int e) { return m_.on_card(l, e); },
                                                [&](int l, int e) { return m_.peer_held(l, e); });
                const uint64_t all = sh.all, card = sh.card;
                if (m_.peer_device() >= 0)
                    basalt::log::info("basalt inkling2", "experts: %llu decode lookups, %.2f%% on the card, %llu on GPU %d, %llu on the CPU",
                                      (unsigned long long) all, all ? 100.0 * (double) card / (double) all : 100.0,
                                      (unsigned long long) sh.peer, m_.peer_device(), (unsigned long long) (all - card - sh.peer));
                else
                    basalt::log::info("basalt inkling2", "experts: %llu decode lookups, %.2f%% on the card, %llu on the CPU",
                                      (unsigned long long) all, all ? 100.0 * (double) card / (double) all : 100.0,
                                      (unsigned long long) (all - card));
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
        const int64_t E = g_.n_embd, R = kMaxRows;
        const size_t row_bytes = (size_t) kv_cache_row_bytes(m_.cache_fp8(), m_.cache_int8());
        const int64_t ctx = m_.context();
        int max_slot = 0;
        for (const Visit& v : g_.visits) max_slot = std::max(max_slot, v.slot);
        vc_.assign((size_t) max_slot + 1, VisitCache{});
        slot_card_.assign(vc_.size(), 0);
        for (const Visit& v : g_.visits) {
            VisitCache& c = vc_[(size_t) v.slot];
            slot_card_[(size_t) v.slot] = m_.card_of(v.layer);
            cudaSetDevice(m_.card_device(m_.card_of(v.layer)));   // a visit's caches on its layer's card
            c.ring = g_.swa[(size_t) v.layer];
            c.cap = c.ring ? (int64_t) g_.window + kMaxRows : ctx;
            const size_t bytes = (size_t) kKvHeads * (size_t) c.cap * row_bytes;
            if (!c.ring && m_.elastic()) {   // the elastic KV: the range reserved, its first kElasticKvBase cells backed
                for (int j = 0; j < 2; ++j) {
                    auto a = std::make_unique<core::ElasticArray>();
                    if (ok && (!a->open(bytes, err) || !a->ensure(cell_bytes() * (uint64_t) core::kElasticKvBase, 0, err))) ok = false;
                    (j ? c.v : c.k) = a->data();
                    elastic_.push_back(std::move(a));
                }
                backed_ = core::kElasticKvBase;
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
        rec_blobs_ = (unsigned long long*) alloc((size_t) g_.visits.size() * kAheadExperts * 8, ok);
        rec_n_ = (int32_t*) alloc((size_t) g_.visits.size() * 4, ok);
        logits_ = f(R * (int64_t) g_.n_vocab);
        pick_ = (int32_t*) alloc(4 * 32, ok);
        if (ok && cudaMallocHost((void**) &pick_h_, 4 * 32) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(pick_h_);
        if (ok && cudaMallocHost((void**) &io_h_, (size_t) kStage * 2 * R * 4) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(io_h_);
        for (cudaEvent_t& e : stage_ev_)
            if (ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) ok = false;
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
        lane_[1].x = &sh_->x1;
        for (const VisitCache& c : vc_) {
            if (slot_card_[(size_t) (&c - vc_.data())] != 0) continue;
            cudaMemsetAsync(c.k_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.v_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.a_ring, 0, (size_t) kConvRing * E * 4, stream_);
            cudaMemsetAsync(c.m_ring, 0, (size_t) kConvRing * E * 4, stream_);
        }
        if (ok) cudaMemsetAsync(ex_tally_, 0, (size_t) expert_pairs() * 4, stream_);
        if (ok) cudaMemsetAsync(rec_n_, 0, (size_t) g_.visits.size() * 4, stream_);
        cudaStreamSynchronize(stream_);
        return dspark_.init(m_.dspark(), g_, m_.context(), std::max(kMaxRows, kPromptChunk), sh_->tap, m_.flags().diag_dump, err);   // DSPARK HOOK
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "inkling2: an empty window"};
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the window runs past the context"};
        std::string err;
        if (!ensure_positions(r.pos + r.n_tokens, err)) return {model::Status::Refused, "inkling2: " + err};
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        hist_.resize((size_t) r.pos);
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, head, err)) return {model::Status::Error, "inkling2: " + err};
            lookups_ += (int64_t) n * g_.n_used * (int64_t) g_.visits.size();
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            head_rows_ = head;
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "inkling2: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits)
                cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        return end_call("the window failed");
    }

    // ---- batch windows (docs/design/batching.md): several sessions' windows in one pass through the weights, on this
    // session's scratch and stream. Each conversation keeps its caches and rings (the kernels take them per segment), its
    // drafter's context, its logits and its draws; a row gives the bits it gives in its own window.
    int batch_rows() const override { return kMaxRows; }

protected:
    // the hooks of the shared driver (model/batch_driver.hpp)
    const char* batch_name() const override { return "inkling2"; }
    int64_t batch_context() const override { return m_.context(); }
    model::BatchShared& batch_shared() override { return sh_->batch; }
    cudaStream_t batch_stream() const override { return stream_; }
    bool batch_reserve(const model::DecodeRequest& r, std::string& err) override { return ensure_positions(r.pos + r.n_tokens, err); }
    bool batch_stage(const model::BatchView& v, model::GraphLru<core::CapturedGraph>::Key& key, std::string& err) override {
        if (!admit(err)) return false;
        Admitted held{*this};   // passed to the driver (batch_end) once the stage succeeds
        key.push_back((uint64_t) stage(v.w.tokens.data(), nullptr, v.w.rows, 0, v.w.positions.data()));
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
        view_ = &v;
        body(v.w.rows, v.w.rows, full_keys_of(v.w.max_pos));
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
        head_rows_ = h;
        if (h == 0) return;
        cudaMemcpyAsync(logits_, blog_ + (size_t) (sg.row0 + sg.rows - h) * V, (size_t) h * V * 4, cudaMemcpyDeviceToDevice, v.stream);
        if (r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (h - 1) * V, (size_t) V * 4, cudaMemcpyDeviceToHost, v.stream);
    }
    const int32_t* batch_draw(const model::BatchView& v, int i, std::string& err) override {
        return draw(v.reqs[i], 0, v.head[i], err, v.stream);
    }

public:
    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the prompt runs past the context"};
        if (r.pos == 0) restart_positions(r.n_tokens);
        if (std::string e; !ensure_positions(r.pos + r.n_tokens, e)) return {model::Status::Refused, "inkling2: " + e};
        hist_.resize((size_t) r.pos);
        if (sh_->prompt && r.n_tokens > 16) return sh_->prompt->read(r, hist_, vc_, dspark_, stream_);   // PROMPT HOOK
        std::string err;
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, 0, err)) return {model::Status::Error, "inkling2: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            if (r.progress && ((t0 / kMaxRows) % 64 == 63 || t0 + n == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        return end_call("the prompt failed");
    }

    // ---- [vision] images (vision.hpp, vision_encoder.hpp): a picture's rows stand in for its pad tokens' embeddings, taken
    // as they are (no embedding norm); the history holds the pad token at their positions.
    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        ImageEncoder& enc = m_.encoder();
        if (!enc.staged()) return {model::Status::NotImplemented, "inkling2: started without --vision"};
        std::string err;
        if (!enc.encode(data, bytes, rows, n_rows, grid, err)) return {model::Status::Error, "inkling2: " + err};
        if (n_rows <= 0 || rows.size() != (size_t) n_rows * (size_t) g_.n_embd)
            return {model::Status::Error, "inkling2: the projector's rows are not the model's width"};
        return {};
    }
    void images_done() override { m_.encoder().release(); }
    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid) override {
        if (pos + n_rows > m_.context()) return {model::Status::Refused, "inkling2: the picture runs past the context"};
        if (std::string e; !ensure_positions(pos + n_rows, e)) return {model::Status::Refused, "inkling2: " + e};
        hist_.resize((size_t) pos);
        if (sh_->prompt && n_rows > 16) return sh_->prompt->read_rows(rows, n_rows, pos, g_.image_token, hist_, vc_, dspark_, stream_);
        std::string err;
        if (!admit(err)) return {model::Status::Error, "inkling2: " + err};
        const Admitted held{*this};
        for (int at = 0; at < n_rows; at += kMaxRows) {
            const int n = std::min(kMaxRows, n_rows - at);
            if (!forward(nullptr, rows + (size_t) at * g_.n_embd, n, pos + at, 0, err)) return {model::Status::Error, "inkling2: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(g_.image_token);
        }
        return end_call("the image rows failed");
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
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: the row is not in the last window"};
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
        if (row0 < 0 || n < 1 || row0 + n > head_rows_) return {model::Status::Error, "inkling2: the session keeps the last window's logits only"};
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
        if (!m_.elastic()) return {};
        cudaStreamSynchronize(stream_);
        const int64_t before = m_.promoted();
        m_.lend(low, stream_);
        basalt::log::info("basalt inkling2", low ? "sleep: %lld expert copies go back to RAM (decode reads them there)" : "wake: %lld expert copies return to the card",
                          (long long) (low ? before : m_.promoted()));
        expert_slots = expert_pairs() - m_.spill().spilled + m_.promoted();
        return {};
    }

    // VRAM: the copies resize to leave `mib` free (< 0: the start's --vram-reserve-mib). A static spill or none has no
    // copies to resize: refused unless the card already leaves that much.
    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        cudaStreamSynchronize(stream_);
        const bool fits = m_.resize_reserve((uint64_t) (mib < 0 ? std::max(m_.flags().vram_reserve_mib, 0) : mib) << 20, stream_);
        expert_slots = m_.elastic() ? expert_pairs() - m_.spill().spilled + m_.promoted() : -1;
        if (!fits) return {model::Status::Error, "inkling2: the card cannot leave " + std::to_string(mib) + " MiB free: the experts it holds are not copies"};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: no logits for that row"};
        if (cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "inkling2: copying the logits failed"};
        return {};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    // DSPARK HOOK: the drafter's windows (the anchor and up to seven drafts) and blocks
    int max_window() const override { return dspark_.on() ? std::clamp(m_.flags().spec, 1, kMaxRows) : 1; }
    bool copy_drafts() const override { return max_window() > 1; }   // prompt lookup: any window's rows verify alike
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override { return dspark_.draft_begin(anchor, pos, n, stream_); }
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override {
        return dspark_.draft_end(ids, conf, max_n, n, need, stream_);
    }

    // ---- conversations: the full-attention visits' cells by position (save_cells / restore_cells), the window rings and
    // the convolutions' rings by checkpoint (running_state). A window ring holds window + kMaxRows positions and a
    // convolution ring kConvRing, so a checkpoint at `pos` is intact while the session has not run more than kMaxRows
    // rows past it.
    size_t row_bytes() const { return (size_t) kv_cache_row_bytes(m_.cache_fp8(), m_.cache_int8()); }
    uint64_t cell_bytes() const { return (uint64_t) kKvHeads * row_bytes(); }

    // The elastic KV: before a run reaches `positions`, the full visits' caches back them (in steps of kElasticKvStep cells),
    // and when the card has no room the RAM experts' card copies go back to RAM first, the least routed first. False when
    // even without any copy there is no room.
    static constexpr int64_t kElasticKvStep = 8192;
    static constexpr uint64_t kElasticKeep = 512ull << 20;   // free bytes a grow leaves (the driver, the graphs)
    bool ensure_positions(int64_t positions, std::string& err) {
        if (elastic_.empty() || positions <= backed_) return true;
        const int64_t target = std::min<int64_t>(m_.context(), (positions + kElasticKvStep - 1) / kElasticKvStep * kElasticKvStep);
        const uint64_t bytes = cell_bytes() * (uint64_t) target;
        uint64_t grow = 64ull << 20;   // what the arrays will map, each rounded as it maps, and a margin
        for (const auto& a : elastic_) grow += a->growth(bytes);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t keep_free = m_.held_free(kElasticKeep);
        if (free_b < grow + keep_free) {
            const uint64_t freed = m_.demote(grow + keep_free - free_b, stream_);
            basalt::log::info("basalt inkling2", "elastic spill: the caches grow to %lld positions; %.2f GiB of expert copies go back to RAM "
                              "(%lld stay on the card)", (long long) target, (double) freed / (double) (1ull << 30), (long long) m_.promoted());
        }
        for (auto& a : elastic_)
            if (!a->ensure(bytes, 0, err)) {
                err = "the caches cannot grow to " + std::to_string(target) + " positions: " + err;
                return false;
            }
        backed_ = target;
        return true;
    }
    // A conversation starts over (a prompt at position 0): the caches give back the cells past what it needs (never below
    // the base), and the RAM experts come back to the card while it has room.
    void restart_positions(int64_t positions) {
        if (elastic_.empty()) return;
        const int64_t keep = std::max<int64_t>(core::kElasticKvBase, (positions + kElasticKvStep - 1) / kElasticKvStep * kElasticKvStep);
        if (keep < backed_) {
            cudaStreamSynchronize(stream_);
            for (auto& a : elastic_) a->trim(cell_bytes() * (uint64_t) keep);
            backed_ = std::min<int64_t>(backed_, (int64_t) (elastic_[0]->backed() / cell_bytes()));
        }
        if (m_.promoted() < m_.spill().spilled && !m_.lent()) {
            cudaStreamSynchronize(stream_);
            m_.promote(m_.held_free(kElasticKeep) + (uint64_t) elastic_.size() * cell_bytes() * (uint64_t) std::max<int64_t>(0, keep - backed_));
        }
    }
    size_t kv_bytes_of(const VisitCache& c) const { return (size_t) kKvHeads * (size_t) c.cap * row_bytes(); }
    size_t conv_ring_bytes() const { return (size_t) kConvRing * (2 * kKvHeads * kHeadDim + 2 * (size_t) g_.n_embd) * 4; }

    size_t running_bytes() const override {
        size_t n = 0;
        for (const VisitCache& c : vc_) n += (c.ring ? 2 * kv_bytes_of(c) : 0) + conv_ring_bytes();
        return n + dspark_.running_bytes();   // DSPARK HOOK
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && (int64_t) hist_.size() - pos > kMaxRows)
            return {model::Status::Error, "inkling2: the rings have moved on more than " + std::to_string(kMaxRows) + " positions past the checkpoint"};
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
        if (positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
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
                    return cudaMemcpy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
                });
            };
            if (!down(c.k, k.k) || !down(c.v, k.v)) return {model::Status::Error, "inkling2: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        if (!dspark_.save_cells(kv, positions, stream_)) return {model::Status::Error, "inkling2: reading the drafter's cache back failed"};   // DSPARK HOOK
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
        if (std::string e; !ensure_positions(positions, e)) return {model::Status::Refused, "inkling2: " + e};
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
                    return cudaMemcpy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
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
        if (!m_.spill().any() || !ex_tally_) return {lookups_, lookups_, 0, 0};
        std::vector<uint32_t> t;
        if (cudaStreamSynchronize(stream_) != cudaSuccess || !read_tally(t)) return {lookups_, lookups_, 0, 0};
        const CardShare sh = card_share(t, g_.n_expert, [&](int l, int e) { return m_.on_card(l, e); },
                                        [&](int l, int e) { return m_.peer_held(l, e); });
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
        void* p = nullptr;
        int dev = 0;
        if (!ok || cudaGetDevice(&dev) != cudaSuccess || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
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
            if (cudaMemcpy(c.data(), k.ex_tally, c.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
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
        if (guess && vi > 0 && m_.card_of(g_.visits[(size_t) vi - 1].layer) == card) {   // its guess was made on this card
            const bk::NativeExpertLayout Lx = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
            cudaEventRecord(k.pf_ev, k.stream);
            cudaStreamWaitEvent(k.pf_st, k.pf_ev, 0);
            cudaStreamWaitEvent(k.pf_st, k.gs_ev, 0);
            expert_prefetch(x.gs_ids_ + (size_t) vi * kGuessExperts, kGuessExperts / rows * rows, w.blob_table, (uint64_t) Lx.down_off & ~(uint64_t) 15,
                            x.pf_sink_, k.pf_st);
            k.pf_on = true;
        }
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
        const Segments seg = segments(v.slot);   // a batch window's: each conversation's caches of this visit
        ap.seg = aa.seg = seg;
        attention_window(ap, aa, rows, c.ring ? (int64_t) g_.window : full_keys, x.att_scr_, k.stream);
        bk::native_q8_planes_mmvq(w.out, x.xo_, x.o_, kOutRows, E, rows, k.stream);
        // a wider window: the experts this visit took in the window before into L2 on the side stream, while the glue
        // before the experts runs (it starts once the output projection is done)
        if (!guess) {
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
        bk::router(nrm, w.router, w.router_bias, rows, ra, x.rlog_, x.ids_, x.rw_, k.stream);
        if (guess_next) {   // the next visit's experts: its router on this visit's MLP input, beside this visit's router
            const LayerWeights& nw = m_.layer(g_.visits[(size_t) vi + 1].layer);
            bk::RouterArgs ga = ra;
            ga.k = kGuessExperts / rows;
            cudaStreamWaitEvent(k.ra_st, k.ra_ev, 0);
            bk::router(nrm, nw.router, nw.router_bias, rows, ga, x.gs_log_, x.gs_ids_ + (size_t) (vi + 1) * kGuessExperts, x.gs_w_, k.ra_st);
            cudaEventRecord(k.gs_ev, k.ra_st);
            k.ra_on = true;
        }
        moe_plan(x.ids_, rows, g_.n_used, g_.n_expert, w.blob_table, x.ex_ptr_, x.ex_start_, x.ex_ng_, x.ex_dst_, x.ex_tok_, k.stream,
                 k.ex_tally + (size_t) l * g_.n_expert, k.rec_blobs + (size_t) vi * kAheadExperts, k.rec_n + vi);
        const bool on_cpu = cpu_ >> l & 1;   // the RAM experts' pairs on the CPU, beside the card's
        if (on_cpu) {   // open until finish has queued the join: a throw between leaves the fork outside the call's sync
            fork_open_ = true;
            sh_->cpu->start(l, rows, nrm, x.ids_, k.stream);
        }
        bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
        L.gate_softcap = g_.gate_softcap;
        L.up_softcap = g_.up_softcap;
        const int pairs = rows * g_.n_used;
        bk::native_expert_grouped(L, x.ex_ptr_, x.ex_start_, x.ex_ng_, x.ex_dst_, x.ex_tok_, std::min(pairs, g_.n_expert), pairs, x.xq_[0], x.ex_scr_,
                                  x.ex_rows_, k.stream);
        if (on_cpu) {
            sh_->cpu->finish(l, rows, x.ids_, x.ex_rows_, k.ex_tally + (size_t) l * g_.n_expert, k.stream);
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
    Segments segments(int slot) const {
        Segments s;
        if (!view_) return s;
        s.at = view_->w.device_segs();
        for (int i = 0; i < view_->n; ++i) {
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

    // The full-attention visits' launch bound of a call whose last row sits at `last`: its keys rounded up to a power of two.
    static int64_t full_keys_of(int64_t last) {
        int64_t full_keys = 1024;
        while (full_keys < last + 1) full_keys <<= 1;
        return full_keys;
    }

    // A call's inputs to the card: its tokens and positions (pos0.., or every row's own in `positions`), its first rows;
    // returns the full visits' launch bound.
    int64_t stage(const int32_t* tokens, const float* emb, int rows, int64_t pos0, const int32_t* positions) {
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        // the tokens and positions through a ring of pinned slots: a slot is written again once its copy has run
        const int slot = (int) (stage_n_++ % kStage);
        cudaEventSynchronize(stage_ev_[slot]);
        int32_t* io = io_h_ + (size_t) slot * 2 * kMaxRows;
        int64_t last = 0;
        for (int i = 0; i < rows; ++i) {
            io[i] = tokens ? tokens[i] : 0;
            io[kMaxRows + i] = positions ? positions[i] : (int32_t) (pos0 + i);
            last = std::max<int64_t>(last, io[kMaxRows + i]);
        }
        cudaMemcpyAsync(io_d_, io, (size_t) 2 * kMaxRows * 4, cudaMemcpyHostToDevice, stream_);
        if (Lane& k = lane_[1]; k.stream) {   // the positions to the second card too, after its last window, before this one
            cudaEventRecord(io_ev0_, stream_);
            cudaSetDevice(k.device);
            cudaStreamWaitEvent(k.stream, io_ev0_, 0);
            cudaMemcpyAsync(k.x->io_d_, io, (size_t) 2 * kMaxRows * 4, cudaMemcpyHostToDevice, k.stream);
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
        return full_keys_of(last);
    }

    // The layers with experts on the CPU (bit l: layer l) for the next window: part of its shape.
    uint64_t cpu_layers() const { return sh_->cpu ? m_.cpu_layers() : 0; }

    // The layers and the head of a window of `rows` rows: replayed from its shape's graph, captured on the shape's second
    // window (its first runs direct, outside a capture).
    bool layers(int rows, int head, int64_t full_keys, std::string& err) {
        const Shape key{cpu_, ((uint64_t) full_keys << 16) | ((uint64_t) head << 8) | (uint64_t) rows};
        auto it = graphs_.find(key);
        if (it != graphs_.end()) return it->second.launch(stream_, err);
        if (direct_.insert(key).second) {
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
        if (!g.end(stream_, err)) return false;
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
        const size_t n = (size_t) rows * g_.n_embd, slot = (size_t) kMaxRows * g_.n_embd;
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
        float* slot = sh_->hand + (size_t) (sh_->tap_slot0 + t) * kMaxRows * g_.n_embd;
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
        if (tapped && !view_) dspark_.commit(pos_d_, rows, stream_);   // DSPARK HOOK: the rows' context for the drafter
        if (tapped && view_)   // a batch window's: each conversation's rows of the shared tap into its own drafter's context
            for (int i = 0; i < view_->n; ++i) {
                const model::BatchWindow::Seg& sg = view_->w.segs[(size_t) i];
                static_cast<Session*>(view_->ss[i])->dspark_.commit(pos_d_ + sg.row0, sg.rows, stream_, sg.row0);
            }
        if (head <= 0) return;
        const size_t q8_row = (size_t) E / 32 * 36;   // the head's rows' q8_1 blocks follow the rows before them
        float* lg = view_ ? blog_ : logits_;          // a batch window's rows go to each session in batch_take
        bk::native_q8_0_mmvq(m_.head(), (const uint8_t*) xq_[0] + (size_t) (rows - head - first) * q8_row, lg, E, g_.n_vocab, head,
                             stream_);
        softcap(lg, head, g_.n_vocab, g_.n_vocab, g_.logit_softcap, stream_);
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
            const size_t n = (size_t) kMaxRows * history_len_;
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
    uint64_t cpu_ = 0;                                 // the window in flight's layers with experts on the CPU
    bool admitted_ = false;                            // holds admission to the CPU experts' windows (admit)
    bool fork_open_ = false;                           // a start whose finish has not queued its join
    const model::BatchView* view_ = nullptr;           // the batch window in flight, while batch_forward runs it
    float* logits_ = nullptr;                // the last window's head rows (row_logprobs, copy_logits, resample_row)
    uint32_t* ex_tally_ = nullptr;           // [layer][expert] the card's running count of each expert's lookups
    unsigned long long* rec_blobs_ = nullptr;   // [visit][kAheadExperts] the experts each visit took in the last window
    int32_t* rec_n_ = nullptr;                  // [visit] how many
    cudaStream_t ra_st_ = nullptr;              // the side stream (the read-ahead, the guesses), its fork and join events
    cudaEvent_t ra_ev_ = nullptr, ra_join_ = nullptr, gs_ev_ = nullptr;   // gs_ev_: a guess is in
    cudaStream_t pf_st_ = nullptr;              // the guesses' prefetch stream, its fork and join events
    cudaEvent_t pf_ev_ = nullptr, pf_join_ = nullptr;
    std::vector<uint32_t> tally_h_;
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
