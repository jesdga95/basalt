// include/basalt/core/verify.hpp - the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU.
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <functional>
#include <cstdio>

#include "basalt/core/expert_source.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::kernels::cpu { class ExpertPool; }
namespace basalt::kernels { struct DmMixLayer; }

namespace basalt::core {

class NativeHead;
class RemoteExpertOpt;
class PeerExperts;
struct DecMoe;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const int32_t* h_res = nullptr;      ///< host [n_layers * n_expert] slot or -1 (when all >= 0 in stage: zero-doorbell)
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   // host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;   // how many (for the device copy)
    int64_t blob = 0;
};

/// --profile: the GPU's idle time between a verify window and the draft round after it, and back (from globaltimer
/// stamps of the window's first and last kernels and the drafter's), summed for profile_report.
struct DecodeGaps {
    unsigned long long verify_end = 0, draft_end = 0, verify_prev_end = 0;
    double v2d = 0, d2v = 0, draft_gpu = 0;
    long long n_v2d = 0, n_d2v = 0;
    // host time between marks on the decode loop's path (mark(i) adds the time since the last mark to hms[i])
    bool host_on = false;
    double hms[32] = {};
    std::chrono::steady_clock::time_point hlast{};
    void mark(int i) {
        if (!host_on) return;
        const auto n = std::chrono::steady_clock::now();
        if (i >= 0) hms[i] += std::chrono::duration<double, std::milli>(n - hlast).count();
        hlast = n;
    }
};
DecodeGaps& decode_gaps();

/// --exp host_ev=1: stream events at the decode loop's graph launches (verify, commit, draft), resolved into GPU-side
/// spans and idle gaps per window at the end of a request.
struct DecodeEvents {
    enum Kind { kVerifyPre, kVerifyPost, kCommitPre, kCommitPost, kDraftPre, kDraftPost, kRunEntry, kKinds };
    bool on = false;
    std::vector<cudaEvent_t> ev;     // [window * kKinds + kind]
    std::vector<char> set;
    size_t window = 0;
    cudaEvent_t cal = nullptr;       // clock alignment: the host time of GPU time zero (ms)
    double cal_host = 0;
    std::vector<unsigned> tags;      // per window: what the host did around it (bit per kind of work)
    void tag(unsigned bits) { if (on && window < tags.size()) tags[window] |= bits; }
    std::vector<std::array<double, 10>> hs;   // per window: host times (ms) of the loop's stations
    void stamp(int i) {
        if (!on || window >= hs.size()) return;
        hs[window][(size_t) i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void enable(size_t max_windows);
    void next_window() { if (on) ++window; }
    void record(Kind k, cudaStream_t s, size_t ahead = 0);
    /// per-window medians and means (ms) of the spans and gaps; clears the trace
    std::string report();
};
DecodeEvents& decode_events();

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;
    void set_remote_expert_opt(RemoteExpertOpt* opt) { remote_opt_ = opt; } // before init/capture
    /// Before init: the window's main stream outranks other streams' kernels (a prompt read running beside the windows).
    void set_main_priority(bool high) { main_high_ = high; }
    /// The peer expert tier the window drives itself (persistent kernel, mapped rings) instead of the pool launching it
    /// per layer.  Before init.
    void set_peer(PeerExperts* peer) { peer_ = peer; }
    /// The CPU pool and the expert source the window's CPU tier runs on: between doorbells the parked workers predict
    /// the next layer's CPU experts (its router on this layer's FFN input) and read the likeliest into their caches.
    /// Before init; only with the device-planned MoE tail.
    void set_cpu_warm(basalt::kernels::cpu::ExpertPool* pool, ExpertSource* src) { warm_pool_ = pool; warm_src_ = src; }

    /// The watchdog's view of the window in flight: the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;
    /// the device plan's words while a window is stalled - the ring CUDA0 last published and its tiers, and on the peer
    /// card the ring it was rung for, the ring it finished and whether its kernel still runs.  Empty without the
    /// device plan.
    std::string dm_note() const;

    /// raise every flag the window's spin kernels wait on past any ring (UINT32_MAX), so a window the GPU cannot finish
    /// drains instead of staying resident, then wait up to `timeout_ms` for its streams.  For the paths that give up on
    /// the engine (a timed-out window, the serve watchdog): the window then ran on whatever the flags guarded, so this
    /// verifier refuses every later window.  True when the streams finished.
    bool release_gpu_waits(int timeout_ms);

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);
    /// The window capacity in rows given to init.  A batch planner must never build a window with more rows than this;
    /// kVerifyMaxT only bounds the graphs, not this verifier.
    int max_t() const { return max_t_; }

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Diagnostics: row `t` of the last window's head logits (n_vocab floats) to the host. Valid after run().
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return next_ ? next_->vocab() : n_vocab_; }
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const basalt::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        sampling_.logit_bias = logit_bias_host_.empty() ? nullptr : logit_bias_device_;
        if (next_) next_->set_sampling(sp);
    }
    /// One request's token bias (one float per vocabulary entry, added to the logits before the sampler chain); empty
    /// clears it.  Applies to this verifier's own sampling only: the drafter never sees it.  Call before set_sampling;
    /// copied to the final stage's device only when it changed.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);
    /// Draws row `t` of the window just run again, with `bias` (one float per vocabulary entry, -inf bans) in place of
    /// the request's logit_bias; the penalties, the filters and the position's draw are the window's own. `pos0` is the
    /// window's first position. The token is in `token`; the window's other rows are untouched.
    bool resample_row(int t, int64_t pos0, const std::vector<float>& bias, int32_t& token, std::string& err);

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len` int32
    /// slots at that stride (`basalt::kernels::penalty_rows` builds them), most recent token LAST, unused front slots
    /// -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t.  Null disables the
    /// penalties entirely - the neutral run's sampling call is byte-for-byte the plain one.  The engine re-uploads the
    /// rows before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose picks are
    /// discarded - a prompt read through windows commits every token - so they cost no sampler launch or sync and
    /// never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }
    /// On: a sampled window also leaves draft_prob(t), the probability row t's final distribution gives the window's
    /// token t + 1 (t < T - 1), for relaxed acceptance.  The picks are unchanged.  Off by default; set per request.
    void set_draft_probe(bool on) { draft_probe_ = on; if (next_) next_->set_draft_probe(on); }
    /// Valid after a sampled run() with the probe on.
    float draft_prob(int t) const { return next_ ? next_->draft_prob(t) : ((volatile float*) h_qp_)[t]; }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }
    /// The 8-bit copies of every layer's hyper-connection matrices (and the head's) on the current device; called
    /// before the expert cache takes the free memory.
    static void prepare_hc8(const WeightTable& wt, const ModelGeometry& g);
    /// The CUDA0 bytes the GDN mixer split by key head loads after the expert cache is sized (CUDA0's packed rows and
    /// columns of its key heads), for CUDA0 on dev0 and the peer on dev1; 0 when the split is not taken.
    static int64_t gs_cuda0_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1);
    /// The peer bytes the split loads after the peer's expert tier is filled (its key heads' rows, columns, gates,
    /// state slices and the mixer's rows); 0 when the split is not taken.
    static int64_t gs_peer_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1);
    /// The peer bytes the head split by vocabulary loads after the peer's expert tier is filled, for a head of
    /// `row_bytes` per row and native `type`, windows of up to max_t tokens; 0 when the split is not taken.
    static int64_t hs_peer_bytes(size_t row_bytes, int type, int64_t n_vocab, int64_t n_embd, int max_t, int dev0, int dev1);
    /// The bytes prepare_hc8 will place when `has` names the tensors the model holds (0 with the copies off).
    static uint64_t hc8_plan_bytes(const std::function<bool(const std::string&)>& has, const ModelGeometry& g);

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.  `parts`: 1 launches
    /// the commit graph, 2 advances `ss.ple_prev` (3: both, in that order).
    bool commit(int n_keep, std::string& err, int parts = 3);
    /// The decode loop's host work off the GPU's critical path (see run() and the serve loop)
    static bool chain_on();
    /// the residency tables uploaded now (between windows) rather than at the next run()
    bool dm_presync(std::string& err) { return dm_ == nullptr || dm_sync(err); }
    /// Work queued on `s` after this call starts only once the window stream has run everything queued on it so far (the
    /// residency uploads of dm_presync), so a large copy on `s` cannot hold the small uploads back in the copy queue.
    bool dm_fence(cudaStream_t s);
    /// `f` runs on the calling thread of the next `run`, right after the window's graph is launched.
    void set_post_launch(std::function<void()> f) { post_launch_ = std::move(f); }
    /// runs it now if no window has since (a request ending)
    void flush_post_launch() { if (post_launch_) { auto f = std::move(post_launch_); post_launch_ = nullptr; f(); } }

    // ================================ SEVERAL SEQUENCES IN ONE WINDOW ================================
    //
    // A batch window holds the SEGMENTS of several INDEPENDENT sequences: segment s is `lens[s]` consecutive tokens
    // of slot s at the slot's own positions, reading and writing slot s's own state (GDN recurrence and conv
    // history, QSA K/V and indexer, PLE history), which lives in `slots[s]` - a session carved like this
    // verifier's own (same layer range, same max_cells).  Everything that is per row (hyper-connections, dense
    // projections, router, shared expert, the routed experts on the GPU and the CPU, the head) runs once over the
    // window's rows, so the weights are read once per window for all the sequences.  A segment's rows are a solo
    // window of that slot, so a slot's greedy tokens are its solo greedy tokens (modulo the multi-token CPU kernel
    // choice).  The window graph is keyed by the row count alone; the split into segments is data.  Each segment
    // then keeps its own number of rows (commit_slots), as a solo commit does.
    //
    // No drafting here: the caller supplies the tokens of every segment.  `init_slots` once after `init` (S <= max_t); a layer
    // split's stages each get their own sessions, and run_slots/commit_slots continue into the next stage.
    bool init_slots(const std::vector<SessionState*>& slots, std::string& err);
    int n_slots() const { return (int) slots_.size(); }
    /// Captures every window width's graph now (the solo windows, and the batch windows when slots exist), so no
    /// request pays a capture while others decode.  After the verifier is configured.
    bool warm_graphs(std::string& err);
    /// One batch window over slots [0, S): tokens[s] at positions pos[s]; out[s] = the greedy pick after it.
    bool run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err);
    /// The same over the S slots `rows` (row t is slot rows[t], any distinct slots in any order): the slots not
    /// listed are not touched, so an idle slot keeps its state (a finished conversation it may continue later).
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    /// A window of S segments: segment s feeds lens[s] tokens (concatenated in `tokens`, at most max_t in all) of
    /// slot slots[s] from position pos0[s]; out[r] = the greedy pick after row r.
    bool run_slot_segs(const int* slots, const int* lens, int S, const int32_t* tokens, const int64_t* pos0,
                       PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Keep every row of the last batch window: each slot's state advances by its tokens.
    bool commit_slots(std::string& err);
    /// Keep the first n_keep[s] (1..lens[s]) rows of segment s: its slot's state advances by those tokens only and
    /// is exactly what a solo commit of that many leaves.
    bool commit_slots(const int* n_keep, std::string& err);
    /// The same in two halves: `begin` stages the counts and launches the commit graph, `end` waits for it.  What
    /// does not read the committed state (the draft layer from the window's rows) can run between them.
    bool commit_slots_begin(const int* n_keep, std::string& err);
    bool commit_slots_end(std::string& err);

    // ---- The stages of a layer split as a PIPELINE.  A batch window over the slot GROUP
    // [base, base + S) is launched on ONE stage with its commit right behind it on the stage's stream (a batch window
    // keeps every row, so the commit needs no host decision), and the host serves the rings of every stage that has
    // a window in flight from one thread (batch_poll does not block).  Stage k can then run group g while stage k+1
    // runs group g-1.  Rows of group `base` use hand-off rows [base, base + S), so groups never share a hand-off row.
    bool batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = this stage's window and commit are done (the last stage's picks are in batch_out), 0 = still running,
    /// -1 = an error (err).  Serves every layer that has rung so far.
    int batch_poll(PoolMultiFn pool, void* user, std::string& err);
    bool batch_busy() const { return b_running_; }
    /// A slot's sampling (temperature / top_p / top_k / min_p / seed; penalties are not applied in batch windows):
    /// its row is drawn again on the last stage with Philox(seed, position), as a solo window draws it.  Greedy by
    /// default.  Set on the first stage, it reaches the last.
    void set_slot_sampling(int slot, const basalt::kernels::SamplerParams& sp) {
        if (slot >= 0 && slot < (int) slot_sp_.size()) slot_sp_[(size_t) slot] = sp;
        if (next_) next_->set_slot_sampling(slot, sp);
    }
    const int32_t* batch_out() const { return b_out_; }
    bool last_stage() const { return g_ != nullptr && le_ == g_->n_layers; }
    /// commit() returns without waiting for its graph (a single-GPU session sets it): the next window follows it on
    /// the same stream and the drafter reads nothing it writes, so it overlaps the draft. Whoever reads or writes
    /// the session from another stream or the host afterwards (a new request, a checkpoint, a snapshot, the prompt
    /// path, the end of a run) calls wait_commit() first.
    static void set_commit_async(bool on);
    /// Waits for the last commit graph when commit() did not (an event recorded after it, not the whole device);
    /// false with `err` when it failed.  Free when nothing is pending.
    bool wait_commit(std::string& err);
    /// The solo windows (run, commit) work on session `s` from here on: a session carved like init's, kept at one
    /// address (its graphs are keyed by it).  Back to init's session: bind_session(&ss).  No copy of any state; a
    /// session's graphs are captured on its first use, or ahead of it by warm_session.
    bool bind_session(SessionState* s, std::string& err);
    /// Captures every solo window width's graph and the commit for session `s` now.
    bool warm_session(SessionState* s, std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    double ms_launch = 0;   ///< host time inside the window graph's launch
    int64_t windows = 0;
    /// --profile: GPU stage times of the windows since the last call (ms per window), as one line; empty when off.
    std::string profile_report();

private:
    RemoteExpertOpt* remote_opt_ = nullptr;
    bool main_high_ = false;
    PeerExperts* peer_ = nullptr;
    basalt::kernels::cpu::ExpertPool* warm_pool_ = nullptr;
    ExpertSource* warm_src_ = nullptr;
    DecMoe* dm_ = nullptr;   // the device-planned MoE tail (null: off)
    bool dm_init(std::string& err);
    // the head split by vocabulary with the peer card, when the peer streams its rows faster than CUDA0 does
    void hs_init();
    void hs_free();
    // the GDN mixer split by key head with the peer card, when the peer streams the mixer's bytes faster than CUDA0
    void gs_init();
    void gs_free();
    bool dm_sync(std::string& err);        ///< the residency tables to the devices, before a window
    bool dm_begin(std::string& err, DecodeGaps* gm);   ///< the tail's window start (tables, flags, the peer window)
    bool dm_end(std::string& err);         ///< the peer window's end
    void dm_free();
    std::string peer_released();           ///< after a release: whether the peer window ended (see dm_note)
    bool capture(int T, std::string& err);
    bool pdl_rec_ = false;   // record_window may use programmatic dependent launch
    bool pdl_off_ = false;                 ///< ... a window failed to capture with it: the later ones go without
    // batch windows (see init_slots)
    std::vector<SessionState*> slots_;
    bool batch_rec_ = false;               ///< record_window is capturing a batch window
    int row_base_ = 0;                     ///< ... its hand-off rows start here (a pipeline group's own rows)
    bool last_batch_ = false;              ///< the last run was a batch window (set_plan_slot: one group)
    /// One window graph per (hand-off base, row count, doorbell variant) and one commit graph per row count.  Which
    /// slots the rows are is data (the staged segment tables), so a membership change re-records nothing.
    std::map<uint64_t, cudaGraphExec_t> exec_bm_;
    std::map<int, cudaGraphExec_t> commit_bm_;
    int last_rows_[8] = {};                ///< the slot of each row of the last batch window
    int last_nseg_ = 0, last_seg_slot_[8] = {}, last_seg_len_[8] = {};   ///< its segments
    std::chrono::steady_clock::time_point commit_t0_;   ///< commit_slots_begin
    std::vector<int> commit_next_keep_;                 ///< its counts, for the next stage
    uint64_t bkey(int S, int hbase) const {
        return batch_key(S, hbase) | (ar_off_ ? (1ull << 63) : 0ull);
    }
    static uint64_t batch_key(int S, int hbase) { return (uint64_t) hbase << 40 | (uint64_t) S << 32; }
    /// Instantiates a finished batch capture as the executable of a new key.
    bool instantiate_batch(cudaGraphExec_t& ex, cudaGraph_t graph, std::string& err, bool priority = false);
    // batch_launch / batch_poll
    bool b_running_ = false;
    int64_t b_k_ = 0, b_steps_ = 0;
    std::chrono::steady_clock::time_point b_last_;
    int32_t b_out_[8] = {};
    std::vector<basalt::kernels::SamplerParams> slot_sp_;
    bool sample_rows(int S, std::string& err);   ///< the sampled slots' rows of the last batch window
    int32_t* h_commitb_ = nullptr; int32_t* m_commitb_ = nullptr;   // per slot [1, 0, pos, -1 ..], stride 2 + max_t
    int32_t* commitb_ = nullptr;
    float* tail_snap_b_ = nullptr;         ///< per (slot, QSA layer) indexer tail snapshot
    /// The window's segment table, one entry per row: staged from the host (mapped) at the start of the window and of
    /// the commit, so the GDN kernels take slot identity as data.
    basalt::kernels::SegRow* h_segb_ = nullptr; basalt::kernels::SegRow* m_segb_ = nullptr;
    basalt::kernels::SegRow* segb_ = nullptr;
    /// The QSA chain's segment tables (qsa_seg.hpp): one entry per row and per segment, staged with the segment table;
    /// the per-layer, per-slot pointers are uploaded once.
    basalt::kernels::QsaSeg* h_qseg_ = nullptr; basalt::kernels::QsaSeg* m_qseg_ = nullptr;
    basalt::kernels::QsaSeg* d_qrows_ = nullptr;   ///< kQsaSegMaxRows row entries, then the segment entries
    basalt::kernels::QsaSeg* d_qsegs_ = nullptr;
    std::vector<basalt::kernels::QsaSegSlot*> d_qslots_;   ///< per QSA layer ordinal: [slot] (null outside the stage)
    float** d_ple_slots_ = nullptr;        ///< each slot's PLE history
    float* ple_work_ = nullptr;            ///< each row's PLE history while the window runs
    bool batch_dm_ = false;                ///< a batch window runs the device-planned tail
    void* arena_b_ = nullptr;
    int64_t last_pos_b_[8] = {};
    bool capture_batch(int R, int hbase, std::string& err);
    void collect_profile();   // --profile: add the last window's stamps to prof_sum_
    bool capture_commit_batch(int R, std::string& err);
    bool stage_batch(const int* slots, const int* lens, int S, int hbase, const int32_t* tokens, const int64_t* pos0,
                     std::string& err);
    bool batch_seg_ple_ = true;            ///< the window's PLE runs the window kernels (segments of several rows)
    basalt::kernels::SamplerParams sampling_ = [] {
        basalt::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   // penalty-history row (set_history); null = no penalties apply
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
    bool draft_probe_ = false;           ///< set_draft_probe
    std::vector<float> logit_bias_host_;
    float* logit_bias_device_ = nullptr;
    float* row_bias_device_ = nullptr;   ///< resample_row's bias
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    std::atomic<bool> released_{false};   // release_gpu_waits ran (maybe on the watchdog thread): no more windows
    bool all_resident_ = false;           ///< 100% of experts in [lb_, le_) resident in VRAM: zero-doorbell graph
    /// The zero-doorbell graph plans from the device residency table alone, so it is right only while every expert of
    /// the stage is in VRAM.  A prompt loan, a VRAM shrink or an adaptive swap marks some -1: the windows then run the
    /// doorbell graph (exec_nr_ / the batch key's top bit).  refresh_ar() reads the host table before each window.
    bool ar_off_ = false;
    bool ar_on() const { return all_resident_ && !ar_off_; }
    void refresh_ar();
    uint32_t* h_plan_err_ = nullptr; uint32_t* m_plan_err_ = nullptr;   // set by resident_plan on a -1 (zero-doorbell graph)
    bool device_plan_ = false;   // resident-only layers planned on the device
    uint32_t* skip_ = nullptr;   // per group, the ring whose plan the device built (0: the host's)
    unsigned long long* slot_off_d_ = nullptr;   // the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    void stage_inputs(int T, const int32_t* tokens, int64_t pos0);
    bool staged_ = false;
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);
    double prof_graph0_ = 0;                        // the graph's start to layer 0's first stamp (ns, summed)
    static constexpr int kProfPer = 33;              // stamps per layer (32 left the hc-read second
                                      // half's up-stamp at slot 32 = the next layer's slot 0: D8)
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaStream_t sh_cs_ = nullptr;
    cudaStream_t sh_hi_ = nullptr;   // the shared expert's side stream at the greatest priority
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr;
    cudaStream_t qs_[3] = {nullptr, nullptr, nullptr};   // a QSA layer's branches
    cudaEvent_t qev_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaStream_t ring_hi_ = nullptr;   // the MoE ring's stream at the greatest priority
    cudaEvent_t ev_ring_ = nullptr, ev_ring_join_ = nullptr;
    cudaStream_t ple_s_ = nullptr;   // layer 1's PLE projections, from the window's start
    cudaEvent_t ev_ple_fork_ = nullptr, ev_ple_join_ = nullptr;
    bool ple_side_ = false;          // the window's graph computes them there (the host raises h_flag_[8] with the rows)
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t exec_nr_[9] = {};   // the doorbell variant of a stage that is all-resident otherwise
    cudaGraphExec_t commit_exec_ = nullptr;
    struct SoloGraphs {
        cudaGraphExec_t exec[9] = {}, exec_nr[9] = {}, commit = nullptr;
    };
    std::map<SessionState*, SoloGraphs> solo_sets_;   // the solo graphs of the sessions not bound now

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    int32_t* h_q_ = nullptr;     int32_t* m_q_ = nullptr;       // the probe's tokens: the window's drafts
    float* h_qp_ = nullptr;      float* m_qp_ = nullptr;        // the probe's probabilities
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    std::function<void()> post_launch_;
    cudaEvent_t fence_ev_ = nullptr;      // dm_fence
    cudaEvent_t commit_done_ = nullptr;   // recorded after an async commit (set_commit_async); see wait_commit
    bool commit_pending_ = false;
    // The async commit's own stream, after an event on cs_.  The next window's first node (the finisher) takes what the
    // commit has not claimed and returns once it is complete; h_pend_ is the epoch whose begin kernel has run.
    cudaStream_t cm_ = nullptr;
    cudaEvent_t ev_commit_go_ = nullptr;
    basalt::kernels::CommitCtl* commit_ctl_ = nullptr;
    uint32_t* h_pend_ = nullptr; uint32_t* m_pend_ = nullptr;   // the epoch the commit's begin kernel last started
    uint32_t commit_epoch_ = 0;
    int fin_blocks_ = 0;
    bool finisher_on() const;                        ///< solo windows start with the finisher
    bool commit_begun() const;                       ///< the begin kernel of the last async commit has run
    void order_after_commit();                       ///< a launch on cs_ outside the solo windows follows the async commit
    basalt::kernels::GdnCommitLayers commit_gdn_args() const;
    static void raise_flag_b(void* ctx);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot();
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* sh_xq_ = nullptr;                                // T columns of q8_1 for shared expert branch
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;                  // the PCIe call's groups side by side (of <= 16)
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    // head split: CUDA0 computes rows [0, hs_rows_), the peer the rest; hs_rows_ 0 means no split
    int hs_rows_ = 0, hs_dev_ = -1;
    void* hs_w_ = nullptr;           // peer: head rows [hs_rows_, n_vocab_)
    void* hs_xq_ = nullptr;          // peer: the head's q8_1 input
    float* hs_y_ = nullptr;          // peer: its logits, T rows of n_vocab_ - hs_rows_
    float* hs_y0_ = nullptr;         // CUDA0: its logits, T rows of hs_rows_
    void* hs_hxq_ = nullptr;         // pinned staging of the input
    float* hs_hy_ = nullptr;         // pinned staging of the peer's logits
    cudaStream_t hs_s_ = nullptr;    // peer
    cudaEvent_t hs_in_ = nullptr;    // CUDA0: the input is staged
    cudaEvent_t hs_out_ = nullptr;   // peer: its logits are staged
    // GDN mixer split by key head: CUDA0 owns key heads [0, gs_nk0_), the peer the rest; gs_nk0_ 0 means no split
    struct GsLayer {   // peer: one GDN layer's weights for its heads (gdn_head_split.hpp layout)
        void *wqkv = nullptr, *wz = nullptr, *wo = nullptr;
        void *wqkv0 = nullptr, *wz0 = nullptr, *wo0 = nullptr;   // CUDA0: its heads' packed qkv/z rows, out columns
        float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *gamma = nullptr;
        uint16_t *wa = nullptr, *wb = nullptr;
        int tq = -1, tz = -1, to = -1;
    };
    int gs_nk0_ = 0, gs_dev_ = -1;
    std::vector<GsLayer> gs_layers_;   // by GDN layer index
    float* gs_state_ = nullptr;        // peer: its heads' recurrent state and conv history, per GDN layer
    // the mixer doorbell (DmPeerArgs::mix): mapped q8_1 rows per ring; peer: which rings have one, the rows' copy
    uint8_t *gs_h_xq_ = nullptr, *gs_m_xq_ = nullptr;
    uint8_t* gs_mix_ = nullptr;
    void* gs_pxq_ = nullptr;
    // mapped: the fp32 rows per ring and the peer's out-projection partial; peer: the layer table and its scratch;
    // CUDA0: its heads' packed qkv / z
    float *gs_h_x_ = nullptr, *gs_m_x_ = nullptr, *gs_h_part_ = nullptr, *gs_m_part_ = nullptr;
    basalt::kernels::DmMixLayer* gs_mixl_ = nullptr;
    float* gs_pscr_ = nullptr;
    float *gs_qkv0_ = nullptr, *gs_z0_ = nullptr;
    uint8_t* vhead_scratch_ = nullptr;   // the two-stage head's histogram and row list (q6k_vhead)
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    float* ple_win_ = nullptr;   // 2 * T * NG_HC_DIM
    float *ple_key_ = nullptr, *ple_val_ = nullptr;   // the PLE key / value projections: T * HC * N, T * N
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

/// The loaded hc8 table (model_file.hpp) has a copy of `name` of this shape.
bool hc8_table_has(const std::string& name, int rows, int cols);
/// Registers the table's copy of `name` on the current device, keyed by the bf16 matrix `w` (hc8_prepare's form) or,
/// with w == nullptr, by the copy itself (hc8_adopt's form; the key is returned).  nullptr: no such copy, or no VRAM.
const void* hc8_table_adopt(const std::string& name, const uint16_t* w, int rows, int cols);
/// --dump-hc8: writes the 8-bit copies of the weight table's hyper-connection matrices, and `extra` (name, key) pairs
/// (the drafter's), to `path` in the layout model_file.hpp describes.
bool hc8_dump(const WeightTable& wt, const ModelGeometry& g, const std::vector<std::pair<std::string, const void*>>& extra,
              const std::string& path, std::string& err);

}  // namespace basalt::core
