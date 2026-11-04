// src/models/qwen4exp/serve/qfn_state.hpp - Flash-Next's serving state, ordered by resource lifetime.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include "../../../program/serve/state.hpp"
#include "qfn_runtime.hpp"
#include "basalt/model/adapter.hpp"
#include "../../../program/cli/tokens.hpp"
#include "loans.hpp"
#include "loan_moves.hpp"
#include "basalt/models/qwen4exp/options.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/models/qwen4exp/kernels/cvec.hpp"
#include "basalt/core/logit_bias.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/pinned.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/program/image_loan.hpp"
#include "basalt/program/multi_gpu_plan.hpp"
#include "basalt/program/tokenizer.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/mrope.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/native_dense.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/ppl_gate.hpp"
#include "basalt/program/probe_rates.hpp"
namespace basalt::models::qwen4exp { std::shared_ptr<basalt::program::ProbePeerPath> make_probe_peer_path(); }
#include "basalt/spec/draft_policy.hpp"
#include "basalt/spec/suffix_drafter.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"
#include <cerrno>
#include <cstddef>   // std::ptrdiff_t

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <set>
#include <tuple>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::models::qwen4exp::serve {
namespace serve_loans = basalt::models::qwen4exp::serve;
using engine_local::Drive;
using engine_local::GpuStage;
using engine_local::JobRunner;
using engine_local::PromptStage;
using engine_local::SplitDrive;
using engine_local::biggest_chunk;
using engine_local::drive_pool_multi;
using engine_local::drive_pool_split;
using engine_local::kSatBoost;
using engine_local::peer_hot_share_order;

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again would make a long session wait for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * positional state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * running state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold its tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ConvStateSizes = basalt::core::ConversationStateSizes;

// A round that found at least kSatRatio times --adapt-swaps misses to swap (the pipeline is saturated) lets the next
// round swap kSatBoost times --adapt-swaps into the main card; the free RAM places are sized for it.  Fewer misses than
// that are not worth the extra copies.
inline constexpr int kSatRatio = 2;
/// Pipelined windows: the least chance (the cumulative draft rule's product up to the guess) a window's successor is
/// launched on, and how far below the window threshold the chain runs on past its window so there are drafts for it.
inline constexpr float kPairTheta = 0.50f;
inline constexpr float kPairChainScale = 0.25f;
inline constexpr double kPairFull = 0.70;   ///< the share of windows accepted whole from which the chain runs on

ConvStateSizes conv_state_sizes(const basalt::core::ModelGeometry& g, const basalt::core::SessionState& ss);
bool checkpoint_save(ConvCheckpoint& c, const basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g);
bool checkpoint_restore(const ConvCheckpoint& c, basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g);
uint64_t conversation_identity(const Options& o, const basalt::core::ModelGeometry& g);
void seed_usage_from_prompt(std::vector<float>& usage, const std::vector<uint32_t>& hist, int64_t tokens,
                            const std::vector<int32_t>& host_res, int64_t n_expert, const basalt::core::PeerExperts& peer);


class QfnRequest;

// Flash-Next's serving state over the shared one.
class QfnRuntime final : public Runtime, private Context {
public:
    explicit QfnRuntime(Context& context)
        : Runtime(context.o, context.model_file, context.err, context.n_vocab, context.grammar_vocab), Context(context) {}
    int init_before_input() override;
    int init_after_input() override;
    void after_input_start() override;
    void print_info() override;
    void at_exit() override;
    bool windows_in_flight() override;
    bool next_batch_window() override;
    bool drain_windows() override;
    void script_command(const std::string& line) override;
    void probe_command(const std::string& line, std::string& e) override;
    void sleep_command(bool down, std::string& e) override;
    void between_requests(const std::string& line) override;
    Flow run_request() override;
    bool slot0_solo() const override { return slot0_is_solo; }
    bool pipelined() const override { return piped; }
    int64_t slot_context(int slot) const override { return slot_cap[(size_t) slot]; }

private:
    friend class QfnRequest;
    using Context::o;
    using Context::model_file;
    using Context::err;
    using Context::n_vocab;
    using Context::grammar_vocab;
    friend class Request;
    basalt::prefill::Prefill sp{};
    void* borrow{};
    uint64_t borrow_bytes{};
    int32_t lend_first{};
    // ---- who borrows, and from whose cache.  One entry per prompt path: CUDA0's (layers [0, split_at[0]), which is
    // the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage borrowing the tail
    // of its own cache.  A loan is sized by the exact `Prefill::bytes_needed` for the chunk, is laid out by
    // `Prefill::relayout`, and is refilled before any window reads - so outside the prompt the whole cache is
    // expert cache.  A stage may only lend the rows for its own layers: `host_res` is one table whose slot values
    // are indices into whichever cache owns the layer, so a loan that marked rows by slot number alone would hand
    // CUDA0 a slot belonging to another stage's cache.
    struct PfPart {
        basalt::core::ExpertCache* cache = nullptr;
        const basalt::core::SessionState* ses = nullptr;
        basalt::prefill::Prefill* sp = nullptr;
        int dev = -1;                  // -1: leave the device alone (CUDA0)
        int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
        int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
        int32_t first_now = -1;        // where its buffers are laid out now
        int64_t lent_chunk = 0;
        std::vector<std::pair<int32_t, int32_t>> lent;
    };
    std::vector<PfPart> pf_parts{};
    // A prompt read beside decoding slots lends at most this many cache slots, so their experts stay in the
    // cache, and its streamed ring is this deep (the loan shrinks with it); otherwise the ring is full depth.
    static constexpr int64_t kLendCapDecoding = 1216;
    static constexpr int kRingDecoding = 64;
    // A read of at most this many tokens runs alone even while slots decode: they wait for it (a few seconds at
    // most) and it lends and reads as on an idle engine.  A longer read goes beside them, within the limits above,
    // and so does the rest of one that gave way to a shorter request (in its chunk).
    static constexpr int64_t kReadAlone = 16384;
    bool any_loan{};
    static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
    int64_t split_chunk_max{};
    bool split_ready{};
    std::vector<uint32_t> split_hist{};   // the routed entries the stages counted in the last split read
    int64_t split_hist_tokens{};
    // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last `penalty_last_n`
    // tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for the widest window; a
    // request without penalties gets a null buffer and takes the byte-for-byte neutral path (no upload, no buffer
    // handed to the sampler).
    static constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap* (size_t) basalt::kernels::kVerifyMaxT;
    int32_t* d_hist{};
    std::vector<int32_t> hist_stage{};
    int hist_dev{};
    basalt::core::Verifier ver{};
    basalt::core::VerifyHits vh{};
    // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
    // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
    basalt::core::Verifier ver_same{};
    SplitDrive split_drive{};
    SplitDrive split_drive_b{};
    basalt::core::Verifier ver_b{};   // pipelined windows: the second chain over the same sessions (stage 0)
    void* pair_snap{};
    int n_stages{};
    bool pipe_on{};
    basalt::core::PoolMultiFn win_pool_fn{};
    void* win_pool_user{};
    // A turn-boundary checkpoint's GDN bytes reach the host after the first token (see the deferral class); this
    // is declared after `checks` so it is destroyed, and its copy finished, before the checkpoints are.
    basalt::core::ConversationCheckpointDeferral ckd{};
    std::vector<basalt::core::SessionState> rviews{};
    basalt::core::SessionState* rsp{};
    // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
    // the next, so each stage saves its own part when it reaches a checkpoint position (the same rule as the main card's:
    // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
    std::mutex part_mu{};
    std::map<int64_t, std::vector<ConvCheckpoint>> part_at{};   // position -> one part per stage
    std::vector<int64_t> part_next{};
    bool all_experts_resident{};
    // --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays empty
    // and nothing below runs).  It needs the adaptive tier's counts and the residency table.
    std::vector<double> heat{};
    Clock::time_point profile_saved_at{};
    cudaStream_t adapt_stream{};
    // swaps in flight - (residency index, slot) admitted when adapt_ev has completed
    std::vector<std::pair<int32_t, int32_t>> pending{};
    std::vector<std::array<int32_t, 3>> promoted{};   // (layer, in, out) of swaps from the peer
    cudaEvent_t adapt_ev{};
    bool peer_landing{};
    std::vector<uint8_t> xlanding{};
    std::vector<std::pair<int32_t, int32_t>> pending_spare{};   // the RAM places the incoming experts free once resident
    // the VRAM tier follows the conversation
    // the peer's experts are candidates too, so the ones a workload routes most move to
    // CUDA0 (from their RAM copy, as any other swap) and its coldest go to the CPU
    // routings a peer expert needs above CUDA0's victim before it moves
    static constexpr float promote_margin = 10.0f;
    // the count at which a CPU-tier expert routed about once lately takes a slot the round leaves unused
    static constexpr float once_bar = 1.0f;
    // at most this many of them a round: more copies cost the CPU tier host memory bandwidth on routing that rarely
    // comes back (JSON output)
    static constexpr size_t kOnceSwaps = 16;
    // the round's swaps, best first: the most-routed missing experts against the least-routed resident ones, per
    // layer, or over all layers (`across`: the incoming expert takes a slot of the outgoing one's layer `lout`)
    struct Swap { float gain; int32_t layer, in, out, lout; };
    // a layer whose peer experts take more than this share of its 10 routed entries per token promotes them without
    // the margin (a layer's peer share, not the step's, sets how late the peer is)
    static constexpr float promote_load = 0.0f;
    std::vector<int32_t> prof_rank{};
    bool adapt_fast{};
    float fast_decay{};
    struct XSwap { int32_t layer, in, out, lout, slot; uint64_t off; bool xchg, promo, p2p; };
    bool peer_excl{};
    bool xq_across{};
    std::vector<XSwap> xq{};
    int xq_stage{};
    bool xq_saturated{};
    // since the start: the swaps the rounds queued, and those a round left for a later one (no free RAM place)
    int64_t xq_swapped{};
    int64_t xq_noplace{};
    // The landed words (Verifier::landed_words): a round's incoming copies are queued in layer order, each layer's
    // followed by a 4-byte copy of the round's number into the layer's word.  The copies' sources are a pinned
    // ring of rounds (every layer's word, then the round's number).
    static constexpr int kXqPinRounds = 4;
    uint32_t xq_no{};
    bool xq_words{};
    uint32_t* xq_pin{};
    cudaEvent_t p2p_ev{};
    bool xq_unsettled{};
    std::vector<std::pair<int32_t, int32_t>> xq_released{};
    int S{};
    int S_mtp{};
    std::vector<int32_t> res_at_start{};
    std::vector<int32_t> peer_at_start{};
    std::vector<int32_t> prof_rank_start{};
    int64_t rounds{};
    std::optional<basalt::spec::SuffixDrafter> sfx;
    std::optional<basalt::spec::DraftPolicy> policy;
    bool mrope_identity{};
    std::vector<float> img_rows{};
    std::vector<const float*> row_ptr{};
    std::vector<std::pair<int32_t, int32_t>> vram_evicted{};   // (residency index, slot) the shrinks took
    int64_t chunk_full{};
    int64_t kKvGrowCells{};
    static constexpr int64_t kKvWindowCells = 2* basalt::kernels::kVerifyMaxT + 64;   // a decoding session's next window
    static constexpr int64_t kPeerTakeCells = 32768;   // a fit backing this many new cells hands its experts to the peer
    int64_t kv_moves{};
    int64_t kv_ceiling_sent = -1;   // the elastic KV's ceiling the server was last told (INFO elastic_kv_ceiling)
    bool kv_short{};
    bool kv_keep_routed{};   // an admission while slots decode: a shrink keeps their routed experts
    // a solo request's fits before its prompt read leave their handoffs to the peer to that read: (peer slot, its
    // expert, the expert it takes), done by the split's loan of the slot or else before the first window
    bool peer_defer{};
    std::vector<std::array<int32_t, 3>> peer_due{};
    int64_t peer_due_off{}, peer_due_lent_n{};
    bool batch_spec{};
    /// the solo conversation's draft K/V waits in slot 0's store while a request in another slot holds the main one
    bool solo_draft_in0{};
    int S_all{};
    std::optional<basalt::spec::DraftPolicy> bpolicy;
    double bt_draft{};
    double bt_draft_last{};
    int64_t bt_slotwin{};
    int64_t bt_dr_off{};
    int64_t bt_dr_acc{};
    std::vector<std::pair<std::vector<int32_t>, uint64_t>> bscript{};
    bool bdigest{};
    bool logit_hash{};
    std::vector<float> lh_buf{};
    bool batch_adapt{};
    int64_t ba_win{};
    int64_t ba_swaps{};
    int64_t ba_rounds{};
    int64_t ba_swaps_l{};
    int64_t ba_rounds_l{};
    int64_t ba_h_l{};
    int64_t ba_m_l{};
    int64_t ba_noplace{};   // the batch rounds' swaps deferred for want of a free RAM place
    double ba_ms{};
    JobRunner ba_thr{};   // queues the adaptive copies beside the windows
    bool ba_job{};
    double bt_run{};
    double bt_commit{};
    double bt_emit{};
    double bt_wait0{};
    double bt_pool0{};
    int64_t bt_miss0{};
    int64_t bt_hits0{};
    int64_t bt_pcie0{};
    int64_t bt_windows{};
    int64_t bt_rows{};
    int64_t bt_tokens{};
    Clock::time_point bt_start{};
    bool ov_active{};
    std::function<bool ()> ov_post{};   // what the overlapped read's thread does right after its last chunk
    bool ov_post_done{};
    int n_pipe{};
    bool piped{};
    int GS{};
    struct PGroup {
        bool inflight = false;
        int stage = 0;                  ///< the stage it runs on or waits for
        int32_t tok[basalt::kernels::kVerifyMaxT] = {};
        int64_t pos[basalt::kernels::kVerifyMaxT] = {};
        int64_t since = 0;              ///< tick it started waiting (fairness)
    };
    std::vector<PGroup> pg{};
    std::vector<int> stage_group{};
    int64_t pipe_tick{};
    int64_t rr{};

    int64_t part_slots(const PfPart& p, int64_t c);
    uint64_t part_bytes(const PfPart& p, int32_t first);
    int init_prompt_paths();
    bool split_region(PromptStage& st, int64_t cells, int64_t chunk, int64_t l0, int64_t l1, uint8_t*& w_at, uint8_t*& s_at, uint8_t*& b_at);
    basalt::core::Verifier& stage_ver(int st);
    int pcie_num_of(double f);
    basalt::core::Verifier& stage_verb(int st);
    void ckd_finish() override;
    bool capture_admit(uint64_t need, uint64_t floor) override;
    bool park_current(size_t held, int64_t kept) override;
    bool gpu_sync_ok();
    bool checkpoint_at(int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool defer = false,
                       ConvCheckpoint::Source source = ConvCheckpoint::kPeriodic);
    void res_upload();
    bool adapt_peers(const std::vector<int32_t>& r0, std::string& perr);
    void apply_pending(bool wait, bool exchanges = true);
    void rank_swaps(std::vector<Swap>& swaps, bool across);
    void rank_once(std::vector<Swap>& swaps, size_t room);
    bool adapt();
    void xq_settle();
    bool xq_round(std::vector<Swap>* ranked = nullptr);
    bool xq_advance(bool wait);
    bool xq_drain();
    void save_profile(const char* why);
    void make_view(basalt::core::SessionState& v, int b);
    int32_t loan_plan(int64_t live, int64_t& chunk);
    void relend();
    bool home_given_up(int64_t live, std::string& e);
    bool vram_command(const std::string& cmd, std::string& e, bool for_sleep, std::string* reply = nullptr) override;
    int64_t cache_floor_slots();
    int64_t kv_ceiling();
    void kv_ceiling_note() override;
    bool move_routed_below(int64_t first, const char* why, std::string& e,
                           const std::vector<serve_loans::LentEntry>* lent_out = nullptr);
    bool kv_drain(std::string& e);
    bool cache_regrow(std::string& e);
    bool peer_take(const std::vector<int32_t>& off, std::string& e);
    void peer_due_lent(std::vector<std::pair<int32_t, int32_t>>& lent);
    bool peer_due_apply(std::string& e);
    bool kv_fit(basalt::core::SessionState& s, int64_t cells, std::string& e, bool shrink = true, int64_t min_cells = -1);
    bool kv_release(basalt::core::SessionState& s, int64_t cells, std::string& e);
    uint64_t row_hash(int t);
    bool kv_give_back(int64_t ss_keep, const basalt::core::SessionState* skip, int skip_slot, const std::function<bool()>& enough, int64_t& given, std::string& e);
    std::vector<std::pair<double, int>> idle_list(const basalt::core::SessionState* s, bool any);
    int64_t park_idle(const basalt::core::SessionState* s, int64_t need, bool any, std::string& e);
    bool park_one(int b, std::string& e);
    bool park_idle_engine(std::string& e) override;
    void enc_command(const std::string& cmd) override;
    bool copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e) override;
    bool hold_slot(int b, const ConvCheckpoint* at, std::string& e);
    bool copy_from_slot(int b, const ConvCheckpoint* at, std::string& e) override;
    void solo_draft_back(bool keep) override;
    bool slot_digest(int b, int64_t upto, uint64_t* part);
    bool batch_step();
    basalt::core::Verifier& stage_verifier(int k);
    bool pipe_inflight();
    bool group_active(int gi);
    bool pump(bool may_start);
    bool pipe_drain();
    size_t spill_to_disk(bool at_exit, size_t limit) override;
    int64_t disk_min_tokens() const override { return 8192; }
    bool probe_adapt(const std::string& cmd, std::string& e);
    int init_prompt();
    int init_verify();
    int init_conversations();
    int init_tiers();
    int init_memory();
    int capture_windows();
    int init_batch();
};

}  // namespace basalt::models::qwen4exp::serve
