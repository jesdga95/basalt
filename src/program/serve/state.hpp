// src/program/serve/state.hpp - private serve state and operations, ordered by resource lifetime.
#pragma once

#include "runtime.hpp"
#include "../cli/tokens.hpp"
#include "loans.hpp"
#include "loan_moves.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/core/logit_bias.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/pinned.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/core/remote_experts.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/program/image_loan.hpp"
#include "basalt/program/multi_gpu_plan.hpp"
#include "basalt/program/tokenizer.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/mrope.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/native_dense.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/ppl_gate.hpp"
#include "basalt/program/probe_rates.hpp"
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

namespace basalt::program::serve {
namespace cli = basalt::program::cli;
namespace serve_loans = basalt::program::serve;
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
using Clock = std::chrono::steady_clock;

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again would make a long session wait for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = basalt::core::ConversationImageKey;
using ConvCheckpoint = basalt::core::ConversationCheckpoint;
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
uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull);
void seed_usage_from_prompt(std::vector<float>& usage, const std::vector<uint32_t>& hist, int64_t tokens,
                            const std::vector<int32_t>& host_res, int64_t n_expert, const basalt::core::PeerExperts& peer);
std::string stage_text();
void stall_report(std::FILE* f, uint64_t layers_during);

enum class Flow { proceed, next, quit, error };

class Request;

class Runtime : private Context {
public:
    explicit Runtime(Context& context) : Context(context) {}
    int execute();

private:
    friend class Request;
    basalt::prefill::Prefill sp{};
    void* borrow{};
    uint64_t borrow_bytes{};
    int32_t lend_first{};
    // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]), which is
    // the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage borrowing the tail
    // of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the chunk, is laid out by
    // `Prefill::relayout`, and is refilled before any window reads - so outside the prompt the whole cache is
    // expert cache.  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose slot values
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
    static constexpr int kPenaltyWindowCap = 4096;
    static constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap* (size_t) basalt::kernels::kVerifyMaxT;
    int32_t* d_hist{};
    std::vector<int32_t> hist_stage{};
    int hist_dev{};
    basalt::core::Verifier ver{};
    basalt::core::VerifyHits vh{};
    // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
    // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
    // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
    basalt::core::Verifier ver_same{};
    SplitDrive split_drive{};
    SplitDrive split_drive_b{};
    basalt::core::Verifier ver_b{};   // pipelined windows: the second chain over the same sessions (stage 0)
    void* pair_snap{};
    int n_stages{};
    bool pipe_on{};
    basalt::core::PoolMultiFn win_pool_fn{};
    void* win_pool_user{};
    std::vector<int64_t> cur{};
    // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens it
    // has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the saved
    // points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain - the radix
    // cache's tree collapsed onto the one branch of history whose cells the session holds.  The chain's root is the
    // deepest point every request so far shared (the end of the system prompt, in practice); the retention policy
    // pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that shares that prefix mounts through it
    // instead of reading it again.
    std::vector<int32_t> live{};
    std::vector<ImgKey> live_imgs{};
    std::vector<ImgKey> req_imgs{};
    bool live_ok{};
    std::vector<ConvCheckpoint> checks{};
    uint64_t check_clock{};
    // A turn-boundary checkpoint's GDN bytes reach the host after the first token (see the deferral class); this
    // is declared after `checks` so it is destroyed, and its copy finished, before the checkpoints are.
    basalt::core::ConversationCheckpointDeferral ckd{};
    bool cvec_cached{};
    std::vector<basalt::core::SessionState> rviews{};
    basalt::core::SessionState* rsp{};
    std::optional<basalt::core::ConversationCache> conversations;
    // the disk tier under it (--conversation-cache-dir): what leaves the session or the RAM cache is written there,
    // and a prompt that starts with a stored conversation reads it back, after a restart too
    basalt::core::ConversationStore disk{};
    static constexpr int64_t kDiskMinTokens = 8192;   // shorter ones are cheaper to read again
    static constexpr int64_t kDiskStaleTokens = 4096;   // a stored copy this close to the chain's end is kept
    static constexpr uint64_t kDiskQueueCap = 4ull << 30;   // images waiting for the writer, in RAM
    uint64_t disk_floor{};
    int64_t pp_total{};
    int64_t pp_from{};
    int64_t pp_next_check{};
    int64_t pp_part_ckpt{};
    int64_t pp_reached{};
    Clock::time_point pp_t0{};
    // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
    // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
    // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
    std::mutex part_mu{};
    std::map<int64_t, std::vector<ConvCheckpoint>> part_at{};   // position -> one part per stage
    std::vector<int64_t> part_next{};
    // why the last checkpoint_at failed - a failed device sync is the GPU itself, not the checkpoint, and the error
    // says so
    std::string ckpt_why{};
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
    // the VRAM tier follows the conversation (the same rule as the speculative loop below)
    // the peer's experts are candidates too, so the ones a workload routes most move to
    // CUDA0 (from their RAM copy, as any other swap) and its coldest go to the CPU
    // routings a peer expert needs above CUDA0's victim before it moves
    static constexpr float promote_margin = 10.0f;
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
    // GENT: the request text arrives as bytes and the reader thread turns it into ids, from the pack's tokenizer tables
    basalt::program::Tokenizer text_tokenizer{};
    std::string text_tokenizer_note{};
    std::mutex gent_mu{};
    std::map<uint64_t, std::vector<int32_t>> gent_ids{};   // tokenized prompts waiting for their request line, by sequence
    uint64_t gent_seq{};
    std::atomic<bool> stop_req{};
    std::atomic<bool> stdin_lost{};
    std::mutex in_mu{};
    std::condition_variable in_cv{};
    std::deque<std::string> in_lines{};
    bool in_eof{};
    std::atomic<bool> ram_stop{};
    uint64_t guard_floor{};
    uint64_t ready_avail{};
    uint64_t ready_rss{};
    uint64_t shed_drop{};
    int S{};
    int S_mtp{};
    std::vector<int32_t> res_at_start{};
    std::vector<int32_t> peer_at_start{};
    std::vector<int32_t> prof_rank_start{};
    std::string line{};
    int64_t rounds{};
    std::optional<basalt::spec::SuffixDrafter> sfx;
    std::optional<basalt::spec::DraftPolicy> policy;
    static constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
    bool mrope_identity{};
    std::vector<float> img_rows{};
    std::vector<const float*> row_ptr{};
    std::vector<std::pair<int32_t, int32_t>> vram_evicted{};   // (residency index, slot) the shrinks took
    int64_t chunk_full{};
    int64_t kKvGrowCells{};
    static constexpr int64_t kKvWindowCells = 2* basalt::kernels::kVerifyMaxT + 64;   // a decoding session's next window
    int64_t kv_moves{};
    bool kv_short{};
    bool kv_keep_routed{};   // an admission while slots decode: a shrink keeps their routed experts
    // ---- --batch: the slots of the batch windows
    struct BSlot {
        bool active = false, stop = false;
        int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
        int64_t p = 0;                 ///< its position
        int64_t produced = 0, max_new = 0;
        Clock::time_point t0;
        // the slot's own conversation cache: the tokens its sessions hold (the prompt, then every token a window
        // fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from it
        // instead of reading its history again (see `slot_source` in the request path).
        std::vector<int32_t> ids;
        bool cached = false;           ///< idle, and its sessions still hold `ids`
        bool cvec = true;              ///< the control vector setting `ids` were read with
        std::vector<ImgKey> imgs;      ///< the pictures in `ids`: a later request reuses them only with the same ones
        // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's header): a
        // client's next turn renders the history again WITHOUT this reply's thinking, so it matches the slot's
        // tokens only up to there - the slot's K/V up to it plus this state continue from it
        std::vector<ConvCheckpoint> checks;
        // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
        // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0;
        // `chunk`: the chunk of a read beside decoding slots, whose rest is read beside them in it, else 0)
        bool partial = false, partial_from0 = false;
        int64_t partial_chunk = 0;
        // BSCRIPT (a test drive of the multi-token segments): the tokens this slot's request is known to produce
        // (script[0] is its first token); the windows draft from it, some drafts wrong, and keep what matches
        std::vector<int32_t> script;
        std::mt19937_64 rng;
        // the request's grammar (the batch windows' form of the solo loop's `gguard`): the window's kept picks are
        // scanned, the first refused one is resampled under the mask, and the segment ends there
        std::unique_ptr<basalt::program::grammar::Guard> guard;
        // the MTP drafter's side of the slot (the draft K/V is the drafter's store of the same index): the drafts and
        // their probabilities for the slot's next window, the request's draft-length rule state, its prompt-lookup
        // history and last proposal
        int32_t dr[16] = {};
        float dprob[16] = {};
        int n_dr = 0;
        basalt::core::DraftCumAdapt cum;
        std::unique_ptr<basalt::spec::SuffixDrafter> sfx;
        int32_t sbuf[16] = {};
        int sk = 0, smatch = 0;
        bool from_sfx = false;
        double acc = 0.6;          ///< drafts accepted per offered, smoothed: who gets the free rows
        int64_t since_draft = 0;   ///< windows since the slot last drafted
        int64_t windows = 0;
        int64_t drafts_accepted = 0, drafts_offered = 0;   ///< the request's draft tokens, for BDONE
    };
    std::vector<BSlot> bs{};
    bool batch_spec{};
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
    int admit_slot{};
    int solo_slot{};
    long long admit_max_new{};
    bool ov_active{};
    std::function<bool ()> ov_post{};   // what the overlapped read's thread does right after its last chunk
    bool ov_post_done{};
    // a slot's conversation held in RAM (`hold_slot`) while its KV cells back the request that continues it, when the
    // two do not fit side by side; copy_from_slot restores from it
    struct SlotHeld {
        int b = -1;
        std::vector<basalt::core::ConversationCheckpoint> ck;
        std::vector<std::vector<basalt::core::ConversationKv>> kv;
    };
    SlotHeld slot_held{};
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
    void ckd_finish();
    bool disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec);
    bool capture_admit(uint64_t need, uint64_t floor);
    bool park_current(size_t held, int64_t kept);
    std::vector<ImgKey> imgs_below(const std::vector<ImgKey>& all, int64_t L);
    bool gpu_sync_ok();
    bool checkpoint_at(int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool defer = false);
    void res_upload();
    bool adapt_peers(const std::vector<int32_t>& r0, std::string& perr);
    void apply_pending(bool wait, bool exchanges = true);
    void rank_swaps(std::vector<Swap>& swaps, bool across);
    bool adapt();
    void xq_settle();
    bool xq_round();
    bool xq_advance(bool wait);
    bool xq_drain();
    void save_profile(const char* why);
    bool next_line(std::string& out);
    int next_line_within(std::string& out, int ms);
    bool halted();
    void shed_if_low();
    void make_view(basalt::core::SessionState& v, int b);
    int32_t loan_plan(int64_t live, int64_t& chunk);
    void relend();
    bool home_given_up(int64_t live, std::string& e);
    bool vram_command(const std::string& cmd, std::string& e, bool for_sleep, std::string* reply = nullptr);
    int64_t cache_floor_slots();
    bool move_routed_below(int64_t first, const char* why, std::string& e,
                           const std::vector<serve_loans::LentEntry>* lent_out = nullptr);
    bool kv_drain(std::string& e);
    bool cache_regrow(std::string& e);
    bool kv_fit(basalt::core::SessionState& s, int64_t cells, std::string& e, bool shrink = true, int64_t min_cells = -1);
    bool kv_release(basalt::core::SessionState& s, int64_t cells, std::string& e);
    uint64_t row_hash(int t);
    bool batch_on();
    bool try_next_line(std::string& out);
    bool kv_give_back(int64_t ss_keep, const basalt::core::SessionState* skip, int skip_slot, const std::function<bool()>& enough, int64_t& given, std::string& e);
    void enc_command(const std::string& cmd);
    bool copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e);
    bool hold_slot(int b, const ConvCheckpoint* at, std::string& e);
    bool copy_from_slot(int b, const ConvCheckpoint* at, std::string& e);
    bool slot_digest(int b, int64_t upto, uint64_t* part);
    bool batch_step();
    basalt::core::Verifier& stage_verifier(int k);
    bool pipe_inflight();
    bool group_active(int gi);
    bool pump(bool may_start);
    bool pipe_drain();
    size_t spill_to_disk(bool at_exit, size_t limit);
    bool probe_adapt(const std::string& cmd, std::string& e);
    int init_prompt();
    int init_verify();
    int init_conversations();
    int init_tiers();
    int init_protocol();
    int announce();
    int init_memory();
    int capture_windows();
    int init_batch();
    int run_requests();
    Flow dispatch();
    int shutdown();
};

}  // namespace basalt::program::serve
