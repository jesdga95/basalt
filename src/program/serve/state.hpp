// src/program/serve/state.hpp - the server's state every model shares, and what a model answers.
#pragma once

#include "../cli/tokens.hpp"
#include "console_stats.hpp"
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/logit_bias.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/ppl_gate.hpp"
#include "basalt/program/tokenizer.hpp"
#include "basalt/spec/draft_policy.hpp"
#include "basalt/spec/suffix_drafter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace basalt::program::serve {

namespace cli = basalt::program::cli;
using Clock = std::chrono::steady_clock;
using ImgKey = basalt::core::ConversationImageKey;
using ConvCheckpoint = basalt::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull);
std::string stage_text();
void stall_report(std::FILE* f, uint64_t layers_during);

enum class Flow { proceed, next, quit, error };

struct SlotWin;

// Below this much free VRAM a later allocation (a first-used window's buffers, the desktop, another program) can make
// a driver that pages GPU memory do so, and a graph spinning on a host flag then never finishes.  A driver that
// fails such an allocation instead needs room only for what is allocated after loading (lazily loaded kernels, graphs,
// a streaming state's residency map).
inline constexpr int kVramStallMib = 256;
inline constexpr int kVramFloorMib = 128;
inline int vram_floor_mib() { return basalt::platform::gpu_memory_pages() ? kVramStallMib : kVramFloorMib; }

// The server's state every model shares: its input, its requests' conversation chain, its slots.
class Runtime {
public:
    Runtime(basalt::program::CommonOptions& options, basalt::core::ModelFile& file, std::string& error, const int64_t& vocab,
            std::optional<basalt::program::grammar::PackVocab>& vocab_tables)
        : o(options), model_file(file), err(error), n_vocab(vocab), grammar_vocab(vocab_tables) {}
    // the stdin reader thread uses this object's state: it ends before the object does
    virtual ~Runtime() {
        reader_stop_.store(true);
        if (reader_.joinable()) reader_.join();
    }
    int execute();
    Flow dispatch();
    int run_requests();
    int shutdown();
    int init_protocol();
    int announce();
    // ---- start and end
    virtual int init_before_input() = 0;
    virtual int init_after_input() = 0;
    virtual void after_input_start() {}
    virtual void print_info() = 0;   // the INFO line
    virtual void at_exit() {}
    /// The pack's tokenizer rule, set before the input starts.
    basalt::model::ModelAdapter::PreTokenizer pre_{};
    static constexpr int kPenaltyWindowCap = 4096;
    // ---- what a model answers: one call per batch window at most, the rest per command or per request
    virtual bool next_batch_window() = 0;
    virtual void probe_command(const std::string& line, std::string& e) = 0;
    virtual bool vram_command(const std::string& cmd, std::string& e, bool for_sleep, std::string* reply = nullptr) = 0;
    virtual void sleep_command(bool down, std::string& e) = 0;
    virtual void between_requests(const std::string& line) = 0;
    virtual Flow run_request() = 0;
    virtual void solo_draft_back(bool) {}
    virtual bool capture_admit(uint64_t need, uint64_t floor) = 0;
    virtual bool park_current(size_t held, int64_t kept) = 0;
    virtual bool copy_from_slot(int b, const ConvCheckpoint* at, std::string& e) = 0;
    /// Waits for the checkpoint save still on its way to the host, if any; a save that failed drops its checkpoint.  Every
    /// reader of a checkpoint's bytes (a restore, a copy, a park) calls it first.
    void ckd_finish();
    /// Writes the `CONV` line with the RAM tier's figures (a restore adds the tokens it returned and its ms), for
    /// every model after it parks or restores a conversation.
    void report_conversations(long long restored = -1, double restored_ms = 0);
    /// Once per drop below the RAM guard's floor, between steps: the parked conversations leave RAM (to the disk tier
    /// when it is on and holds no copy of them yet, the others are dropped), the checkpoint blocks kept for reuse are
    /// freed and the heap's free pages return to the OS. The checkpoints stay: a request may already resume from one.
    void shed_if_low();
    /// From READY on the RAM guard watches MemAvailable against the reserve (--ram-reserve-mib); with 0 it never
    /// starts. The disk tier keeps a GiB above that floor either way.
    void start_guard();
    virtual int64_t slot_context(int slot) const = 0;
    virtual bool copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e) = 0;
    virtual bool slots_share_cells() const { return true; }   // one slot's positional state can be copied to another
    virtual bool slot0_solo() const = 0;   // slot 0 of the batch windows is the solo session
    virtual void enc_command(const std::string& cmd) = 0;
    virtual size_t spill_to_disk(bool at_exit, size_t limit) = 0;
    virtual bool park_idle_engine(std::string& e) = 0;
    virtual void kv_ceiling_note() {}   // a model with an elastic KV sends its ceiling again when it moves
    // ---- the console's figures (console_stats.hpp), once the server sends STATS
    ConsoleStats stats{};
    /// The model's expert counts so far (none for a dense model).
    virtual LookupCounts lookup_counts() { return {}; }
    /// The expert map's sample: the pairs of all, each one's home and its lookups since the last call. False for a model
    /// without routed experts. The first call starts the counting.
    virtual bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) {
        (void) total; (void) homes; (void) routed;
        return false;
    }
    /// Per card (CUDA ordinal): the experts it holds and the VRAM their slots take, the KV cache's VRAM (-1: not known), and
    /// the host-to-device bandwidth as the start measured it (0: not measured). Sent as
    /// `INFO cards=<bus id>/<expert slots>/<GB/s or ->/<expert MiB>/<KV MiB or ->;...` whenever it changes.
    struct CardFacts {
        int device = 0;
        int64_t expert_slots = 0;
        double h2d_gbps = 0;
        uint64_t expert_bytes = 0;
        int64_t kv_bytes = -1;
    };
    virtual std::vector<CardFacts> card_facts() { return {}; }
    std::string cards_sent_{};
    /// A decode window of `slot` (see ConsoleStats::window); around a window or a batch round, its lookups.
    void stats_window(int slot, int offered, int accepted, int tokens) {
        if (stats.on) stats.window(slot, offered, accepted, tokens);
    }
    void stats_lookups_begin() { if (stats.on) stats.lookups_begin(lookup_counts()); }
    void stats_lookups_end() { if (stats.on) stats.lookups_end(lookup_counts()); }
    /// Writes the XSLOT and XMAP lines when due (`force`: now, as a request ends).
    void stats_tick(bool force = false);
    basalt::program::CommonOptions& o;
    basalt::core::ModelFile& model_file;
    std::string& err;
    const int64_t& n_vocab;
    std::optional<basalt::program::grammar::PackVocab>& grammar_vocab;
    // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens it
    // has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the saved
    // points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain - the radix
    // cache's tree collapsed onto the one branch of history whose cells the session holds.  The chain's root is the
    // deepest point every request so far shared (the end of the system prompt, in practice); the retention policy
    // pins it and rotates the rest LRU (conv_cache.hpp), so a new chat that shares that prefix mounts through it
    // instead of reading it again.
    std::vector<int32_t> live{};
    std::vector<ImgKey> live_imgs{};
    std::vector<ImgKey> req_imgs{};
    bool live_ok{};
    std::vector<ConvCheckpoint> checks{};
    /// The checkpoint save in flight (see ckd_finish).  Declared after `checks`: its wait runs before their bytes are freed.
    basalt::core::CheckpointDeferral ckd{};
    /// A model split across two cards: the second card's part of that save, on its device.
    basalt::core::CheckpointDeferral ckd_peer{};
    int ckd_peer_dev = -1;
    /// A checkpoint of `to` tokens already in the chain is used again (and keeps the higher `source`): true.
    bool touch_check(int64_t to, ConvCheckpoint::Source source);
    /// Adds a checkpoint to the chain; the least recently used one goes first past --prompt-cache.
    void keep_check(ConvCheckpoint&& c);
    uint64_t check_clock{};
    bool cvec_cached{};
    std::optional<basalt::core::ConversationCache> conversations;
    // the disk tier under it (--conversation-cache-dir): what leaves the session or the RAM cache is written there,
    // and a prompt that starts with a stored conversation reads it back, after a restart too
    basalt::core::ConversationStore disk{};
    virtual int64_t disk_min_tokens() const = 0;
    static constexpr int64_t kDiskStaleTokens = 4096;   // a stored copy this close to the chain's end is kept
    static constexpr uint64_t kDiskQueueCap = 4ull << 30;   // images waiting for the writer, in RAM
    uint64_t disk_floor{};
    bool disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec);
    int64_t pp_total{};
    int64_t pp_from{};
    int64_t pp_next_check{};
    int64_t pp_part_ckpt{};
    int64_t pp_reached{};
    Clock::time_point pp_t0{};
    // why the last checkpoint_at failed - a failed device sync is the GPU itself, not the checkpoint, and the error
    // says so
    std::string ckpt_why{};
    // GENT: the request text arrives as bytes and the reader thread turns it into ids, from the pack's tokenizer tables
    basalt::program::Tokenizer text_tokenizer{};
    std::string text_tokenizer_note{};
    /// The raw bytes a token stands for; empty when the pack's tokenizer tables are not loaded or the id has none.
    std::string_view token_bytes(int32_t id) const {
        return !text_tokenizer_note.empty() || id < 0 || (size_t) id >= text_tokenizer.vocab_size()
                   ? std::string_view() : std::string_view(text_tokenizer.token_bytes(id));
    }
    std::mutex gent_mu{};
    std::map<uint64_t, std::vector<int32_t>> gent_ids{};   // tokenized prompts waiting for their request line, by sequence
    uint64_t gent_seq{};
    // ---- --batch: a prompt the server sends once (`PROMPT`: text the engine tokenizes, `PROMPTI`: ids) and then names by
    // handle (`#<handle>[+<id>,...]` where a GEN line has its ids), so a request that runs in several engine runs or
    // waits for a slot does not send or tokenize it again.  The input thread takes it in and answers `PROMPT <handle>
    // <n>` at once, even while a request runs; `PMATCH <handle>` then says what each idle slot holds of it.  Guarded by gent_mu.
    struct HeldPrompt {
        std::vector<int32_t> ids;
        bool media = false;   ///< its pictures are placed by the request that reads it: a held conversation meets it only as an exact prefix
    };
    std::map<uint64_t, HeldPrompt> prompts{};
    static constexpr size_t kPromptsMax = 64;
    /// The tokens an idle slot's conversation holds for its next turn; null when it holds none (or is decoding).
    const std::vector<int32_t>* idle_ids(int b) const;
    struct SlotMeet {
        int64_t match = 0;   ///< the tokens of the prompt the slot's conversation continues (another conversation sharing a head: 0)
        int64_t held = 0;    ///< the tokens it holds
    };
    std::vector<SlotMeet> slot_meets(const std::vector<int32_t>& prompt, bool media) const;
    /// Holds a prompt and returns its handle.
    uint64_t hold_prompt(HeldPrompt&& p);
    /// PROMPTI [img] <id,id,...>, from the input thread: held and answered.
    void take_prompt_ids(const std::string& line);
    /// PMATCH <handle> | PFREE <handle>, from the main loop.
    void prompt_command(const std::string& line);
    std::atomic<bool> stop_req{};
    std::atomic<bool> stdin_lost{};
    std::thread reader_{};
    std::atomic<bool> reader_stop_{};
    std::mutex in_mu{};
    std::condition_variable in_cv{};
    std::deque<std::string> in_lines{};
    bool in_eof{};
    std::atomic<bool> ram_stop{};
    uint64_t guard_floor{};
    uint64_t ready_avail{};
    uint64_t ready_rss{};
    uint64_t shed_drop{};
    std::string line{};
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
        double idle_busy = 0;          ///< the busy clock (busy_now) when it last became idle holding `ids`
        double park_noted = -1;        ///< the idle_busy whose stay in VRAM for want of RAM was logged
        bool cvec = true;              ///< the control vector setting `ids` were read with
        std::vector<ImgKey> imgs;      ///< the pictures in `ids`: a later request reuses them only with the same ones
        // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's header): a
        // client's next turn renders the history again without this reply's thinking, so it matches the slot's
        // tokens only up to there - the slot's K/V up to it plus this state continue from it
        std::vector<ConvCheckpoint> checks;
        // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
        // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0;
        // `chunk`: the chunk of a read beside decoding slots, whose rest is read beside them in it, else 0)
        bool partial = false, partial_from0 = false;
        int64_t partial_chunk = 0;
        // the request's grammar (the batch windows' form of the solo loop's `gguard`): the window's kept picks are
        // scanned, the first refused one is resampled under the mask, and the segment ends there
        std::unique_ptr<basalt::program::grammar::Guard> guard;
        int lp_top = -1;   ///< the request's logprobs=K: a BP line ahead of the BT line for every kept pick (-1: off)
        // the drafter's side of the slot: the drafts its session left for the slot's next window and their chances, its
        // prompt-lookup history and last proposal
        int32_t dr[basalt::spec::DraftPolicy::kMaxT] = {};
        float dprob[basalt::spec::DraftPolicy::kMaxT] = {};
        int n_dr = 0;
        std::unique_ptr<basalt::spec::SuffixDrafter> sfx;
        int32_t sbuf[basalt::spec::DraftPolicy::kMaxT] = {};
        int pen_n = 0;                 ///< the request's penalty window (penalty_last_n; 0: no penalties)
        int64_t windows = 0;   ///< the request's windows in this slot, for BDONE
        int64_t drafts_accepted = 0, drafts_offered = 0;   ///< the request's draft tokens, for BDONE
    };
    std::vector<BSlot> bs{};
    // ---- --batch: the slots' rounds (batch_round.cpp), the same for every model
    /// The session slot `b` decodes in.
    virtual basalt::model::ModelSession& slot_session(int b) = 0;
    /// False while active slot `b` sits out the next round (its model cannot run its window yet); it keeps its place.
    virtual bool slot_ready(int b) { (void) b; return true; }
    /// One window of every ready active slot: the windows share batch_rows() rows, each group runs as one batch
    /// (decode_batch) or a lone window (decode_window), then every slot takes its tokens and drafts its next window.
    bool batch_round();
    /// The batch windows' draft policy, which the runtime sets up, and each slot's sampling (slot_set_sampling).
    std::optional<basalt::spec::DraftPolicy> bpolicy{};
    std::vector<basalt::kernels::SamplerParams> slot_sampling{};
    void slot_window_begin(int b, SlotWin& w, int limit);
    bool slot_window_finish(int b, SlotWin& w);
    bool slot_window_accept(int b, SlotWin& w);
    bool slot_window_end(int b, SlotWin& w);
    /// Active slot `b`'s request ends with finish `fin` (BDONE); the slot keeps its conversation for the next turn.
    void end_slot(int b, const char* fin);
    // ---- idle conversations out of VRAM.  An idle conversation (a cached batch slot's, or the solo session's while
    // no request works there) keeps its KV in VRAM until the work beside it has run kParkAfterBusy seconds: the
    // expert-cache slots its cells hold have then cost that work about what moving it out and back costs, both
    // in proportion to its tokens.  From then on, the VRAM a session is short of comes from it before the expert
    // cache's slots, and an idle engine moves it out to regrow the cache.  It goes to the RAM tier outside its
    // budget when RAM admits it, else it stays only on the disk tier when that holds it, else in VRAM; its next
    // turn reads it back like any parked conversation.  Before a prompt is refused for want of VRAM, any idle
    // conversation goes.
    static constexpr double kParkAfterBusy = 15.0;
    /// the solo session's conversation while a request works in another slot (that request's, moved out of `live`)
    struct SoloBook {
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs;
        bool live_ok = false, cvec = true;
        std::vector<ConvCheckpoint> checks;
    };
    SoloBook* solo_out{};
    bool solo_busy{};                    ///< a request reads or decodes in the solo session
    double solo_idle_busy{};             ///< the busy clock when the solo conversation last became idle
    double solo_park_noted = -1;         ///< as BSlot::park_noted
    int park_keep = -1;                  ///< the slot whose conversation the request being admitted continues
    // the busy clock: seconds some request or batch slot ran (dispatch counts them)
    double busy_s{};
    Clock::time_point busy_from{};
    bool busy_on{};
    double busy_now() const {
        return busy_s + (busy_on ? std::chrono::duration<double>(Clock::now() - busy_from).count() : 0.0);
    }
    int admit_slot{};
    int solo_slot{};
    long long admit_max_new{};
    // a slot's conversation held in RAM (`hold_slot`) while its KV cells back the request that continues it, when the
    // two do not fit side by side; copy_from_slot restores from it
    struct SlotHeld {
        int b = -1;
        std::vector<basalt::core::ConversationCheckpoint> ck;
        std::vector<std::vector<basalt::core::ConversationKv>> kv;
    };
    SlotHeld slot_held{};
    std::vector<ImgKey> imgs_below(const std::vector<ImgKey>& all, int64_t L);
    bool next_line(std::string& out);
    std::vector<int64_t> cur{};
    int next_line_within(std::string& out, int ms);
    bool halted();
    bool batch_on();
    bool try_next_line(std::string& out);
};

}  // namespace basalt::program::serve
