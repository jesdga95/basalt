// src/program/serve/state.hpp - the server's state every model shares, and what a model answers.
#pragma once

#include "../cli/tokens.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/draft_cum.hpp"
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
    virtual bool windows_in_flight() = 0;
    virtual bool next_batch_window() = 0;
    virtual bool drain_windows() = 0;
    virtual void script_command(const std::string& line) = 0;
    virtual void probe_command(const std::string& line, std::string& e) = 0;
    virtual bool vram_command(const std::string& cmd, std::string& e, bool for_sleep, std::string* reply = nullptr) = 0;
    virtual void sleep_command(bool down, std::string& e) = 0;
    virtual void between_requests(const std::string& line) = 0;
    virtual Flow run_request() = 0;
    virtual void solo_draft_back(bool) {}
    virtual bool capture_admit(uint64_t need, uint64_t floor) = 0;
    virtual bool park_current(size_t held, int64_t kept) = 0;
    virtual bool copy_from_slot(int b, const ConvCheckpoint* at, std::string& e) = 0;
    virtual void ckd_finish() {}
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
    virtual bool pipelined() const { return false; }   // batch groups pipelined through layer split stages
    virtual int64_t slot_context(int slot) const = 0;
    virtual bool copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e) = 0;
    virtual bool slots_share_cells() const { return true; }   // one slot's positional state can be copied to another
    virtual bool slot0_solo() const = 0;   // slot 0 of the batch windows is the solo session
    virtual void enc_command(const std::string& cmd) = 0;
    virtual size_t spill_to_disk(bool at_exit, size_t limit) = 0;
    virtual bool park_idle_engine(std::string& e) = 0;
    virtual void kv_ceiling_note() {}   // a model with an elastic KV sends its ceiling again when it moves
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
    std::mutex gent_mu{};
    std::map<uint64_t, std::vector<int32_t>> gent_ids{};   // tokenized prompts waiting for their request line, by sequence
    uint64_t gent_seq{};
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
        int pen_n = 0;                 ///< the request's penalty window (penalty_last_n; 0: no penalties)
        bool from_sfx = false;
        double acc = 0.6;          ///< drafts accepted per offered, smoothed: who gets the free rows
        int64_t since_draft = 0;   ///< windows since the slot last drafted
        int64_t windows = 0;
        int64_t drafts_accepted = 0, drafts_offered = 0;   ///< the request's draft tokens, for BDONE
    };
    std::vector<BSlot> bs{};
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
