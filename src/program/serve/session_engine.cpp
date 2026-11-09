// src/program/serve/session_engine.cpp - the shared server over a model's ModelSession: one token per window.
#include "session_engine.hpp"

#include "decode_loop.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/device.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/protocol.hpp"
#include "basalt/spec/draft_policy.hpp"

#include <iterator>
#include <sstream>

namespace basalt::program::serve {

namespace {

constexpr int32_t kEmbeddingsMagic = 0x31455653;

class SessionRuntime;

/// The drafts to make for the next window: up to one more than the window just verified used (at least 2), so a long
/// window grows a row at a time where its drafts pay and the drafter's head runs only the rows a window can take.
int drafts_ahead(int T, int most) { return std::clamp(std::max(T, 2), 1, most - 1); }

/// A picture's embedding rows, placed at prompt position `at`.
struct Block {
    int64_t at = 0;
    std::vector<float> rows;
    int n_rows = 0;
};

class SessionRuntime final : public Runtime {
public:
    SessionRuntime(CommonOptions& options, basalt::core::ModelFile& file, std::string& error, basalt::model::LoadedModel& model)
        : Runtime(options, file, error, model.desc().n_vocab, vocab_), model_(model) {}

    int init_before_input() override {
        const int slots = std::max(o.batch, 1);
        sessions_.resize((size_t) slots);
        for (auto& s : sessions_) {
            const basalt::model::Result r = model_.make_session(s);
            if (!r.ok()) { basalt::log::error("basalt serve", "%s", r.message.c_str()); return 1; }
        }
        // one policy for the solo windows and one for the batch windows, whose rounds cost differently
        policy.emplace(session().max_window());
        bpolicy.emplace(session().max_window());
        if (o.batch > 0) {
            bs.resize((size_t) o.batch);
            slot_sampling_.resize((size_t) o.batch);
        }
        conversations.emplace((size_t) o.conversation_cache_mib << 20, (size_t) o.conversation_cache_slots);
        if (o.conversation_cache_disk_mib > 0) {
            const auto t0 = Clock::now();
            const std::string text = "basalt conversation state " + model_.desc().arch + " vision_in_engine " + (o.vision_in_engine ? "1" : "0");
            const uint64_t identity = basalt::core::conversation_store_hash_file(
                o.model, basalt::core::conversation_store_hash(text.data(), text.size(), 1));
            std::string de;
            if (o.prompt_cache <= 0) {
                basalt::log::warn("basalt serve", "conversation cache on disk off: --prompt-cache 0");
            } else if (!disk.open(o.conversation_cache_dir, identity, (uint64_t) o.conversation_cache_disk_mib << 20, de)) {
                basalt::log::warn("basalt serve", "conversation cache on disk off: %s", de.c_str());
            } else {
                conversations->set_spill(true);
                const auto st = disk.stats();
                basalt::log::info("basalt serve", "conversation cache on disk: %s, %zu conversations of this model "
                                  "(%.2f GiB), budget %lld MiB, indexed in %.0f ms", o.conversation_cache_dir.c_str(),
                                  st.files, (double) st.bytes / 1073741824.0, (long long) o.conversation_cache_disk_mib,
                                  std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        }
        if (o.grammar) {   // the token texts the grammar walks, read from the pack's tokenizer files
            std::vector<uint8_t> vj, tj;
            std::string gerr;
            basalt::program::grammar::PackVocab pv;
            std::vector<int32_t> ends(o.eos_ids.begin(), o.eos_ids.end());
            if (!basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, gerr) ||
                !basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, gerr) ||
                !basalt::program::grammar::vocab_from_pack(std::string_view((const char*) vj.data(), vj.size()),
                                                           std::string_view((const char*) tj.data(), tj.size()),
                                                           (size_t) n_vocab, ends, pv, gerr)) {
                basalt::log::error("basalt serve", "--grammar on: %s", gerr.c_str());
                return 2;
            }
            vocab_ = std::move(pv);
        }
        return 0;
    }
    // the pack's quantization label and the PCI bus ids of the cards this engine computes on, primary first: what a
    // benchmark row records
    std::string identity_words() const {
        const basalt::core::Json* nm = model_file.meta().get("name");
        std::string ids;
        for (int d : basalt::program::engine_gpu_order(o.peer_device, o.peer_devices, {})) {
            std::string bus;
            if (basalt::core::device_pci_bus_id(d, bus)) ids += (ids.empty() ? "" : ",") + bus;
        }
        return "quant=" + basalt::program::pack_quant_label(nm != nullptr && nm->is_string() ? nm->s : std::string()) +
               " gpus=" + ids + " ";
    }
    int init_after_input() override {
        for (auto& s : sessions_)
            if (const basalt::model::Result r = s->prepare(); !r.ok()) {
                basalt::log::error("basalt serve", "%s", r.message.c_str());
                return 1;
            }
        start_guard();
        std::printf("INFO %s\n", (identity_words() + sessions_[0]->info_words()).c_str());
        std::printf("READY %lld stop\n", (long long) o.max_context);
        std::fflush(stdout);
        return 0;
    }
    void print_info() override {
        const std::string slots = o.batch > 0 ? " batch_slots=" + std::to_string(o.batch) + " slot_cache=" +
                                                    std::to_string(o.prompt_cache > 0 ? 1 : 0) + " slot0_solo=1 solo_any_slot=1"
                                              : std::string();
        std::printf("INFO logit_bias=%lld context=%lld spec=%d mtp_max=0 lookup=0%s engine=" BASALT_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context, o.spec, slots.c_str());
    }
    bool windows_in_flight() override { return false; }
    bool next_batch_window() override;
    bool drain_windows() override { return true; }
    void script_command(const std::string&) override {
        std::printf("ERR BSCRIPT: this model's slots take no scripts\n");
        std::fflush(stdout);
    }
    void probe_command(const std::string& line, std::string& e) override {
        basalt::program::ProbeEnv env;
        if (const basalt::model::Result r = session().probe_env(env); !r.ok()) { e = "PROBE: " + r.message; return; }
        if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K: a live value for the next requests
            std::istringstream kvs(line.substr(10));
            for (std::string kv; kvs >> kv;) {
                const int v = kv.rfind("workers=", 0) == 0 ? std::atoi(kv.c_str() + 8) : -1;
                if (v >= 1) env.pool->set_active_workers(v);
                else e = "PROBE: bad setting " + kv;
            }
            if (e.empty()) std::printf("PROBE end workers=%d\n", env.pool->active_workers());
        } else if (line == "PROBE rates") {
            basalt::program::probe_rates(env, stdout, e);
        } else {
            e = "PROBE: expected PROBE rates or PROBE set";
        }
    }
    // VRAM <mib>: the session leaves that much free from now on; VRAM alone: the start's reserve.  One reply line.
    bool vram_command(const std::string& cmd, std::string& e, bool, std::string* reply) override {
        char* end = nullptr;
        const long long reserve = cmd.size() > 4 ? std::strtoll(cmd.c_str() + 4, &end, 10) : -1;
        if (cmd.size() > 4 && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
        int64_t slots = 0;
        if (const basalt::model::Result r = session().set_vram_reserve(reserve, slots); !r.ok()) {
            e = r.message;
            return false;
        }
        return memory_reply(reserve, slots, reply);
    }
    void sleep_command(bool down, std::string& e) override {
        std::string reply;
        int64_t slots = 0;
        if (const basalt::model::Result r = session().set_low_memory(down, slots); !r.ok()) {
            e = r.message;
            return;
        }
        memory_reply(-1, slots, &reply);
        std::fputs(reply.c_str(), stdout);
    }
    void between_requests(const std::string&) override {}
    Flow run_request() override;
    void enc_command(const std::string& cmd) override;
    // The conversations the engine holds go to the disk tier when no stored copy reaches them: the RAM tier's at exit,
    // the solo session's and every idle slot's.
    int64_t disk_min_tokens() const override { return sessions_[0]->disk_min_tokens(); }
    size_t spill_to_disk(bool at_exit, size_t limit) override {
        size_t written = 0;
        if (at_exit)
            while (conversations->size() > 0) {
                basalt::core::SavedConversation s = conversations->take(conversations->size() - 1);
                if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec) && disk.put(std::move(s), kDiskQueueCap)) ++written;
            }
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
        const auto spill = [&](const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
            if (at_exit) disk.flush();   // one image in RAM at a time
            const int64_t positions = (int64_t) ids.size();
            if (!capture_admit(session().running_bytes() + session().cells_bytes(positions), floor)) {
                basalt::log::info("basalt serve", "conversation cache: %zu tokens not written to disk (physical RAM admission)", ids.size());
                return;
            }
            basalt::core::SavedConversation image;
            std::string se;
            if (!capture(image, ids, imgs, cks, cvec, se)) {
                basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
                return;
            }
            if (disk.put(std::move(image), kDiskQueueCap)) ++written;
        };
        const int active = active_;
        const bool slot0_cached = o.batch > 0 && !bs.empty() && bs[0].cached;
        if (live_ok && !live.empty() && !slot0_cached && !disk_has(live, live_imgs, checks, cvec_cached)) {
            set_active(0);
            spill(live, live_imgs, checks, cvec_cached);
        }
        for (int b = 0; b < (int) bs.size() && written < limit; ++b) {
            BSlot& sl = bs[(size_t) b];
            if (!sl.cached || sl.active || sl.partial || disk_has(sl.ids, sl.imgs, sl.checks, sl.cvec)) continue;
            set_active(b);
            spill(sl.ids, sl.imgs, sl.checks, sl.cvec);
        }
        set_active(active);
        if (at_exit) disk.flush();
        return written;
    }
    bool park_idle_engine(std::string&) override { return false; }
    bool capture_admit(uint64_t need, uint64_t floor) override {
        return basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok &&
               basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), need, floor);
    }
    /// The session's conversation as an image: its running state and its positional state by layer.
    bool capture(basalt::core::SavedConversation& image, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                 const std::vector<ConvCheckpoint>& cks, bool cvec, std::string& e) {
        const int64_t positions = (int64_t) ids.size();
        image.live.ids = ids;
        image.live.imgs = imgs;
        image.live.state_a.resize(session().running_bytes());
        basalt::model::Result r = session().save_running(image.live.state_a.data(), positions);
        if (r.ok()) r = session().save_cells(image.kv, positions);
        if (!r.ok()) { e = r.message; return false; }
        image.checkpoints = cks;
        image.cvec = cvec;
        return true;
    }
    // The outgoing conversation leaves the session for the RAM tier, or the disk tier when there is no room or the
    // session goes on holding it (`kept`) and only the disk lacks it.
    bool park_current(size_t held, int64_t kept) override {
        if ((!conversations->enabled() && !disk.enabled()) || !live_ok || live.empty()) return true;
        if (o.conversation_cache_min_tokens > 0 && (int64_t) live.size() < o.conversation_cache_min_tokens) return true;
        int64_t deepest = 0;
        for (const ConvCheckpoint& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
        const bool disk_wants = disk.enabled() && !(kept > 0 && kept >= deepest) && !disk_has(live, live_imgs, checks, cvec_cached);
        bool to_disk = !conversations->enabled() || kept > 0;
        if (to_disk && !disk_wants) return true;
        const int64_t positions = (int64_t) live.size();
        const size_t estimate = session().running_bytes() + session().cells_bytes(positions);
        const bool ram_room = !to_disk && conversations->make_room(estimate, held);
        if (!to_disk && !ram_room && disk_wants) to_disk = true;
        if (to_disk ? disk.queued_bytes() + estimate > kDiskQueueCap : !ram_room) return true;
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk.enabled() ? disk_floor : 0);
        if (!capture_admit(estimate, floor)) {
            basalt::log::info("basalt serve", "conversation cache: %zu tokens not parked (physical RAM admission)", live.size());
            return true;
        }
        basalt::core::SavedConversation image;
        if (!capture(image, live, live_imgs, checks, cvec_cached, err)) return false;
        (void) (to_disk ? disk.put(std::move(image), kDiskQueueCap) : conversations->put(std::move(image), held));
        for (auto& s : conversations->take_spilled())
            if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec)) disk.put(std::move(s), kDiskQueueCap);
        return true;
    }
    bool copy_from_slot(int, const ConvCheckpoint*, std::string& e) override { e = "this model's slots copy no cells"; return false; }
    // slot 0 is the solo session, the only way a request reaches this with its state elsewhere
    bool copy_to_slot(int b, const std::vector<int32_t>&, std::string& e) override {
        if (b == 0) return true;
        e = "this model's slots copy no cells";
        return false;
    }
    bool slots_share_cells() const override { return false; }
    int64_t slot_context(int) const override { return o.max_context; }
    bool slot0_solo() const override { return o.batch > 0; }

    /// The running state at `to`, kept in the chain; the oldest goes first past --prompt-cache.
    bool checkpoint_at(int64_t to, ConvCheckpoint::Source source, const std::vector<int64_t>& prompt) {
        if (o.prompt_cache <= 0 || to < 1) return true;
        for (ConvCheckpoint& c : checks)
            if ((int64_t) c.ids.size() == to) {
                c.used = ++check_clock;
                if (source > c.source) c.source = source;
                return true;
            }
        ConvCheckpoint c;
        c.source = source;
        c.ids.assign(prompt.begin(), prompt.begin() + to);
        c.imgs = imgs_below(req_imgs, to);
        c.state_a.resize(session().running_bytes());
        if (const basalt::model::Result r = session().save_running(c.state_a.data(), to); !r.ok()) {
            err = r.message;
            return false;
        }
        c.used = ++check_clock;
        checks.push_back(std::move(c));
        while ((int) checks.size() > o.prompt_cache) {
            std::vector<uint64_t> stamps;
            stamps.reserve(checks.size());
            for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
            checks.erase(checks.begin() + (std::ptrdiff_t) basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(), o.prompt_cache));
        }
        return true;
    }

    basalt::model::ModelSession& session() { return *sessions_[(size_t) active_]; }
    void set_active(int slot) { active_ = slot; }
    void set_slot_sampling(int slot, const basalt::kernels::SamplerParams& sp) { slot_sampling_[(size_t) slot] = sp; }
    const basalt::model::ModelDesc& desc() const { return model_.desc(); }
    /// The draft windows' policy (basalt/spec/draft_policy.hpp): learns this machine's round times and the drafter's
    /// acceptance across requests.
    std::optional<basalt::spec::DraftPolicy> policy, bpolicy;
    /// --draft-log FILE, opened once.
    FILE* draft_log() {
        static FILE* const f = std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
        return f;
    }

private:
    bool memory_reply(long long reserve, int64_t slots, std::string* reply) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        char line[160];
        std::snprintf(line, sizeof line, "VRAM reserve_mib=%lld expert_slots=%lld vram_free_mib=%lld\n", reserve,
                      (long long) slots, (long long) (free_b >> 20));
        if (reply != nullptr) *reply = line;
        else std::fputs(line, stdout);
        return true;
    }

    std::optional<basalt::program::grammar::PackVocab> vocab_;
    basalt::model::LoadedModel& model_;
    std::vector<std::unique_ptr<basalt::model::ModelSession>> sessions_;
    std::vector<basalt::kernels::SamplerParams> slot_sampling_;
    std::vector<int32_t> win_, out_;   // a batch window's fed tokens and picks
    int active_ = 0;
};

class SessionRequest final : public Request {
public:
    explicit SessionRequest(SessionRuntime& runtime) : Request(runtime), rt_(runtime) {}

    static constexpr bool kSpeculates = true;
    /// No adaptive step: the model's tiers move by themselves.
    struct TierStep {
        explicit TierStep(SessionRequest&) {}
        void before_window(int, int, bool) {}
        void after_commit() {}
        void finish() {}
        bool ok() const { return true; }
    };
    Flow decode_begin() { return Flow::proceed; }
    void seed_tiers() {}
    void decode_state_begin() {
        draws_ = 0;
        n_draft_ = 0;
        drafts_.assign((size_t) max_window(), 0);
        conf_.assign((size_t) max_window(), 0.0f);
    }
    void decode_stats_begin() {
        counts0_ = rt_.session().expert_counts();
        draft_log_ = basalt::core::diagnostics().draft_log.empty() ? nullptr : rt_.draft_log();
        if (draft_log_ != nullptr) {
            std::fprintf(draft_log_, "R temp=%g S=%d\n", (double) req_temperature, max_window());
            std::fflush(draft_log_);
        }
    }
    int max_window() const { return rt_.session().max_window(); }
    int draft_window() {
        const int T = rt_.policy->window(conf_.data(), n_draft_);
        return (int) std::max<int64_t>(1, std::min<int64_t>(T, runtime_.o.max_context - p));
    }
    void lookup_window(int&, bool&, int&) {}
    void fill_drafts(int T, bool) {
        for (int i = 1; i < T; ++i) window[(size_t) i] = drafts_[(size_t) i - 1];
    }
    int before_window(int, int64_t) { return 0; }
    int32_t* history_stage() {
        history_.resize((size_t) max_window() * hist_n);
        return history_.data();
    }
    void upload_history(int) {}
    void mark_before_run() {}
    bool run_window(int T, int, int&) {
        basalt::model::DecodeRequest d;
        d.tokens = window.data();
        d.n_tokens = T;
        d.pos = p;
        (T > 1 ? d.row_tokens : d.next_token) = outv.data();
        d.sampling = &req_sp;
        d.history = hist_n > 0 ? history_.data() : nullptr;
        d.history_len = hist_n;
        req_sp.counter = draws_;
        const basalt::model::Result r = rt_.session().decode_window(d);
        if (!r.ok()) error_ = r.message;
        return r.ok();
    }
    bool take_held(int) { return false; }
    bool window_failed() const { return false; }
    const char* window_error() const { return error_.c_str(); }
    void relax_note_forced(int) {}
    int accept_drafts(int T) {
        int a = 0;
        while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
        return a;
    }
    bool resample_row(int row, int64_t, const std::vector<float>& bias, int32_t& tok, std::string& e) {
        const basalt::model::Result r = rt_.session().resample_row(row, bias.data(), tok);
        if (!r.ok()) e = r.message;
        return r.ok();
    }
    bool copy_logits(int, float* out) { return rt_.session().copy_logits(out).ok(); }
    bool after_accept(int, int&, int, bool, bool, bool, bool& hold) {
        hold = false;
        return true;
    }
    void note_draft_probs(bool) {
        window_conf_.assign(conf_.begin(), conf_.begin() + n_draft_);
    }
    void stamp_before_commit() {}
    // each row's pick is the draw of its own index, whatever the width of the window
    bool commit(int keep, std::string&, int phase) {
        if (phase == 2) draws_ += (uint64_t) keep;
        return true;
    }
    void on_token(int32_t) {}
    void count_round() {}
    // The next window's drafts start before this window's tokens go out, which run in their shadow.
    template <class After>
    bool draft_next(int T, int a, int, int, bool, bool, bool will_draft, After& after_launch) {
        n_draft_ = 0;
        if (!will_draft || max_window() < 2) return true;
        const int n = drafts_ahead(T, max_window());
        if (basalt::model::Result r = rt_.session().draft_begin(outv[(size_t) a], p + a + 1, n); !r.ok()) {
            if (r.status == basalt::model::Status::NotImplemented) return true;
            runtime_.err = r.message;
            return false;
        }
        after_launch();
        if (basalt::model::Result r = rt_.session().draft_end(drafts_.data(), conf_.data(), n, n_draft_); !r.ok()) {
            runtime_.err = r.message;
            return false;
        }
        return true;
    }
    bool draft_fatal() const { return false; }
    void window_timing(int T, int a, bool, Clock::time_point round0, Clock::time_point tw0, Clock::time_point tw1, Clock::time_point) {
        if (draft_log_ == nullptr) return;
        const auto msd = [](Clock::time_point x, Clock::time_point y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
        std::fprintf(draft_log_, "W %d %d 0 %.3f %.3f %.3f", T, a, msd(round0, tw0), msd(tw0, tw1), msd(tw1, Clock::now()));
        for (float c : window_conf_) std::fprintf(draft_log_, " %.4f", (double) c);
        std::fputc('\n', draft_log_);
    }
    void observe_policy(bool, int T, int a, int, Clock::time_point round0) {
        if (ppl != nullptr) return;   // a quality gate's windows are forced, not drafted
        rt_.policy->observe_window(window_conf_.data(), T, a, std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
    }
    void mark_after_window() {}
    void mark_loop_end() {}
    void shed_low() { rt_.shed_if_low(); }
    void stamp_loop_end() {}
    void decode_end() {}

private:
    Flow prepare_image() override;
    Flow decode() override { return decode_loop(*this); }
    bool release_room() override { return true; }
    bool checkpoint_mount(const ConvCheckpoint& c) override {
        const basalt::model::Result r = rt_.session().restore_running(c.state_a.data(), (int64_t) c.ids.size());
        if (!r.ok()) runtime_.err = r.message;
        return r.ok();
    }
    Flow set_sampling() override {
        hist_n = req_sp.penalty_last_n;
        const basalt::model::Result r = rt_.session().set_logit_bias(req_bias_present ? req_bias.data() : nullptr);
        if (!r.ok()) {
            std::printf("ERR %s\n", r.message.c_str());
            return Flow::next;
        }
        return Flow::proceed;
    }
    Flow read_to(int64_t to) override;
    Flow slot_bind() override { rt_.set_active(slot_b); return Flow::proceed; }
    Flow restore_image() override {
        const int64_t positions = (int64_t) incoming->live.ids.size();
        basalt::model::Result r = rt_.session().restore_cells(incoming->kv, positions);
        if (r.ok()) r = rt_.session().restore_running(incoming->live.state_a.data(), positions);
        if (r.ok()) r = rt_.session().set_history(incoming->live.ids.data(), positions);
        if (!r.ok()) {
            basalt::log::warn("basalt serve", "conversation cache: restoring a parked conversation failed (%s); reading the prompt from the start", r.message.c_str());
            incoming.reset();
            runtime_.live.clear();
            runtime_.live_imgs.clear();
            runtime_.checks.clear();
            resume = 0;
            from_live = false;
        }
        return Flow::proceed;
    }
    void adopt_checkpoints() override { runtime_.checks = std::move(incoming->checkpoints); }
    bool slot_leave() override { rt_.set_active(0); return true; }
    void slot_set_sampling(int slot) override { rt_.set_slot_sampling(slot, req_sp); }
    Flow prompt_end() override {
        rt_.session().end_prompt();
        return Flow::proceed;
    }
    bool turn_checkpoint_at_last_token() const override { return true; }
    DoneCounts done_counts() override {
        const basalt::model::ModelSession::ExpertCounts c = rt_.session().expert_counts();
        DoneCounts dc;
        dc.hits = c.hits - counts0_.hits;
        dc.lookups = c.lookups - counts0_.lookups;
        dc.offloaded = c.offloaded - counts0_.offloaded;
        dc.file_blobs = c.files - counts0_.files;
        return dc;
    }
    void report_request(const DoneCounts& dc) override { report_expert_counts(dc); }

    void progress_line(const char* tag, int64_t at_pos) {
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
        basalt::protocol::emit("%s %lld %lld %.0f %.1f\n", tag, (long long) at_pos, (long long) n, ms,
                               ms > 0.0 ? 1000.0 * (double) (at_pos - resume) / ms : 0.0);
    }

    SessionRuntime& rt_;
    std::vector<Block> blocks_;
    std::vector<int32_t> history_;
    uint64_t draws_ = 0;
    std::vector<int32_t> drafts_;
    std::vector<float> conf_, window_conf_;
    int n_draft_ = 0;
    FILE* draft_log_ = nullptr;
    basalt::model::ModelSession::ExpertCounts counts0_;
    std::string error_;
    size_t next_block_ = 0;
    double live_ms_ = -1e9;
};

// One window of every active slot, one session at a time: the slot's drafts as a solo window takes them, then the
// drafts of its next window, which run while its tokens go out.
bool SessionRuntime::next_batch_window() {
    std::vector<int32_t> history;
    std::vector<float> grammar_bias;
    const auto failed = [&](const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        set_active(0);
        return false;
    };
    for (int b = 0; b < (int) bs.size(); ++b) {
        BSlot& sl = bs[(size_t) b];
        if (!sl.active) continue;
        set_active(b);
        const Clock::time_point round0 = Clock::now();
        const int most = (int) std::min<int64_t>({(int64_t) session().max_window(), o.max_context - sl.p, (int64_t) std::size(sl.dr) + 1});
        const int T = std::max(1, std::min(bpolicy->window(sl.dprob, sl.n_dr), most));
        float chance[std::size(sl.dprob)];
        std::copy(sl.dprob, sl.dprob + (T - 1), chance);
        win_.assign((size_t) T, 0);
        out_.assign((size_t) T, -1);
        win_[0] = sl.x;
        std::copy(sl.dr, sl.dr + (T - 1), win_.begin() + 1);
        basalt::kernels::SamplerParams& sp = slot_sampling_[(size_t) b];
        basalt::model::DecodeRequest d;
        d.tokens = win_.data();
        d.n_tokens = T;
        d.pos = sl.p;
        (T > 1 ? d.row_tokens : d.next_token) = out_.data();
        d.sampling = &sp;
        if (sl.pen_n > 0) {
            history.resize((size_t) T * sl.pen_n);
            basalt::kernels::penalty_rows(sl.ids.data(), (int64_t) sl.ids.size(), win_.data(), T, sl.pen_n, history.data());
            d.history = history.data();
            d.history_len = sl.pen_n;
        }
        sp.counter = (uint64_t) sl.produced;
        if (const basalt::model::Result r = session().decode_window(d); !r.ok()) return failed(r.message);
        int a = 0;
        while (a < T - 1 && win_[(size_t) a + 1] == out_[(size_t) a]) ++a;
        if (sl.guard != nullptr && sl.guard->on()) {
            // the first pick the grammar refuses is drawn again under its mask, and the window ends there
            const int miss = sl.guard->scan(out_.data(), a + 1);
            if (miss >= 0) {
                sl.guard->mask_bias({}, grammar_bias);
                int32_t y = -1;
                if (const basalt::model::Result r = session().resample_row(miss, grammar_bias.data(), y); !r.ok()) return failed(r.message);
                out_[(size_t) miss] = y;
                a = miss;
                if (!sl.guard->take(y)) basalt::log::info("basalt grammar", "slot %d: the masked draw was token %d", b, (int) y);
            }
        }
        sl.drafts_offered += T - 1;
        sl.drafts_accepted += a;
        std::string line = "BT " + std::to_string(b);
        const char* fin = nullptr;
        for (int i = 0; i <= a && fin == nullptr; ++i) {
            const int32_t y = out_[(size_t) i];
            sl.ids.push_back(win_[(size_t) i]);
            ++sl.produced;
            ++sl.p;
            sl.x = y;
            line += " " + std::to_string(y);
            const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
            fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new || sl.p + 2 > o.max_context ? "length" : nullptr;
        }
        sl.n_dr = 0;
        bool drafting = false;
        if (fin == nullptr && session().max_window() > 1) {
            const basalt::model::Result r = session().draft_begin(sl.x, sl.p, drafts_ahead(T, session().max_window()));
            if (!r.ok() && r.status != basalt::model::Status::NotImplemented) return failed(r.message);
            drafting = r.ok();
        }
        basalt::protocol::emit_text(line + "\n");
        basalt::core::progress_beat();
        if (drafting) {
            const int max_n = std::min(drafts_ahead(T, session().max_window()), (int) std::size(sl.dr));
            if (const basalt::model::Result r = session().draft_end(sl.dr, sl.dprob, max_n, sl.n_dr); !r.ok()) return failed(r.message);
            bpolicy->observe_window(chance, T, a, std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
        }
        if (fin != nullptr) {
            std::printf("BDONE %d %lld %s %.1f %lld %lld\n", b, (long long) sl.produced, fin,
                        std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                        (long long) sl.drafts_accepted, (long long) sl.drafts_offered);
            sl.active = false;
            sl.cached = o.prompt_cache > 0;
            sl.idle_busy = busy_now();
            sl.guard.reset();
        }
        std::fflush(stdout);
    }
    set_active(0);
    return true;
}

Flow SessionRuntime::run_request() {
    SessionRequest request(*this);
    return request.execute();
}

// Each run of pad tokens takes the next picture's rows from the request's embeddings file.
Flow SessionRequest::prepare_image() {
    // a session keeps one row of logits, so the gate scores one row per window
    if (const Flow flow = begin_quality_gate(runtime_.o.ppl_base, runtime_.o.ppl_out, 1); flow != Flow::proceed) return flow;
    if (!geni) return Flow::proceed;
    std::string bad;
    std::ifstream ef(emb_path, std::ios::binary);
    if (!ef) bad = "cannot open " + emb_path;
    for (int64_t i = 0; bad.empty() && i < n;) {
        if (ids[(size_t) i] != runtime_.o.image_pad_id) { ++i; continue; }
        int32_t hdr[5];
        if (!ef.read((char*) hdr, sizeof hdr) || hdr[0] != kEmbeddingsMagic) { bad = "bad embeddings file"; break; }
        int64_t run = 0;
        while (i + run < n && ids[(size_t) (i + run)] == runtime_.o.image_pad_id) ++run;
        if (hdr[1] != run) { bad = "the prompt and its images do not match"; break; }
        Block b;
        b.at = i;
        b.n_rows = hdr[1];
        b.rows.resize((size_t) hdr[1] * (size_t) hdr[4]);
        if (!ef.read((char*) b.rows.data(), (std::streamsize) (b.rows.size() * sizeof(float)))) { bad = "short embeddings file"; break; }
        const int64_t grid[3] = {hdr[1], hdr[2], hdr[3]};
        uint64_t h = fnv1a(grid, sizeof grid);
        h = fnv1a(b.rows.data(), b.rows.size() * sizeof(float), h);
        runtime_.req_imgs.push_back({i, h});
        blocks_.push_back(std::move(b));
        i += run;
    }
    if (bad.empty() && ids[(size_t) (n - 1)] == runtime_.o.image_pad_id) bad = "the prompt must end with text";
    if (!bad.empty()) {
        std::printf("ERR %s\n", bad.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    return Flow::proceed;
}

// Positions at..to: text through the model's prompt path, pictures as rows, then the checkpoint of a turn or root part.
Flow SessionRequest::read_to(int64_t to) {
    if (at == read_from)
        for (next_block_ = 0; next_block_ < blocks_.size() && blocks_[next_block_].at < at;) ++next_block_;
    while (at < to) {
        const bool image = next_block_ < blocks_.size() && blocks_[next_block_].at == at;
        basalt::model::Result r;
        if (image) {
            const Block& b = blocks_[next_block_++];
            r = rt_.session().prefill_embeddings(b.rows.data(), b.n_rows, at);
            at += b.n_rows;
        } else {
            const int64_t end = next_block_ < blocks_.size() ? std::min(to, blocks_[next_block_].at) : to;
            basalt::model::PrefillRequest pr;
            pr.tokens = ids32.data() + at;
            pr.n_tokens = (int) (end - at);
            pr.pos = at;
            pr.total_tokens = (int) (n - 1 - at);
            pr.rows_after = to - end;
            pr.more = end < n - 1;
            const int64_t base = at;
            pr.progress = [this, base](int k) { progress_line("PP", base + k); basalt::core::progress_beat(); };
            pr.live = [this, base](int k) {
                basalt::core::progress_beat();
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
                if (ms - live_ms_ < 500.0 && ms >= live_ms_) return;
                live_ms_ = ms;
                progress_line("PL", base + k);
            };
            pr.stop = [this] { return runtime_.halted(); };
            r = rt_.session().prefill_chunk(pr);
            at = end;
        }
        if (!r.ok()) {
            basalt::log::error("basalt serve", "%s", r.message.c_str());
            std::printf("ERR %s\n", r.message.c_str());
            return Flow::error;
        }
        if (runtime_.halted()) { cancelled = true; return Flow::quit; }
        progress_line("PP", at);
    }
    if ((to == turn_at || to == root_at) &&
        !rt_.checkpoint_at(to, to == root_at ? ConvCheckpoint::kRoot : ConvCheckpoint::kTurnEnd, ids)) {
        std::printf("ERR saving a conversation checkpoint failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    return Flow::proceed;
}

// ENC <image> <out> [<image> <out> ...]: one `ENC OK <n> <nx> <ny> <ms>` per image, or one `ENC ERR <message>`.
void SessionRuntime::enc_command(const std::string& cmd) {
    std::istringstream in(cmd);
    std::string word;
    in >> word;
    std::vector<std::string> words;
    while (in >> word) words.push_back(word);
    std::vector<std::string> replies;
    std::string fail;
    for (size_t i = 0; i + 1 < words.size() && fail.empty(); i += 2) {
        std::ifstream f(words[i], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<float> rows;
        int rows_n = 0;
        const auto te = Clock::now();
        const basalt::model::Result er = bytes.empty() ? basalt::model::Result{basalt::model::Status::Error, "cannot read the image " + words[i]}
                                                       : session().encode_image(bytes.data(), bytes.size(), rows, rows_n);
        if (!er.ok()) { fail = er.message; break; }
        const int32_t nx = 0, ny = 0, width = (int32_t) (rows.size() / (size_t) rows_n);
        const int32_t hdr[5] = {kEmbeddingsMagic, rows_n, nx, ny, width};
        std::FILE* fo = std::fopen(words[i + 1].c_str(), "wb");
        const bool ok = fo && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                        std::fwrite(rows.data(), sizeof(float), rows.size(), fo) == rows.size();
        if (fo) std::fclose(fo);
        if (!ok) { fail = "cannot write " + words[i + 1]; break; }
        replies.push_back("ENC OK " + std::to_string(rows_n) + " " + std::to_string(nx) + " " + std::to_string(ny) + " " +
                          std::to_string((int) std::chrono::duration<double, std::milli>(Clock::now() - te).count()));
    }
    session().images_done();
    if (!fail.empty()) std::printf("ENC ERR %s\n", fail.c_str());
    else for (const std::string& r : replies) std::printf("%s\n", r.c_str());
    std::fflush(stdout);
}

}  // namespace

int run_session(CommonOptions& o, basalt::core::ModelFile& file, const basalt::model::ModelAdapter& adapter,
                basalt::model::LoadedModel& model) {
    std::string err;
    SessionRuntime runtime(o, file, err, model);
    runtime.pre_ = adapter.pretokenizer();
    return runtime.execute();
}

}  // namespace basalt::program::serve
