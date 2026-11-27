// src/program/serve/session_engine.cpp - the shared server over a model's ModelSession: one token per window.
#include "basalt/core/gpu_mem_info.hpp"
#include "session_engine.hpp"

#include "batch_plan.hpp"
#include "decode_loop.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/protocol.hpp"
#include "basalt/spec/draft_policy.hpp"
#include "basalt/vision/ffmpeg_api.hpp"
#include "basalt/vision/video.hpp"

#include <iterator>
#include <sstream>

namespace basalt::program::serve {

namespace {

constexpr int32_t kEmbeddingsMagic = 0x31455653;

class SessionRuntime;

/// A picture's (or a video group's) embedding rows, placed at prompt position `at`, and its grid.
struct Block {
    int64_t at = 0;
    std::vector<float> rows;
    int n_rows = 0;
    basalt::model::ImageGrid grid;
};

class SessionRuntime final : public Runtime {
public:
    SessionRuntime(CommonOptions& options, basalt::core::ModelFile& file, std::string& error, basalt::model::Started& started)
        : Runtime(options, file, error, started.model->desc().n_vocab, vocab_), model_(*started.model),
          sessions_(std::move(started.sessions)) {}

    int init_before_input() override {
        if (sessions_.size() != (size_t) std::max(o.batch, 1)) {   // the start makes one per slot
            basalt::log::error("basalt serve", "the model has %zu sessions for %d slots", sessions_.size(), std::max(o.batch, 1));
            return 1;
        }
        // one policy for the solo windows and one for the batch windows, whose rounds cost differently
        policy.emplace(session().max_window());
        bpolicy.emplace(session().max_window());
        if (lookup_on()) sfx.emplace(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        if (o.batch > 0) {
            bs.resize((size_t) o.batch);
            slot_sampling.resize((size_t) o.batch);
        }
        conversations.emplace((size_t) o.conversation_cache_mib << 20, (size_t) o.conversation_cache_slots);
        if (o.conversation_cache_disk_mib > 0) {
            const auto t0 = Clock::now();
            std::string text = "basalt conversation state " + model_.desc().arch + " vision_in_engine " + (o.vision_in_engine ? "1" : "0");
            if (const std::string rope = basalt::model::rope_flag_words(o.rope); !rope.empty()) text += " " + rope;
            const uint64_t identity = basalt::core::conversation_store_hash_file(
                o.model, basalt::core::conversation_store_hash(text.data(), text.size(), 1));
            std::string de;
            disk.set_row_cells(session().disk_row_cells());
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
        for (const basalt::model::GpuCards::Card& c : o.gpus.cards) {
            const int d = c.device;
            std::string bus;
            if (basalt::core::device_pci_bus_id(d, bus)) ids += (ids.empty() ? "" : ",") + bus;
        }
        std::string quant = basalt::program::pack_quant_label(nm != nullptr && nm->is_string() ? nm->s : std::string());
        if (quant == "unknown" && !o.community_quant.empty()) quant = o.community_quant;   // the model names it when the pack's name does not
        return "quant=" + quant + " gpus=" + ids + " ";
    }
    int init_after_input() override {
        reserve_checkpoints();
        for (auto& s : sessions_)
            if (const basalt::model::Result r = s->prepare(); !r.ok()) {
                basalt::log::error("basalt serve", "%s", r.message.c_str());
                return 1;
            }
        start_guard();
        // whether the in-engine encoder reads video: FFmpeg found, for a model that reads the shared patches
        if (o.vision_in_engine && sessions_[0]->reads_patches()) std::printf("%s\n", basalt::vision::video_line().c_str());
        std::printf("INFO %s\n", (identity_words() + sessions_[0]->info_words()).c_str());
        kv_ceiling_note();
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
    bool next_batch_window() override { return batch_round(); }
    basalt::model::ModelSession& slot_session(int b) override { return *sessions_[(size_t) b]; }
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
        if (reserve >= 0) {
            const int floor = vram_floor_mib();
            if (reserve < floor) {
                e = "VRAM reserve is below " + std::to_string(floor) + " MiB";
                return false;
            }
            if ((uint64_t) reserve > (UINT64_MAX >> 20)) {
                e = "VRAM reserve is too large";
                return false;
            }
        }
        int64_t slots = 0;
        if (const basalt::model::Result r = session().set_vram_reserve(reserve, slots); !r.ok()) {
            e = r.message;
            return false;
        }
        return memory_reply(reserve, slots, reply);
    }
    void sleep_command(bool down, std::string& e) override {
        std::string reply;
        int64_t slots = -1;
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
        ckd_finish();   // a turn checkpoint's state may still be on its way to the host
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
            if (!disk.fits(session().running_bytes() + session().cells_bytes(positions))) return;   // no copy the writer drops
            const uint64_t snapshot = session().running_bytes() + session().cells_bytes(positions);
            if (!admit_parking(snapshot, floor, "disk write", ids.size(), snapshot)) return;
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
    // a model with an elastic KV: its ceiling as an INFO line whenever it moves (the server reports it in /v1/status)
    int64_t kv_ceiling_sent_ = -1;
    void kv_ceiling_note() override {
        const int64_t n = sessions_[0]->kv_ceiling();
        if (n < 0 || n == kv_ceiling_sent_) return;
        kv_ceiling_sent_ = n;
        std::printf("INFO elastic_kv_ceiling=%lld\n", (long long) n);
        std::fflush(stdout);
    }
    LookupCounts lookup_counts() override {
        const basalt::model::ModelSession::ExpertCounts c = sessions_[0]->expert_counts();
        return {c.hits, c.lookups, c.offloaded, c.files};
    }
    bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) override {
        return sessions_[0]->expert_map(total, homes, routed);
    }
    std::vector<CardFacts> card_facts() override {
        std::vector<CardFacts> out;
        for (const basalt::model::ModelSession::CardExperts& c : sessions_[0]->experts_by_card())
            out.push_back({c.device, c.slots, 0.0, c.bytes, out.empty() ? -1 : 0});
        // every session's KV is on the main card; a second card holds experts only
        int64_t kv = 0;
        for (const auto& s : sessions_) kv = kv < 0 || s->kv_bytes() < 0 ? -1 : kv + s->kv_bytes();
        if (!out.empty()) out[0].kv_bytes = kv;
        return out;
    }
    /// The session's conversation as an image: its running state and its positional state by layer.
    bool capture(basalt::core::SavedConversation& image, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                 const std::vector<ConvCheckpoint>& cks, bool cvec, std::string& e) {
        const int64_t positions = (int64_t) ids.size();
        image.live.ids = ids;
        image.live.imgs = imgs;
        basalt::model::Result r = save_running_now(image.live.state_a, positions);
        if (r.ok()) r = session().save_cells(image.kv, positions);
        if (!r.ok()) { e = r.message; return false; }
        image.checkpoints = cks;
        image.cvec = cvec;
        return true;
    }
    // The outgoing conversation leaves the session for the RAM tier, or the disk tier when there is no RAM tier or
    // no room in it.
    bool park_current(size_t held, int64_t kept) override {
        ckd_finish();   // a turn checkpoint's state may still be on its way to the host
        if ((!conversations->enabled() && !disk.enabled()) || !live_ok || live.empty()) return true;
        if (o.conversation_cache_min_tokens > 0 && (int64_t) live.size() < o.conversation_cache_min_tokens) return true;
        int64_t deepest = 0;
        for (const ConvCheckpoint& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
        const bool disk_wants = disk.enabled() && !(kept > 0 && kept >= deepest) && !disk_has(live, live_imgs, checks, cvec_cached);
        bool to_disk = !conversations->enabled();
        if (to_disk && !disk_wants) return true;
        if (!to_disk)
            if (const size_t dropped = conversations->drop_superseded(live, live_imgs, checks, cvec_cached))
                basalt::log::info("basalt serve", "conversation cache: dropped %zu superseded cop%s of this "
                                  "conversation; parked=%zu", dropped, dropped == 1 ? "y" : "ies", conversations->size());
        const int64_t positions = (int64_t) live.size();
        size_t estimate = session().running_bytes() + session().cells_bytes(positions) +
                          live.size() * sizeof(int32_t) + live_imgs.size() * sizeof(ImgKey);
        for (const ConvCheckpoint& c : checks) estimate += c.bytes();
        const bool ram_room = !to_disk && conversations->make_room(estimate, held);
        if (!to_disk && !ram_room && disk_wants) to_disk = true;
        if (to_disk && !disk.fits(estimate)) return true;   // a full disk: no copy off the card that the writer drops
        if (to_disk ? disk.queued_bytes() + estimate > kDiskQueueCap : !ram_room) return true;
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk.enabled() ? disk_floor : 0);
        if (!admit_parking(estimate, floor, "parking", live.size(), estimate)) return true;
        basalt::core::SavedConversation image;
        if (!capture(image, live, live_imgs, checks, cvec_cached, err)) return false;
        (void) (to_disk ? disk.put(std::move(image), kDiskQueueCap) : conversations->put(std::move(image), held));
        report_conversations();
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
        ckd_finish();   // the save before this one lands first; one that failed has left the chain
        if (touch_check(to, source)) return true;
        ConvCheckpoint c;
        c.source = source;
        c.ids.assign(prompt.begin(), prompt.begin() + to);
        c.imgs = imgs_below(req_imgs, to);
        basalt::core::RunningState rs;
        if (const basalt::model::Result r = session().running_state(to, true, rs); !r.ok()) {
            err = r.message;
            return false;
        }
        c.state_a.resize(rs.bytes());
        std::memcpy(c.state_a.data(), rs.head.data(), rs.head.size());
        // the state is snapshotted on the device now and moves to the host while the request goes on (each card's part on
        // its own device)
        const std::vector<basalt::core::SaveSegment> seg{{c.state_a.data() + rs.head.size(), rs.regions}};
        if (!ckd.save(seg, (cudaStream_t) rs.stream, c.state_a.data(), err)) return false;
        if (!rs.peer_regions.empty()) {
            const basalt::core::OnDevice on(rs.peer_device);
            ckd_peer_dev = rs.peer_device;
            const std::vector<basalt::core::SaveSegment> ps{{c.state_a.data() + rs.head.size() + rs.regions_bytes(), rs.peer_regions}};
            if (!ckd_peer.save(ps, (cudaStream_t) rs.peer_stream, c.state_a.data(), err)) return false;
        }
        keep_check(std::move(c));
        return true;
    }

    /// The running state as it is now, into `to`: when this returns it has landed.
    basalt::model::Result save_running_now(basalt::core::CheckpointBytes& to, int64_t pos) {
        basalt::core::RunningState rs;
        if (const basalt::model::Result r = session().running_state(pos, true, rs); !r.ok()) return r;
        to.resize(rs.bytes());
        std::memcpy(to.data(), rs.head.data(), rs.head.size());
        std::string e;
        if (!basalt::core::running_copy_out(rs, to.data(), e)) return {basalt::model::Status::Error, e};
        return {};
    }
    /// A checkpoint's or a parked conversation's running state back into the session.
    basalt::model::Result restore_running_from(const basalt::core::CheckpointBytes& from, int64_t pos) {
        basalt::core::RunningState rs;
        if (const basalt::model::Result r = session().running_state(pos, false, rs); !r.ok()) return r;
        if (from.size() != rs.bytes()) return {basalt::model::Status::Error, "a checkpoint's running state has the wrong size"};
        std::string e;
        if (!basalt::core::running_copy_in(rs, from.data(), e)) return {basalt::model::Status::Error, e};
        return session().running_restored(pos, from.data());
    }

    /// The host blocks the checkpoints take (page-locked, their pages written) and the device buffer a save is
    /// snapshotted into, held before the models' RAM and VRAM tiers are sized so they plan beside them.
    void reserve_checkpoints() {
        if (o.prompt_cache <= 0) return;
        const size_t bytes = session().running_bytes();
        if (bytes == 0) return;
        const size_t keep = (size_t) (o.prompt_cache + 1) * (size_t) std::max(o.batch, 1);
        if (!basalt::core::CheckpointBytes::allocator_type::reserve(bytes, keep, &basalt::core::kCheckpointPin))
            basalt::log::warn("basalt serve", "checkpoint blocks not reserved: out of host memory");
        if (std::string e; !ckd.reserve(bytes, e)) basalt::log::warn("basalt serve", "deferred checkpoint copy off: %s", e.c_str());
        basalt::core::RunningState rs;   // a split model's second card keeps a spare of its own part
        if (session().running_state(0, false, rs).ok() && !rs.peer_regions.empty()) {
            const basalt::core::OnDevice on(rs.peer_device);
            ckd_peer_dev = rs.peer_device;
            if (std::string e; !ckd_peer.reserve(basalt::core::RunningState::sum(rs.peer_regions), e))
                basalt::log::warn("basalt serve", "deferred checkpoint copy off on the second card: %s", e.c_str());
        }
    }

    basalt::model::ModelSession& session() { return *sessions_[(size_t) active_]; }
    void set_active(int slot) { active_ = slot; }
    void set_slot_sampling(int slot, const basalt::kernels::SamplerParams& sp) { slot_sampling[(size_t) slot] = sp; }
    const basalt::model::ModelDesc& desc() const { return model_.desc(); }
    /// The solo windows' policy (basalt/spec/draft_policy.hpp; the batch windows' is bpolicy): learns this machine's round
    /// times and the drafter's acceptance across requests.
    std::optional<basalt::spec::DraftPolicy> policy;
    /// Prompt lookup (--suffix-draft) for a model whose sessions take copied drafts: the solo request's history.
    bool lookup_on() const { return o.suffix_draft > 0 && sessions_[0]->copy_drafts(); }
    std::optional<basalt::spec::SuffixDrafter> sfx;
    /// --draft-log FILE, opened once.
    FILE* draft_log() {
        static FILE* const f = std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
        return f;
    }

private:
    bool memory_reply(long long reserve, int64_t slots, std::string* reply) {
        size_t free_b = 0, total_b = 0;
        basalt::core::gpu_mem_info(&free_b, &total_b);
        char line[160];
        char held[48] = "";   // a session with no cache names no slots
        if (slots >= 0) std::snprintf(held, sizeof held, " expert_slots=%lld", (long long) slots);
        std::snprintf(line, sizeof line, "VRAM reserve_mib=%lld%s vram_free_mib=%lld\n", reserve, held,
                      (long long) (free_b >> 20));
        if (reply != nullptr) *reply = line;
        else std::fputs(line, stdout);
        return true;
    }

    std::optional<basalt::program::grammar::PackVocab> vocab_;
    basalt::model::LoadedModel& model_;
    std::vector<std::unique_ptr<basalt::model::ModelSession>> sessions_;
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
        hybrid_drafts_ = false;
        drafts_.assign((size_t) max_window(), 0);
        conf_.assign((size_t) max_window(), 0.0f);
        sbuf_.assign((size_t) max_window(), 0);
        if (rt_.sfx && (!in_slot || solo_req))
            rt_.sfx->sync(ids32.data(), ids32.size());
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
        return (int) std::max<int64_t>(1, std::min<int64_t>(T, most_rows()));
    }
    // the context's end, or a position the session changes state at (window_limit)
    int64_t most_rows() const {
        const int limit = rt_.session().window_limit(p);
        const int64_t room = runtime_.o.max_context - p;
        return limit > 0 ? std::min<int64_t>(room, limit) : room;
    }
    // a lookup proposal takes the window when its first token is the drafter's (or the drafter has none) and the policy
    // expects more tokens per ms from it
    void lookup_window(int& T, bool& from_sfx, int& sfx_match) {
        if (!rt_.sfx || first_window || ppl != nullptr) return;
        const int most = (int) std::min<int64_t>(max_window(), most_rows());
        if (most < 2) return;
        const int k = rt_.sfx->propose(most - 1, sbuf_.data());
        sfx_match = rt_.sfx->last_match();
        const bool allow_override = sfx_match >= basalt::spec::SuffixDrafter::LONG_ORDER && (n_draft_ == 0 || conf_[0] < 0.85f);
        if (k <= 0 || (n_draft_ > 0 && sbuf_[0] != drafts_[0] && !allow_override)) return;
        const basalt::spec::DraftPolicy::Pick pk = rt_.policy->choose_window(conf_.data(), n_draft_, k, sfx_match);
        if (pk.lookup) {
            T = std::min(pk.t, most);
            from_sfx = true;
        }
    }
    void fill_drafts(int T, bool from_sfx) {
        for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf_[(size_t) i - 1] : drafts_[(size_t) i - 1];
        own_drafts_ = !from_sfx && !hybrid_drafts_;
    }
    int before_window(int, int64_t) { return 0; }
    int32_t* history_stage() {
        history_.resize((size_t) std::max<int64_t>({(int64_t) max_window(), (int64_t) rt_.session().prompt_fold_window(), n - fold_p0}) * hist_n);
        return history_.data();
    }
    void upload_history(int) {}
    void mark_before_run() {}
    bool run_window(int T, int e0, int&) {
        basalt::model::DecodeRequest d;
        d.tokens = window.data();
        d.n_tokens = T;
        d.pos = p;
        const bool tail_only = e0 == T - 1 && T > 1 && req_logprobs < 0 && (gguard == nullptr || !gguard->on()) && ppl == nullptr;
        d.drawn_drafts = own_drafts_ && e0 == 0 && ppl == nullptr && T > 1;
        if (tail_only) d.next_token = &outv[(size_t) e0];
        else (T > 1 ? d.row_tokens : d.next_token) = outv.data();
        d.sampling = &req_sp;
        d.history = hist_n > 0 ? history_.data() + (tail_only ? (size_t) e0 * hist_n : 0) : nullptr;
        d.history_len = hist_n;
        req_sp.counter = draws_;
        const basalt::model::Result r = rt_.session().decode_window(d);
        refused_ = r.status == basalt::model::Status::Refused;
        if (refused_) basalt::log::warn("basalt serve", "%s", r.message.c_str());
        if (!r.ok()) error_ = r.message;
        return r.ok();
    }
    bool take_held(int) { return false; }
    bool window_failed() const { return false; }
    bool window_refused() const { return refused_; }
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
    bool copy_logits(int row, float* out) { return rt_.session().copy_logits(row, out).ok(); }
    bool row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) {
        return rt_.session().row_logprobs(row0, n, picks, top, out).ok();
    }
    bool after_accept(int, int&, int, bool, bool, bool, bool& hold) {
        hold = false;
        return true;
    }
    void note_draft_probs(bool from_sfx) {
        if (from_sfx) window_conf_.clear();
        else window_conf_.assign(conf_.begin(), conf_.begin() + n_draft_);
    }
    void stamp_before_commit() {}
    // each row's pick is the draw of its own index, whatever the width of the window
    bool commit(int keep, std::string& e, int phase) {
        if (phase != 2) return true;
        draws_ += (uint64_t) keep;
        basalt::model::ModelSession* s = &rt_.session();
        const basalt::model::Result r = s->decode_kept(&s, &keep, 1);
        if (!r.ok()) e = r.message;
        return r.ok();
    }
    void on_token(int32_t tok) {
        if (rt_.sfx) rt_.sfx->append(tok);
    }
    void count_round() {}
    // The next window's drafts start before this window's tokens go out, which run in their shadow.
    template <class After>
    bool draft_next(int T, int a, int e0, int, bool, bool, bool will_draft, After& after_launch) {
        n_draft_ = 0;
        hybrid_drafts_ = false;
        if (!will_draft || max_window() < 2) return true;
        // Skip launching the GPU drafter when prompt lookup already has a long high-confidence exact match at the new tail.
        if (rt_.sfx && ppl == nullptr && a >= e0) {
            int pre_match = 0;
            const int pre_k = rt_.sfx->propose_after(outv.data() + e0, (size_t) (a - e0 + 1), max_window() - 1, sbuf_.data(), &pre_match);
            if (pre_match >= 2 * basalt::spec::SuffixDrafter::LONG_ORDER && pre_k >= basalt::spec::SuffixDrafter::LONG_ORDER &&
                rt_.policy->lookup_rate(pre_match) >= 0.90) {
                after_launch();
                return true;
            }
        }
        const int n = drafts_ahead(T, max_window());
        if (basalt::model::Result r = rt_.session().draft_begin(outv[(size_t) a], p + a + 1, n); !r.ok()) {
            if (r.status == basalt::model::Status::NotImplemented) return true;
            runtime_.err = r.message;
            return false;
        }
        after_launch();
        // room for every draft the session makes (a block drafter may make more than asked, the policy picks); a draft
        // no window can hold need not be made, but a lookup proposal is matched against the first
        const auto need = [this](const float* c, int j, int m) { return (j == 0 && rt_.sfx) || rt_.policy->may_verify(c, j, m); };
        if (basalt::model::Result r = rt_.session().draft_end(drafts_.data(), conf_.data(), max_window() - 1, n_draft_, need); !r.ok()) {
            runtime_.err = r.message;
            return false;
        }
        // Hybrid drafter -> suffix continuation: when confident drafts re-enter verbatim prompt text after an edit,
        // extend the window up to max_window() - 1 using the suffix index.
        if (rt_.sfx && ppl == nullptr && n_draft_ >= 4 && n_draft_ < max_window() - 1) {
            double joint = 1.0;
            for (int i = 0; i < n_draft_; ++i) joint *= (double) conf_[(size_t) i];
            if (joint >= 0.60) {
                int tail_match = 0;
                const int ext = rt_.sfx->propose_after(drafts_.data(), (size_t) n_draft_, max_window() - 1 - n_draft_,
                                                       drafts_.data() + n_draft_, &tail_match);
                if (ext > 0 && tail_match >= 6) {
                    const float q = (float) rt_.policy->lookup_rate(tail_match);
                    for (int j = 0; j < ext; ++j) conf_[(size_t) (n_draft_ + j)] = q;
                    n_draft_ += ext;
                    hybrid_drafts_ = true;
                }
            }
        }
        return true;
    }
    bool draft_fatal() const { return false; }
    void window_timing(int T, int a, bool from_sfx, Clock::time_point round0, Clock::time_point tw0, Clock::time_point tw1, Clock::time_point) {
        if (draft_log_ == nullptr) return;
        const auto msd = [](Clock::time_point x, Clock::time_point y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
        std::fprintf(draft_log_, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0), msd(tw0, tw1), msd(tw1, Clock::now()));
        for (float c : window_conf_) std::fprintf(draft_log_, " %.4f", (double) c);
        std::fputc('\n', draft_log_);
    }
    void observe_policy(bool from_sfx, int T, int a, int sfx_match, Clock::time_point round0) {
        if (ppl != nullptr) return;   // a quality gate's windows are forced, not drafted
        const double ms = rt_.session().window_cold() ? 0.0 : std::chrono::duration<double, std::milli>(Clock::now() - round0).count();
        if (from_sfx) rt_.policy->observe(true, T, a, sfx_match, ms);
        else rt_.policy->observe_window(window_conf_.data(), T, a, ms);
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
        const basalt::model::Result r = rt_.restore_running_from(c.state_a, (int64_t) c.ids.size());
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
    // a slot's prompt lookup history: the conversation, then the token its first window feeds
    void slot_init(BSlot& sl) override {
        if (!rt_.lookup_on()) return;
        if (!sl.sfx)
            sl.sfx = std::make_unique<basalt::spec::SuffixDrafter>(std::max(1, runtime_.o.suffix_draft), 64,
                                                                   (size_t) runtime_.o.max_context + 4096);
        sl.sfx->sync(runtime_.live.data(), runtime_.live.size());
        sl.sfx->append(x);
    }
    Flow restore_image() override {
        const int64_t positions = (int64_t) incoming->live.ids.size();
        // a conversation stored by a build whose running state had another layout is read again, not mounted
        bool sized = incoming->live.state_a.size() == rt_.session().running_bytes();
        for (const ConvCheckpoint& c : incoming->checkpoints) sized = sized && c.state_a.size() == rt_.session().running_bytes();
        basalt::model::Result r = sized ? rt_.session().restore_cells(incoming->kv, positions)
                                        : basalt::model::Result{basalt::model::Status::Error, "its running state has another size"};
        if (r.ok()) r = rt_.restore_running_from(incoming->live.state_a, positions);
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
    bool own_drafts_ = false;
    bool hybrid_drafts_ = false;   // the window's drafts are the drafter's (fill_drafts), not a lookup proposal
    std::vector<int32_t> drafts_, sbuf_;
    std::vector<float> conf_, window_conf_;
    int n_draft_ = 0;
    FILE* draft_log_ = nullptr;
    basalt::model::ModelSession::ExpertCounts counts0_;
    std::string error_;
    bool refused_ = false;   // the last window was refused (Status::Refused): the request ends, the engine serves on
    size_t next_block_ = 0;
    double live_ms_ = -1e9;
};

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
    const int64_t vpad = runtime_.o.video_pad_id;
    const auto is_pad = [&](int64_t id) { return id == runtime_.o.image_pad_id || (vpad >= 0 && id == vpad); };
    for (int64_t i = 0; bad.empty() && i < n;) {
        if (!is_pad(ids[(size_t) i])) { ++i; continue; }
        int32_t hdr[5];
        if (!ef.read((char*) hdr, sizeof hdr) || hdr[0] != kEmbeddingsMagic) { bad = "bad embeddings file"; break; }
        // a run is one record's rows: a picture, or one two-frame group of a video (groups sit between their times)
        int64_t run = 0;
        while (i + run < n && run < hdr[1] && ids[(size_t) (i + run)] == ids[(size_t) i]) ++run;
        if (hdr[1] != run || (i + run < n && ids[(size_t) (i + run)] == ids[(size_t) i])) { bad = "the prompt and its images do not match"; break; }
        Block b;
        b.at = i;
        b.n_rows = hdr[1];
        if (hdr[2] > 0 && hdr[3] > 0 && (int64_t) hdr[2] * hdr[3] == hdr[1]) b.grid = {hdr[2], hdr[3]};
        b.rows.resize((size_t) hdr[1] * (size_t) hdr[4]);
        if (!ef.read((char*) b.rows.data(), (std::streamsize) (b.rows.size() * sizeof(float)))) { bad = "short embeddings file"; break; }
        const int64_t grid[3] = {hdr[1], hdr[2], hdr[3]};
        uint64_t h = fnv1a(grid, sizeof grid);
        h = fnv1a(b.rows.data(), b.rows.size() * sizeof(float), h);
        runtime_.req_imgs.push_back({i, h});
        blocks_.push_back(std::move(b));
        i += run;
    }
    if (bad.empty() && is_pad(ids[(size_t) (n - 1)])) bad = "the prompt must end with text";
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
    int64_t fold_tail = 0;
    const int fw = rt_.session().prompt_fold_window();
    if (fw > 1 && to == n - 1 && !part_checkpoint(to) && next_block_ >= blocks_.size() && (!in_slot || solo_req) &&
        ppl == nullptr && to > at && to - at < fw) {
        fold_tail = to - at;
    }
    const int64_t pre_to = to - fold_tail;
    while (at < pre_to) {
        const bool image = next_block_ < blocks_.size() && blocks_[next_block_].at == at;
        basalt::model::Result r;
        if (image) {
            const Block& b = blocks_[next_block_++];
            r = rt_.session().prefill_embeddings(b.rows.data(), b.n_rows, at, b.grid);
            at += b.n_rows;
        } else {
            const int64_t end = next_block_ < blocks_.size() ? std::min(pre_to, blocks_[next_block_].at) : pre_to;
            basalt::model::PrefillRequest pr;
            pr.tokens = ids32.data() + at;
            pr.n_tokens = (int) (end - at);
            pr.pos = at;
            pr.total_tokens = (int) (n - 1 - at);
            pr.rows_after = to - end;
            pr.more = end < n - 1;
            const int64_t base = at;
            pr.progress = [this, base](int k) {
                runtime_.pp_reached = std::max(runtime_.pp_reached, base + k);
                progress_line("PP", base + k);
                basalt::core::progress_beat();
            };
            pr.live = [this, base](int k) {
                basalt::core::progress_beat();
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
                if (ms - live_ms_ < 500.0 && ms >= live_ms_) return;
                live_ms_ = ms;
                progress_line("PL", base + k);
            };
            pr.stop = [this] { return runtime_.halted(); };
            r = rt_.session().prefill_chunk(pr);
            if (!runtime_.halted()) at = end;
        }
        if (r.status == basalt::model::Status::Refused) {   // no room for this prompt: the engine serves on
            basalt::log::warn("basalt serve", "%s", r.message.c_str());
            rt_.session().end_prompt();
            std::printf("ERR %s\n", r.message.c_str());
            return Flow::next;
        }
        if (!r.ok()) {
            basalt::log::error("basalt serve", "%s", r.message.c_str());
            std::printf("ERR %s\n", r.message.c_str());
            return Flow::error;
        }
        if (runtime_.halted()) { cancelled = true; return Flow::quit; }
        runtime_.pp_reached = std::max(runtime_.pp_reached, at);
        progress_line("PP", at);
    }
    if (fold_tail > 0) {
        fold_p0 = pre_to;
        at = to;
    }
    if (part_checkpoint(to) &&
        !rt_.checkpoint_at(to, to == root_at ? ConvCheckpoint::kRoot : ConvCheckpoint::kTurnEnd, ids)) {
        std::printf("ERR saving a conversation checkpoint failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    return Flow::proceed;
}

// ENC <image> <out> [<image> <out> ...]: one `ENC OK <n> <nx> <ny> <ms>` per image, or one `ENC ERR <message>`.
// ENCV <video> <out> [k=v ...] (a model that reads the shared preprocessor's patches): one `ENC OK <groups> <nx> <ny>
// <ms> <time,time,...>`, the file holding one record per two-frame group.
void SessionRuntime::enc_command(const std::string& cmd) {
    const bool video = cmd.rfind("ENCV ", 0) == 0;
    std::vector<std::string> replies;
    std::string fail;
    const auto ms_since = [](Clock::time_point t) { return (int) std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    const auto write_record = [&](std::FILE* fo, const std::vector<float>& rows, int n, basalt::model::ImageGrid grid) {
        const int32_t width = n > 0 ? (int32_t) (rows.size() / (size_t) n) : 0;
        const int32_t hdr[5] = {kEmbeddingsMagic, n, grid.nx, grid.ny, width};
        return std::fwrite(hdr, sizeof hdr, 1, fo) == 1 && std::fwrite(rows.data(), sizeof(float), rows.size(), fo) == rows.size();
    };
    if (video) {
        std::string vpath, opath;
        basalt::vision::VideoOptions vo;
        const basalt::vision::Geometry geo;
        vo.group_tokens = geo.max_tokens;
        basalt::vision::Video v;
        const auto te = Clock::now();
        if (!session().reads_patches()) fail = "this model reads no video";
        else if (!basalt::vision::parse_encv(cmd, vpath, opath, vo, fail)) {
        } else if (!basalt::vision::read_video(vpath, vo, v, fail)) {
        } else {
            std::vector<float> patches, rows;
            basalt::vision::patchify_video(v.frames, (int) v.indices.size(), v.canvas, geo, patches);
            const int groups = v.groups(), nx = v.canvas.w / 32, ny = v.canvas.h / 32;
            const size_t per = patches.size() / (size_t) groups;
            std::FILE* fo = std::fopen(opath.c_str(), "wb");
            if (!fo) fail = "cannot write " + opath;
            for (int gi = 0; fail.empty() && gi < groups; ++gi) {
                const std::vector<float> part(patches.begin() + (ptrdiff_t) (gi * per), patches.begin() + (ptrdiff_t) ((gi + 1) * per));
                const basalt::model::Result er = session().encode_patches(part, v.canvas.w, v.canvas.h, rows);
                if (!er.ok()) fail = er.message;
                else if (!write_record(fo, rows, nx * ny, {nx, ny})) fail = "short write to " + opath;
            }
            if (fo) std::fclose(fo);
            if (fail.empty())
                replies.push_back("ENC OK " + std::to_string(groups) + " " + std::to_string(nx) + " " + std::to_string(ny) + " " +
                                  std::to_string(ms_since(te)) + " " +
                                  basalt::vision::times_text(basalt::vision::group_times(v.indices, v.fps)));
        }
    } else {
        std::istringstream in(cmd);
        std::string word;
        in >> word;
        std::vector<std::string> words;
        while (in >> word) words.push_back(word);
        for (size_t i = 0; i + 1 < words.size() && fail.empty(); i += 2) {
            std::ifstream f(words[i], std::ios::binary);
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::vector<float> rows;
            int rows_n = 0;
            basalt::model::ImageGrid grid;
            const auto te = Clock::now();
            const basalt::model::Result er = bytes.empty() ? basalt::model::Result{basalt::model::Status::Error, "cannot read the image " + words[i]}
                                                           : session().encode_image(bytes.data(), bytes.size(), rows, rows_n, grid);
            if (!er.ok()) { fail = er.message; break; }
            std::FILE* fo = std::fopen(words[i + 1].c_str(), "wb");
            const bool ok = fo && write_record(fo, rows, rows_n, grid);
            if (fo) std::fclose(fo);
            if (!ok) { fail = "cannot write " + words[i + 1]; break; }
            replies.push_back("ENC OK " + std::to_string(rows_n) + " " + std::to_string(grid.nx) + " " + std::to_string(grid.ny) + " " +
                              std::to_string(ms_since(te)));
        }
    }
    session().images_done();
    if (!fail.empty()) std::printf("ENC ERR %s\n", fail.c_str());
    else for (const std::string& r : replies) std::printf("%s\n", r.c_str());
    std::fflush(stdout);
}

}  // namespace

int run_session(CommonOptions& o, basalt::core::ModelFile& file, const basalt::model::ModelAdapter& adapter,
                basalt::model::Started& started) {
    std::string err;
    SessionRuntime runtime(o, file, err, started);
    const basalt::core::GpuLeftRunningExit left_running;
    runtime.pre_ = adapter.pretokenizer();
    return runtime.execute();
}

}  // namespace basalt::program::serve
