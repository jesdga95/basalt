// src/program/serve/session_engine.cpp - the shared server over a model's ModelSession: one token per window.
#include "session_engine.hpp"

#include "decode_loop.hpp"
#include "basalt/program/conv_cache.hpp"

namespace basalt::program::serve {

namespace {

constexpr int32_t kEmbeddingsMagic = 0x31455653;

class SessionRuntime;

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
        const basalt::model::Result r = model_.make_session(session_);
        if (!r.ok()) { basalt::log::error("basalt serve", "%s", r.message.c_str()); return 1; }
        conversations.emplace((size_t) 0, (size_t) 0);
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
    int init_after_input() override {
        if (const basalt::model::Result r = session_->prepare(); !r.ok()) {
            basalt::log::error("basalt serve", "%s", r.message.c_str());
            return 1;
        }
        std::printf("READY %lld stop\n", (long long) o.max_context);
        std::fflush(stdout);
        return 0;
    }
    void print_info() override {
        std::printf("INFO logit_bias=%lld context=%lld spec=0 mtp_max=0 lookup=0 engine=" BASALT_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context);
    }
    bool pack_tokenizer_supported(std::string_view) const override { return false; }
    bool windows_in_flight() override { return false; }
    bool next_batch_window() override { return true; }
    bool drain_windows() override { return true; }
    void script_command(const std::string&) override {
        std::printf("ERR BSCRIPT: this model has no batch windows\n");
        std::fflush(stdout);
    }
    void probe_command(const std::string&, std::string& e) override { e = "PROBE: this model has no probe"; }
    // VRAM <mib>: what the session can give back goes; VRAM alone: it is taken again.  One reply line.
    bool vram_command(const std::string& cmd, std::string& e, bool, std::string* reply) override {
        char* end = nullptr;
        const long long reserve = cmd.size() > 4 ? std::strtoll(cmd.c_str() + 4, &end, 10) : -1;
        if (cmd.size() > 4 && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
        return set_memory(reserve >= 0, reserve, e, reply);
    }
    void sleep_command(bool down, std::string& e) override {
        std::string reply;
        if (set_memory(down, -1, e, &reply)) std::fputs(reply.c_str(), stdout);
    }
    void between_requests(const std::string&) override {}
    Flow run_request() override;
    void enc_command(const std::string& cmd) override;
    size_t spill_to_disk(bool, size_t) override { return 0; }
    bool park_idle_engine(std::string&) override { return false; }
    bool capture_admit(uint64_t, uint64_t) override { return false; }
    bool park_current(size_t, int64_t) override { return true; }
    bool copy_from_slot(int, const ConvCheckpoint*, std::string& e) override { e = "this model has no slots"; return false; }
    bool copy_to_slot(int, const std::vector<int32_t>&, std::string& e) override { e = "this model has no slots"; return false; }
    void shed_if_low() override {
        if (basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok && !checks.empty()) checks.clear();
    }
    int64_t slot_context(int) const override { return o.max_context; }
    bool slot0_solo() const override { return false; }

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
        c.gdn.resize(session_->running_bytes());
        if (const basalt::model::Result r = session_->save_running(c.gdn.data(), to); !r.ok()) {
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

    basalt::model::ModelSession& session() { return *session_; }
    const basalt::model::ModelDesc& desc() const { return model_.desc(); }

private:
    bool set_memory(bool low, long long reserve, std::string& e, std::string* reply) {
        int64_t slots = 0;
        if (const basalt::model::Result r = session_->set_low_memory(low, slots); !r.ok()) {
            e = r.message;
            return false;
        }
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
    std::unique_ptr<basalt::model::ModelSession> session_;
};

class SessionRequest final : public Request {
public:
    explicit SessionRequest(SessionRuntime& runtime) : Request(runtime), rt_(runtime) {}

    static constexpr bool kSpeculates = false;
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
    void decode_state_begin() { draws_ = 0; }
    void decode_stats_begin() {}
    int max_window() const { return 1; }
    int before_window(int, int64_t) { return 0; }
    int32_t* history_stage() {
        history_.resize((size_t) hist_n);
        return history_.data();
    }
    void upload_history(int) {}
    void mark_before_run() {}
    bool run_window(int T, int, int&) {
        basalt::model::DecodeRequest d;
        d.tokens = window.data();
        d.n_tokens = T;
        d.pos = p;
        d.next_token = outv.data();
        d.sampling = &req_sp;
        d.history = hist_n > 0 ? history_.data() : nullptr;
        d.history_len = hist_n;
        req_sp.counter = draws_++;
        const basalt::model::Result r = rt_.session().decode_window(d);
        if (!r.ok()) error_ = r.message;
        return r.ok();
    }
    bool window_failed() const { return false; }
    const char* window_error() const { return error_.c_str(); }
    bool resample_row(int, int64_t, const std::vector<float>& bias, int32_t& tok, std::string& e) {
        const basalt::model::Result r = rt_.session().resample_last(bias.data(), tok);
        if (!r.ok()) e = r.message;
        return r.ok();
    }
    bool copy_logits(int, float* out) { return rt_.session().copy_logits(out).ok(); }
    void stamp_before_commit() {}
    bool commit(int, std::string&, int) { return true; }
    void on_token(int32_t) {}
    void count_round() {}
    void window_timing(int, int, bool, Clock::time_point, Clock::time_point, Clock::time_point, Clock::time_point) {}
    void mark_after_window() {}
    void mark_loop_end() {}
    void shed_low() { rt_.SessionRuntime::shed_if_low(); }
    void stamp_loop_end() {}
    void decode_end() {}

private:
    Flow prepare_image() override;
    Flow decode() override { return decode_loop(*this); }
    bool release_room() override { return true; }
    bool checkpoint_mount(const ConvCheckpoint& c) override {
        const basalt::model::Result r = rt_.session().restore_running(c.gdn.data(), (int64_t) c.ids.size());
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
    Flow prompt_end() override {
        rt_.session().end_prompt();
        return Flow::proceed;
    }

    void progress_line(const char* tag, int64_t at_pos) {
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
        std::printf("%s %lld %lld %.0f %.1f\n", tag, (long long) at_pos, (long long) n, ms,
                    ms > 0.0 ? 1000.0 * (double) (at_pos - resume) / ms : 0.0);
        std::fflush(stdout);
    }

    SessionRuntime& rt_;
    std::vector<Block> blocks_;
    std::vector<int32_t> history_;
    uint64_t draws_ = 0;
    std::string error_;
    size_t next_block_ = 0;
    double live_ms_ = -1e9;
};

Flow SessionRuntime::run_request() {
    SessionRequest request(*this);
    return request.execute();
}

// Each run of pad tokens takes the next picture's rows from the request's embeddings file.
Flow SessionRequest::prepare_image() {
    ppl = nullptr;
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
            r = rt_.session().prefill_chunk(pr);
            at = end;
        }
        if (!r.ok()) {
            basalt::log::error("basalt serve", "%s", r.message.c_str());
            std::printf("ERR %s\n", r.message.c_str());
            return Flow::error;
        }
        progress_line("PP", at);
        if (runtime_.halted()) { cancelled = true; return Flow::quit; }
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
                                                       : session_->encode_image(bytes.data(), bytes.size(), rows, rows_n);
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
    if (!fail.empty()) std::printf("ENC ERR %s\n", fail.c_str());
    else for (const std::string& r : replies) std::printf("%s\n", r.c_str());
    std::fflush(stdout);
}

}  // namespace

int run_session(CommonOptions& o, basalt::core::ModelFile& file, basalt::model::LoadedModel& model) {
    std::string err;
    SessionRuntime runtime(o, file, err, model);
    return runtime.execute();
}

}  // namespace basalt::program::serve
