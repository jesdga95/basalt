// src/program/generate/adapter.cpp - models served by their adapter session: the CLI path and the --serve line protocol.
#include "context.hpp"
#include "run.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/program/prefix_reuse.hpp"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace basalt::program::generate {

namespace {

bool g_diag_serial_prompt = false;   // --diag-serial-prompt
std::FILE* g_diag_logits = nullptr;  // --diag-logits FILE: each decoded row's logits (the serial prompt and the last prompt token)
int64_t g_n_vocab = 0;

/// A picture's embedding block, placed before the prompt token at `at`.
struct AdapterImage {
    int64_t at = 0;
    std::vector<float> rows;
    int n_rows = 0;
};

struct AdapterTimes {
    double prompt_ms = 0;
    int64_t prompt_tokens = 0;   ///< positions in the prompt: text tokens and picture rows
    int64_t reused = 0;          ///< leading positions the session already held
};

/// Runs the prompt (text and picture blocks, in order) through the session and decodes its last token, which must be
/// text. `next` is the token after the prompt; `pos` the next free position. The first `reuse` tokens (no pictures)
/// are already in the session. `report`: print `PP <position> <positions in the prompt> <ms> <tok/s>` as the prompt is
/// read, the reused positions counted as read and left out of the speed.
bool adapter_prompt(basalt::model::ModelSession& sess, const std::vector<int32_t>& toks,
                    const std::vector<AdapterImage>& images, int32_t& next, int64_t& pos, AdapterTimes& tm,
                    std::string& err, float* logits = nullptr, size_t reuse = 0, bool report = false,
                    bool* checkpointed = nullptr) {
    const auto t0 = Clock::now();
    pos = (int64_t) reuse;
    int64_t total = (int64_t) toks.size();
    for (const AdapterImage& im : images) total += im.n_rows;
    // live progress inside a prefill chunk, as PL lines at most twice a second (from the layer ticks' thread)
    auto live_at = [&](int64_t at) {
        static std::atomic<double> last_ms{-1e9};
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const double prev = last_ms.load();
        if (ms - prev < 500.0 && ms >= prev) return;
        last_ms.store(ms);
        std::printf("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                    ms > 0.0 ? 1000.0 * (double) (at - (int64_t) reuse) / ms : 0.0);
        std::fflush(stdout);
    };
    auto report_at = [&](int64_t at) {
        if (!report) return;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                    ms > 0.0 ? 1000.0 * (double) (at - (int64_t) reuse) / ms : 0.0);
        std::fflush(stdout);
    };
    size_t done = reuse;   // prompt tokens consumed
    const size_t n_pre = toks.size() - 1;
    basalt::model::Result r;
    for (size_t i = 0; r.ok() && i <= images.size(); ++i) {
        const size_t end = i < images.size() ? (size_t) images[i].at : n_pre;
        if (end > done) {
            basalt::model::PrefillRequest pr;
            pr.tokens = toks.data() + done;
            pr.n_tokens = (int) (end - done);
            pr.pos = pos;
            if (g_diag_serial_prompt) {   // one token per step, through the decode path
                for (int t = 0; r.ok() && t < pr.n_tokens; ++t) {
                    int32_t row_next = -1;
                    basalt::model::DecodeRequest one;
                    one.tokens = pr.tokens + t;
                    one.n_tokens = 1;
                    one.pos = pos + t;
                    one.next_token = &row_next;
                    std::vector<float> row_logits;
                    if (g_diag_logits) {
                        row_logits.resize((size_t) g_n_vocab);
                        one.logits = row_logits.data();
                    }
                    r = sess.decode_window(one);
                    if (g_diag_logits) std::fwrite(row_logits.data(), sizeof(float), row_logits.size(), g_diag_logits);
                    std::printf("R %lld %d\n", (long long) (pos + t), row_next);
                }
                pos += pr.n_tokens;
                done = end;
                continue;
            }
            const int64_t base = pos;
            if (report) pr.progress = [&, base](int n) { report_at(base + n); };
            if (report) pr.live = [&, base](int n) { live_at(base + n); };
            const auto tp = Clock::now();
            r = sess.prefill_chunk(pr);
            basalt::log::info("basalt generate", "batched prefill %d tokens in %.0f ms", pr.n_tokens,
                         std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
            pos += pr.n_tokens;
            done = end;
            if (r.ok()) report_at(pos);
        }
        if (r.ok() && i < images.size()) {
            const auto tp = Clock::now();
            r = sess.prefill_embeddings(images[i].rows.data(), images[i].n_rows, pos);
            basalt::log::info("basalt generate", "image prefill %d rows in %.0f ms", images[i].n_rows,
                         std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
            pos += images[i].n_rows;
            if (r.ok()) report_at(pos);
        }
    }
    // the state before the last prompt token: a next turn renders the same text up to here, whatever this reply thinks
    if (r.ok() && checkpointed != nullptr && images.empty() && n_pre > 0) *checkpointed = sess.checkpoint(pos).ok();
    if (r.ok()) {
        basalt::model::DecodeRequest dr;
        dr.tokens = toks.data() + n_pre;
        dr.n_tokens = 1;
        dr.pos = pos;
        dr.next_token = &next;
        std::vector<float> diag_row;
        if (g_diag_logits && logits == nullptr) {
            diag_row.resize((size_t) g_n_vocab);
            logits = diag_row.data();
        }
        dr.logits = logits;
        r = sess.decode_window(dr);
        if (g_diag_logits) std::fwrite(logits, sizeof(float), (size_t) g_n_vocab, g_diag_logits);
        ++pos;
    }
    tm.prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    tm.prompt_tokens = pos;
    tm.reused = (int64_t) reuse;
    if (!r.ok()) err = r.message;
    return r.ok();
}

}  // namespace

// DeepSeek (adapter session) CLI path: greedy decode of a pre-tokenized prompt, one `T <id>` line per token.
// Each --image's embedding block goes before the prompt token at its --image-at index.
int adapter_generate(basalt::model::LoadedModel& loaded, const std::vector<int64_t>& prompt, int64_t max_new,
                            const std::vector<int64_t>& stop_ids, const std::vector<std::string>& images,
                            const std::vector<int64_t>& image_at) {
    std::unique_ptr<basalt::model::ModelSession> sess;
    const basalt::model::Result mr = loaded.make_session(sess);
    if (!mr.ok()) {
        basalt::log::error("basalt generate", "%s", mr.message.c_str());
        return 1;
    }
    if (images.size() != image_at.size()) {
        basalt::log::error("basalt generate", "every --image needs an --image-at");
        return 2;
    }
    std::vector<int32_t> toks(prompt.begin(), prompt.end());
    // the pictures, encoded before the prompt runs
    std::vector<AdapterImage> blocks(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        if (image_at[i] < 0 || image_at[i] >= (int64_t) toks.size() || (i > 0 && image_at[i] < image_at[i - 1])) {
            basalt::log::error("basalt generate", "--image-at must rise and stay inside the prompt");
            return 2;
        }
        blocks[i].at = image_at[i];
        std::ifstream f(images[i], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const auto te = Clock::now();
        const basalt::model::Result er = sess->encode_image(bytes.data(), bytes.size(), blocks[i].rows, blocks[i].n_rows);
        if (!er.ok()) {
            basalt::log::error("basalt generate", "%s: %s", images[i].c_str(), er.message.c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "image %s -> %d rows in %.0f ms", images[i].c_str(), blocks[i].n_rows,
                     std::chrono::duration<double, std::milli>(Clock::now() - te).count());
    }
    int32_t next = -1;
    int64_t pos = 0;
    AdapterTimes tm;
    std::string err;
    const auto t0 = Clock::now();
    if (!adapter_prompt(*sess, toks, blocks, next, pos, tm, err)) {
        basalt::log::error("basalt generate", "%s", err.c_str());
        return 1;
    }
    const auto t1 = Clock::now();
    basalt::model::DecodeRequest dr;
    dr.next_token = &next;
    basalt::model::Result r;
    int64_t made = 0;
    while (made < max_new) {
        std::printf("T %d\n", next);
        std::fflush(stdout);
        ++made;
        bool stop = false;
        for (int64_t s : stop_ids) stop = stop || s == next;
        if (stop || made >= max_new) break;
        int32_t cur = next;
        dr.tokens = &cur;
        dr.n_tokens = 1;
        dr.pos = pos++;
        r = sess->decode_window(dr);
        if (!r.ok()) {
            basalt::log::error("basalt generate", "%s", r.message.c_str());
            return 1;
        }
    }
    const auto t2 = Clock::now();
    basalt::log::info("basalt generate", "prompt %.0f ms, %lld tokens in %.0f ms",
                 std::chrono::duration<double, std::milli>(t1 - t0).count(), (long long) made,
                 std::chrono::duration<double, std::milli>(t2 - t1).count());
    return 0;
}

// DeepSeek --serve: the line protocol the Python server speaks (READY, ENC, GEN, GENI, STOP, QUIT; T and DONE
// replies). A prompt that continues the tokens the session holds reads only its new tail (`RESUME <n>`, and <n> in
// DONE's reuse field); any other prompt starts from an empty sequence. Sampling keys: temperature, top_p,
// top_k, min_p, seed; the rest are ignored.
int adapter_serve(basalt::model::LoadedModel& loaded, int64_t max_context, int64_t pad_id,
                         const std::vector<int64_t>& stop_ids) {
    std::unique_ptr<basalt::model::ModelSession> sess;
    const basalt::model::Result mr = loaded.make_session(sess);
    if (!mr.ok()) {
        basalt::log::error("basalt serve", "%s", mr.message.c_str());
        return 1;
    }
    // stdin on its own thread, so STOP is seen between tokens
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    // read(2) on the descriptor, not std::cin: exit() flushes every stdio stream and waits for stdin's lock, which
    // getline holds while it waits for input, so an engine ending on a kernel failure would hang in exit()
    std::thread reader([&] {
        std::string buf;
        char chunk[4096];
        auto push = [&](std::string l) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(std::move(l));
            cv.notify_all();
        };
        for (;;) {
            const std::ptrdiff_t n = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            buf.append(chunk, (size_t) n);
            for (size_t nl; (nl = buf.find('\n')) != std::string::npos; buf.erase(0, nl + 1)) push(buf.substr(0, nl));
        }
        if (!buf.empty()) push(buf);
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_all();
    });
    reader.detach();
    auto next_line = [&](std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = lines.front();
        lines.pop_front();
        return true;
    };
    auto stop_seen = [&] {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end(); ++it)
            if (*it == "STOP") { lines.erase(it); return true; }
        return false;
    };
    if (const basalt::model::Result pr = sess->prepare(); !pr.ok()) {
        basalt::log::error("basalt serve", "%s", pr.message.c_str());
        return 1;
    }
    std::printf("READY %lld stop\n", (long long) max_context);
    std::fflush(stdout);
    std::string line;
    std::vector<int32_t> held;   // the tokens whose state the session holds, position by position (no pictures)
    std::vector<int32_t> turn;   // the tokens of the session's checkpoint: the last prompt, without its final token
    while (next_line(line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "QUIT") break;
        if (cmd == "ENC") {
            std::vector<std::string> words;
            for (std::string w; in >> w;) words.push_back(w);
            std::vector<std::string> replies;
            std::string fail_msg;
            for (size_t i = 0; i + 1 < words.size() && fail_msg.empty(); i += 2) {
                std::ifstream f(words[i], std::ios::binary);
                const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                std::vector<float> rows;
                int n = 0;
                const auto te = Clock::now();
                const basalt::model::Result er = bytes.empty() ? basalt::model::Result{basalt::model::Status::Error, "cannot read the image " + words[i]}
                                                               : sess->encode_image(bytes.data(), bytes.size(), rows, n);
                if (!er.ok()) { fail_msg = er.message; break; }
                const int32_t nx = 0, ny = 0, width = (int32_t) (rows.size() / (size_t) n);
                const int32_t hdr[5] = {0x31455653, n, nx, ny, width};
                std::FILE* fo = std::fopen(words[i + 1].c_str(), "wb");
                const bool ok = fo && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                                std::fwrite(rows.data(), sizeof(float), rows.size(), fo) == rows.size();
                if (fo) std::fclose(fo);
                if (!ok) { fail_msg = "cannot write " + words[i + 1]; break; }
                replies.push_back("ENC OK " + std::to_string(n) + " " + std::to_string(nx) + " " + std::to_string(ny) + " " +
                                  std::to_string((int) std::chrono::duration<double, std::milli>(Clock::now() - te).count()));
            }
            if (!fail_msg.empty()) std::printf("ENC ERR %s\n", fail_msg.c_str());
            else for (const std::string& r : replies) std::printf("%s\n", r.c_str());
            std::fflush(stdout);
            continue;
        }
        if (cmd != "GEN" && cmd != "GENI") continue;   // other lines (VRAM, ...) do not apply here
        long long max_new = 0;
        in >> max_new;
        float temperature = 0.f, top_p = 1.f, min_p = 0.f;
        int top_k = 0;
        uint64_t seed = 0;
        std::string emb_path, ids_text, w;
        while (in >> w) {
            const size_t eq = w.find('=');
            if (cmd == "GENI" && emb_path.empty() && eq == std::string::npos && w.find(',') == std::string::npos) { emb_path = w; continue; }
            if (eq == std::string::npos) { ids_text = w; continue; }
            const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
            if (k == "temperature") temperature = (float) std::atof(v.c_str());
            else if (k == "top_p") top_p = (float) std::atof(v.c_str());
            else if (k == "top_k") top_k = std::atoi(v.c_str());
            else if (k == "min_p") min_p = (float) std::atof(v.c_str());
            else if (k == "seed") seed = (uint64_t) std::atoll(v.c_str());
        }
        std::vector<int64_t> all;
        std::string perr;
        if (!cli::parse_i64_list(ids_text.c_str(), all, perr)) { std::printf("ERR %s\n", perr.c_str()); std::fflush(stdout); continue; }
        // the SVE records: each run of pad tokens becomes one picture block before the next text token
        std::vector<AdapterImage> blocks;
        std::vector<int32_t> toks;
        std::string bad;
        if (!emb_path.empty()) {
            std::ifstream ef(emb_path, std::ios::binary);
            size_t k = 0;
            for (size_t i = 0; i < all.size(); ++i) {
                if (all[i] != pad_id) { toks.push_back((int32_t) all[i]); continue; }
                int32_t hdr[5];
                if (!ef.read((char*) hdr, sizeof hdr) || hdr[0] != 0x31455653) { bad = "bad embeddings file"; break; }
                size_t run = 0;
                while (i + run < all.size() && all[i + run] == pad_id) ++run;
                if ((size_t) hdr[1] != run) { bad = "the prompt and its images do not match"; break; }
                AdapterImage im;
                im.at = (int64_t) toks.size();
                im.n_rows = hdr[1];
                im.rows.resize((size_t) hdr[1] * (size_t) hdr[4]);
                if (!ef.read((char*) im.rows.data(), (std::streamsize) (im.rows.size() * sizeof(float)))) { bad = "short embeddings file"; break; }
                blocks.push_back(std::move(im));
                i += run - 1;
                ++k;
            }
        } else {
            toks.assign(all.begin(), all.end());
        }
        if (bad.empty() && (toks.empty() || (!blocks.empty() && blocks.back().at >= (int64_t) toks.size()))) bad = "the prompt must end with text";
        if (bad.empty() && (int64_t) toks.size() >= max_context) bad = "the prompt (" + std::to_string(toks.size()) + " tokens) does not fit the context of " + std::to_string(max_context) + " tokens";
        if (!bad.empty()) { std::printf("ERR %s\n", bad.c_str()); std::fflush(stdout); continue; }
        max_new = std::min<long long>(max_new, (long long) max_context - (long long) toks.size());
        int32_t next = -1;
        int64_t pos = 0;
        AdapterTimes tm;
        std::string err;
        const char* finish = "length";
        long long made = 0;
        std::mt19937_64 rng(seed ? seed : std::random_device{}());
        std::vector<float> logits;
        size_t reuse = blocks.empty() ? basalt::program::prefix_reuse::reusable(held, toks) : 0;
        if (reuse == 0 && blocks.empty() && basalt::program::prefix_reuse::reusable(turn, toks) > 0 &&
            sess->rewind_to_checkpoint().ok()) {
            reuse = turn.size();
            held = turn;
        }
        if (!held.empty() && reuse == 0) {
            size_t same = 0;
            while (same < held.size() && same < toks.size() && held[same] == toks[same]) ++same;
            basalt::log::info("basalt serve", "the prompt (%zu tokens) shares %zu of the %zu held tokens: read from the start",
                         toks.size(), same, held.size());
        }
        held.clear();   // refilled below once the state is known to match
        std::printf("RESUME %lld\n", (long long) reuse);
        std::fflush(stdout);
        auto t_dec = Clock::now();
        const bool sample = temperature > 0.f;
        if (sample) logits.resize((size_t) loaded.desc().n_vocab);
        bool checkpointed = false;
        turn.clear();
        if (!adapter_prompt(*sess, toks, blocks, next, pos, tm, err, sample ? logits.data() : nullptr, reuse, true, &checkpointed)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (blocks.empty()) held = toks;
        if (checkpointed) turn.assign(toks.begin(), toks.end() - 1);
        t_dec = Clock::now();
        basalt::model::DecodeRequest dr;
        int32_t cur = 0;
        // sampling over the logits of the last decode (temperature, then top-k, min-p and top-p)
        auto pick = [&]() -> int32_t {
            std::vector<std::pair<float, int32_t>> c(logits.size());
            for (size_t i = 0; i < logits.size(); ++i) c[i] = {logits[i], (int32_t) i};
            const size_t keep = top_k > 0 ? std::min<size_t>((size_t) top_k, c.size()) : std::min<size_t>(c.size(), 1024);
            std::partial_sort(c.begin(), c.begin() + (long) keep, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
            c.resize(keep);
            const float mx = c[0].first;
            double sum = 0;
            std::vector<double> pr(keep);
            for (size_t i = 0; i < keep; ++i) { pr[i] = std::exp((double) (c[i].first - mx) / temperature); sum += pr[i]; }
            double acc = 0;
            size_t n = 0;
            for (; n < keep; ++n) {
                if (pr[n] / sum < (double) min_p * (pr[0] / sum) && n > 0) break;
                acc += pr[n] / sum;
                if (acc >= (double) top_p) { ++n; break; }
            }
            double tot = 0;
            for (size_t i = 0; i < n; ++i) tot += pr[i];
            double u = std::uniform_real_distribution<double>(0.0, tot)(rng), a2 = 0;
            for (size_t i = 0; i < n; ++i) { a2 += pr[i]; if (u <= a2) return c[i].second; }
            return c[0].second;
        };
        if (sample) next = pick();
        bool failed = false;
        while (made < max_new) {
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++made;
            bool stop = false;
            for (int64_t s : stop_ids) stop = stop || s == next;
            if (stop) { finish = "stop"; break; }
            if (made >= max_new) break;
            if (stop_seen()) { finish = "stop"; break; }
            cur = next;
            dr.tokens = &cur;
            dr.n_tokens = 1;
            dr.pos = pos++;
            dr.next_token = &next;
            dr.logits = sample ? logits.data() : nullptr;
            const basalt::model::Result rr = sess->decode_window(dr);
            if (!rr.ok()) { std::printf("ERR %s\n", rr.message.c_str()); failed = true; break; }
            if (!held.empty()) held.push_back(cur);
            if (sample) next = pick();
        }
        if (failed) held.clear();
        if (!failed)
            std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld\n", made, (long long) tm.prompt_tokens, tm.prompt_ms,
                        std::chrono::duration<double, std::milli>(Clock::now() - t_dec).count(), finish,
                        (long long) tm.reused);
        std::fflush(stdout);
    }
    return 0;
}


std::optional<int> run_adapter_model(const Options& o, const basalt::core::ModelFile& model_file) {
    g_diag_serial_prompt = o.diag_serial_prompt;
    if (!o.diag_logits.empty()) g_diag_logits = std::fopen(o.diag_logits.c_str(), "wb");
    const basalt::core::Json* kv = model_file.meta().get("gguf_kv");
    const basalt::core::Json* arch = kv ? kv->get("general.architecture") : nullptr;
    const std::string name = arch && arch->is_string() ? arch->s : std::string();
    const basalt::model::ModelAdapter* adapter = basalt::model::find_adapter(name);
    if (adapter == nullptr) return fatal_start_error("--model: no model adapter for architecture '" + name + "'");
    try {
        const basalt::GgufFile model_gguf(o.native_shards.front());
        std::string err = adapter->validate(model_gguf);
        if (!err.empty()) return fatal_start_error(err);
        const basalt::model::ModelDesc desc = adapter->describe(model_gguf);
        if (desc.rope_from_model) {
            // the model's own YaRN extension is part of its weights' contract; a flag cannot change it
            const bool rope_flag = !o.rope_scaling.empty() || o.rope_scale != 0 || o.rope_freq_base != 0 ||
                                   o.rope_freq_scale != 0 || o.yarn_orig_ctx != 0 || o.yarn_ext_factor >= 0 ||
                                   o.yarn_attn_factor != 1.0 || o.yarn_beta_fast != 32.0 || o.yarn_beta_slow != 1.0;
            if (rope_flag)
                return fatal_start_error("the --rope-* and --yarn-* flags do not apply to " + desc.arch +
                                         ": its rotary scaling comes from the model file");
        }
        g_n_vocab = desc.n_vocab;
        std::unique_ptr<basalt::model::LoadedModel> loaded;
        basalt::model::LoadRequest lreq;
        lreq.paths = o.native_shards;
        lreq.diag_dump = o.diag_dump;
        lreq.diag_tier_ram = o.diag_tier_ram;
        lreq.diag_tier_vram = o.diag_tier_vram;
        lreq.max_context = o.max_context;
        lreq.max_context_set = o.max_context_set;
        const basalt::model::Result r = adapter->load(lreq, loaded);
        if (r.status == basalt::model::Status::UseEnginePath) return std::nullopt;
        if (!r.ok() || !loaded) return fatal_start_error(r.message);
        std::vector<int64_t> stop = {1};
        if (o.stop_eos) stop.insert(stop.end(), o.eos_ids.begin(), o.eos_ids.end());
        else stop.clear();
        if (o.serve)
            return adapter_serve(*loaded, loaded->context() ? loaded->context() : o.max_context, o.image_pad_id, stop);
        return adapter_generate(*loaded, o.tokens, o.max_new, stop, o.images, o.image_at);
    } catch (const std::exception& e) {
        return fatal_start_error(e.what());
    }
}

}  // namespace basalt::program::generate
