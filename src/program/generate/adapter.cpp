// src/program/generate/adapter.cpp - models served by their adapter session: the CLI path and the --serve line protocol.
#include "run.hpp"
#include "session_model.hpp"
#include "../cli/tokens.hpp"
#include "../serve/session_engine.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/shipped_profiles.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/program/options.hpp"
#include "basalt/protocol.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

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

using Clock = std::chrono::steady_clock;
namespace cli = basalt::program::cli;

// The name of the engine's expert profile for a model (data/profiles/<key>.prof).
std::string profile_key(const basalt::core::Json& meta) {
    using T = basalt::core::Json::Type;
    const basalt::core::Json* kv = meta.get("gguf_kv");
    const basalt::core::Json* arch = kv ? kv->get("general.architecture") : nullptr;
    if (arch == nullptr || !arch->is_string() || arch->s.empty()) return {};
    const basalt::core::Json* nl = kv->get(arch->s + ".block_count");
    const basalt::core::Json* ne = kv->get(arch->s + ".expert_count");
    if (nl == nullptr || ne == nullptr || nl->type != T::Int || ne->type != T::Int || nl->neg || ne->neg) return {};
    return arch->s + "-" + std::to_string(nl->u64()) + "x" + std::to_string(ne->u64());
}

int fatal_start_error(const std::string& text) {
    basalt::log::error("basalt", "%s", text.c_str());
    return 1;
}

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
                    std::string& err, float* logits = nullptr, size_t reuse = 0, bool report = false) {
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
        basalt::protocol::emit("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                               ms > 0.0 ? 1000.0 * (double) (at - (int64_t) reuse) / ms : 0.0);
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
            pr.rows_after = total - 1 - (pos + pr.n_tokens);
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

// The CLI path of a session model: greedy decode of a pre-tokenized prompt, one `T <id>` line per token.
// Each --image's embedding block goes before the prompt token at its --image-at index.
int adapter_generate(basalt::model::LoadedModel& loaded, const std::vector<int64_t>& prompt, int64_t max_new,
                            const std::vector<int64_t>& stop_ids, const std::vector<std::string>& images,
                            const std::vector<int64_t>& image_at, const std::vector<int64_t>& oracle, bool teacher) {
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
    if (!images.empty()) sess->images_done();
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
    // A window holds the last pick and the drafts after it, from the drafter or from the expected continuation.
    const int cap = sess->max_window();
    std::vector<int32_t> window((size_t) cap), rows((size_t) cap), drafts((size_t) cap);
    std::vector<float> conf((size_t) cap);
    std::vector<float> diag_rows;
    int64_t made = 0, windows = 0, offered = 0, accepted = 0, differ = 0;
    double ms_draft = 0, ms_verify = 0;
    bool stop = false;
    const auto emit = [&](int32_t tok) {
        std::printf("T %d\n", tok);
        ++made;
        for (int64_t s : stop_ids) stop = stop || s == tok;
    };
    emit(next);
    std::fflush(stdout);
    if (teacher) next = (int32_t) oracle[0];
    while (!stop && made < max_new) {
        const auto w0 = Clock::now();
        int n_draft = 0;
        if (!oracle.empty()) {
            while (n_draft < cap - 1 && made + n_draft < (int64_t) oracle.size()) {
                drafts[(size_t) n_draft] = (int32_t) oracle[(size_t) (made + n_draft)];
                ++n_draft;
            }
        } else if (cap > 1) {
            basalt::model::Result dres = sess->draft_begin(next, pos);
            if (dres.ok()) dres = sess->draft_end(drafts.data(), conf.data(), cap - 1, n_draft);
            if (!dres.ok() && dres.status != basalt::model::Status::NotImplemented) {
                basalt::log::error("basalt generate", "%s", dres.message.c_str());
                return 1;
            }
        }
        const int T = 1 + n_draft;
        window[0] = next;
        std::copy(drafts.begin(), drafts.begin() + n_draft, window.begin() + 1);
        const auto w1 = Clock::now();
        basalt::model::DecodeRequest dr;
        dr.tokens = window.data();
        dr.n_tokens = T;
        dr.pos = pos;
        (T > 1 ? dr.row_tokens : dr.next_token) = rows.data();
        if (g_diag_logits) {
            diag_rows.resize((size_t) cap * (size_t) g_n_vocab);
            dr.logits = diag_rows.data();
        }
        const basalt::model::Result r = sess->decode_window(dr);
        if (!r.ok()) {
            basalt::log::error("basalt generate", "%s", r.message.c_str());
            return 1;
        }
        const auto w2 = Clock::now();
        ms_draft += std::chrono::duration<double, std::milli>(w1 - w0).count();
        ms_verify += std::chrono::duration<double, std::milli>(w2 - w1).count();
        int a = 0;
        while (a < T - 1 && window[(size_t) a + 1] == rows[(size_t) a]) ++a;
        ++windows;
        offered += T - 1;
        accepted += a;
        if (g_diag_logits) std::fwrite(diag_rows.data(), sizeof(float), (size_t) (teacher ? T : a + 1) * (size_t) g_n_vocab, g_diag_logits);
        if (teacher) {
            // the expected text goes on whatever the picks: every row runs, and each pick that differs is counted
            for (int i = 0; i < T && !stop && made < max_new; ++i) {
                const int64_t at = made;
                if (at < (int64_t) oracle.size() && oracle[(size_t) at] != rows[(size_t) i]) ++differ;
                emit(rows[(size_t) i]);
            }
            std::fflush(stdout);
            pos += T;
            if (made >= (int64_t) oracle.size()) break;
            next = (int32_t) oracle[(size_t) made - 1];
            continue;
        }
        for (int i = 0; i <= a && !stop && made < max_new; ++i) emit(rows[(size_t) i]);
        std::fflush(stdout);
        pos += a + 1;
        next = rows[(size_t) a];
    }
    const auto t2 = Clock::now();
    basalt::log::info("basalt generate", "prompt %.0f ms, %lld tokens in %.0f ms",
                 std::chrono::duration<double, std::milli>(t1 - t0).count(), (long long) made,
                 std::chrono::duration<double, std::milli>(t2 - t1).count());
    if (cap > 1)
        basalt::log::info("basalt generate", "windows %lld of %.2f tokens: %lld of %lld drafts accepted, %lld picks differ from the oracle; draft %.2f ms, verify %.2f ms per window",
                          (long long) windows, windows ? (double) (accepted + windows) / (double) windows : 0.0, (long long) accepted,
                          (long long) offered, (long long) differ, windows ? ms_draft / (double) windows : 0.0, windows ? ms_verify / (double) windows : 0.0);
    return 0;
}

int run_session_model(basalt::program::CommonOptions& o, const basalt::model::ModelAdapter& model_adapter) {
    const basalt::model::ModelAdapter* adapter = &model_adapter;
    basalt::core::ModelFile model_file;
    {
        std::string merr;
        if (!model_file.open(o.model, merr) || !model_file.load_meta(merr) || !model_file.register_gguf_view(merr))
            return fatal_start_error("--model: " + merr);
    }
    o.native_shards = {o.model};
    g_diag_serial_prompt = o.diag_serial_prompt;
    if (!o.diag_logits.empty()) g_diag_logits = std::fopen(o.diag_logits.c_str(), "wb");
    try {
        const basalt::GgufFile model_gguf(o.native_shards.front());
        std::string err = adapter->validate(model_gguf);
        if (!err.empty()) return fatal_start_error(err);
        const basalt::model::ModelDesc desc = adapter->describe(model_gguf);
        g_n_vocab = desc.n_vocab;
        std::unique_ptr<basalt::model::LoadedModel> loaded;
        basalt::model::LoadRequest lreq;
        lreq.paths = o.native_shards;
        lreq.diag_dump = o.diag_dump;
        lreq.diag_tier_ram = o.diag_tier_ram;
        lreq.diag_tier_vram = o.diag_tier_vram;
        lreq.peer_devices = o.peer_devices;
        lreq.max_context = o.max_context;
        lreq.sessions = std::max(o.batch, 1);
        lreq.vision = o.vision;
        lreq.expert_profile = o.expert_profile;
        const std::string key = profile_key(model_file.meta());
        {   // what the engine ships for this pack must be there; --expert-profile stands in for the main one
            std::vector<std::string> need;
            for (const std::string& name : adapter->shipped_profiles(key)) {
                if (!(name == key + ".prof" && !o.expert_profile.empty())) need.push_back(name);
            }
            const std::string miss = basalt::core::missing_shipped_profile(basalt::platform::install_root() + "/data/profiles", need);
            if (!miss.empty()) return fatal_start_error(miss);
        }
        if (lreq.expert_profile.empty()) {   // the engine's own for this model, when it ships one
            const std::string prof = basalt::platform::install_root() + "/data/profiles/" + key + ".prof";
            uint64_t bytes = 0;
            if (!key.empty() && basalt::platform::file_size(prof, bytes) && bytes > 0) lreq.expert_profile = prof;
        }
        lreq.expert_cache = o.expert_cache;
        lreq.resident_budget = o.resident_budget;
        lreq.pool_workers = o.pool_workers;
        lreq.pool_affinity = o.pool_affinity;
        lreq.pool_affinity_given = o.pool_affinity_given;
        lreq.peer_reserve_mib = o.peer_reserve_mib;
        lreq.prompt_skip = !o.no_prompt_skip;
        lreq.vram_reserve_mib = o.vram_reserve_given ? o.vram_reserve_mib : 0;
        lreq.spec = o.spec;
        lreq.draft = o.spec >= 2 && o.spec_oracle.empty();
        lreq.drafter_device = o.drafter_device;
        lreq.profile_dir = basalt::platform::install_root() + "/data/profiles";
        lreq.draft_experts = o.draft_experts;
        lreq.diag_tier_frozen = o.diag_tier_frozen;
        lreq.kv = o.kv;
        const basalt::model::Result r = adapter->load(lreq, loaded);
        if (!r.ok() || !loaded) return fatal_start_error(r.message);
        std::vector<int64_t> stop;
        if (o.stop_eos) stop = o.eos_ids;
        if (loaded->context() != 0) o.max_context = loaded->context();
        if (o.serve) return basalt::program::serve::run_session(o, model_file, *adapter, *loaded);
        std::vector<int64_t> oracle;
        if (const auto code = cli::load_spec_oracle(o.spec_oracle, oracle)) return *code;
        if (o.diag_teacher && oracle.empty()) return fatal_start_error("--diag-teacher needs --spec-oracle");
        return adapter_generate(*loaded, o.tokens, o.max_new, stop, o.images, o.image_at, oracle, o.diag_teacher);
    } catch (const std::exception& e) {
        return fatal_start_error(e.what());
    }
}

}  // namespace basalt::program::generate
