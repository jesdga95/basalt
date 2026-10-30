// src/program/serve/protocol.cpp - protocol operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

// ---- what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard
};
MemSample mem_sample() {
    const basalt::platform::ProcessMemory p = basalt::platform::process_memory();
    MemSample m;
    m.faults = p.faults;
    m.rss_mib = p.rss_mib;
    m.avail_mib = p.avail_mib;
    m.commit_mib = p.commit_mib;
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in
std::string stage_text() {
    const basalt::core::Progress& p = basalt::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

// the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): many threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    const basalt::platform::ThreadWaits w = basalt::platform::threads_in_disk_wait();
    if (w.threads == 0) return;
    std::string list;
    for (const auto& [name, n] : w.where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", w.waiting, w.threads, list.c_str(),
                 w.waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (--ple-io ram), or a drive too slow for the reads asked of it" : "");
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    basalt::core::Progress& p = basalt::core::progress();
    std::fprintf(f, "basalt serve: error: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", BASALT_VERSION, stage_text().c_str(),
                 (long long) ((basalt::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = basalt::core::diag_pool_fn().load()) fn(f);
        if (auto fn = basalt::core::diag_verify_fn().load()) fn(f);
        disk_wait_threads(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
        );
        std::fflush(f);
    }
}

bool Runtime::next_line(std::string& out) {
    std::unique_lock<std::mutex> lk(in_mu);
    in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
    if (in_lines.empty()) return false;
    out = std::move(in_lines.front());
    in_lines.pop_front();
    return true;
}

// the same, waiting at most `ms`: 1 a line, 0 none came, -1 the input ended
int Runtime::next_line_within(std::string& out, int ms) {
    std::unique_lock<std::mutex> lk(in_mu);
    if (!in_cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return !in_lines.empty() || in_eof; }))
        return 0;
    if (in_lines.empty()) return -1;
    out = std::move(in_lines.front());
    in_lines.pop_front();
    return 1;
}

bool Runtime::halted() {
    if (stop_req.load() || stdin_lost.load()) return true;
    if (basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Critical) return false;
    ram_stop.store(true);
    return true;
}

bool Runtime::try_next_line(std::string& out) {
    std::lock_guard<std::mutex> lk(in_mu);
    if (in_lines.empty()) return false;
    out = std::move(in_lines.front());
    in_lines.pop_front();
    return true;
}

int Runtime::init_protocol() {

    text_tokenizer_note = "this pack has no qwen35 tokenizer tables";
    {
        std::vector<uint8_t> vj, mj, tj, cj;
        std::string terr;
        if (basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/merges.txt"), mj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, terr) &&
            basalt::core::read_range(model_file.range("tokenizer/tokenizer.json"), cj, terr) &&
            std::string_view((const char*) cj.data(), cj.size()).find("\"pre\": \"qwen35\"") != std::string_view::npos) {
            if (text_tokenizer.load(std::string_view((const char*) vj.data(), vj.size()),
                                    std::string_view((const char*) mj.data(), mj.size()),
                                    std::string_view((const char*) tj.data(), tj.size()), terr))
                text_tokenizer_note.clear();
            else
                text_tokenizer_note = terr;
        }
    }

    gent_seq = 0;
    stop_req.store(false);
    stdin_lost.store(false);   // set once at EOF: the peer is gone, so no request keeps running

    in_eof = false;
    // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
    // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
    std::thread([&] {
        // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
        // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
        // std::exit) would hang in exit(), and the server would wait for it forever
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
                const std::ptrdiff_t n = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        // exactly n raw bytes: what the line reader already holds, then the pipe
        auto read_exact = [&](size_t n, std::string& out) -> bool {
            while (buf.size() < n) {
                const std::ptrdiff_t got = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                if (got < 0 && errno == EINTR) continue;
                if (got <= 0) return false;
                buf.append(chunk, (size_t) got);
            }
            out.assign(buf, 0, n);
            buf.erase(0, n);
            return true;
        };
        // GENT/SGENT/BGENT <...> <max_new> [k=v ...] <nbytes> [plain=a:b,...] + nbytes of UTF-8: tokenize here and hand
        // the main loop the matching GEN/SGEN/BGEN line, with "@<seq>" where the ids would be.  False: the input ends
        // inside the payload.
        auto take_text = [&](const std::string& head_line, std::string& line_out) -> bool {
            std::vector<std::string> tk;
            {
                std::istringstream is(head_line);
                for (std::string w; is >> w;) tk.push_back(w);
            }
            const size_t lead = tk[0] == "GENT" ? 2 : 3;   // the command, [the slot,] max_new
            std::string head = tk[0].substr(0, tk[0].size() - 1);
            long long nbytes = -1;
            std::string plain_text;
            for (size_t i = 1; i < tk.size(); ++i) {
                if (i < lead) head += " " + tk[i];
                else if (tk[i].rfind("plain=", 0) == 0) plain_text = tk[i].substr(6);
                else if (tk[i].find('=') != std::string::npos) head += " " + tk[i];
                else if (nbytes < 0) nbytes = std::atoll(tk[i].c_str());
            }
            if (tk.size() < lead || nbytes < 0 || nbytes > (1LL << 31)) {
                line_out = "ERR_GENT bad header: GENT <max_new> [k=v ...] <nbytes> [plain=a:b,...]";
                return true;
            }
            std::string payload;
            if (!read_exact((size_t) nbytes, payload)) { line_out = "ERR_GENT the prompt text was cut short"; return false; }
            if (!text_tokenizer_note.empty()) { line_out = "ERR_GENT " + text_tokenizer_note; return true; }
            std::vector<basalt::program::Tokenizer::Span> spans;
            for (size_t p = 0; p < plain_text.size();) {
                char* e1 = nullptr;
                const unsigned long long a = std::strtoull(plain_text.c_str() + p, &e1, 10);
                if (*e1 != ':') break;
                char* e2 = nullptr;
                const unsigned long long b = std::strtoull(e1 + 1, &e2, 10);
                spans.push_back({(size_t) a, (size_t) b});
                p = (size_t) (e2 - plain_text.c_str());
                if (p < plain_text.size() && plain_text[p] == ',') ++p;
            }
            std::vector<int32_t> tids;
            std::string terr;
            if (!text_tokenizer.encode(payload, true, spans, tids, terr)) { line_out = "ERR_GENT tokenizer: " + terr; return true; }
            std::lock_guard<std::mutex> lk(gent_mu);
            const uint64_t seq = ++gent_seq;
            gent_ids[seq] = std::move(tids);
            while (gent_ids.size() > 8) gent_ids.erase(gent_ids.begin());   // a request that never claimed its ids
            line_out = head + " @" + std::to_string(seq);
            return true;
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            if (l.rfind("GENT ", 0) == 0 || l.rfind("SGENT ", 0) == 0 || l.rfind("BGENT ", 0) == 0) {
                std::string rewritten;
                const bool framed = take_text(l, rewritten);
                l = rewritten;
                if (!framed) {   // the input ended inside the payload: report it
                    std::lock_guard<std::mutex> lk(in_mu);
                    in_lines.push_back(l);
                    in_cv.notify_one();
                    break;
                }
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_lines.push_back(l);
            in_cv.notify_one();
        }
        stdin_lost.store(true);
        stop_req.store(true);
        std::lock_guard<std::mutex> lk(in_mu);
        in_eof = true;
        in_cv.notify_one();
    }).detach();

    // a request also stops when the RAM guard judges the RAM critical; `ram_stop` keeps that it did, so the request
    // ends on an error instead of a cancel
    ram_stop.store(false);

    sp.should_stop = [&] { return halted(); };
    {
        // what is left once everything is allocated: a GPU filled to the brim may not fail but page, and a page-in
        // while the verify graph spins on a host flag stalls the request for good
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t free_mib = (int64_t) (free_b >> 20);
        if (free_mib >= vram_floor_mib()) {
            basalt::log::info("basalt serve", "%lld MiB of VRAM free with everything loaded", (long long) free_mib);
        } else {
            basalt::log::warn("basalt", "only %lld MiB of VRAM free, requests may stall; close other "
                                 "programs that use the GPU", (long long) free_mib);
        }
    }
    return 0;
}

int Runtime::announce() {
    // what the server's Monitor tab shows (the server skips unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // "Experts in VRAM" is every tier, not one card's.  The tiers are disjoint by construction
        // (remote_experts.cpp skips any pair a stage already claimed), so they add.
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        int64_t slots_all = slots_primary, mib_all = mib_primary;
        for (const auto& st : stages) {
            slots_all += (int64_t) st->cache.slots();
            mib_all += (int64_t) (st->cache.bytes() >> 20);
        }
        for (int r = 0; r < 3; ++r)
            if (o.expert_cache_remote[(size_t) r] > 0) {
                slots_all += remote_experts[(size_t) r].resident();
                mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
            }
        // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
        // bus ids of the cards this engine computes on, primary first: what a benchmark row records
        std::string quant_label, gpu_ids;
        {
            const basalt::core::Json* nm = model_file.meta().get("name");
            std::string nmv = nm != nullptr && nm->is_string() ? nm->s : std::string();
            quant_label = nmv.substr(nmv.rfind(' ') == std::string::npos ? 0 : nmv.rfind(' ') + 1);
            for (const char* pre : {"NVIDIA-", "UD-"})
                if (quant_label.rfind(pre, 0) == 0) quant_label.erase(0, std::strlen(pre));
            if (quant_label.empty()) quant_label = "unknown";
            for (int d : basalt::program::engine_gpu_order(o.peer_device, o.peer_devices,
                                                           multi_gpu ? split_devs : std::vector<int>())) {
                char bus[32] = {};
                if (cudaDeviceGetPCIBusId(bus, sizeof bus, d) == cudaSuccess)
                    gpu_ids += (gpu_ids.empty() ? "" : ",") + std::string(bus);
            }
        }
        std::printf("INFO logit_bias=%lld context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                    "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                    "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                    "conversation_cache_mib=%lld conversation_cache_slots=%d "
                    "conversation_cache_min_free_mib=%lld tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" BASALT_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context, o.kv.c_str(),
                    (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                     ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                    (long long) slots_all, (long long) mib_all,
                    (long long) slots_primary, (long long) mib_primary,
                    o.spec, o.mtp_max_t,
                    o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                    (long long) ((o.mmap_experts ? src.resident_bytes() : basalt::kernels::cpu::expert_layout().total) >> 20),
                    pool.workers(), o.pcie_frac,
                    (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                    (long long) o.conversation_cache_min_free_mib, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                    o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                   " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                   " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                   " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                    quant_label.c_str(), gpu_ids.c_str());
    }
    if (text_tokenizer_note.empty()) std::printf("INFO text_prompt=1\n");
    // a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on a flag nobody
    // will raise - end the engine with where it was, so the server starts it again instead of the GPU spinning
    // forever.  Before it does, it reports what every part was doing (stall_report), so one occurrence says where
    // the wait is.
    {
        const int limit = 60;   // one step (a prompt layer, a verify window) takes seconds
        if (limit > 0)
            std::thread([limit] {
                basalt::core::Progress& p = basalt::core::progress();
                uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                auto since = std::chrono::steady_clock::now();
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    const auto now = std::chrono::steady_clock::now();
                    const uint64_t b = p.beats.load();
                    if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                    if (now - since < std::chrono::seconds(limit)) continue;
                    basalt::log::error("basalt serve", "no progress for %d s during a request (%s) - stopping "
                                         "the engine so the server starts it again",
                                 limit, stage_text().c_str());
                    stall_report(basalt::log::stream(), p.ticks.load() - ticks_at);
                    basalt::core::release_gpu_waits(basalt::log::stream());   // no spin kernel outlives the process
                    std::abort();
                }
            }).detach();
    }
    return 0;
}

}  // namespace basalt::program::serve
