// src/models/qwen4exp/serve/qfn_commands.cpp - the protocol commands that reach Flash-Next's tiers.
#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/qwen4exp/adapter.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "basalt/model/batch_check.hpp"
#include "basalt/vision/ffmpeg_api.hpp"
#include "../../../program/tokenizer/unicode_flags.hpp"
#include "../../../program/cli/tokens.hpp"

namespace basalt::models::qwen4exp::serve {

int64_t QfnRuntime::gpu_slots() const {
    int64_t n = xcache.slots();
    for (const auto& st : stages) n += st->cache.slots();
    if (peer.valid()) n += peer.cache().slots();
    for (const auto& xpe : xpeers)
        if (xpe->valid()) n += xpe->cache().slots();
    return n;
}

// the main card (the caller's device) with its cache and the link the start probed, each layer split stage's card, each tier card
// with its link when the prompt split probed it
std::vector<QfnRuntime::CardFacts> QfnRuntime::card_facts() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    const auto link = [&](int d) {
        for (const auto& ps : pstages) if (ps->dev == d) return ps->link_gbps;
        return 0.0;
    };
    // a tier card holds experts only; the main card's KV: the solo session's and every slot session's cells as backed now
    std::vector<const basalt::core::SessionState*> on_main{&ss};
    if (!bslot_ss.empty())
        for (const auto& sl : bslot_ss[0])
            if (sl && std::find(on_main.begin(), on_main.end(), sl.get()) == on_main.end()) on_main.push_back(sl.get());
    int64_t kv = 0;
    for (const basalt::core::SessionState* s : on_main)
        kv += (int64_t) basalt::core::session_kv_bytes_for(*s, basalt::core::session_kv_cells(*s));
    std::vector<CardFacts> out{{dev, (int64_t) xcache.slots(), main_link_gbps, (uint64_t) xcache.mapped_bytes(), kv}};
    for (const auto& st : stages)
        out.push_back({st->dev, (int64_t) st->cache.slots(), link(st->dev), (uint64_t) st->cache.mapped_bytes(), -1});
    if (peer.valid())
        out.push_back({peer.device(), (int64_t) peer.cache().slots(), link(peer.device()), (uint64_t) peer.cache().mapped_bytes(), 0});
    for (const auto& xpe : xpeers)
        if (xpe->valid())
            out.push_back({xpe->device(), (int64_t) xpe->cache().slots(), link(xpe->device()), (uint64_t) xpe->cache().mapped_bytes(), 0});
    return out;
}

basalt::program::serve::LookupCounts QfnRuntime::lookup_counts() {
    const basalt::core::ExpertDispatch& d = drive.d;   // DONE's counts (done_counts)
    return {d.cache_hits, d.cache_hits + d.cache_admitted + d.cache_refused, d.offload_entries, src.file_reads()};
}

// a sampled expert's home as the start's tier report (report_expert_tiers) counts it: a card's cache, the peer tier, RAM, the file
bool QfnRuntime::expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) {
    basalt::core::ExpertMapTally& m = drive.d.map;
    if (!m.is_open()) m.open((int) g.n_layers, (int) g.n_expert);
    total = m.total();
    const bool ram_ok = srcp == &src && src.complement_ready(), arena = srcp != &src;
    homes.clear();
    for (const int32_t p : m.pairs()) {
        const int64_t l = p / g.n_expert, e = p % g.n_expert;
        bool on_gpu = (size_t) p < host_res.size() && host_res[(size_t) p] >= 0;
        for (const auto& st : stages) on_gpu = on_gpu || st->cache.slot_of(l, e) >= 0;
        bool on_peer = peer.valid() && peer.has(l, e);
        for (const auto& xpe : xpeers) on_peer = on_peer || (xpe->valid() && xpe->has(l, e));
        homes.push_back((uint8_t) (on_gpu ? basalt::core::ExpertHome::Vram : on_peer ? basalt::core::ExpertHome::Peer
                                   : arena || (ram_ok && src.has_resident(l, e)) ? basalt::core::ExpertHome::Ram
                                                                                 : basalt::core::ExpertHome::File));
    }
    m.take(routed);
    return true;
}


void QfnRuntime::probe_command(const std::string& line, std::string& perr) {
if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            std::istringstream kvs(line.substr(10));
            for (std::string kv; kvs >> kv;) {
                const size_t eq = kv.find('=');
                const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                else perr = "PROBE: bad setting " + kv;
            }
            if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
        }
    } else if (line == "PROBE adapt" || line.rfind("PROBE adapt ", 0) == 0) {
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            (void) probe_adapt(line, perr);
        }
    } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates, PROBE set or PROBE adapt";
    else {
        if (ba_job) { ba_thr.join(); ba_job = false; }
        basalt::program::ProbeEnv penv;
        penv.pool = &pool;
        penv.ram_blob = [this](int64_t l, int64_t e) { return srcp->resident_blob(l, e); };
        if (const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout(); lay.native) penv.fmt = lay.fmt;
        penv.peer_device = peer.valid() ? peer.device() : -1;
        penv.peer_path = basalt::models::qwen4exp::make_probe_peer_path();
        penv.n_layers = g.n_layers;
        penv.n_expert = g.n_expert;
        basalt::program::probe_rates(penv, stdout, perr);
    }
}

void QfnRuntime::sleep_command(bool down, std::string& verr) {
    // A peer card's tier goes too.  SLEEP then drops the RAM copy's registration with the driver (a process
    // checkpoint would map the copy into every context again on restore); WAKE registers it again while
    // both caches refill from the ranges already back.  A tier or a registration that cannot come back
    // fails WAKE: the server then starts the engine again.
    std::string perr, reply, uerr;
    const auto tp = Clock::now();
    const bool repin = !down && src.complement_unpinned();
    std::function<bool(const void*, uint64_t)> ready;
    if (repin) {
        src.repin_start(4);
        ready = [this](const void* p, uint64_t n) { return src.repin_wait(p, n); };
    }
    if (down && kv_pool && kv_pool->size() > 0) {   // the pooled conversations' VRAM goes back too
        basalt::log::info("basalt serve", "SLEEP: %zu pooled conversations dropped", kv_pool->size());
        (void) kv_pool->clear();
    }
    const bool peer_ok = !peer.valid() || (down ? peer.sleep(perr) : peer.wake(perr, ready));
    vram_command(down ? "VRAM " + std::to_string(1ll << 30) : std::string("VRAM"), verr, true, &reply);
    if (repin) {
        if (!src.repin_finish(uerr) && verr.empty()) verr = "WAKE: " + uerr;
        basalt::log::info("basalt serve", "WAKE: the RAM copy registered again beside the caches' refill "
                          "in %.0f ms", std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
    }
    if (down && verr.empty() && src.complement_ready()) {
        const auto tu = Clock::now();
        if (src.unpin_complement(uerr))
            basalt::log::info("basalt serve", "SLEEP: the RAM copy's registration dropped in %.0f ms",
                              std::chrono::duration<double, std::milli>(Clock::now() - tu).count());
        else
            basalt::log::info("basalt serve", "SLEEP: the RAM copy stays registered (%s)", uerr.c_str());
    }
    if (!down && peer.valid() && peer_ok && !peer.wake_wait(perr)) {
        if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
    } else if (!down && !peer_ok) {
        if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
    } else if (!peer_ok) {
        basalt::log::warn("basalt serve", "SLEEP: the peer tier stays as it is (%s)", perr.c_str());
    } else if (peer.valid()) {
        basalt::log::info("basalt serve", "%s: the peer tier %s; %.0f ms with the main cache",
                          line.c_str(), down ? "gave its VRAM back" : "is back",
                          std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
    }
    if (verr.empty()) std::fputs(reply.c_str(), stdout);
}

void QfnRuntime::between_requests(const std::string& line) {
    // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
    if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
}

void QfnRuntime::after_input_start() { sp.should_stop = [&] { return halted(); }; }

void QfnRuntime::print_info() {
    // what the server's Monitor tab shows (the server skips unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        basalt::core::gpu_mem_info(&free_b, &total_b);
        // "Experts in VRAM" is every stage's cache, not one card's.  The caches hold disjoint pairs, so they add.
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        const int64_t slots_all = gpu_slots();
        int64_t mib_all = mib_primary;
        for (const auto& st : stages) mib_all += (int64_t) (st->cache.bytes() >> 20);
        // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
        // bus ids of the cards this engine computes on, primary first: what a benchmark row records
        std::string quant_label, gpu_ids;
        {
            const basalt::core::Json* nm = model_file.meta().get("name");
            quant_label = basalt::program::pack_quant_label(nm != nullptr && nm->is_string() ? nm->s : std::string());
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
                    "conversation_cache_min_free_mib=%lld conversation_cache_min_tokens=%d tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s "
                    "ram_experts_mib=%lld engine=" BASALT_VERSION "\n",
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
                    (long long) o.conversation_cache_min_free_mib, o.conversation_cache_min_tokens, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                    o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                   " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                   " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                   " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                    quant_label.c_str(), gpu_ids.c_str(),
                    // the experts held in RAM: the pinned copy, or the arena of all of them; none when the file's pages are read
                    (long long) ((srcp == &src ? (src.complement_ready() ? src.resident_bytes() : 0)
                                               : basalt::kernels::cpu::expert_layout().total) >> 20));
        // whether the in-engine encoder reads video (FFmpeg found at run time), for the server's start line
        if (o.vision_in_engine) std::printf("%s\n", basalt::vision::video_line().c_str());
    }
}

Flow QfnRuntime::run_request() {
    QfnRequest request(*this);
    return request.execute();
}

int QfnRuntime::init_before_input() {
    if (const int code = init_prompt()) return code;
    if (const int code = init_verify()) return code;
    if (const int code = init_conversations()) return code;
    return init_tiers();
}

int QfnRuntime::init_after_input() {
    if (const int code = init_memory()) return code;
    if (const int code = capture_windows()) return code;
    return init_batch();
}

void QfnRuntime::at_exit() {
    for (int st = 0; st < std::max(n_stages, 1); ++st)
        for (const std::string& r : stage_ver(st).route_report()) basalt::log::info("basalt serve", "%s", r.c_str());
    save_profile("exit");   // QUIT, or the server closed stdin
}

// --diag-batch-check: the shared batch check over the slots' sessions (model/batch_check.hpp) in place of serving
int QfnRuntime::batch_check() {
    if (const int code = init_before_input()) return code;
    if (const int code = init_after_input()) return code;
    if (slot_ss.size() < 2) {
        basalt::log::error("basalt batch", "the batch check needs --batch 2 or more slots");
        return 1;
    }
    std::vector<basalt::model::ModelSession*> ss;
    for (const auto& s : slot_ss) ss.push_back(s.get());
    const std::vector<int32_t> prompt(o.tokens.begin(), o.tokens.end());
    basalt::model::BatchCheck bc;
    basalt::model::Result r = basalt::model::check_batch(ss.data(), (int) ss.size(), prompt, (int) std::max<int64_t>(o.max_new, 4), n_vocab, bc);
    if (std::string ce; r.ok() && !commit_settle(ce)) r = {basalt::model::Status::Error, ce};
    std::printf("%s", bc.report.c_str());
    if (!r.ok()) {
        basalt::log::error("basalt batch", "%s", r.message.c_str());
        return 1;
    }
    std::printf("batch check: %d windows compared, %d differ: %s\n", bc.compared, bc.differ, bc.differ ? "FAILED" : "PASSED");
    return bc.differ ? 1 : 0;
}

int run(Context& context) {
    QfnRuntime runtime(context);
    runtime.pre_ = {"qwen35", basalt::program::tokenizer::pretok_qwen35};
    return context.o.diag_batch_check ? runtime.batch_check() : runtime.execute();
}

}  // namespace basalt::models::qwen4exp::serve
