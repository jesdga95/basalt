// src/models/qwen4exp/serve/qfn_commands.cpp - the protocol commands that reach Flash-Next's tiers.
#include "models/qwen4exp/adapter.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "../../../program/tokenizer/unicode_flags.hpp"
#include "../../../program/cli/tokens.hpp"

namespace basalt::models::qwen4exp::serve {

bool QfnRuntime::windows_in_flight() { return piped && pipe_inflight(); }
bool QfnRuntime::next_batch_window() { return piped ? pump(true) : batch_step(); }
// a request reads its prompt through every stage: the groups in flight finish first
bool QfnRuntime::drain_windows() { return !piped || pipe_drain(); }

void QfnRuntime::script_command(const std::string& line) {
    char* e1 = nullptr;
    const long b = std::strtol(line.c_str() + 8, &e1, 10);
    char* e2 = nullptr;
    const unsigned long long seed = std::strtoull(e1, &e2, 10);
    std::vector<int64_t> sids;
    std::string se;
    if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
        std::printf("ERR BSCRIPT: bad slot or list\n");
        std::fflush(stdout);
        return;
    }
    bscript[(size_t) b].first.assign(sids.begin(), sids.end());
    bscript[(size_t) b].second = seed;
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
        cudaMemGetInfo(&free_b, &total_b);
        // "Experts in VRAM" is every stage's cache, not one card's.  The caches hold disjoint pairs, so they add.
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        int64_t slots_all = slots_primary, mib_all = mib_primary;
        for (const auto& st : stages) {
            slots_all += (int64_t) st->cache.slots();
            mib_all += (int64_t) (st->cache.bytes() >> 20);
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
                    "conversation_cache_min_free_mib=%lld conversation_cache_min_tokens=%d tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" BASALT_VERSION "\n",
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
                    quant_label.c_str(), gpu_ids.c_str());
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

int run(Context& context) {
    QfnRuntime runtime(context);
    runtime.pre_ = {"qwen35", basalt::models::qwen4exp::tokenizer::pretok_qwen35};
    return runtime.execute();
}

}  // namespace basalt::models::qwen4exp::serve
