// src/program/serve/qfn_commands.cpp - the protocol commands that reach Flash-Next's tiers.
#include "request.hpp"
#include "../cli/tokens.hpp"

namespace basalt::program::serve {

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
        penv.src = srcp;
        penv.peer = peer.valid() ? &peer : nullptr;
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

Flow QfnRuntime::run_request() {
    QfnRequest request(*this);
    return request.execute();
}

}  // namespace basalt::program::serve
