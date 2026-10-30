// src/program/serve/commands.cpp - scheduling and protocol command dispatch.
#include "request.hpp"

namespace basalt::program::serve {

Flow Runtime::dispatch() {
    if (batch_on() || (piped && pipe_inflight())) {
        if (!try_next_line(line)) {
            if (!(piped ? pump(true) : batch_step())) return Flow::error;
            return Flow::next;
        }
        // a request reads its prompt through every stage: the groups in flight finish first
        if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return Flow::error;
    } else {
        // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
        // lose goes to disk, one conversation at a time while no line waits
        bool got = false;
        if (disk.enabled()) {
            const int r = next_line_within(line, 1000);
            if (r < 0) return Flow::quit;
            got = r == 1;
            while (!got && !(got = try_next_line(line)) && spill_to_disk(false, 1) > 0) {}
        }
        if (!got && !next_line(line)) return Flow::quit;
    }
    // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
    // request's prompt and first token as a GEN 1, then continues it in that slot
    if (line.rfind("BSTOP ", 0) == 0) {
        const int b = std::atoi(line.c_str() + 6);
        if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
        return Flow::next;
    }
    if (line.rfind("BSCRIPT ", 0) == 0) {   // BSCRIPT <slot> <seed> <id,id,...>: before the slot's BGEN
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + 8, &e1, 10);
        char* e2 = nullptr;
        const unsigned long long seed = std::strtoull(e1, &e2, 10);
        std::vector<int64_t> sids;
        std::string se;
        if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
            std::printf("ERR BSCRIPT: bad slot or list\n");
            std::fflush(stdout);
            return Flow::next;
        }
        bscript[(size_t) b].first.assign(sids.begin(), sids.end());
        bscript[(size_t) b].second = seed;
        return Flow::next;
    }
    if (line.rfind("BYIELD", 0) == 0) return Flow::next;   // for a prompt read that has ended meanwhile
    if (line.rfind("ENC ", 0) == 0) {   // the in-engine image encoder, between requests and windows
        enc_command(line);
        return Flow::next;
    }
    if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
        std::string perr;
        if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
            if (batch_on()) perr = "PROBE: not while batch slots are decoding";
            else {
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
            if (batch_on()) perr = "PROBE: not while batch slots are decoding";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                (void) probe_adapt(line, perr);
            }
        } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates, PROBE set or PROBE adapt";
        else if (batch_on()) perr = "PROBE: not while batch slots are decoding";
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
        if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
        std::string verr;
        if (batch_on()) verr = "VRAM: not while batch slots are decoding";
        else vram_command(line, verr, false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    // SLEEP / WAKE, between requests: the expert cache gives its VRAM back down to its smallest size (every
    // expert keeps a RAM copy and the slot it held is remembered), so the process can be suspended with little
    // VRAM left to save; WAKE maps the cache again and puts each expert back in its slot.  Reply: a VRAM line.
    if (line == "SLEEP" || line == "WAKE") {
        std::string verr;
        const bool down = line == "SLEEP";
        if (batch_on()) {
            verr = line + ": not while batch slots are decoding";
        } else {
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
        // asleep, the driver holds the engine's GPU state in RAM until the wake: the RAM guard judges no reading
        // then (no shed, no admission wait from it), and judges a fresh one at the wake
        if (down && verr.empty()) basalt::platform::ram_guard_pause(true);
        if (!down) basalt::platform::ram_guard_pause(false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("ERR_GENT ", 0) == 0) {
        std::printf("ERR %s\n", line.c_str() + 9);
        std::fflush(stdout);
        return Flow::next;
    }
    basalt::platform::ram_guard_pause(false);   // a request without a WAKE first: the guard judges again
    admit_slot = -1;
    solo_slot = -1;
    if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
        const bool simg = line.rfind("SGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
        if (!slot0_is_solo || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
            std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
            std::fflush(stdout);
            return Flow::next;
        }
        solo_slot = (int) b;
        line = std::string(simg ? "GENI" : "GEN") + e1;
    }
    if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
        const bool bimg = line.rfind("BGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
        char* e2 = nullptr;
        const long long mn = std::strtoll(e1, &e2, 10);
        if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
            std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
            std::fflush(stdout);
            return Flow::next;
        }
        admit_slot = (int) b;
        admit_max_new = mn;
        line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
    }
    // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
    if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
    if (line == "QUIT") return Flow::quit;
    // the watchdog watches a request from here until this iteration ends, whichever way it ends
    struct BusyScope {
        BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
        ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
    } busy_scope;
    Request request(*this);
    return request.execute();
}

}  // namespace basalt::program::serve
