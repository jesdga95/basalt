// src/program/serve/commands.cpp - scheduling and protocol command dispatch.
#include "state.hpp"
#include "../cli/tokens.hpp"

namespace basalt::program::serve {

namespace {
// the busy clock runs while a request or a batch window does
struct BusyClock {
    double& total;
    bool& on;
    Clock::time_point t0;
    BusyClock(double& total_, bool& on_, Clock::time_point& from) : total(total_), on(on_), t0(Clock::now()) {
        from = t0;
        on = true;
    }
    ~BusyClock() {
        total += std::chrono::duration<double>(Clock::now() - t0).count();
        on = false;
    }
};
}  // namespace

uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::vector<ImgKey> Runtime::imgs_below(const std::vector<ImgKey>& all, int64_t L) {
    std::vector<ImgKey> v;
    for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
    return v;
}

bool Runtime::batch_on() { for (const BSlot& b : bs) if (b.active) return true; return false; }

Flow Runtime::dispatch() {
    if (batch_on() || windows_in_flight()) {
        if (!try_next_line(line)) {
            const BusyClock busy(busy_s, busy_on, busy_from);
            if (!next_batch_window()) return Flow::error;
            return Flow::next;
        }
        if (line.rfind("BSTOP ", 0) != 0 && !drain_windows()) return Flow::error;
    } else {
        kv_ceiling_note();   // between requests too: the free VRAM also moves without the KV
        // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
        // lose goes to disk, then the idle conversations whose time is up leave VRAM, one at a time while no line
        // waits
        bool got = false, parking = true;
        std::string pe;
        while (!got && (disk.enabled() || parking)) {
            const int r = next_line_within(line, 1000);
            if (r < 0) return Flow::quit;
            got = r == 1;
            while (!got && !(got = try_next_line(line)) && disk.enabled() && spill_to_disk(false, 1) > 0) {}
            if (got) break;
            parking = parking && park_idle_engine(pe);
            if (!pe.empty()) { std::printf("ERR %s\n", pe.c_str()); return Flow::error; }
            if (!parking) break;
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
        script_command(line);
        return Flow::next;
    }
    if (line.rfind("BYIELD", 0) == 0) return Flow::next;   // for a prompt read that has ended meanwhile
    if (line.rfind("ENC ", 0) == 0 || line.rfind("ENCV ", 0) == 0) {   // the in-engine encoder, between requests and windows
        enc_command(line);
        return Flow::next;
    }
    if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
        std::string perr;
        if (batch_on()) perr = "PROBE: not while batch slots are decoding";
        else probe_command(line, perr);
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
            sleep_command(down, verr);
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
        if (!slot0_solo() || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
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
    between_requests(line);
    if (line == "QUIT") return Flow::quit;
    // the watchdog watches a request from here until this iteration ends, whichever way it ends
    struct BusyScope {
        BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
        ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
    } busy_scope;
    const BusyClock busy(busy_s, busy_on, busy_from);
    return run_request();
}

}  // namespace basalt::program::serve
