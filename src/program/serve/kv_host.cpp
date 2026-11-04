// src/program/serve/kv_host.cpp - the per-request host K/V: admission, pool growth and the way back to VRAM.
#include "state.hpp"
#include "basalt/program/auto_plan.hpp"

#include <cstring>

namespace basalt::program::serve {

void Runtime::kv_host_init() {
    // by default only beside the experts' RAM copy: with a file mapping the pin would take the file cache's RAM
    const bool kv_host_ram_copy = srcp == &src && src.complement_ready();
    // and only with a cache loan: a streamed prompt read stages the K/V through it
    const char* kv_host_why = !basalt::core::session_kv_can_stream(ss) ? "the session's KV cannot stream"
                            : pf_parts.empty() || pf_parts[0].first < 0 ? "the prompt path lends no cache slots"
                            : nullptr;
    kv_host_live = kv_host_on && kv_host_why == nullptr && (kv_host_forced || kv_host_ram_copy);
    if (kv_host_on && kv_host_forced && kv_host_why != nullptr)
        basalt::log::warn("basalt serve", "host K/V is off: %s; every request keeps its K/V in VRAM", kv_host_why);
    // a request that starts decoding below this depth (cells) decodes faster from the elastic tail than streamed
    kv_host_cross = (int64_t) basalt::core::exp_flag("kv_host_cross", 99328);
    // a test's stand-in for the RAM free now, MiB (forces the fallback); negative: MemAvailable
    kv_host_ram = (int64_t) basalt::core::exp_flag("kv_host_ram_mib", -1);
    if (kv_host_live)
        basalt::log::info("basalt serve", "host K/V: on, %lld-cell slot pool, streamed from %lld cells, %.2f GiB of RAM "
                          "kept free", (long long) kKvHostSlotCells, (long long) kv_host_cross,
                          (double) kv_host_kept() / 1073741824.0);
    // the output cells a pool is pinned for, and its slack (no more than one page of output); a request that writes
    // past its pool grows it between windows.  A test pins less to cross it.
    kv_host_out = std::max<int64_t>(0, (int64_t) basalt::core::exp_flag("kv_host_out", 16384));
    // A test's free-RAM reading for admission and mid-request growth; negative uses the ordinary reading.
    kv_host_grow_ram = (int64_t) basalt::core::exp_flag("kv_host_grow_ram_mib", -1);
    kv_host_late = false;
    // a session not streamed yet takes a pool only for a request that reads at most this many prompt tokens: a long
    // streamed read costs more than its decode wins back, and the conversation's next turn (a short read) streams
    kv_host_fresh = (int64_t) basalt::core::exp_flag("kv_host_fresh", 32768);
    // a session not streamed yet prefaults its first pool from this many cells below kv_host_cross (0: at the take)
    kv_host_lead = std::max<int64_t>(0, (int64_t) basalt::core::exp_flag("kv_host_lead", 8192));
}

// a pin leaves the guard's floor free and what serving can still take: its budget less the part it holds now
uint64_t Runtime::kv_host_kept() {
    return basalt::program::kv_host_keep(guard_floor, serve_budget.total(), serve_held());
}

int64_t Runtime::kv_host_pin_cells(int64_t at, int64_t new_tokens) {
    return basalt::program::kv_host_cells(at, std::min<int64_t>(std::max<int64_t>(new_tokens, 0), kv_host_out),
                                          std::min<int64_t>(4096, kv_host_out), 16384, ss.max_cells);
}

// the RAM free now, counting what the prefaulted pool holds (its take reuses it)
uint64_t Runtime::kv_ram_free() {
    const uint64_t ahead = kv_ahead.prefaulted_bytes();   // first: a reading after it is not higher
    return basalt::platform::available_memory() + ahead;
}

uint64_t Runtime::kv_ram_now() {
    return kv_host_ram >= 0 ? (uint64_t) kv_host_ram << 20 : kv_ram_free();
}

// a session nearing the crossover prefaults its first pool when the take's keep admits it now; one farther from it,
// or already streamed, holds none
void Runtime::kv_host_ahead(int64_t need) {
    if (kv_pool.valid() || kv_host_lead <= 0 || need < kv_host_cross - kv_host_lead) {
        kv_ahead_drop("shallow");
        return;
    }
    if (kv_ahead.prefaulting()) return;
    basalt::program::KvHostInputs in;
    in.cells = kv_host_pin_cells(kv_host_cross, kv_host_out);
    in.crossover_cells = kv_host_cross;
    in.bytes_per_cell = (basalt::core::KvHostPool::bytes_for(ss, g, in.cells) + (uint64_t) in.cells - 1) /
                        (uint64_t) std::max<int64_t>(in.cells, 1);
    in.ram_available = kv_ram_now();
    in.ram_kept = kv_host_kept();
    in.ram_low = basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok;
    if (!basalt::program::choose_kv_host(in).host) return;
    std::string pe;
    if (!kv_ahead.prefault(ss, g, in.cells, pe)) return;
    basalt::log::info("basalt serve", "host K/V: prefaulting a %lld-cell pool at %lld cells, %.2f GiB, %.2f GiB free, "
                      "%.2f GiB kept free", (long long) in.cells, (long long) need,
                      (double) ((uint64_t) in.cells * in.bytes_per_cell) / 1073741824.0,
                      (double) in.ram_available / 1073741824.0, (double) in.ram_kept / 1073741824.0);
}

void Runtime::kv_ahead_drop(const char* why) {
    if (!kv_ahead.prefaulting()) return;
    const uint64_t bytes = kv_ahead.prefaulted_bytes();
    kv_ahead.release();
    basalt::log::info("basalt serve", "host K/V: the prefaulted pool released (%s), %.2f GiB", why,
                      (double) bytes / 1073741824.0);
}

// A larger host extent, registered in place or copied into a replacement: 1 done, 0 refused (`pe`, the old pool
// stays bound), -1 failed (`e`). The device is drained.
int Runtime::kv_host_regrow(int64_t cells, int64_t live_c, std::string& pe, std::string& e) {
    if (kv_pool.grows_in_place()) {
        if (!kv_pool.grow(cells, pe)) return 0;
        basalt::core::session_kv_host_cells(ss, kv_pool.cells());
        return 1;
    }
    basalt::core::KvHostPool np;
    if (!np.reserve(ss, g, cells, pe) || !np.copy_from(kv_pool, live_c, pe) ||
        !basalt::core::session_kv_set_mode(ss, g, 1, &np, live_c, nullptr, pe))
        return 0;
    if (!ver.drop_kv_graphs(1, e)) return -1;   // they point at the old pool
    kv_pool = std::move(np);
    return ver.warm_session(&ss, e) ? 1 : -1;
}

// back to the elastic VRAM tail: the K/V arrays backed for `at` cells (out of the expert cache), then the first
// `live_c` cells come in from the pool, which is freed: 1 done, 0 the tail cannot take them (`pe`, still
// streamed), -1 failed (`e`).  The device is drained.
int Runtime::kv_host_to_vram(int64_t at, int64_t live_c, std::string& pe, std::string& e) {
    basalt::core::session_kv_cap(ss, false);
    if (!kv_fit(ss, at + kKvGrowCells, pe, true, at + kKvWindowCells)) {
        basalt::core::session_kv_cap(ss, true);
        if (!kv_short) { e = pe; return -1; }
        return 0;
    }
    if (!basalt::core::session_kv_set_mode(ss, g, 0, nullptr, live_c, nullptr, e) || !ver.sync_kv_mode(e) ||
        !ver.drop_kv_graphs(1, e))
        return -1;
    kv_pool.release();
    relend();   // the prompt path's layout follows the mode (no staging pool in VRAM)
    return 1;
}

// 1 admitted (in either mode), 0 refused (`e`, the engine stays up), -1 failed
int Runtime::kv_host_admit(int64_t need, int64_t new_tokens, int64_t live_cells, int64_t read, std::string& e) {
    kv_host_late = false;
    if (!kv_host_live) return 1;
    const int64_t live_c = std::min<int64_t>(std::max<int64_t>(live_cells, 0), ss.max_cells);
    const int64_t at = std::max(need, live_c);
    // shallow (the depth it starts decoding at): the VRAM tail (a streamed session moves there after the park)
    if (need < kv_host_cross) {
        kv_host_late = kv_pool.valid();
        kv_host_ahead(need);
        return 1;
    }
    const int64_t cells = kv_host_pin_cells(at, new_tokens);
    // the K/V past the slot pool goes back to the expert cache at a take
    const int64_t unmap_to = std::min<int64_t>(std::max(kv_floor, at + kKvGrowCells), ss.max_cells);
    basalt::program::KvHostInputs in;
    in.cells = cells;
    in.crossover_cells = kv_host_cross;
    in.bytes_per_cell = (basalt::core::KvHostPool::bytes_for(ss, g, cells) + (uint64_t) cells - 1) /
                        (uint64_t) std::max<int64_t>(cells, 1);
    in.grow_in_place = kv_pool.grows_in_place() && cells > kv_pool.cells();
    if (in.grow_in_place) in.growth_bytes = kv_pool.growth_bytes(cells);
    const bool growing = kv_pool.valid() && cells > kv_pool.cells();
    in.ram_available = growing && kv_host_grow_ram >= 0 ? (uint64_t) kv_host_grow_ram << 20 : kv_ram_now();
    in.ram_kept = kv_host_kept();
    in.ram_low = basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok;
    in.read_cells = std::max<int64_t>(read, 0);
    in.fresh_cells = std::max<int64_t>(kv_host_fresh, 0);
    const auto t0 = Clock::now();
    const basalt::program::KvHostPlan plan = basalt::program::plan_kv_host(in, kv_pool.valid() ? kv_pool.cells() : 0);
    uint64_t pin_free = in.ram_available;
    auto said = [&](const char* what, const char* why = nullptr) {
        basalt::log::info("basalt serve", "host K/V: %s (%s), %lld cells, %.2f GiB pinned, %.2f GiB free, %.2f GiB "
                          "kept free, the expert cache %lld slots, in %.1f ms, %.2f GiB free before pin", what,
                          why != nullptr ? why : plan.why, (long long) cells,
                          (double) kv_pool.bytes() / 1073741824.0, (double) in.ram_available / 1073741824.0,
                          (double) in.ram_kept / 1073741824.0, (long long) xcache.slots(),
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                          (double) pin_free / 1073741824.0);
    };
    std::string pe;
    switch (plan.act) {
    case basalt::program::KvHostAct::none:
        // a deep request whose pin does not fit: said, since it decodes from the VRAM tail
        if (std::strcmp(plan.why, "fresh") == 0)
            basalt::log::info("basalt serve", "host K/V: kept in VRAM (a %lld-token prompt read), %lld cells; a "
                              "later turn that reads at most %lld streams", (long long) in.read_cells,
                              (long long) cells, (long long) in.fresh_cells);
        if (std::strcmp(plan.why, "no RAM") == 0) {
            kv_ahead_drop("no RAM");
            basalt::log::info("basalt serve", "host K/V: kept in VRAM (no RAM), %lld cells need %.2f GiB pinned, "
                              "%.2f GiB free, %.2f GiB kept free", (long long) cells,
                              (double) ((uint64_t) cells * in.bytes_per_cell) / 1073741824.0,
                              (double) in.ram_available / 1073741824.0, (double) in.ram_kept / 1073741824.0);
        }
        return 1;
    case basalt::program::KvHostAct::keep:
        return 1;
    case basalt::program::KvHostAct::take: {
        // each phase's time, ms, said with the take: drain, slot pool, pin, move, unmap, cache, verifier, loan, graphs
        double ph[9] = {};
        auto mark = t0;
        auto lap = [&](int i) {
            const auto now = Clock::now();
            ph[i] = std::chrono::duration<double, std::milli>(now - mark).count();
            mark = now;
        };
        if (!kv_drain(e)) return -1;
        lap(0);
        // the slot pool's pages backed before the switch caps the K/V arrays at them
        if (!kv_fit(ss, kKvHostSlotCells, pe)) {
            if (!kv_short) { e = pe; return -1; }
            basalt::log::warn("basalt serve", "host K/V: %s; the request keeps its K/V in VRAM", pe.c_str());
            return 1;
        }
        lap(1);
        // the RAM read again just before the pin: the drain and the slot pool may have taken some
        pin_free = std::min(in.ram_available, kv_ram_free());
        basalt::program::KvHostInputs pin = in;
        pin.ram_available = pin_free;
        pin.ram_kept = kv_host_kept();
        pin.ram_low = basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok;
        if (!basalt::program::choose_kv_host(pin).host) {
            kv_ahead_drop("no RAM");
            said("kept in VRAM", "no RAM");
            return 1;
        }
        const uint64_t cache_homes0 = src.homes_bytes();
        // a prefaulted pool is registered in place: the pin finds its pages in RAM
        const bool ahead = kv_ahead.prefaulting();
        double ahead_ms = 0, ahead_wait = 0;
        uint64_t ahead_bytes = 0;
        if (ahead) {
            const auto w0 = Clock::now();
            ahead_ms = kv_ahead.prefault_wait();
            ahead_wait = std::chrono::duration<double, std::milli>(Clock::now() - w0).count();
            ahead_bytes = kv_ahead.prefaulted_bytes();
            kv_pool = std::move(kv_ahead);
        }
        bool moved = kv_pool.reserve(ss, g, plan.cells, pe);
        lap(2);
        if (ahead)
            basalt::log::info("basalt serve", "host K/V: the take pins a prefaulted pool, %.2f GiB written in %.1f ms, "
                              "%.1f ms of it waited", (double) ahead_bytes / 1073741824.0, ahead_ms, ahead_wait);
        moved = moved && basalt::core::session_kv_set_mode(ss, g, 1, &kv_pool, live_c, nullptr, pe);
        if (!moved) {
            kv_pool.release();
            basalt::log::warn("basalt serve", "host K/V: %s; the request keeps its K/V in VRAM", pe.c_str());
            return 1;
        }
        lap(3);
        if (!basalt::core::session_kv_unmap(ss, unmap_to, e)) return -1;
        lap(4);
        if (!cache_regrow(e)) return -1;
        lap(5);
        const uint64_t homes1 = src.homes_bytes();
        if (!ver.sync_kv_mode(e)) return -1;
        lap(6);
        relend();   // the prompt path's layout follows the mode: its loan now carries the staging pool
        lap(7);
        if (!ver.warm_session(&ss, e)) return -1;
        lap(8);
        said("streamed");
        basalt::log::info("basalt serve", "host K/V: the take in ms: drain %.1f, slot pool %.1f, pin %.1f, move %.1f, "
                          "unmap %.1f, cache %.1f (%.0f MiB of RAM homes freed), verifier %.1f, loan %.1f, graphs %.1f",
                          ph[0], ph[1], ph[2], ph[3], ph[4], ph[5],
                          (double) (cache_homes0 > homes1 ? cache_homes0 - homes1 : 0) / 1048576.0,
                          ph[6], ph[7], ph[8]);
        return 1;
    }
    case basalt::program::KvHostAct::grow: {
        if (!kv_drain(e)) return -1;
        const int r = kv_host_regrow(plan.cells, live_c, pe, e);
        if (r < 0) return -1;
        if (r > 0) {
            said("a larger pool");
            return 1;
        }
        basalt::log::warn("basalt serve", "host K/V: a larger pool: %s; back to VRAM", pe.c_str());
    }
        [[fallthrough]];
    case basalt::program::KvHostAct::drop: {
        if (!kv_drain(e)) return -1;
        const int r = kv_host_to_vram(at, live_c, pe, e);
        if (r < 0) return -1;
        if (r == 0) {
            if (basalt::program::kv_host_pool_serves(kv_pool.cells(), at, kKvWindowCells)) {   // it stays streamed
                basalt::log::warn("basalt serve", "host K/V: %s; the request streams from the pool it has", pe.c_str());
                return 1;
            }
            e = pe;
            return 0;
        }
        said("back to VRAM");
        return 1;
    }
    }
    return 1;
}

// the shallow request's switch (kv_host_late), after the park: back to the VRAM tail with the first `keep` cells,
// the arrays backed for `need`.  1 done or still streamed, -1 failed (`e`).
int Runtime::kv_host_back(int64_t need, int64_t keep, std::string& e) {
    const auto t0 = Clock::now();
    if (!kv_drain(e)) return -1;
    std::string pe;
    const int r = kv_host_to_vram(need, keep, pe, e);
    if (r < 0) return -1;
    if (r == 0)
        basalt::log::warn("basalt serve", "host K/V: %s; the request streams from the pool it has", pe.c_str());
    else
        basalt::log::info("basalt serve", "host K/V: back to VRAM (shallow), %lld cells kept, the expert cache %lld "
                          "slots, in %.1f ms", (long long) keep, (long long) xcache.slots(),
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return 1;
}

// a streamed request about to write past its pool (between windows, at position `p` with `left` tokens to go,
// the next window reaching `need` cells, its first `live_c` cells written): a larger pool when the RAM above the
// guard's floor takes it, else back to the VRAM tail.  1 done, 0 neither (`e`: the request stops there), -1 failed.
int Runtime::kv_host_more(int64_t p, int64_t need, int64_t left, int64_t live_c, std::string& e) {
    const int64_t had = kv_pool.cells();
    // a pool of the whole context holds every cell a window writes (none goes past it), as the tail does there
    if (had >= ss.max_cells) return 1;
    const auto t0 = Clock::now();
    if (!kv_drain(e)) return -1;
    const int64_t cells = kv_host_pin_cells(need, left);
    const uint64_t bytes = kv_pool.grows_in_place() ? kv_pool.growth_bytes(cells)
                                                 : basalt::core::KvHostPool::bytes_for(ss, g, cells);
    const uint64_t avail = kv_host_grow_ram >= 0 ? (uint64_t) kv_host_grow_ram << 20 : kv_ram_now();
    // the keep admission uses (it is never below the RAM homes' floor)
    const uint64_t floor = std::max<uint64_t>(guard_floor + (256ull << 20), kv_host_kept());
    auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
    std::string pe = "no RAM";
    if (cells > had && avail > floor && bytes <= avail - floor) {
        const int r = kv_host_regrow(cells, live_c, pe, e);
        if (r < 0) return -1;
        if (r > 0) {
            basalt::log::info("basalt serve", "host K/V: a larger pool mid-request, %lld -> %lld cells, %.2f GiB "
                              "pinned, %.2f GiB free, in %.1f ms", (long long) had, (long long) cells,
                              (double) kv_pool.bytes() / 1073741824.0, (double) avail / 1073741824.0, ms());
            return 1;
        }
    }
    const std::string why = pe;
    const int r = kv_host_to_vram(p, live_c, pe, e);
    if (r < 0) return -1;
    if (r == 0) { e = "host K/V: the pool cannot grow (" + why + ") and " + pe; return 0; }
    basalt::log::info("basalt serve", "host K/V: back to VRAM mid-request (%s), %lld cells, %.2f GiB free, the expert "
                      "cache %lld slots, in %.1f ms", why.c_str(), (long long) live_c,
                      (double) avail / 1073741824.0, (long long) xcache.slots(), ms());
    return 1;
}

}  // namespace basalt::program::serve
