// src/program/serve/memory.cpp - memory operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

uint64_t serve_growth_bytes(const Options& o, const basalt::core::ModelGeometry& g) {
    if (!o.serve) return 0;
    uint64_t bytes = basalt::platform::kRequestHostBytes;
    basalt::core::ConversationStateSizes z;
    std::string error;
    if (o.prompt_cache > 0 && basalt::core::conversation_state_sizes(g, z, error)) {
        const uint64_t one = z.gdn + z.ple + (uint64_t) g.n_qsa_layers() * (z.tail + z.dead + z.block_pos);
        bytes += (uint64_t) (o.prompt_cache + 1) * (uint64_t) std::max(o.batch, 1) * one;
        bytes += (uint64_t) o.conversation_cache_mib << 20;
    }
    if (!o.vision_mmproj.empty()) {
        // the canvas (8-bit RGB), its normalized planar floats and the patches
        const basalt::vision::Geometry vg;
        const uint64_t pixels = (uint64_t) vg.max_tokens * (uint64_t) (vg.patch * vg.merge) * (uint64_t) (vg.patch * vg.merge);
        bytes += pixels * 3 * (1 + 2 * sizeof(float));
    }
    return bytes;
}

void Runtime::shed_if_low() {
    if (basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok) return;
    const uint64_t drop = basalt::platform::ram_guard_drops();
    if (drop == shed_drop) return;
    shed_drop = drop;
    const size_t parked = conversations->size();
    const size_t freed = conversations->clear();
    size_t to_disk = 0;
    for (auto& sc : conversations->take_spilled())
        if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
            disk.put(std::move(sc), kDiskQueueCap))
            ++to_disk;
    if (freed > 0)
        basalt::log::info("basalt serve", "RAM guard: %zu parked conversations (%zu MiB) leave RAM, %zu of them "
                          "to disk", parked, freed >> 20, to_disk);
    if (const size_t pooled = basalt::core::CheckpointBytes::allocator_type::release_pooled(); pooled > 0)
        basalt::log::info("basalt serve", "RAM guard: freed %zu MiB of checkpoint blocks kept for reuse",
                          pooled >> 20);
    const uint64_t before = basalt::platform::available_memory();
    basalt::platform::release_free_heap();
    const uint64_t after = basalt::platform::available_memory();
    basalt::log::info("basalt serve", "RAM guard: free heap returned to the OS, %lld MiB free now (%+lld MiB)",
                      (long long) (after >> 20), ((long long) after - (long long) before) / (1ll << 20));
}

// the prompt path's loan with `live` slots: its chunk as large as fits, and the first slot it covers
int32_t Runtime::loan_plan(int64_t live, int64_t& chunk) {
    PfPart& p = pf_parts[0];
    auto fits = [&](int64_t c) {
        const int64_t k = part_slots(p, c);
        return k > 0 && k + 128 <= live && !(o.prefill_auto && k * 100 > auto_lend_pct * live);
    };
    // auto: the largest chunk on the start-up's 256-token grid, so a few slots lent away cost a few tokens
    // of chunk, not half of it
    int64_t c = chunk_full;
    if (o.prefill_auto && !fits(c)) {
        c = engine_local::biggest_chunk(chunk_full, fits);
        if (c == 0) c = 256;
    }
    while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
    chunk = c;
    return (int32_t) std::max<int64_t>(0, live - part_slots(p, c));
}

void Runtime::relend() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
    if (pf_parts.empty() || pf_parts[0].first < 0) return;
    PfPart& p = pf_parts[0];
    int64_t c = 0;
    p.first = loan_plan(xcache.slots(), c);
    o.prefill_chunk = c;
    p.first_now = -1;   // laid out again at the next loan
}

// The resident RAM mode holds only the experts no slot holds, so an expert in a slot about to go has no RAM
// copy: it is copied from its slot into a RAM home first and read from there like the compact copy's.  Refused
// (nothing copied) when the RAM above the floor cannot take them.  `live`: the slots that stay.
bool Runtime::home_given_up(int64_t live, std::string& e) {
    if (!src.complement_ready()) return true;
    // the slots the loan will overwrite keep a RAM copy too, so a prompt never costs a file read
    int64_t from = live;
    if (!pf_parts.empty() && pf_parts[0].first >= 0) {
        int64_t c = 0;
        from = std::min<int64_t>(live, loan_plan(live, c));
    }
    std::vector<std::pair<int32_t, int32_t>> give;
    std::vector<int32_t> slots;
    for (size_t i = 0; i < host_res.size(); ++i) {
        const int64_t layer = (int64_t) i / g.n_expert, ex = (int64_t) i % g.n_expert;
        if (host_res[i] >= from && !src.has_resident(layer, ex)) {
            give.emplace_back((int32_t) layer, (int32_t) ex);
            slots.push_back(host_res[i]);
        }
    }
    if (give.empty()) return true;
    std::vector<uint8_t*> dst;
    if (!src.homes_take(give, e, dst)) return false;
    for (size_t k = 0; k < give.size(); ++k) {
        const size_t bytes = (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(give[k].first);
        if (cudaMemcpyAsync(dst[k], xcache.device_slot(slots[k]), bytes, cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
            e = std::string("copying an expert to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
            (void) cudaDeviceSynchronize();   // queued copies stop writing before their regions are freed
            (void) cudaGetLastError();
            src.homes_abort();
            return false;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        e = std::string("copying the experts to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
        (void) cudaDeviceSynchronize();
        (void) cudaGetLastError();
        src.homes_abort();
        return false;
    }
    src.homes_commit();
    basalt::log::info("basalt serve", "VRAM: %zu experts moved from the cache to RAM (%.0f MiB held)", give.size(),
                 (double) src.homes_bytes() / 1048576.0);
    return true;
}

// `for_sleep` (SLEEP and WAKE): the cache goes to its smallest size and comes back to the same placement, with no
// request between the two, so neither --vram-elastic (any segmented cache does) nor a peer card's tier stands in
// the way
bool Runtime::vram_command(const std::string& cmd, std::string& e, bool for_sleep, std::string* reply) {
    // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
    char* end = nullptr;
    const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
    const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
    if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
    if (reserve < vram_floor_mib() && !for_sleep) {
        e = "VRAM: a reserve below " + std::to_string(vram_floor_mib()) + " MiB is too small for what is allocated "
            "after loading; the smallest safe value is " + std::to_string(vram_floor_mib());
        return false;
    }
    if ((!o.vram_elastic && !for_sleep) || !xcache.segmented()) {
        e = for_sleep ? "SLEEP needs a segmented expert cache (an elastic KV cache or --vram-elastic)"
                      : "VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)";
        return false;
    }
    if (peer.valid() && !for_sleep) { e = "VRAM: not with a second GPU as expert tier"; return false; }
    for (const PfPart& p : pf_parts)
        if (!p.lent.empty()) { e = "VRAM: a prompt's loan is still out"; return false; }
    if (!ver.wait_commit(e)) return false;
    apply_pending(true);                     // the adaptive tier's swaps in flight land first
    if (!xq_drain()) { e = "VRAM: an adaptive swap failed"; return false; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        e = std::string("VRAM: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const auto t0 = Clock::now();
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
    const int64_t before = xcache.slots();
    // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
    const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
        128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
    std::string note;
    if ((int64_t) free_b < want_free) {
        const int64_t floor_b = xcache.bytes_of(floor_slots);
        int64_t keep = mapped - (want_free - (int64_t) free_b);
        if (keep < floor_b) {
            keep = floor_b;
            note = " (the cache keeps its smallest size: the prompt path's buffers)";
        }
        if (!home_given_up(xcache.slots_after_shrink(keep), e)) { e = "VRAM: " + e + " (nothing changed)"; return false; }
        // the homes are committed by now: whatever the shrink left or took is followed from the cache itself
        const bool shrunk = xcache.shrink(keep, e);
        const int64_t live = xcache.slots();
        for (size_t i = 0; i < host_res.size(); ++i)
            if (host_res[i] >= live) {
                vram_evicted.emplace_back((int32_t) i, host_res[i]);
                host_res[i] = basalt::core::kNotResident;
            }
        res_upload();
        if (!shrunk) return false;
    } else if (mapped < xcache.full_bytes()) {
        std::string gerr;
        if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
        const int64_t live = xcache.slots();
        std::string ferr;
        std::vector<std::pair<int32_t, int32_t>> keep_out;
        struct Fill { int32_t slot; const uint8_t* b; int64_t bytes; };
        std::vector<Fill> fills;
        for (const auto& [i, slot] : vram_evicted) {
            if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
            const int64_t layer = i / g.n_expert;
            int64_t pick = i;
            if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                pick = -1;
                float best = -1.0f;
                for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                    const size_t j = (size_t) (layer * g.n_expert + ex);
                    if (host_res[j] >= 0 || (src.complement_ready() && !src.has_resident(layer, ex))) continue;
                    const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                    if (u > best) { best = u; pick = (int64_t) j; }
                }
            }
            if (pick < 0) continue;
            const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
            if (b == nullptr) {
                e = "VRAM: refilling the cache failed: an expert has no RAM copy";
                return false;
            }
            fills.push_back({slot, b, (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer)});
            host_res[(size_t) pick] = slot;
        }
        // after a SLEEP the RAM copy is registered again in address order (WAKE): the copies follow it
        const bool repinning = src.complement_unpinned();
        if (repinning)
            std::sort(fills.begin(), fills.end(),
                      [](const Fill& x, const Fill& y) { return std::less<const uint8_t*>()(x.b, y.b); });
        for (const Fill& f : fills) {
            if (repinning && !src.repin_wait(f.b, (uint64_t) f.bytes)) {
                e = "VRAM: the RAM copy could not be registered again";
                return false;
            }
            if (!xcache.fill_slot_queued(f.slot, f.b, ferr, f.bytes)) {
                e = "VRAM: refilling the cache failed: " + ferr;
                return false;
            }
        }
        vram_evicted.swap(keep_out);
        if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
        res_upload();
        relend();
        // the slots hold the experts again: every expert in a slot below the loan has its RAM home freed (the
        // copy is the slots' now: those given up, those the loan range kept, and those the adaptive tier moved
        // back).  The loan's own slots keep theirs: a prompt overwrites them and reads those experts from RAM.
        if (src.complement_ready()) {
            const int64_t loan_from = pf_parts.empty() || pf_parts[0].first < 0 ? INT64_MAX : pf_parts[0].first;
            size_t released = 0;
            for (size_t j = 0; j < host_res.size(); ++j)
                if (host_res[j] >= 0 && host_res[j] < loan_from &&
                    src.home_release((int64_t) j / g.n_expert, (int64_t) j % g.n_expert))
                    ++released;
            const uint64_t freed = src.homes_reclaim() + src.homes_compact();
            basalt::log::info("basalt serve", "VRAM: %zu RAM homes released, %.0f MiB freed, %.0f MiB kept for the "
                              "prompt loan (slots from %lld)", released, (double) freed / 1048576.0,
                              (double) src.homes_bytes() / 1048576.0, (long long) std::min<int64_t>(loan_from, live));
        }
    }
    relend();
    cudaMemGetInfo(&free_b, &total_b);
    basalt::log::info("basalt serve", "VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                         "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s",
                 reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                 (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                 (long long) (free_b >> 20), (long long) o.prefill_chunk,
                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
    char line[400];
    std::snprintf(line, sizeof line, "VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld "
                  "expert_cache_mib=%lld expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld "
                  "ram_homes_mib=%lld\n", reserve, (long long) xcache.slots(), (long long) xcache.full_slots(),
                  (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                  (long long) (free_b >> 20), (long long) o.prefill_chunk, (long long) (src.homes_bytes() >> 20));
    if (reply != nullptr) *reply = line;
    else std::fputs(line, stdout);
    return true;
}

int Runtime::init_memory() {
    // the load's own buffers are gone: RAM still free keeps the next-hottest experts the first sizing left out
    if (srcp == &src && src.complement_ready() && o.resident_budget == 0) {
        int64_t n_top = 0;
        // what the engine still takes once serving (see serve_growth_bytes) and the file cache's floor while
        // experts are read from the SSD (none when every expert has a tier)
        const uint64_t later = serve_growth_bytes(o, g) +
                               (ram_floor_bytes - std::min(ram_floor_bytes, basalt::platform::ram_margin()));
        const uint64_t add = src.top_up(stages.empty() ? profile : profile_all, [&](int32_t l, int32_t e) {
            return host_res[(size_t) l * g.n_expert + (size_t) e] >= 0 || (peer.valid() && peer.has(l, e));
        }, o.resident_pin, n_top, later);
        if (add > 0) {
            basalt::platform::watch_resident_ram(true, src.resident_bytes() + add);
            basalt::log::info("basalt generate", "RAM top-up: +%.2f GiB, %lld experts (total %.2f GiB)",
                         (double) add / 1073741824.0, (long long) n_top,
                         (double) (src.resident_bytes() + add) / 1073741824.0);
        }
    }
    report_expert_tiers();
    basalt::platform::stop_memory_watchdog();
    // from here the RAM guard watches: below its floor the engine sheds and new requests wait, and a running
    // request stops only once the RAM is critical
    guard_floor = basalt::platform::ram_guard_floor(basalt::platform::ram_margin(), tier_file_bytes, prompt_file_bytes);
    basalt::platform::start_ram_guard(guard_floor);
    disk_floor = guard_floor + (1ull << 30);   // a disk-tier capture or read stays a GiB above the guard
    // what a stop's error compares against, to tell the engine's own growth from other programs'
    ready_avail = basalt::platform::available_memory();
    ready_rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
    // once per drop below the guard's floor, on the serve thread between steps: the parked conversations leave RAM
    // (to the disk tier when it is on and holds no copy of them yet: queued for its writer, whose writes free them;
    // the others are dropped), the checkpoint blocks kept for reuse are freed and the heap's free pages return to
    // the OS.  The checkpoints stay: a live request's rewind and the deferred copy in flight use them.
    shed_drop = 0;

    // a VRAM shrink's RAM homes leave the guard's floor above the margin and the top-up's own slack free
    src.set_homes_reserve(guard_floor - std::min(guard_floor, basalt::platform::ram_margin()) + (256ull << 20));
    if (const uint64_t avail = basalt::platform::available_memory(); avail > 0)
        basalt::log::info("basalt serve", "RAM %.2f GiB free, guard %.2f GiB, %.2f GiB kept for serving",
                     (double) avail / 1073741824.0, (double) guard_floor / 1073741824.0,
                     (double) serve_growth_bytes(o, g) / 1073741824.0);
    return 0;
}

}  // namespace basalt::program::serve
