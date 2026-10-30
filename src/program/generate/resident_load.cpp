#include "context.hpp"

namespace basalt::program::generate {

int resident_load(Context& context) {
    auto& o = *context.o;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& profile = *context.profile;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& ss = *context.ss;
    auto& stages = *context.stages;
    auto& profile_all = *context.profile_all;
    auto& src = *context.src;
    auto& srcp = *context.srcp;
    auto& ram_available_at_plan = *context.ram_available_at_plan;
    auto& auto_resident_bytes = *context.auto_resident_bytes;
    auto& xcache = *context.xcache;
    auto& kv_tail_slots = *context.kv_tail_slots;
    auto& cache_fill_bytes = *context.cache_fill_bytes;
    auto& peer = *context.peer;
    auto& measure_pinned_share = context.measure_pinned_share;
    auto& lend_pct_for_share = context.lend_pct_for_share;
    auto& auto_lend_pct = *context.auto_lend_pct;
    auto& kMaxAutoChunk = *context.kMaxAutoChunk;
    auto& auto_ceiling = *context.auto_ceiling;
    auto& largest_layer_bytes = *context.largest_layer_bytes;
    auto& prompt_file_bytes = *context.prompt_file_bytes;
    auto& adapt_fast_on = *context.adapt_fast_on;
    auto& late = *context.late;
    auto& resident_headroom_base = *context.resident_headroom_base;
    auto& lend_from = *context.lend_from;
    auto& stage_held = *context.stage_held;
    auto& peer_only = *context.peer_only;
    auto& peer_only_bytes = *context.peer_only_bytes;

    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from the experts section into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    uint64_t ram_floor_bytes = basalt::platform::ram_margin();   // the RAM the copy leaves free; the serve guard's floor
    if (o.resident_cpu_experts) {
        // the RAM copy is sized from the available RAM: without a reading the experts would silently stay in the file
        if (basalt::core::detail::HostMemory hm; !basalt::core::detail::host_available_memory(hm))
            return fatal_start_error(
                "cannot determine the available RAM (no MemAvailable in /proc/meminfo, or a malformed cgroup memory "
                "limit or memory.stat), which the resident RAM mode (--resident-experts) sizes its copy of the "
                "experts from");
        // --peer-exclusive: peer swaps take places too (an evicted peer expert goes to one)
        if (adapt_fast_on)
            src.set_spares(o.peer_exclusive ? o.adapt_swaps * kSatBoost + std::max(o.peer_adapt_swaps, o.adapt_swaps)
                                            : o.adapt_swaps * kSatBoost);
        uint64_t resident_headroom = resident_headroom_base;
        // The experts no GPU tier holds, and the RAM there is for them: when the copy cannot hold them all, the rest
        // are read from the file through the OS file cache, so the copy leaves that cache its floor beside the margin.
        {
            const uint64_t gpu_held = std::min(expert_total_bytes, cache_fill_bytes + peer_only_bytes);
            const uint64_t complement = expert_total_bytes - gpu_held;
            const uint64_t avail = ram_available_at_plan;
            const uint64_t margin = basalt::platform::ram_margin();
            const uint64_t copy = std::min(o.resident_budget > 0 ? o.resident_budget : auto_resident_bytes, complement);
            const uint64_t file_bytes = complement - copy;
            ram_floor_bytes =
                basalt::platform::ram_floor(margin, file_bytes, complement, largest_layer_bytes, prompt_file_bytes);
            if (avail > 0 && file_bytes > 0) {
                const uint64_t safe = basalt::platform::max_safe_copy(avail, late, ram_floor_bytes);
                constexpr double kGiB = 1073741824.0;
                if (o.resident_budget > 0 && o.resident_budget > safe) {
                    return fatal_start_error(
                        "--resident-budget-gib " + gib_text(o.resident_budget) + " too high: " + gib_text(avail) +
                        " GiB free, " + gib_text(ram_floor_bytes) + " GiB floor" +
                        (late > 0 ? " + " + gib_text(late) + " GiB for serving" : std::string()) + "; max " +
                        gib_text(safe));
                }
                resident_headroom = std::max(resident_headroom, ram_floor_bytes + late);
                basalt::log::info("basalt generate", "RAM floor %.1f GiB (%.1f margin + %.1f cache), %.1f GiB "
                                     "experts on SSD, RAM copy <= %.1f of %.1f GiB",
                             (double) ram_floor_bytes / kGiB, (double) margin / kGiB,
                             (double) (ram_floor_bytes - margin) / kGiB, (double) file_bytes / kGiB,
                             (double) (o.resident_budget > 0 ? std::min(o.resident_budget, safe) : safe) / kGiB,
                             (double) avail / kGiB);
            }
        }
        std::vector<std::pair<int32_t, int32_t>> gpu_only = peer_only;
        gpu_only.insert(gpu_only.end(), stage_held.begin(), stage_held.end());
        const auto& rank = stages.empty() ? profile : profile_all;
        bool resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, gpu_only, lend_from, resident_headroom,
                                                    o.resident_budget, &rank);
        std::string whole_err;
        if (!resident_ok && o.resident_soft) {
            // the whole complement does not fit - keep what does, the hottest by the profile, through the budget
            // path (sized by the RAM alone) instead of none: the misses outside it read the same file bytes the mmap
            // fallback reads, so the answers are unchanged.  Nothing pinned: the fallback below.
            whole_err = err;
            resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, gpu_only, -1, resident_headroom,
                                                   basalt::core::FileExpertSource::kResidentWhatFits, &rank);
            if (resident_ok) {
                basalt::log::warn("basalt generate", "the whole complement does not fit in RAM (%s); the hottest %.2f GiB "
                                     "are kept in RAM", whole_err.c_str(), (double) src.resident_bytes() / 1073741824.0);
            }
            else
                err = whole_err + "; " + err;
        }
        if (resident_ok && o.prefill_auto_fit && lend_from >= 0) {
            // A lent slot whose expert the complement holds is refilled from RAM; the others are read from the file.
            // Allow the backed slots plus a fixed amount of file reading per prompt.  Those reads are evictable page
            // cache that the refill passes through once, so two chunks' worth are funded; the read-time cap
            // (kMaxAutoChunk) bounds the chunk, so the ceiling stops there too.
            const int64_t allow = std::max<int64_t>(0, src.resident_lent_slots() - kv_tail_slots) +
                                  (int64_t) (2 * basalt::platform::kPromptFileReadBytes / std::max<uint64_t>(1, basalt::kernels::cpu::expert_layout().max_blob));
            auto_ceiling = std::max<int64_t>(8192, engine_local::biggest_chunk(std::min<int64_t>(auto_ceiling, kMaxAutoChunk),
                                                                 [&](int64_t t) { return serve_loans::lend_slots(xcache, g, ss, t) <= allow; }));
        }
        if (resident_ok) {
            if (o.serve) basalt::platform::watch_resident_ram(true, src.resident_bytes());
            if (o.adapt_every > 0 && o.adapt_swaps > 0 && !adapt_fast_on &&
                !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
                basalt::log::error("basalt", "CPU expert residency: %s", err.c_str());
                return 1;
            }
            basalt::log::info("basalt generate", "resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                                 "cache; adaptive swaps %s",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                         (long long) xcache.resident(),
                         o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                                : "off");
            measure_pinned_share();   // the loans sized from here on see the pinned complement
            auto_lend_pct = lend_pct_for_share();
            if (o.prefill_auto)
                basalt::log::info("basalt generate", "prompt path lends at most %lld%% of the cache (%.0f%% of the expert "
                                     "bytes pinned)", (long long) auto_lend_pct,
                             100.0 * basalt::prefill::Prefill::pinned_share());
        } else if (o.resident_soft) {
            basalt::log::warn("basalt generate", "the resident RAM mode does not fit (%s)", err.c_str());
        } else if (o.resident_budget > 0) {
            // a RAM budget that cannot be kept is not a reason to stop - the experts it would have held are read from
            // the files like the ones outside it (pin_cache_complement leaves nothing half-built)
            basalt::log::warn("basalt generate", "the RAM budget (--resident-budget-gib) cannot be kept (%s)",
                         err.c_str());
        } else {
            basalt::log::error("basalt generate", "CPU expert residency: %s", err.c_str());
            return 1;
        }
    }
    // experts no RAM tier holds (none was asked for, or the RAM is short) are read from the file when needed
    if (srcp == &src && !src.complement_ready() && expert_total_bytes > cache_fill_bytes) {
        basalt::core::LoadProgress::skipped(expert_total_bytes - cache_fill_bytes - peer_only_bytes);
        basalt::core::LoadProgress::plan("ram", 0);
    }
    // the peer's swaps keep their experts in RAM or on a GPU, never only in the file
    if (o.peer_exclusive && peer.valid() && srcp == &src && src.complement_ready()) peer.set_exclusive(&src);

    context.ram_floor_bytes = &ram_floor_bytes;
    return prompt_split(context);
}

}
