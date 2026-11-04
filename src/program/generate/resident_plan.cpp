#include "context.hpp"

namespace basalt::program::generate {

int resident_plan(Context& context) {
    auto& o = *context.o;
    auto& set_stream_all_min = context.set_stream_all_min;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& profile = *context.profile;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& ss = *context.ss;
    auto& stages = *context.stages;
    auto& srcp = *context.srcp;
    auto& pool_store = *context.pool_store;
    auto& ram_available_at_plan = *context.ram_available_at_plan;
    auto& auto_resident_bytes = *context.auto_resident_bytes;
    auto& peer_prompt_bytes = context.peer_prompt_bytes;
    auto& xcache = *context.xcache;
    auto& kv_tail_slots = *context.kv_tail_slots;
    auto& cache_fill_bytes = *context.cache_fill_bytes;
    auto& peer_split_bytes = context.peer_split_bytes;
    auto& peer_expert_room = context.peer_expert_room;
    auto& peer = *context.peer;
    auto& xpeers = *context.xpeers;
    auto& drive = *context.drive;
    auto& d_res = *context.d_res;

    // ================================ THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below)
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk (`PL`, the same fields, inside a chunk: the position is an estimate), and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's lend cap); measured again once the RAM
    // complement is pinned, since a source pins nothing before that
    auto measure_pinned_share = [&]() {
        if (srcp == nullptr || o.prefill_chunk <= 0) return;
        uint64_t pinned = 0, total = 0;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        basalt::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    };
    measure_pinned_share();
    // The share of a cache a lent prompt chunk may take: 90% when the copies are DMA from pinned RAM, 85% when host
    // copies are the limit (lending more only streams more through them).
    auto lend_pct_for_share = [] { return basalt::prefill::Prefill::pinned_share() >= 0.9 ? (int64_t) 90 : (int64_t) 85; };
    int64_t auto_lend_pct = lend_pct_for_share();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the context,
    // but never below 8192.  A bare `auto` is cut back once the RAM complement is built (below) to what the lend can
    // fund, and again to the fewest-equal-chunks cap when a prompt is read; without the resident RAM mode there is no
    // complement to size it from, so it stays at 8192.
    const int64_t kMaxAutoChunk = basalt::kernels::cpu::expert_layout().max_blob > (3u << 19) ? 32768 : 16384;
    int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    if (o.prefill_auto_fit && !o.resident_cpu_experts) auto_ceiling = 8192;
    // The peer tier's fill.  `exclusive` (--peer-exclusive): done before the RAM complement, which then leaves the
    // peer's experts out, so they are read from the file here and held only on the peer.
    auto open_peer = [&](bool exclusive) -> bool {
        const auto tp0 = Clock::now();
        // the GDN mixer split loads the peer's key heads after its tier is filled
        int d0 = 0;
        cudaGetDevice(&d0);
        const int peer_reserve = o.peer_reserve_mib + (int) ((peer_split_bytes(d0) + peer_prompt_bytes() + (1 << 20) - 1) >> 20);
        if (o.peer_devices.size() <= 1) {
            if (!peer.open(o.peer_device, profile, xcache, *srcp, g.n_layers, g.n_expert, peer_reserve, o.peer_slots,
                           err, exclusive)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
            basalt::log::info("basalt generate", "peer GPU %d: %lld experts, %.2f GiB (filled in %.1f s); with the primary's "
                                 "%lld that is %lld of %lld on the GPUs", o.peer_device, (long long) peer.resident(), peer.gib(),
                         std::chrono::duration<double>(Clock::now() - tp0).count(), (long long) xcache.slots(),
                         (long long) (peer.resident() + xcache.slots()), (long long) (g.n_layers * g.n_expert));
            return true;
        }
        // several tiers: every card's buffers first, then the pairs the primary does not hold dealt out by room and
        // bandwidth (gpu_select.hpp), each card's room shared by the tiers it holds
        const size_t np = o.peer_devices.size();
        xpeers.clear();
        for (size_t i = 1; i < np; ++i) xpeers.push_back(std::make_unique<basalt::core::PeerExperts>());
        auto tier = [&](size_t i) -> basalt::core::PeerExperts& { return i == 0 ? peer : *xpeers[i - 1]; };
        for (size_t i = 0; i < np; ++i)
            if (!tier(i).prepare(o.peer_devices[i], *srcp, g.n_layers, g.n_expert, err)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
        std::vector<basalt::program::TierRoom> rooms(np);
        for (size_t i = 0; i < np; ++i) {
            const int dev = o.peer_devices[i];
            size_t fb = 0, tb = 0;
            {
                const basalt::core::OnDevice on(dev);
                cudaMemGetInfo(&fb, &tb);
            }
            const uint64_t reserve = ((uint64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                     (dev == o.peer_device ? (uint64_t) peer_split_bytes(d0) + peer_prompt_bytes() : 0);
            const uint64_t share = (uint64_t) std::count(o.peer_devices.begin(), o.peer_devices.end(), dev);
            rooms[i].bytes = fb > reserve ? ((uint64_t) fb - reserve) / share : 0;
            rooms[i].gbps = o.peer_gbps[i];
        }
        std::vector<std::pair<int32_t, int32_t>> ranked;
        std::vector<uint64_t> bytes;
        for (const auto& pr : profile)
            if (xcache.slot_of(pr.first, pr.second) < 0) {
                ranked.push_back(pr);
                bytes.push_back(basalt::core::PeerExperts::slot_bytes(pr.first));
            }
        const std::vector<int> owner = basalt::program::spread_experts(bytes, rooms);
        int64_t held = xcache.slots();
        for (size_t i = 0; i < np; ++i) {
            std::vector<std::pair<int32_t, int32_t>> pick;
            for (size_t j = 0; j < ranked.size(); ++j)
                if (owner[j] == (int) i) pick.push_back(ranked[j]);
            if (!tier(i).fill(pick, err)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
            held += tier(i).resident();
            basalt::log::info("basalt generate", "peer GPU %d: %lld experts, %.2f GiB (tier %zu of %zu)", o.peer_devices[i],
                              (long long) tier(i).resident(), tier(i).gib(), i + 1, np);
        }
        basalt::log::info("basalt generate", "%zu peer tiers filled in %.1f s; with the primary's %lld that is %lld of %lld "
                             "on the GPUs", np, std::chrono::duration<double>(Clock::now() - tp0).count(),
                          (long long) xcache.slots(), (long long) held, (long long) (g.n_layers * g.n_expert));
        return true;
    };
    // every expert of the largest layer: what one prompt layer can read from the file (see platform::file_cache_floor)
    uint64_t largest_layer_bytes = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        largest_layer_bytes = std::max<uint64_t>(largest_layer_bytes,
                                                 (uint64_t) g.n_expert * basalt::kernels::cpu::expert_layout().blob_bytes(l));
    // what a prompt chunk reads from the file for the lent slots the RAM copy leaves out (see platform::file_cache_floor)
    uint64_t prompt_file_bytes = 0;
    // --serve: free places in the RAM copy take the evicted experts (no exchange buffers)
    const bool adapt_fast_on = o.serve && o.adapt_every > 0 && o.adapt_swaps > 0;
    // A server also leaves room for what it allocates after the RAM copy, before it is ready (the top-up below takes
    // back whatever that does not use), and for what it takes while serving.
    // Exchange buffers are reserved after the copy when the adaptive swaps do not use spares; the copy leaves them room.
    const uint64_t exchange_bytes =
        (o.adapt_every > 0 && o.adapt_swaps > 0 && !adapt_fast_on)
            ? (uint64_t) std::min<int64_t>(o.adapt_swaps, 96) * basalt::kernels::cpu::expert_layout().max_blob
            : 0;
    const uint64_t late = (o.serve ? basalt::platform::kLateHostBytes + serve_growth_bytes(o, g) : 0) + exchange_bytes;
    const uint64_t resident_headroom_base =
        o.serve ? std::max(o.resident_headroom, basalt::platform::ram_margin() + late)
                : std::max(o.resident_headroom, basalt::platform::ram_margin() + basalt::platform::kLateHostBytes) +
                      exchange_bytes;   // what loads after the copy, and the buffers, fit beside the margin
    // the GPU cache's last slots are lent to the prompt path (and the elastic KV): their experts are kept in RAM too
    int64_t lend_from = -1;
    int64_t lend_chunk = o.prefill_chunk;
    if (o.prefill_chunk > 0 && d_res != nullptr && xcache.slots() > 0) {
        int64_t chunk = o.prefill_chunk;
        const int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, auto_ceiling,
                                                (int) auto_lend_pct);
        if (k > 0) { lend_from = xcache.slots() - k; lend_chunk = chunk; }
    }
    // a layer split: the stages' caches are GPU tiers the RAM copy leaves out, except each one's last slots, which the
    // prompt path lends for the chunk the first card's loan was sized for
    std::vector<std::pair<int32_t, int32_t>> stage_held;
    uint64_t stage_lent_bytes = 0;
    for (const auto& stp : stages) {
        const int64_t first = lend_from >= 0
            ? stp->cache.slots() - serve_loans::cache_slots_for(stp->cache, basalt::prefill::Prefill::bytes_needed(g, stp->ss, lend_chunk))
            : stp->cache.slots();
        for (const auto& pr : stp->profile) {
            const int32_t slot = stp->cache.slot_of(pr.first, pr.second);
            if (slot < 0) continue;
            if (slot < first) stage_held.push_back(pr);
            else stage_lent_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(pr.first);
        }
    }
    // the elastic KV's tail keeps its experts in RAM too (a slot lent to the KV becomes a CPU miss read from RAM),
    // below it the prompt path's lend region as before: the RAM decides how far down that goes, as it does today
    if (kv_tail_slots > 0)
        lend_from = lend_from >= 0 ? std::max<int64_t>(0, lend_from - kv_tail_slots) : xcache.slots() - kv_tail_slots;
    // ---- the start-up picks for the settings the user left out (peer-exclusive, resident RAM, pool affinity and
    // size): one pure rule over what this machine and pack have, so a given setting is never replaced.  The pool is
    // built here, with the picks.
    {
        namespace bp = basalt::program;
        namespace cpu = basalt::kernels::cpu;
        const auto topo = basalt::platform::detect_cpu_topology(true, basalt::platform::CpuAffinity::All);
        bp::AutoInputs in;
        in.ram_total = basalt::platform::total_physical_memory();
        // the container-aware reading (a memory cap below the machine's RAM), the one the RAM copy is checked against
        in.ram_available = basalt::platform::available_memory();
        if (basalt::core::detail::HostMemory hm; basalt::core::detail::host_available_memory(hm) && hm.available > 0)
            in.ram_available = in.ram_available > 0 ? std::min(in.ram_available, hm.available) : hm.available;
        in.peer = o.peer_devices.size() == 1 && o.resident_cpu_experts;   // peer-exclusive: one tier card
        in.serve = o.serve;
        in.worker_cores = (int) topo.worker_cores.size();
        in.p_cores = topo.p_cores;
        in.e_cores = topo.e_cores;
        in.hybrid = topo.is_hybrid;
        in.cpu_quota = cpu::cgroup_cpu_quota();
        const uint64_t margin = basalt::platform::ram_margin();
        const uint64_t head = std::max(resident_headroom_base, margin);
        const uint64_t lent_bytes =
            (lend_from >= 0 ? (uint64_t) (xcache.slots() - lend_from) * basalt::kernels::cpu::expert_layout().max_blob : 0) +
            stage_lent_bytes;
        in.complement_full = expert_total_bytes - std::min(expert_total_bytes, cache_fill_bytes);
        in.complement_peer_excl = in.complement_full - std::min(in.complement_full, peer_expert_room());
        // the RAM copy also holds the free places the adaptive swaps use (see set_spares)
        const uint64_t blob = basalt::kernels::cpu::expert_layout().max_blob;
        const uint64_t spares_full = adapt_fast_on ? (uint64_t) o.adapt_swaps * kSatBoost * blob : 0;
        const uint64_t spares_excl =
            adapt_fast_on ? (uint64_t) (o.adapt_swaps * kSatBoost + std::max(o.peer_adapt_swaps, o.adapt_swaps)) * blob : 0;
        // the lent slots' copies and the spares are kept as far as RAM allows: they do not decide whether the complement fits
        in.extra_full = lent_bytes + spares_full;
        in.extra_peer_excl = lent_bytes + spares_excl;
        // the floor when the whole complement is placed is the headroom; with experts left on the SSD the plan adds the file cache's
        in.floor_bytes = head;
        in.margin_bytes = margin;
        in.late_bytes = late;
        in.layer_bytes = largest_layer_bytes;
        prompt_file_bytes = std::min<uint64_t>(lent_bytes, basalt::platform::kPromptFileReadBytes);
        in.prompt_file_bytes = prompt_file_bytes;
        bp::AutoGiven& gv = in.given;
        if (!(in.peer && in.serve) || o.peer_exclusive_given) {   // not applicable, or given
            gv.peer_exclusive = true;
            gv.peer_exclusive_value = o.peer_exclusive;
        }
        if (!o.resident_cpu_experts || o.resident_budget_given) {
            gv.resident = true;
            gv.resident_gb = (double) o.resident_budget / 1073741824.0;
        }
        if (o.pool_affinity_given) {
            gv.pool_affinity = true;
            gv.pool_affinity_value = o.pool_affinity == cpu::PoolAffinity::All    ? bp::AutoAffinity::All
                                     : o.pool_affinity == cpu::PoolAffinity::PCores ? bp::AutoAffinity::PCores
                                                                                    : bp::AutoAffinity::Auto;
        }
        if (o.pool_workers_given && o.pool_workers > 0) {
            gv.pool_workers = true;
            gv.pool_workers_value = o.pool_workers;
        }
        const bp::AutoPlan plan = bp::choose_auto(in);
        auto_resident_bytes = plan.resident_bytes;
        ram_available_at_plan = in.ram_available;
        if (plan.peer_exclusive_src == bp::AutoSource::Auto && plan.peer_exclusive != o.peer_exclusive) {
            o.peer_exclusive = plan.peer_exclusive;
            set_stream_all_min();
        }
        const cpu::PoolAffinity affinity = plan.pool_affinity == bp::AutoAffinity::All    ? cpu::PoolAffinity::All
                                           : plan.pool_affinity == bp::AutoAffinity::PCores ? cpu::PoolAffinity::PCores
                                                                                            : cpu::PoolAffinity::Auto;
        const std::string line = bp::auto_flags_line(plan);
        if (!line.empty()) {
            // the server reads the engine's status lines on stdout
            if (o.serve) { std::printf("%s\n", line.c_str()); std::fflush(stdout); }
            else basalt::log::info("basalt generate", "%s", line.c_str());
        }
        auto& pool = pool_store.emplace(plan.pool_workers, affinity);
        if (pool.is_hybrid() && pool.affinity() != cpu::PoolAffinity::All) {
            const char* aff_str = pool.affinity() == cpu::PoolAffinity::PCores ? "p-cores" : "auto";
            basalt::log::info("basalt generate", "hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s",
                         pool.p_cores(), pool.p_threads(), pool.e_cores(), pool.workers(), aff_str);
        }
        char quota_note[64] = "";
        if (plan.pool_workers_src == bp::AutoSource::Auto && in.cpu_quota > 0 &&
            plan.pool_workers == (std::max)(1, (int) in.cpu_quota - 1))
            std::snprintf(quota_note, sizeof quota_note, " (quota %.2f CPUs)", in.cpu_quota);
        basalt::log::info("basalt generate", "%d expert-pool workers%s + the host thread%s", pool.workers(), quota_note,
                     o.no_pool ? " (UNUSED: --no-pool)" : "");
        drive.d.pool = &pool;
    }
    auto& pool = *pool_store;
    std::vector<std::pair<int32_t, int32_t>> peer_only;   // experts only the peer holds (--peer-exclusive)
    uint64_t peer_only_bytes = 0;
    if (o.resident_cpu_experts && o.peer_exclusive && o.peer_device >= 1) {
        if (!open_peer(true)) return 1;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (peer.has(l, e)) {
                    peer_only.emplace_back((int32_t) l, (int32_t) e);
                    peer_only_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(l);
                }
    }

    context.measure_pinned_share = measure_pinned_share;
    context.lend_pct_for_share = lend_pct_for_share;
    context.auto_lend_pct = &auto_lend_pct;
    context.kMaxAutoChunk = &kMaxAutoChunk;
    context.auto_ceiling = &auto_ceiling;
    context.open_peer = open_peer;
    context.largest_layer_bytes = &largest_layer_bytes;
    context.prompt_file_bytes = &prompt_file_bytes;
    context.adapt_fast_on = &adapt_fast_on;
    context.late = &late;
    context.resident_headroom_base = &resident_headroom_base;
    context.lend_from = &lend_from;
    context.stage_held = &stage_held;
    context.pool = &pool;
    context.peer_only = &peer_only;
    context.peer_only_bytes = &peer_only_bytes;
    return resident_load(context);
}

}
