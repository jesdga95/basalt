// src/program/serve/request_finish.cpp - request complete operations.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::complete() {
    decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
    // the last commit (set_commit_async): the session is complete before anything reads or copies it
    if (!runtime_.ver.wait_commit(runtime_.err) || (runtime_.pipe_on && !runtime_.ver_b.wait_commit(runtime_.err))) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if (runtime_.pipe_on)
        basalt::log::info("basalt serve", "pipelined windows: %lld launched, %lld held", (long long) pair_runs,
                          (long long) pair_holds);
    if (dec_timing && dec_windows > 0) {
        const DecSnap d1 = dec_snap();
        const double w = (double) dec_windows, L = (double) runtime_.g.n_layers;
        auto fmt2 = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return std::string(b); };
        basalt::log::info("basalt decode timing", "%lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                             "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                             "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                             "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f%s",
                     (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                     (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                     (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                     (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                     (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L),
                     runtime_.drive.d.peer == nullptr ? "" : (", peer GPU " + fmt2((d1.peer - ds0.peer) / (w * L)) +
                                                     " (the pool waits " + fmt2((d1.peer_wait - ds0.peer_wait) / w) +
                                                     " ms/window for it)").c_str());
        const std::string evr = basalt::core::decode_events().report();
        if (!evr.empty()) basalt::log::raw("%s\n", evr.c_str());
        const std::string pr = runtime_.ver.profile_report();
        if (!pr.empty()) basalt::log::raw("basalt decode GPU stages (ms/window):%s\n", pr.c_str());
    }
    if (!cancelled) {
        // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
        // (the checkpoints taken while reading it are still good)
        runtime_.live.swap(consumed);
        runtime_.live_imgs = runtime_.imgs_below(runtime_.req_imgs, (int64_t) runtime_.live.size());
        runtime_.live_ok = runtime_.o.prompt_cache > 0;
    }
    runtime_.ckd_finish();
    req_hits = runtime_.drive.d.cache_hits - decode_hits0;
    req_look = (runtime_.drive.d.cache_hits + runtime_.drive.d.cache_admitted + runtime_.drive.d.cache_refused) - decode_look0;
    req_offload = runtime_.drive.d.offload_entries - offload0;
    // the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a cancel
    // stopped it part-way
    fresh = n - resume;
    read_n = cancelled ? std::clamp<int64_t>(runtime_.pp_reached - resume, 0, fresh) : fresh;
    // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
    //      [RAM blobs] [file blobs] [file MB]   (file tiers; appended, so an older server reads the rest)
    //      [prompt tokens read]   (fewer than <prompt> - <reused> when a cancel stopped the read)
    //      [offloaded]   (the decode's routed experts the GPU read over PCIe or another GPU computed;
    //                    not in [lookups])
    ram_out = runtime_.ram_stop.load() || !kv_stop.empty();
    if (!kv_stop.empty()) {
        basalt::log::warn("basalt serve", "request stopped after %lld tokens: %s", (long long) produced_n,
                          kv_stop.c_str());
        std::printf("ERR %s\n", kv_stop.c_str());
    } else if (ram_out) {
        // the client gets the reason as the request's error; the engine stays up for the next request
        uint64_t avail = 0, floor = 0;
        basalt::platform::MemoryStalls stalls;
        (void) basalt::platform::ram_guard_level(&avail, &floor, &stalls);
        // what the engine grew by since it was ready, and what the rest of the drop in free RAM went to
        const uint64_t rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
        const uint64_t own = rss > runtime_.ready_rss ? rss - runtime_.ready_rss : 0;
        const uint64_t drop = runtime_.ready_avail > avail ? runtime_.ready_avail - avail : 0;
        const std::string msg = basalt::platform::ram_stop_message(avail, floor, basalt::platform::kRamHardFloor,
                                                                   stalls, drop > own ? drop - own : 0, own);
        basalt::log::error("basalt", "%s", msg.c_str());
        std::printf("ERR %s\n", msg.c_str());
    }
    if (!ram_out && yielded_at >= 0) std::printf("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
    if (!ram_out) std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                (long long) produced_n,
                (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                (long long) resume, (long long) req_hits, (long long) req_look,
                (long long) (runtime_.src.ram_reads() - ram0), (long long) (runtime_.src.file_reads() - files0),
                (double) (runtime_.src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                (long long) req_offload);
    std::fflush(stdout);
    // no swap stays in flight between requests: the last round's copies land once the reply is out
    if (!runtime_.xq_drain()) {
        std::printf("ERR an adaptive refill failed\n");
        return Flow::error;
    }
    runtime_.xq_saturated = false;   // the next request starts at the base cap
    if (runtime_.admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
        bool cont = ram_out ? false : in_slot ? !cancelled && (int64_t) runtime_.live.size() == p
                            : !cancelled && produced_n == 1 && runtime_.admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                              (int64_t) runtime_.live.size() == p;
        const auto tc0 = Clock::now();
        // the draft layer's cell for the prompt's last token (it pairs with the first generated one), then its K/V
        if (cont && !in_slot && runtime_.batch_spec && !runtime_.mtp.prefill(runtime_.ver.final_R_all(), &x, 1, p - 1, runtime_.err)) {
            basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
            cont = false;
        }
        if (cont && in_slot) {   // the slot holds the prompt already; only its draft K/V is staged
            basalt::core::SessionState& t = *runtime_.bslot_ss[0][(size_t) runtime_.admit_slot];
            t.ple_prev[0] = (*rview).ple_prev[0];
            t.ple_prev[1] = (*rview).ple_prev[1];
            t.ple_token = (*rview).ple_token;
            if (runtime_.batch_spec && !runtime_.mtp.copy_store(runtime_.mtp.main_store(), runtime_.admit_slot, (int64_t) runtime_.live.size(), runtime_.err)) {
                basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
                cont = false;
            }
        } else if (cont && !runtime_.copy_to_slot(runtime_.admit_slot, runtime_.live, runtime_.err)) {
            basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
            runtime_.bs[(size_t) runtime_.admit_slot].cached = false;   // its sessions may be half written
            cont = false;
        }
        if (cont) {
            if (runtime_.slot0_is_solo && runtime_.admit_slot == 0) runtime_.live_ok = false;   // the batch windows move the state on
            runtime_.ver.set_slot_sampling(runtime_.admit_slot, req_sp);   // the request's own sampling, row by row
            BSlot& sl = runtime_.bs[(size_t) runtime_.admit_slot];
            sl = BSlot{};
            sl.active = true;
            sl.script = std::move(runtime_.bscript[(size_t) runtime_.admit_slot].first);
            sl.rng.seed(runtime_.bscript[(size_t) runtime_.admit_slot].second);
            runtime_.bscript[(size_t) runtime_.admit_slot].first.clear();
            if (runtime_.batch_spec && runtime_.o.suffix_draft > 0) {
                sl.sfx = std::make_unique<basalt::spec::SuffixDrafter>(std::max(1, runtime_.o.suffix_draft), 64,
                                                                       (size_t) runtime_.o.max_context + 4096);
                for (int32_t t : runtime_.live) sl.sfx->append(t);
                sl.sfx->append(x);
            }
            sl.x = x;
            sl.p = p;
            sl.produced = in_slot ? 0 : 1;
            sl.max_new = runtime_.admit_max_new;
            sl.t0 = Clock::now();
            sl.ids = runtime_.live;
            sl.cvec = runtime_.cvec_cached;
            sl.imgs = runtime_.live_imgs;
                            sl.guard = std::move(gguard);   // the request's grammar, already past the tokens the request decoded here
            // the slot keeps its conversation's longest checkpoint (its next turn) and its shortest (the
            // system prompt, which other requests may share)
            const ConvCheckpoint *best = nullptr, *root = nullptr;
            for (const ConvCheckpoint& c : runtime_.checks)
                if (c.ids.size() < runtime_.live.size() &&
                    basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), runtime_.live.data())) {
                    if (best == nullptr || c.ids.size() > best->ids.size()) best = &c;
                    if (root == nullptr || c.ids.size() < root->ids.size()) root = &c;
                }
            if (root != nullptr && root != best) sl.checks.push_back(*root);
            if (best != nullptr) sl.checks.push_back(*best);
            basalt::log::info("basalt batch", "slot %d takes %lld tokens (copied in %.1f ms)", runtime_.admit_slot,
                         (long long) runtime_.live.size(), std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
        }
        if (!ram_out)   // (after ERR the server reads no BADM)
            std::printf("BADM %d %d\n", runtime_.admit_slot, cont ? (in_slot ? 2 : 1) : 0);   // 2: no token yet
        std::fflush(stdout);
        runtime_.admit_slot = -1;
    }
    if (solo_req) {   // the conversation stays in the slot it ran in
        basalt::core::SessionState& t = *runtime_.bslot_ss[0][(size_t) slot_b];
        t.ple_prev[0] = (*rview).ple_prev[0];
        t.ple_prev[1] = (*rview).ple_prev[1];
        t.ple_token = (*rview).ple_token;
        if (yielded_at < 0) {   // (a read that gave way was parked in the slot already)
            BSlot& sb = runtime_.bs[(size_t) slot_b];
            sb = BSlot{};
            sb.ids = runtime_.live;
            sb.cvec = runtime_.cvec_cached;
            sb.imgs = runtime_.live_imgs;
            sb.cached = runtime_.live_ok && runtime_.o.prompt_cache > 0;
            if (sb.cached) sb.checks = std::move(runtime_.checks);
            if (sb.cached && runtime_.batch_spec && !runtime_.mtp.copy_store(runtime_.mtp.main_store(), slot_b, (int64_t) runtime_.live.size(), runtime_.err)) {
                runtime_.err.clear();
                sb.cached = false;
            }
        }
    }
    if (in_slot) {   // the solo session and its bookkeeping again
        if (solo_req && !runtime_.ver.bind_session(&runtime_.ss, runtime_.err)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        runtime_.rsp = &runtime_.ss;
        runtime_.sp.set_session(runtime_.ss);
        if (runtime_.mtp_on) runtime_.mtp.set_ple_session(&runtime_.ss);
        runtime_.live = std::move(solo_book.live);
        runtime_.live_imgs = std::move(solo_book.live_imgs);
        runtime_.live_ok = solo_book.live_ok;
        runtime_.cvec_cached = solo_book.cvec;
        runtime_.checks = std::move(solo_book.checks);
        if (solo_moved) {   // its cells went to this request (hold_slot)
            runtime_.live.clear();
            runtime_.live_imgs.clear();
            runtime_.live_ok = false;
            runtime_.checks.clear();
        }
    }
    if (runtime_.drive.routing != nullptr) std::fflush(runtime_.drive.routing);   // the routing trace survives a crash and is watchable mid-session

    if (cancelled)
        std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
    else
        std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
    basalt::log::info("basalt serve", "prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                         "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s",
                 (long long) n, (long long) resume, read_txt, prompt_ms,
                 prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                 decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                 (long long) draft_offered, runtime_.checks.size(), cancelled ? " (cancelled)" : "");
    if (relax_on)
        basalt::log::info("basalt serve", "relaxed drafting %.2f: %lld of the %lld accepted drafts were kept by it",
                     (double) req_relax, (long long) relaxed_kept, (long long) draft_accepted);
    // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU to
    // read are in neither count, so that share is said beside it
    if (req_look > 0) {
        char off[160] = "";
        if (req_offload > 0)
            std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                          "all %lld routed)", (long long) req_offload,
                          100.0 * (double) req_offload / (double) (req_look + req_offload),
                          (long long) (req_look + req_offload));
        basalt::log::info("basalt serve", "decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s",
                     100.0 * (double) req_hits / (double) req_look,
                     (long long) req_hits, (long long) req_look, off);
    }
    // the resident RAM mode, cumulative: experts read from the experts section since the copy was made (what the plain
    // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
    if (runtime_.src.complement_ready())
        basalt::log::info("basalt serve", "resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                             "VRAM tier, %lld blob reads from the file",
                     (double) runtime_.src.resident_bytes() / 1073741824.0, (long long) runtime_.src.exchanges(),
                     (long long) runtime_.src.file_reads());
    // the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
    if (runtime_.srcp == &runtime_.src)
        basalt::log::info("basalt serve", "expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                             "%.1f MB read", (long long) req_hits, (long long) runtime_.src.ram_reads(),
                     (long long) runtime_.src.file_reads(), (double) runtime_.src.file_read_bytes() / 1e6);
    if (runtime_.g.n_qsa_layers() > 0 && runtime_.ss.qsa_states[runtime_.ss.qsa_primary()].kv_mode == 1) {
        // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
        // (this device's owned ordinals; a split's other stages hold theirs)
        uint64_t miss = 0, look = 0;
        bool over = false;
        for (int64_t j = 0; j < runtime_.ss.qsa_alloc; ++j) {
            const basalt::kernels::KvStreamCounters c =
                basalt::kernels::kv_stream_counters(runtime_.ss.qsa_states[runtime_.ss.qsa_ord0 + j].map);
            miss += c.misses; look += c.lookups; over = over || c.overflow;
        }
        basalt::log::info("basalt serve", "KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                             "from RAM%s", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                     (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                     over ? " - OVERFLOW (too few resident cells)" : "");
    }
    if (sfx_windows > 0)
        basalt::log::info("basalt serve", "suffix drafts: %lld windows, %lld of %lld drafts accepted",
                     (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
    for (int r = 0; r < 3; ++r) if (runtime_.o.expert_cache_remote[(size_t) r] > 0)
        basalt::log::info("basalt serve", "CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                             "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                             "waiting for it in this request", r + 1,
                     (long long) (runtime_.remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                     (long long) (runtime_.remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                     (double) (runtime_.remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                     (double) (runtime_.remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                     runtime_.remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                     runtime_.remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);

    return Flow::next;
}

}  // namespace basalt::program::serve
