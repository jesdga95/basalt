// src/program/serve/request_finish.cpp - request complete operations.
#include "request.hpp"

namespace basalt::program::serve {

// the VRAM share of the experts looked up while decoding; experts a bus copy or another GPU served are in neither
// count, so that share is said beside it
void Request::report_expert_counts(const DoneCounts& dc) const {
    if (dc.lookups <= 0) return;
    char off[160] = "";
    if (dc.offloaded > 0)
        std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                      "all %lld routed)", (long long) dc.offloaded,
                      100.0 * (double) dc.offloaded / (double) (dc.lookups + dc.offloaded),
                      (long long) (dc.lookups + dc.offloaded));
    basalt::log::info("basalt serve", "decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s",
                      100.0 * (double) dc.hits / (double) dc.lookups, (long long) dc.hits, (long long) dc.lookups, off);
}

Flow Request::complete() {
    decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
    if (const Flow flow = complete_begin(); flow != Flow::proceed) return flow;
    if (!cancelled) {
        // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
        // (the checkpoints taken while reading it are still good)
        runtime_.live.swap(consumed);
        runtime_.live_imgs = runtime_.imgs_below(runtime_.req_imgs, (int64_t) runtime_.live.size());
        runtime_.live_ok = runtime_.o.prompt_cache > 0;
    }
    runtime_.ckd_finish();
    const DoneCounts dc = done_counts();
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
                (long long) resume, (long long) dc.hits, (long long) dc.lookups,
                (long long) dc.ram_blobs, (long long) dc.file_blobs, dc.file_mb, (long long) read_n,
                (long long) dc.offloaded);
    std::fflush(stdout);
    // no swap stays in flight between requests: the last round's copies land once the reply is out
    if (!tiers_settle()) {
        std::printf("ERR an adaptive refill failed\n");
        return Flow::error;
    }
    if (runtime_.admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
        bool cont = ram_out ? false : in_slot ? !cancelled && (int64_t) runtime_.live.size() == p
                            : !cancelled && produced_n == 1 && runtime_.admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                              (int64_t) runtime_.live.size() == p;
        const auto tc0 = Clock::now();
        // the draft layer's cell for the prompt's last token (it pairs with the first generated one), then its K/V
        if (cont && !in_slot && !admit_draft()) {
            basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
            cont = false;
        }
        if (cont && in_slot) {   // the slot holds the prompt already; only its draft K/V is staged
            slot_state_back(runtime_.admit_slot);
            if (!slot_draft_keep(runtime_.admit_slot)) {
                basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
                cont = false;
            }
        } else if (cont && !runtime_.copy_to_slot(runtime_.admit_slot, runtime_.live, runtime_.err)) {
            basalt::log::error("basalt serve", "batch admission failed: %s", runtime_.err.c_str());
            runtime_.bs[(size_t) runtime_.admit_slot].cached = false;   // its sessions may be half written
            cont = false;
        }
        if (cont) {
            if (runtime_.slot0_solo() && runtime_.admit_slot == 0) runtime_.live_ok = false;   // the batch windows move the state on
            slot_set_sampling(runtime_.admit_slot);   // the request's own sampling, row by row
            BSlot& sl = runtime_.bs[(size_t) runtime_.admit_slot];
            sl = BSlot{};
            sl.active = true;
            sl.pen_n = req_sp.penalty_last_n;
            slot_init(sl);
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
        slot_state_back(slot_b);
        if (yielded_at < 0) {   // (a read that gave way was parked in the slot already)
            BSlot& sb = runtime_.bs[(size_t) slot_b];
            sb = BSlot{};
            sb.ids = runtime_.live;
            sb.cvec = runtime_.cvec_cached;
            sb.imgs = runtime_.live_imgs;
            sb.cached = runtime_.live_ok && runtime_.o.prompt_cache > 0;
            sb.idle_busy = runtime_.busy_now();
            if (sb.cached) sb.checks = std::move(runtime_.checks);
            if (sb.cached && !slot_draft_keep(slot_b)) {
                runtime_.err.clear();
                sb.cached = false;
            }
        }
    }
    if (in_slot) {   // the solo session and its bookkeeping again
        if (!slot_leave()) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        runtime_.solo_out = nullptr;
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
        runtime_.solo_draft_back(!solo_moved);
    } else {
        runtime_.solo_idle_busy = runtime_.busy_now();
    }
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
    report_request(dc);

    return Flow::next;
}

}  // namespace basalt::program::serve
