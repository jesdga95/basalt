// src/program/serve/request_restore.cpp - request restore operations.
#include "request.hpp"

namespace basalt::program::serve {

// ---- where this request starts reading: the live session, or a checkpoint, whose tokens and pictures are
// exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
bool Request::starts_with(const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) {
    const int64_t L = (int64_t) pre.size();
    if (L < 1 || L > n - 1) return false;
    if (!basalt::core::conversation_ids_equal(pre.data(), pre.size(), ids32.data())) return false;
    return runtime_.imgs_below(runtime_.req_imgs, L) == pre_imgs;
}

Flow Request::restore() {
    remote_before = {};
    launches_before = {};
    compact_before = {};
    full_before = {};
    // ms_begin/ms_wait are cumulative since boot; take deltas here like every other column, so the host time
    // reads as belonging to this request.
    begin_before = {};
    wait_before = {};
    for (int r = 0; r < 3; ++r) if (runtime_.o.expert_cache_remote[(size_t) r] > 0)
    {
        remote_before[(size_t) r] = runtime_.remote_experts[(size_t) r].computed();
        launches_before[(size_t) r] = runtime_.remote_experts[(size_t) r].launched_layers();
        compact_before[(size_t) r] = runtime_.remote_experts[(size_t) r].returned_bytes();
        full_before[(size_t) r] = runtime_.remote_experts[(size_t) r].full_row_bytes();
        begin_before[(size_t) r] = runtime_.remote_experts[(size_t) r].ms_begin();
        wait_before[(size_t) r] = runtime_.remote_experts[(size_t) r].ms_wait();
    }
    runtime_.cur = ids;
    // The conversation cache's keys are int32: one narrow view per request, so a parked entry or a checkpoint
    // is matched by one bulk compare over the slice instead of a per-token scan.
    ids32 = std::vector<int32_t>(ids.begin(), ids.end());
    r0 = Clock::now();
    if (!runtime_.drive.d.usage.empty()) runtime_.sp.route_hist_reset();   // counts this request's prompt rows for the usage seed

    want_cvec = basalt::kernels::cvec().loaded() ? req_cvec != 0 : true;
    // the last request's final commit may still be running on the verifier's stream (set_commit_async):
    // everything below reads, restores or zeroes the session from other streams and the host (the end of the
    // last request waited already; this covers a request that ended on an error path)
    if (!runtime_.ver.wait_commit(runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    // A request for a slot other than 0 reads its prompt in that slot: the request path works on a view of the
    // solo session that carries the slot's state, with the slot's own conversation bookkeeping swapped in
    // for the request (the solo one is put back after it)
    slot_b = runtime_.solo_slot >= 0 ? runtime_.solo_slot : runtime_.admit_slot;
    in_slot = runtime_.slot0_is_solo && slot_b > 0;
    solo_req = in_slot && runtime_.solo_slot > 0;   // decoded here, solo-style, on the slot's own state
    rview = &(runtime_.rviews[(size_t) std::max(slot_b, 0)]);

    solo_moved = false;   // the solo session gave its cells to this request (hold_slot): it holds nothing after
    own_partial = false;
    own_partial_from0 = false;   // the slot holds a read that gave way (BYIELD)
    own_partial_chunk = 0;
    if (in_slot) {
        solo_book = {std::move(runtime_.live), std::move(runtime_.live_imgs), runtime_.live_ok, runtime_.cvec_cached, std::move(runtime_.checks)};
        BSlot& sb = runtime_.bs[(size_t) slot_b];
        runtime_.live.clear();
        runtime_.live_imgs.clear();
        runtime_.checks.clear();
        runtime_.live_ok = false;
        runtime_.cvec_cached = sb.cvec;
        if (sb.cached) {
            runtime_.live = sb.ids;
            runtime_.live_imgs = sb.imgs;
            runtime_.live_ok = runtime_.o.prompt_cache > 0;
            runtime_.checks = std::move(sb.checks);
            if (runtime_.batch_spec && !runtime_.mtp.copy_store(slot_b, runtime_.mtp.main_store(), (int64_t) runtime_.live.size(), runtime_.err)) {
                runtime_.err.clear();
                runtime_.live_ok = false;
            }
            own_partial = sb.partial;
            own_partial_from0 = sb.partial_from0;
            own_partial_chunk = sb.partial_chunk;
        }
        sb.cached = false;
        sb.partial = false;
        sb.checks.clear();
        runtime_.make_view((*rview), slot_b);
        runtime_.rsp = &(*rview);
        if (solo_req && !runtime_.ver.bind_session(&(*rview), runtime_.err)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        runtime_.sp.set_session((*rview));
        if (runtime_.mtp_on) runtime_.mtp.set_ple_session(&(*rview));
    }
    // slot 0 is the solo session: what an idle slot 0 holds is the live conversation, in place
    if (runtime_.slot0_is_solo && !in_slot && !runtime_.bs.empty() && runtime_.bs[0].cached && !runtime_.bs[0].active) {
        BSlot& s0 = runtime_.bs[0];
        runtime_.live = std::move(s0.ids);
        runtime_.live_imgs = std::move(s0.imgs);
        runtime_.live_ok = runtime_.o.prompt_cache > 0;
        runtime_.cvec_cached = s0.cvec;
        for (ConvCheckpoint& c : s0.checks) runtime_.checks.push_back(std::move(c));
        s0.checks.clear();
        s0.cached = false;
        if (runtime_.batch_spec && !runtime_.mtp.copy_store(0, runtime_.mtp.main_store(), (int64_t) runtime_.live.size(), runtime_.err)) {   // the draft K/V with it
            runtime_.err.clear();
            runtime_.live_ok = false;
        }
    }
    resume = 0;
    from_live = false;
    if (runtime_.o.prompt_cache > 0 && want_cvec == runtime_.cvec_cached) {
        if (runtime_.live_ok && starts_with(runtime_.live, runtime_.live_imgs)) { resume = (int64_t) runtime_.live.size(); from_live = true; }
        for (const ConvCheckpoint& c : runtime_.checks)
            if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                resume = (int64_t) c.ids.size();
                from_live = false;
            }
    }
    // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a
    // source too - its sessions are copied back below, so only the new part is read
    slot_source = -1;
    slot_tokens = 0;
    resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
    resumed_chunk = 0;    // its chunk, if it was read beside decoding slots: its first part's plan
    slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
    // the slot this request reads into: its own conversation is `live` already
    own_slot = in_slot ? slot_b : runtime_.slot0_is_solo ? 0 : -1;
    // A decoding slot's checkpoints are sources too: the K/V before a checkpoint never changes, so requests that
    // share a start (a system prompt) read it once, whichever slot holds it.
    if (runtime_.o.prompt_cache > 0)
        for (int b = 0; b < (int) runtime_.bs.size(); ++b) {
            const BSlot& sl = runtime_.bs[(size_t) b];
            if (b == own_slot || (!sl.active && !sl.cached) || sl.cvec != want_cvec) continue;
            if (!in_slot && !sl.active && (int64_t) sl.ids.size() > std::max(resume, slot_tokens) &&
                starts_with(sl.ids, sl.imgs)) {
                slot_source = b;
                slot_tokens = (int64_t) sl.ids.size();
                slot_ck = nullptr;
            }
            for (const ConvCheckpoint& c : sl.checks)
                if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                    slot_source = b;
                    slot_tokens = (int64_t) c.ids.size();
                    slot_ck = &c;
                }
        }
    // a request in another slot: the solo session's chain is slot 0's conversation, a source like the rest
    if (runtime_.o.prompt_cache > 0 && in_slot && runtime_.slot0_is_solo && solo_book.cvec == want_cvec)
        for (const ConvCheckpoint& c : solo_book.checks)
            if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                slot_source = 0;
                slot_tokens = (int64_t) c.ids.size();
                slot_ck = &c;
            }
    parked = runtime_.conversations->best(ids32, runtime_.req_imgs, want_cvec);

    in_tokens = parked.tokens;   // where the incoming image continues: the RAM tier's match or the disk's
    in_live = parked.live;
    // the disk tier, when it holds a longer start than every other source: read back cut at its match, after the
    // RAM guard's wait below (the read takes RAM)
    on_disk = runtime_.disk.enabled() ? runtime_.disk.best(ids32.data(), ids32.size(), runtime_.req_imgs, want_cvec)
                       : basalt::core::ConversationStore::Match{};
    from_disk = on_disk.tokens > std::max({resume, slot_tokens, parked.tokens});
    if (!from_disk && parked.tokens > std::max(resume, slot_tokens)) incoming.emplace(runtime_.conversations->take(parked.index));
    // RAM below the guard's floor (another program's peak): shed (after the conversation this request
    // continues is taken out of the cache), then let it come back before the request starts; a STOP ends the
    // wait, and the request then ends as cancelled.  A request that only continues the live conversation by a
    // few tokens (the server's wrap-up after a thinking budget) is part of a reply already streaming: no wait.
    if (!runtime_.batch_on() && basalt::platform::ram_guard_level() != basalt::platform::RamLevel::Ok) {
        runtime_.shed_if_low();
        if (!basalt::platform::ram_continues_live(from_live && !incoming && !from_disk && slot_source < 0,
                                                   n - resume)) {
            uint64_t avail = 0, floor = 0;
            (void) basalt::platform::ram_guard_level(&avail, &floor);
            basalt::log::info("basalt serve", "RAM low (%llu MiB free < %llu MiB guard): the request waits up "
                              "to %llu s", (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20),
                              (unsigned long long) (basalt::platform::kRamAdmitWaitMs / 1000));
            basalt::core::progress_at("RAM wait");
            const auto w0 = Clock::now();
            auto waited = [&] {
                return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - w0).count();
            };
            while (!runtime_.stop_req.load() && !runtime_.stdin_lost.load() &&
                   basalt::platform::ram_admit_waits(basalt::platform::ram_guard_level(), waited())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                basalt::core::progress_beat();
            }
            // a STOP during the wait ends the request here, before anything moves for it: the conversation it
            // would continue goes back to the cache, the live one stays as it is (a batch slot's request ends
            // in its prompt read, which hands the slot back)
            if ((runtime_.stop_req.load() || runtime_.stdin_lost.load()) && !in_slot && runtime_.admit_slot < 0) {
                if (incoming) (void) runtime_.conversations->put(std::move(*incoming));
                basalt::log::info("basalt serve", "RAM wait: the request was stopped after %.1f s",
                                  waited() / 1000.0);
                basalt::log::info("basalt serve", "prompt %lld tokens = 0 reused + 0 of %lld read in 0 ms "
                                  "(0.0 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, %zu "
                                  "checkpoints (cancelled)", (long long) n, (long long) n, runtime_.checks.size());
                std::printf("DONE 0 %lld 0.0 0.0 cancel 0 0 0 0 0 0 0 0.0 0 0\n", (long long) n);
                std::fflush(stdout);
                return Flow::next;
            }
            const basalt::platform::RamLevel level = basalt::platform::ram_guard_level(&avail, &floor);
            if (level == basalt::platform::RamLevel::Ok)
                basalt::log::info("basalt serve", "RAM back after %.1f s (%llu MiB free): the request starts",
                                  waited() / 1000.0, (unsigned long long) (avail >> 20));
            else if (!runtime_.stop_req.load() && !runtime_.stdin_lost.load())
                basalt::log::warn("basalt serve", "RAM still low after %.1f s (%llu MiB free < %llu MiB guard): "
                                  "the request starts", waited() / 1000.0, (unsigned long long) (avail >> 20),
                                  (unsigned long long) (floor >> 20));
            basalt::core::progress_at("request");
        }
    }
    if (from_disk) {
        const uint64_t need = runtime_.disk.load_bytes(on_disk);
        const uint64_t floor = std::max<uint64_t>((uint64_t) runtime_.o.conversation_cache_min_free_mib << 20, runtime_.disk_floor);
        const auto t0 = Clock::now();
        basalt::core::SavedConversation image;
        std::string de;
        if (!runtime_.capture_admit(need, floor)) {
            basalt::log::info("basalt serve", "conversation cache: %lld tokens on disk not read (physical RAM "
                              "admission; need %llu MiB)", (long long) on_disk.tokens,
                              (unsigned long long) (need >> 20));
        } else if (!runtime_.disk.load(on_disk, image, de)) {
            basalt::log::warn("basalt serve", "conversation cache: reading from disk failed (%s); the file is "
                              "dropped", de.c_str());
        } else {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            basalt::log::info("basalt serve", "conversation cache: read %lld tokens from disk in %.1f ms "
                              "(%.2f GB of state)", (long long) on_disk.tokens, ms, (double) image.bytes() / 1e9);
            incoming.emplace(std::move(image));
            in_tokens = on_disk.tokens;
            in_live = true;
        }
        if (!incoming) {   // not read: the RAM tier's match, when the shed left one
            const auto again = runtime_.conversations->best(ids32, runtime_.req_imgs, want_cvec);
            if (again.tokens > std::max(resume, slot_tokens)) {
                incoming.emplace(runtime_.conversations->take(again.index));
                in_tokens = again.tokens;
                in_live = again.live;
            }
        }
    }
    if (incoming) slot_source = -1;
    // a read that gave way in this slot, the same request again: on with it as if it had not stopped
    if (own_partial && from_live && !incoming && slot_source < 0) {
        resumed_from0 = own_partial_from0;
        resumed_chunk = own_partial_chunk;
    }
    // Reject the entire image before parking/overwriting the outgoing
    // state. Invalid entries can safely fall back to its existing prefix.
    if (incoming && incoming->stage_images.size() != runtime_.stages.size()) {
        basalt::log::info("basalt serve", "conversation cache: discard snapshot (stage count)");
        incoming.reset();
    }
    if (incoming) for (size_t i = 0; i < runtime_.stages.size(); ++i) {
        const basalt::core::OnDevice on(runtime_.stages[i]->dev);
        if (!basalt::core::conversation_snapshot_validate(incoming->stage_images[i], runtime_.stages[i]->ss, runtime_.g,
                nullptr, runtime_.err)) {
            basalt::log::warn("basalt serve", "conversation cache: discard invalid stage snapshot (%s)",
                         runtime_.err.c_str());
            incoming.reset();
            runtime_.err.clear();
            break;
        }
    }
    if (incoming && !basalt::core::conversation_snapshot_validate(*incoming, *runtime_.rsp, runtime_.g,
            runtime_.mtp_kv, runtime_.err)) {
        basalt::log::warn("basalt serve", "conversation cache: discard invalid snapshot (%s)", runtime_.err.c_str());
        incoming.reset();
        runtime_.err.clear();
    }
    // Preserve the outgoing branch before any checkpoint rewind, reset,
    // or incoming restore overwrites the positional state it requires.
    // the elastic KV: every cell this request writes is backed before anything (a restore, a slot's copy, the
    // read) writes it
    // (a restore writes the snapshot's whole live K/V, which can be longer than this prompt)
    kv_need = std::max<int64_t>(n, incoming ? (int64_t) incoming->live.ids.size() : 0);
    keep_routed.emplace(runtime_.kv_keep_routed);
    runtime_.kv_keep_routed = runtime_.batch_on();
    kv_ok = runtime_.kv_fit(*runtime_.rsp, kv_need + runtime_.kKvGrowCells, runtime_.err, true, kv_need + runtime_.kKvWindowCells);
    if (!kv_ok && runtime_.kv_short) {   // the other sessions' spare cells first
        int64_t given = 0;
        std::string ge;
        if (!runtime_.kv_give_back(in_slot ? (int64_t) solo_book.live.size() : (int64_t) runtime_.live.size(), runtime_.rsp,
                          in_slot ? slot_b : -1, nullptr, given, ge)) {
            runtime_.err = ge;
            runtime_.kv_short = false;
        } else if (given > 0) {
            runtime_.err.clear();
            kv_ok = runtime_.kv_fit(*runtime_.rsp, kv_need + runtime_.kKvGrowCells, runtime_.err, true, kv_need + runtime_.kKvWindowCells);
        }
    }
    // the idle slot this request continues: its conversation waits in RAM and its cells come here, so the two
    // copies never hold VRAM at once
    if (!kv_ok && runtime_.kv_short && slot_source >= 0 && !runtime_.bs[(size_t) slot_source].active &&
        (runtime_.bslot_ss[0][(size_t) slot_source].get() != &runtime_.ss || in_slot) &&
        runtime_.bslot_ss[0][(size_t) slot_source]->gdn_state != runtime_.rsp->gdn_state) {
        basalt::core::SessionState& from = *runtime_.bslot_ss[0][(size_t) slot_source];
        const int64_t had = basalt::core::session_kv_cells(from);
        std::string he;
        if (runtime_.hold_slot(slot_source, slot_ck, he) && runtime_.kv_release(from, 0, he)) {
            runtime_.bs[(size_t) slot_source].cached = false;   // its cells are this request's now
            solo_moved = &from == &runtime_.ss;
            basalt::log::info("basalt batch", "slot %d's conversation waits in RAM while its %lld KV cells back "
                              "this request", slot_source, (long long) (had - basalt::core::session_kv_cells(from)));
            runtime_.err.clear();
            kv_ok = runtime_.kv_fit(*runtime_.rsp, kv_need + runtime_.kKvGrowCells, runtime_.err, true, kv_need + runtime_.kKvWindowCells);
        } else {
            runtime_.slot_held = SlotHeld{};
            runtime_.err = he;
            runtime_.kv_short = false;
        }
    }
    if (!kv_ok && runtime_.kv_short) {
        // declined: the engine stays up, the solo session and its bookkeeping as they were
        runtime_.slot_held = SlotHeld{};
        basalt::log::warn("basalt serve", "prompt of %lld tokens refused: %s", (long long) n, runtime_.err.c_str());
        std::printf("ERR %s\n", runtime_.err.c_str());
        std::fflush(stdout);
        if (in_slot) {
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
            if (solo_moved) {
                runtime_.live.clear();
                runtime_.live_imgs.clear();
                runtime_.live_ok = false;
                runtime_.checks.clear();
            }
        }
        runtime_.admit_slot = -1;
        return Flow::next;
    }
    if (!kv_ok) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if ((!from_live || incoming || slot_source >= 0) &&
        !runtime_.park_current(incoming ? incoming->bytes() : 0, incoming || slot_source >= 0 ? 0 : resume)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if (slot_source >= 0) {
        const auto t0 = Clock::now();
        if (!runtime_.copy_from_slot(slot_source, slot_ck, runtime_.err)) {
            // the main session may be half written: read this prompt from the start
            basalt::log::warn("basalt batch", "restoring slot %d failed (%s); reading the prompt",
                         slot_source, runtime_.err.c_str());
            runtime_.err.clear();
            runtime_.slot_held = SlotHeld{};
            runtime_.live.clear();
            runtime_.live_imgs.clear();
            runtime_.checks.clear();
            resume = 0;
            from_live = false;
        } else {
            runtime_.checks.clear();   // the main session's checkpoints were of the conversation it held before
            runtime_.conversations->limit_reuse(0);   // its cells now hold the slot's conversation, not the retained buffers'
            // a read that gave way, the same request again (into its own slot): on with it - its segments and
            // checkpoints as if it had not stopped
            if (slot_ck == nullptr && runtime_.bs[(size_t) slot_source].partial && runtime_.admit_slot == slot_source) {
                resumed_from0 = runtime_.bs[(size_t) slot_source].partial_from0;
                resumed_chunk = runtime_.bs[(size_t) slot_source].partial_chunk;
                for (const ConvCheckpoint& c : runtime_.bs[(size_t) slot_source].checks) runtime_.checks.push_back(c);
                runtime_.bs[(size_t) slot_source].partial = false;
            }
            if (slot_ck != nullptr) {
                runtime_.live = slot_ck->ids;
                runtime_.checks.push_back(*slot_ck);   // the main session's chain has it now
                runtime_.checks.back().used = ++runtime_.check_clock;
            } else {
                runtime_.live = runtime_.bs[(size_t) slot_source].ids;
            }
            runtime_.live_imgs = slot_ck != nullptr ? slot_ck->imgs : runtime_.bs[(size_t) slot_source].imgs;
            resume = slot_tokens;
            from_live = true;
            basalt::log::info("basalt batch", "slot %d gave back %lld tokens of this conversation (%s) in "
                         "%.1f ms", slot_source, (long long) slot_tokens,
                         slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
    }
    restore_t0 = Clock::now();
    if (incoming) {
        // a snapshot that cannot be restored is dropped and the prompt read from the start
        bool restored = basalt::core::conversation_snapshot_restore(*incoming, *runtime_.rsp,
                runtime_.g, runtime_.mtp_kv, runtime_.err) == basalt::core::ConversationRestore::restored;
        int failed_stage = -1;
        for (size_t i = 0; restored && i < runtime_.stages.size(); ++i) {   // the later stages' parts
            const basalt::core::OnDevice on(runtime_.stages[i]->dev);
            if (basalt::core::conversation_snapshot_restore(incoming->stage_images[i], runtime_.stages[i]->ss, runtime_.g,
                    nullptr, runtime_.err) !=
                basalt::core::ConversationRestore::restored) {
                restored = false;
                failed_stage = runtime_.stages[i]->dev;
                break;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR stage sync\n"); return Flow::error; }
        }
        if (!restored) {
            basalt::log::warn("basalt serve", "conversation cache: restoring a parked conversation failed%s "
                         "(%s); reading the prompt from the start",
                         failed_stage >= 0 ? (" on CUDA" + std::to_string(failed_stage)).c_str() : "",
                         runtime_.err.c_str());
            (void) cudaGetLastError();
            if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR restore sync\n"); return Flow::error; }
            runtime_.err.clear();
            incoming.reset();
            runtime_.live.clear();
            runtime_.live_imgs.clear();
            runtime_.checks.clear();
            resume = 0;
            from_live = false;
        }
    }
    if (incoming) {
        runtime_.live = std::move(incoming->live.ids);
        runtime_.live_imgs = std::move(incoming->live.imgs);
        if (runtime_.stages.empty()) {
            runtime_.checks = std::move(incoming->checkpoints);
        } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
            basalt::core::ConversationCheckpointSplit cs;
            cs.stage0 = std::move(incoming->checkpoints);
            for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
            if (!basalt::core::conversation_checkpoints_merge(std::move(cs), runtime_.checks)) runtime_.checks.clear();
        }
        runtime_.cvec_cached = incoming->cvec;
        resume = in_tokens;
        from_live = in_live;
        std::vector<std::vector<basalt::core::ConversationKv>> stage_kv;   // every stage's, for its next park
        for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
        runtime_.conversations->retain(std::move(incoming->kv), int64_t(runtime_.live.size()), std::move(stage_kv), runtime_.rsp);
        incoming.reset();   // its running state and checkpoints are in the session now
        basalt::log::info("basalt serve", "conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu",
                     (long long) resume, from_live ? "live" : "checkpoint",
                     std::chrono::duration<double, std::milli>(Clock::now() - restore_t0).count(),
                     runtime_.conversations->size(), runtime_.conversations->bytes());
    }
    runtime_.ckd_finish();
    if (want_cvec != runtime_.cvec_cached) {
        runtime_.live_ok = false;
        runtime_.checks.clear();
        runtime_.cvec_cached = want_cvec;
    }
    if (basalt::kernels::cvec().loaded()) basalt::kernels::cvec_set_enabled(want_cvec);
    // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
    // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
    runtime_.checks.erase(std::remove_if(runtime_.checks.begin(), runtime_.checks.end(), [&](const ConvCheckpoint& c) {
                     return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                 }), runtime_.checks.end());
    runtime_.live_ok = false;   // until this request has finished, the session is in between
    if (resume == 0) {
        basalt::core::session_zero(*runtime_.rsp, runtime_.g, nullptr, runtime_.main_cs);
        cudaStreamSynchronize(runtime_.main_stream);
        for (auto& st : runtime_.stages) {
            const basalt::core::OnDevice on(st->dev);
            basalt::core::session_zero(st->ss, runtime_.g, nullptr, (void*) st->stream);
            cudaStreamSynchronize(st->stream);
        }
        runtime_.checks.clear();
    } else if (!from_live) {
        ConvCheckpoint* c = nullptr;
        for (ConvCheckpoint& k : runtime_.checks) if ((int64_t) k.ids.size() == resume) c = &k;
        if (c != nullptr) c->used = ++runtime_.check_clock;   // mounting through it is the use LRU counts
        if (c == nullptr || !checkpoint_restore(*c, *runtime_.rsp, runtime_.g) || c->stage_parts.size() != runtime_.stages.size() ||
                   [&] {
                       for (size_t i = 0; i < runtime_.stages.size(); ++i) {
                           const basalt::core::OnDevice on(runtime_.stages[i]->dev);
                           if (!checkpoint_restore(c->stage_parts[i], runtime_.stages[i]->ss, runtime_.g)) return true;
                       }
                       return false;
                   }()) {
            std::printf("ERR restoring a conversation checkpoint failed\n");
            return Flow::error;
        }
    }
    // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
    // host copies and slots are always current (every writer writes both), so they need nothing
    if (runtime_.mtp_on) {
        if (resume > 0) runtime_.mtp.kv_restore(resume);
        runtime_.mtp.set_prompt_len(n);
        runtime_.mtp.begin_request(ids.data(), (int64_t) ids.size());
    }
    read_from = resume;
    runtime_.conversations->drop_reuse_unless(runtime_.rsp);   // another session may have overwritten the cells since
    runtime_.conversations->limit_reuse(read_from);
    runtime_.pp_total = n;
    runtime_.pp_from = read_from;
    runtime_.pp_reached = read_from;
    runtime_.pp_t0 = r0;
    runtime_.pp_next_check = resume + runtime_.o.prompt_cache_every;
    {
        std::lock_guard<std::mutex> lk(runtime_.part_mu);
        runtime_.part_at.clear();
        std::fill(runtime_.part_next.begin(), runtime_.part_next.end(), runtime_.pp_next_check);
    }
    // the cells past this request's room hold nothing it reads (the outgoing conversation is parked by now):
    // a longer one before it gives them back to the expert cache
    if (!runtime_.kv_release(*runtime_.rsp, n + runtime_.kKvGrowCells, runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
    basalt::core::progress_at("reading the prompt, from token", read_from);
    std::fflush(stdout);

    return Flow::proceed;
}

}  // namespace basalt::program::serve
