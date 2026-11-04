// src/models/qwen4exp/serve/qfn_restore.cpp - what restoring a conversation does to Flash-Next's sessions.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"

namespace basalt::models::qwen4exp::serve {

bool QfnRequest::cvec_wanted() const { return basalt::kernels::cvec().loaded() ? req_cvec != 0 : true; }
void QfnRequest::set_cvec() { if (basalt::kernels::cvec().loaded()) basalt::kernels::cvec_set_enabled(want_cvec); }
void QfnRequest::select_view() { rview = &(runtime_.rviews[(size_t) std::max(slot_b, 0)]); }
bool QfnRequest::slot_draft_in(int slot) {
    if (runtime_.batch_spec && !runtime_.mtp.copy_store(slot, runtime_.mtp.main_store(), (int64_t) runtime_.live.size(), runtime_.err)) {
        runtime_.err.clear();
        return false;
    }
    return true;
}
void QfnRequest::drop_stale_reuse() { runtime_.conversations->drop_reuse_unless(runtime_.rsp); }
bool QfnRequest::release_room() { return runtime_.kv_release(*runtime_.rsp, n + runtime_.kKvGrowCells, runtime_.err); }
bool QfnRequest::checkpoint_mount(const ConvCheckpoint& c) {
    if (!checkpoint_restore(c, *runtime_.rsp, runtime_.g) || c.stage_parts.size() != runtime_.stages.size()) return false;
    for (size_t i = 0; i < runtime_.stages.size(); ++i) {
        const basalt::core::OnDevice on(runtime_.stages[i]->dev);
        if (!checkpoint_restore(c.stage_parts[i], runtime_.stages[i]->ss, runtime_.g)) return false;
    }
    return true;
}

bool QfnRequest::restore_begin() {
    if (!runtime_.drive.d.usage.empty()) runtime_.sp.route_hist_reset();   // counts this request's prompt rows for the usage seed
    // the last request's final commit may still be running on the verifier's stream (set_commit_async):
    // everything below reads, restores or zeroes the session from other streams and the host (the end of the
    // last request waited already; this covers a request that ended on an error path)
    if (!runtime_.ver.wait_commit(runtime_.err)) {
        return false;
    }
    return true;
}

void QfnRequest::slot_draft_out() {
        // this request drafts on the main sequence: the solo conversation's draft K/V waits in slot 0's store (the
        // solo session's) until the request ends, unless that store holds slot 0's own cached conversation
        if (runtime_.batch_spec && !runtime_.solo_draft_in0 && runtime_.live_ok && !runtime_.live.empty() &&
            !runtime_.bs[0].cached) {
            if (runtime_.mtp.copy_store(runtime_.mtp.main_store(), 0, (int64_t) runtime_.live.size(), runtime_.err))
                runtime_.solo_draft_in0 = true;
            runtime_.err.clear();
        }
}

Flow QfnRequest::slot_bind() {
        runtime_.make_view((*rview), slot_b);
        runtime_.rsp = &(*rview);
        if (solo_req && !runtime_.ver.bind_session(&(*rview), runtime_.err)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        runtime_.sp.set_session((*rview));
        if (runtime_.mtp_on) runtime_.mtp.set_ple_session(&(*rview));
    return Flow::proceed;
}

void QfnRequest::validate_incoming() {
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
}

Flow QfnRequest::make_room() {
    // the elastic KV: every cell this request writes is backed before anything (a restore, a slot's copy, the
    // read) writes it
    // (a restore writes the snapshot's whole live K/V, which can be longer than this prompt)
    kv_need = std::max<int64_t>(n, incoming ? (int64_t) incoming->live.ids.size() : 0);
    runtime_.park_keep = slot_source;   // the conversation this request continues stays where it is
    keep_routed.emplace(runtime_.kv_keep_routed);
    runtime_.kv_keep_routed = runtime_.batch_on();
    runtime_.peer_due.clear();
    runtime_.peer_due_off = runtime_.peer_due_lent_n = 0;
    defer_peer.emplace(runtime_.peer_defer);
    runtime_.peer_defer = !in_slot;
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
            if (solo_moved) runtime_.solo_out = nullptr;
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
            runtime_.solo_out = nullptr;
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
            runtime_.solo_draft_back(!solo_moved);
        }
        runtime_.admit_slot = -1;
        return Flow::next;
    }
    if (!kv_ok) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    return Flow::proceed;
}

Flow QfnRequest::restore_image() {
        // the draft layer's K/V goes to the main sequence, which then holds it as written
        if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        // a snapshot that cannot be restored is dropped and the prompt read from the start
        bool restored = basalt::core::conversation_snapshot_restore(*incoming, *runtime_.rsp,
                runtime_.g, runtime_.mtp_kv, runtime_.err) == basalt::core::ConversationRestore::restored;
        if (restored && runtime_.mtp_on) runtime_.mtp.wrote(0, (int64_t) incoming->live.ids.size());
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
    return Flow::proceed;
}

void QfnRequest::adopt_checkpoints() {
        if (runtime_.stages.empty()) {
            runtime_.checks = std::move(incoming->checkpoints);
        } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
            basalt::core::ConversationCheckpointSplit cs;
            cs.stage0 = std::move(incoming->checkpoints);
            for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
            if (!basalt::core::conversation_checkpoints_merge(std::move(cs), runtime_.checks)) runtime_.checks.clear();
        }
}

void QfnRequest::retain_image() {
        std::vector<std::vector<basalt::core::ConversationKv>> stage_kv;   // every stage's, for its next park
        for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
        runtime_.conversations->retain(std::move(incoming->kv), int64_t(runtime_.live.size()), std::move(stage_kv), runtime_.rsp);
}

bool QfnRequest::session_reset() {
        basalt::core::session_zero(*runtime_.rsp, runtime_.g, nullptr, runtime_.main_cs);
        cudaStreamSynchronize(runtime_.main_stream);
        for (auto& st : runtime_.stages) {
            const basalt::core::OnDevice on(st->dev);
            basalt::core::session_zero(st->ss, runtime_.g, nullptr, (void*) st->stream);
            cudaStreamSynchronize(st->stream);
        }
    return true;
}

void QfnRequest::draft_begin() {
    // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
    // host copies and slots are always current (every writer writes both), so they need nothing
    if (runtime_.mtp_on) {
        if (resume > 0) runtime_.mtp.kv_restore(resume);
        runtime_.mtp.set_prompt_len(n);
        runtime_.mtp.begin_request(ids.data(), (int64_t) ids.size());
    }
}

void QfnRequest::prompt_parts_reset() {
    runtime_.pp_next_check = resume + runtime_.o.prompt_cache_every;
    {
        std::lock_guard<std::mutex> lk(runtime_.part_mu);
        runtime_.part_at.clear();
        std::fill(runtime_.part_next.begin(), runtime_.part_next.end(), runtime_.pp_next_check);
    }
}

}  // namespace basalt::models::qwen4exp::serve
