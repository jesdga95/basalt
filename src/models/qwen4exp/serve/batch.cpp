// src/models/qwen4exp/serve/batch.cpp - the batch slots' rounds on Flash-Next: the shared rounds (batch_round.cpp) over
// the slots' sessions (slot_session.cpp), with the elastic KV and the adaptive tier between rounds.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_state.hpp"

namespace basalt::models::qwen4exp::serve {

uint64_t QfnRuntime::row_hash(int t) {
    lh_buf.resize((size_t) n_vocab);
    if (!ver.copy_logits(t, lh_buf.data())) return 0;
    uint64_t h = 1469598103934665603ull;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(lh_buf.data());
    for (size_t i = 0; i < lh_buf.size(); ++i) { h ^= w[i]; h *= 1099511628211ull; }
    return h;
}

// one round of the active slots (the shared rounds): an idle slot is not touched, so it keeps the conversation it holds
bool QfnRuntime::next_batch_window() {
    constexpr int MR = basalt::kernels::kVerifyMaxT;
    // the elastic KV: each slot's next cells backed ahead of its windows; beside a prompt read only from free
    // VRAM, and a slot that needs more than that waits for the read to end
    kv_wait.assign(bs.size(), 0);
    if (kv_floor > 0)
        for (int b = 0; b < (int) bs.size(); ++b) {
            if (!bs[(size_t) b].active) continue;
            basalt::core::SessionState& u = *bslot_ss[0][(size_t) b];
            const int64_t backed = basalt::core::session_kv_cells(u), at = bs[(size_t) b].p;
            if (at + 2 * MR + 64 <= backed) continue;
            std::string ke;
            if (!ov_active && ba_job) { ba_thr.join(); ba_job = false; }   // its copies land before the drain
            if (kv_fit(u, at + kKvGrowCells, ke, !ov_active, at + 2 * MR + 64)) continue;
            if (!ke.empty() && kv_short) {   // the expert cache's floor: this slot's reply ends here
                BSlot& sl = bs[(size_t) b];
                basalt::log::warn("basalt batch", "slot %d stopped at %lld cells: %s", b, (long long) at, ke.c_str());
                std::printf("BDONE %d %lld length %.1f %lld %lld %lld\n", b, (long long) sl.produced,
                            std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                            (long long) sl.drafts_accepted, (long long) sl.drafts_offered, (long long) sl.windows);
                sl.active = false;
                sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids
                sl.idle_busy = busy_now();
                sl.guard.reset();
                kv_wait[(size_t) b] = 1;
                continue;
            }
            if (!ke.empty()) { std::printf("ERR %s\n", ke.c_str()); return false; }
            if (at + 2 * MR > backed) kv_wait[(size_t) b] = 1;
        }
    bool any = false;
    for (int b = 0; b < (int) bs.size(); ++b) any = any || (bs[(size_t) b].active && !kv_wait[(size_t) b]);
    if (!any) return true;
    const bool was_busy = basalt::core::progress().busy.load();
    basalt::core::progress().busy.store(true);
    if (!ov_active) apply_pending(false);
    if (batch_adapt && !ov_active) {
        // the tier follows what all the slots' rows route: a round every --adapt-every windows.  Its steps
        // are taken once their copies are done; the copies out of the RAM tier (which may read the experts
        // file) are queued from a worker thread beside the windows, so no window waits for one.
        const Clock::time_point ta = Clock::now();
        bool ok = true, acted = false;
        ++ba_win;
        if (ba_job && ba_thr.finished()) { ba_thr.join(); ba_job = false; }
        if (!ba_job) {
            if (xq_stage != 0) {
                if (cudaEventQuery(adapt_ev) == cudaSuccess) {
                    const int was = xq_stage;
                    ok = xq_advance(true);
                    if (ok && was == 1 && xq_stage == 2) {
                        ba_thr.run([&] { ver.flush_post_launch(); });
                        ba_job = true;
                    }
                    acted = true;
                }
            } else if (ba_win % o.adapt_every == 0) {
                const int64_t np0 = xq_noplace;
                ok = xq_round();
                ba_swaps += (int64_t) xq.size();
                ba_noplace += xq_noplace - np0;
                ++ba_rounds;
                acted = true;
            }
        }
        std::string perr;
        if (ok && acted && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
        if (!ok) { std::printf("ERR an adaptive refill failed\n"); return false; }
        ba_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
        if (ba_win % 2000 == 0 && ba_win > 0) {   // the swap rate and the hit rate over the last 2000 windows
            const int64_t h = drive.d.cache_hits, m = drive.d.multi_misses;
            basalt::log::info("basalt batch adapt", "window %lld: %lld swaps in %lld rounds, VRAM hits %.2f and CPU "
                                 "experts %.2f per layer-window", (long long) ba_win, (long long) (ba_swaps - ba_swaps_l),
                         (long long) (ba_rounds - ba_rounds_l), (double) (h - ba_h_l) / (2000.0 * g.n_layers),
                         (double) (m - ba_m_l) / (2000.0 * g.n_layers));
            ba_swaps_l = ba_swaps; ba_rounds_l = ba_rounds; ba_h_l = h; ba_m_l = m;
        }
    }
    if (bt_windows == 0) {
        bt_start = Clock::now();
        bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
        bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
    }
    bool ok = batch_round();
    std::string ce;
    if (ok && !commit_settle(ce)) {   // a window whose slots all ended leaves its commit to here
        std::printf("ERR %s\n", ce.c_str());
        ok = false;
    }
    basalt::core::progress().busy.store(was_busy);
    if (!ok) return false;
    stats_tick();   // also when a prompt read runs these windows beside it, outside the command loop
    if (!batch_on() && bt_windows > 0) {
        const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
        const double L = (double) g.n_layers;
        basalt::log::info("basalt batch", "%lld windows, avg %.2f rows, run %.2f ms/window (CUDA0 GPU-reach wait %.2f + "
                             "CPU experts %.2f) + commit %.2f + draft %.2f; per layer-window: CPU experts %.2f, VRAM hits "
                             "%.2f, PCIe %.2f; %.1f rows/s over %.0f ms of wall time (admissions included)",
                     (long long) bt_windows, bt_rows / w, bt_run / w, (ver.ms_wait - bt_wait0) / w,
                     (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_draft / w, (drive.d.multi_misses - bt_miss0) / (w * L),
                     (drive.d.cache_hits - bt_hits0) / (w * L), (drive.d.pcie_experts - bt_pcie0) / (w * L),
                     1000.0 * bt_rows / std::max(wall, 1e-9), wall);
        if (batch_adapt) {
            basalt::log::info("basalt batch adapt", "%lld rounds, %lld swaps queued (%.1f per round), %lld deferred "
                                 "(no free RAM place), %.3f ms/window of host time, peer swaps total %lld",
                         (long long) ba_rounds, (long long) ba_swaps,
                         ba_rounds > 0 ? (double) ba_swaps / (double) ba_rounds : 0.0,
                         (long long) ba_noplace, ba_ms / w,
                         (long long) (peer.valid() ? peer.swaps() : 0));
            ba_swaps = ba_rounds = ba_swaps_l = ba_rounds_l = ba_noplace = 0;
            ba_ms = 0;
        }
        if (bt_slotwin > 0)
            basalt::log::info("basalt batch spec", "%.2f tokens/window/slot (%.2f slots/window, %.2f rows/window), "
                                 "drafts %lld accepted of %lld, draft %.2f ms/window (%.2f ms/slot-window), %.1f "
                                 "tokens/s", (double) bt_tokens / (double) bt_slotwin, (double) bt_slotwin / w,
                         bt_rows / w, (long long) bt_dr_acc, (long long) bt_dr_off, bt_draft / w,
                         bt_draft / (double) bt_slotwin, 1000.0 * (double) bt_tokens / std::max(wall, 1e-9));
        for (size_t k = 0; k <= stages.size(); ++k) {
            const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
            if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
        }
        bt_run = bt_commit = bt_draft = 0;
        bt_windows = bt_rows = bt_tokens = bt_slotwin = bt_dr_off = bt_dr_acc = 0;
    }
    return true;
}

// One window of the slots' segments on the verifier: segment i is reqs[i] in slot ss[i]'s state.  The last window's
// commit lands first; each row's pick goes to win_out.
bool QfnRuntime::slot_windows(QfnSlotSession* const* ss, const basalt::model::DecodeRequest* reqs, int n, std::string& e) {
    constexpr int MR = basalt::kernels::kVerifyMaxT;
    if (!commit_settle(e)) return false;
    int slots[MR] = {}, lens[MR] = {};
    int64_t pos[MR] = {};
    int32_t tok[MR] = {};
    int R = 0, pen_h = 0;
    if (n < 1 || n > MR) { e = "a batch window of " + std::to_string(n) + " segments"; return false; }
    for (int i = 0; i < n; ++i) {
        const basalt::model::DecodeRequest& r = reqs[i];
        if (r.n_tokens < 1 || R + r.n_tokens > MR) { e = "a batch window of more than " + std::to_string(MR) + " rows"; return false; }
        QfnSlotSession& s = *ss[i];
        slots[i] = s.b_;
        lens[i] = r.n_tokens;
        pos[i] = r.pos;
        s.r0_ = R;
        s.rows_ = r.n_tokens;
        s.pos0_ = r.pos;
        s.own_ = r.drawn_drafts;
        std::copy(r.tokens, r.tokens + r.n_tokens, tok + R);
        R += r.n_tokens;
        if (r.history != nullptr) pen_h = std::max(pen_h, r.history_len);
    }
    // the penalty histories, one row of pen_h per window row: a segment's rows as its window gave them, -1 in front (the
    // sampler reads only a row's last penalty_last_n); a window without penalties hands the sampler no buffer
    if (pen_h > 0) {
        for (int i = 0; i < n; ++i) {
            const basalt::model::DecodeRequest& r = reqs[i];
            const int hl = r.history != nullptr ? r.history_len : 0;
            for (int j = 0; j < r.n_tokens; ++j) {
                int32_t* row = hist_stage.data() + (size_t) (ss[i]->r0_ + j) * (size_t) pen_h;
                std::fill(row, row + (pen_h - hl), -1);
                if (hl > 0) std::copy(r.history + (size_t) j * hl, r.history + (size_t) (j + 1) * hl, row + (pen_h - hl));
            }
        }
        const basalt::core::OnDevice on_h(hist_dev);
        if (cudaMemcpy(d_hist, hist_stage.data(), (size_t) R * (size_t) pen_h * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            e = "batch: the penalty histories could not be staged";
            return false;
        }
    }
    ver.set_batch_history(pen_h > 0 ? d_hist : nullptr, pen_h);
    drive.d.layers = 0;
    drive.d.experts = 0;
    drive.d.failed = false;
    const Clock::time_point w0 = Clock::now();
    if (!ver.run_slot_segs(slots, lens, n, tok, pos, win_pool_fn, win_pool_user, win_out, e) || drive.d.failed) {
        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
        return false;
    }
    win_rows = R;
    win_segs = n;
    bt_run += std::chrono::duration<double, std::milli>(Clock::now() - w0).count();
    ++bt_windows;
    bt_rows += R;
    return true;
}

// The last window's commit, when it ran beside the drafts, lands (once).
bool QfnRuntime::commit_settle(std::string& e) {
    if (!commit_open) return true;
    commit_open = false;
    const Clock::time_point c0 = Clock::now();
    const bool ok = ver.commit_slots_end(e);
    bt_commit += std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
    return ok;
}

int QfnRuntime::init_batch() {
    bs = std::vector<BSlot>((size_t) std::max(o.batch, 0));
    slot_sampling.assign(bs.size(), basalt::kernels::SamplerParams{});
    slot_ss.clear();
    for (int b = 0; b < (int) bs.size(); ++b) slot_ss.push_back(std::make_unique<QfnSlotSession>(*this, b));
    kv_wait.assign(bs.size(), 0);
    commit_open = false;
    // the real MTP drafts in the batch windows (a draft K/V per slot); the scripted test drive keeps its own
    batch_spec = mtp_on && mtp.main_store() > 0 && o.batch > 0;
    S_all = S;
    bpolicy.emplace(S, 0.03, false);   // the lookup-or-MTP choice, costs per row
    bt_draft = 0;
    bt_slotwin = 0;
    bt_dr_off = 0;
    bt_dr_acc = 0;
    // concur9_logit_hash: a hash of the head logits of every accepted row, on stderr (solo and batch alike)
    logit_hash = basalt::core::exp_flag("concur9_logit_hash", 0) != 0;

    // adaptive tier rounds in the batch windows (routing counts from every row of every slot)
    batch_adapt = adapt_fast && !drive.d.usage.empty();
    ba_win = 0;
    ba_swaps = 0;
    ba_rounds = 0;
    ba_swaps_l = 0;
    ba_rounds_l = 0;
    ba_h_l = 0;
    ba_m_l = 0;
    ba_noplace = 0;
    ba_ms = 0;
    ba_job = false;
    // timing of the batch windows since the slots were last all idle (one stderr line then)
    bt_run = 0;
    bt_commit = 0;
    bt_wait0 = 0;
    bt_pool0 = 0;
    bt_miss0 = 0;
    bt_hits0 = 0;
    bt_pcie0 = 0;
    bt_windows = 0;
    bt_rows = 0;
    bt_tokens = 0;
    bt_start = Clock::now();
    admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
    solo_slot = -1;                ///< SGEN: the slot a solo request runs in, on that slot's own state
    admit_max_new = 0;

    // a prompt read on its own thread beside the slots' windows: the tier stays as it is meanwhile
    ov_active = false;
    ov_post_done = false;
    return 0;
}

}  // namespace basalt::models::qwen4exp::serve
