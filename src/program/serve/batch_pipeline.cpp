// src/program/serve/batch_pipeline.cpp - batch pipeline operations.
#include "request.hpp"

namespace basalt::program::serve {

basalt::core::Verifier & Runtime::stage_verifier(int k) { return k == 0 ? ver : stages[(size_t) k - 1]->ver; }

bool Runtime::pipe_inflight() { for (const PGroup& x : pg) if (x.inflight) return true; return false; }

bool Runtime::group_active(int gi) {
    for (int t = 0; t < GS; ++t) if (bs[(size_t) (gi * GS + t)].active) return true;
    return false;
}

bool Runtime::pump(bool may_start) {
    ++pipe_tick;
    if (bt_windows == 0 && !pipe_inflight()) {
        bt_start = Clock::now();
        bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
        bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
    }
    for (int k = 0; k < n_pipe; ++k) {
        const int gi = stage_group[(size_t) k];
        if (gi < 0) continue;
        basalt::core::Verifier& vk = stage_verifier(k);
        const int r = vk.batch_poll(win_pool_fn, win_pool_user, err);
        if (r < 0) { std::printf("ERR %s\n", err.c_str()); return false; }
        if (r == 0) continue;
        stage_group[(size_t) k] = -1;
        PGroup& G = pg[(size_t) gi];
        if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
        // the last stage: the group's picks
        const int32_t* outb = vk.batch_out();
        for (int t = 0; t < GS; ++t) {
            BSlot& sl = bs[(size_t) (gi * GS + t)];
            if (!sl.active) continue;
            const int32_t y = outb[t];
            std::printf("BT %d %d\n", gi * GS + t, (int) y);
            ++sl.produced;
            ++bt_rows;
            const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
            const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                            : sl.p + 2 > slot_cap[(size_t) (gi * GS + t)] ? "length" : nullptr;
            sl.ids.push_back(sl.x);
            if (fin != nullptr) {
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                std::printf("BDONE %d %lld %s %.1f 0 0\n", gi * GS + t, (long long) sl.produced, fin, ms);
                sl.active = false;
                sl.cached = false;   // the pipeline's pad rows: a pipelined slot is not reused as a cache
            } else {
                sl.x = y;
                sl.p += 1;
            }
        }
        std::fflush(stdout);
        ++bt_windows;
        G.inflight = false;
    }
    // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
    for (int k = n_pipe - 1; k >= 0; --k) {
        if (stage_group[(size_t) k] >= 0) continue;
        int pick = -1;
        if (k > 0) {
            for (int gi = 0; gi < (int) pg.size(); ++gi)
                if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                    (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                    pick = gi;
            if (pick < 0) continue;
        } else {
            if (!may_start) continue;
            for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                const int gi = (int) ((rr + j) % (int64_t) pg.size());
                if (!pg[(size_t) gi].inflight && group_active(gi)) pick = gi;
            }
            if (pick < 0) continue;
            rr = pick + 1;
            PGroup& G = pg[(size_t) pick];
            for (int t = 0; t < GS; ++t) {
                BSlot& sl = bs[(size_t) (pick * GS + t)];
                G.tok[t] = sl.active ? sl.x : 0;
                G.pos[t] = sl.active ? sl.p : 0;
                if (!sl.active) sl.cached = false;   // its pad row writes its state
            }
            G.inflight = true;
            G.stage = 0;
            drive.d.failed = false;
            apply_pending(false);
        }
        PGroup& G = pg[(size_t) pick];
        basalt::core::progress().busy.store(true);
        if (!stage_verifier(k).batch_launch(pick * GS, GS, G.tok, G.pos, err)) {
            std::printf("ERR %s\n", err.c_str());
            return false;
        }
        stage_group[(size_t) k] = pick;
    }
    if (drive.d.failed) { std::printf("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed"); return false; }
    if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: the timing line, as batch_step
        const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
        basalt::log::raw("basalt batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = "
                             "%.1f rows/s (admissions included)\n", o.batch_groups, GS, (long long) bt_windows,
                     (long long) bt_rows, wall, 1000.0 * bt_rows / std::max(wall, 1e-9));
        for (int k = 0; k < n_pipe; ++k) {
            const std::string pr = stage_verifier(k).profile_report();
            if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %d (ms/window):%s\n", k + 1, pr.c_str());
        }
        bt_windows = bt_rows = 0;
        basalt::core::progress().busy.store(false);
    }
    return true;
}

bool Runtime::pipe_drain() {
    while (pipe_inflight())
        if (!pump(false)) return false;
    return true;
}

}  // namespace basalt::program::serve
