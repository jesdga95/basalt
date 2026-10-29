// src/program/serve/verify.cpp - verify operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

basalt::core::Verifier & Runtime::stage_ver(int st) {
    return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
}

int Runtime::pcie_num_of(double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); }

basalt::core::Verifier & Runtime::stage_verb(int st) { return st == 0 ? ver_b : stages[(size_t) st - 1]->ver_b; }

int Runtime::init_verify() {

    d_hist = nullptr;
    hist_stage = std::vector<int32_t>(kHistSlots, -1);
    hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
    if (const basalt::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
        basalt::log::error("basalt serve", "the penalty-history allocation failed");
        return 1;
    }

    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
    vh.n_slots = xcache.slots();

    pair_snap = nullptr;

    n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
    // two stages, the drafter and one conversation at a time: windows are pipelined over two chains of verifiers
    pipe_on = pipe_want && n_stages == 2;

    if (n_stages > 1) {
        const size_t hb = (size_t) basalt::kernels::kVerifyMaxT *
                          (size_t) basalt::core::Verifier::handoff_floats(g) * sizeof(float);
        std::vector<float*> hand((size_t) n_stages - 1, nullptr);
        for (float*& h : hand) {
            float* hh = nullptr;
            if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                basalt::log::error("basalt serve", "the layer-split hand-off allocation failed");
                return 1;
            }
            std::memset(hh, 0, hb);
        }
        split_drive.base = &drive;
        split_drive.n = n_stages;
        for (int st = 0; st < n_stages; ++st) {
            stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                    st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
            split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
            split_drive.cache_base[st] = drive.d.cache_base;
            split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
            split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
        }
        for (int st = 1; st < n_stages; ++st) {
            bool ok_s = false;
            if (split_same) {
                ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err);
            } else {
                GpuStage& gs = *stages[(size_t) st - 1];
                const basalt::core::OnDevice on(gs.dev);
                basalt::core::VerifyHits vs;
                vs.d_res = gs.d_res;
                vs.h_res = host_res.empty() ? nullptr : host_res.data();
                vs.cache_base = gs.cache.device_slot(0);
                vs.blob = thits.blob;
                vs.slot_off = gs.cache.slot_offsets();
                vs.n_slots = gs.cache.slots();
                gs.ver.set_remote_expert_opt(remote_opt.get());
                ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err);
                split_drive.cache_base[st] = gs.cache.device_slot(0);
                split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
            }
            if (!ok_s) {
                basalt::log::error("basalt serve", "layer split, stage %d: %s", st + 1, err.c_str());
                return 1;
            }
        }
        for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
        if (pipe_on) {
            GpuStage& gs = *stages[0];
            float* hb_host = nullptr;
            float* hand_b = nullptr;
            if (cudaHostAlloc((void**) &hb_host, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &hand_b, hb_host, 0) != cudaSuccess ||
                cudaMalloc(&pair_snap, basalt::core::Verifier::pair_snapshot_bytes(g, ss, 0, split_at[0])) != cudaSuccess) {
                basalt::log::error("basalt serve", "the pipelined windows' buffers failed");
                return 1;
            }
            std::memset(hb_host, 0, hb);
            ver_b.set_stage(0, split_at[0], nullptr, hand_b);
            gs.ver_b.set_stage(split_at[0], -1, hand_b, nullptr);
            ver.set_pipelined(pair_snap);
            ver_b.set_pipelined(pair_snap);
            const basalt::core::OnDevice on(gs.dev);
            basalt::core::VerifyHits vs;
            vs.d_res = gs.d_res;
            vs.h_res = host_res.empty() ? nullptr : host_res.data();
            vs.cache_base = gs.cache.device_slot(0);
            vs.blob = thits.blob;
            vs.slot_off = gs.cache.slot_offsets();
            vs.n_slots = gs.cache.slots();
            gs.ver_b.set_remote_expert_opt(remote_opt.get());
            if (!gs.ver_b.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
                basalt::log::error("basalt serve", "pipelined windows, stage 2: %s", err.c_str());
                return 1;
            }
            ver_b.set_next(&gs.ver_b, &split_drive_b);
            split_drive_b = split_drive;
            basalt::log::info("basalt serve", "pipelined windows: on");
        }
        std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
        for (int st = 1; st < n_stages; ++st)
            plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                      " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
        basalt::log::info("basalt serve", "layer split: layers %s, one hand-off per window", plan_s.c_str());
    }
    ver.set_remote_expert_opt(remote_opt.get());
    ver.set_main_priority(o.batch > 0);   // beside a prompt read the slots' windows must win the SMs
    if (peer.valid()) ver.set_peer(&peer);   // the window drives the peer tier itself
    for (const auto& xpe : xpeers)
        if (xpe->valid()) ver.add_peer(xpe.get());
    ver.set_cpu_warm(drive.d.pool, drive.d.src);
    if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err) ||
        (mtp_on && !mtp.bind(wt, multi_gpu ? &mtp_head : &native_head, ver.final_R_all(), err))) {
        basalt::log::error("basalt serve", "%s", err.c_str());
        return 1;
    }
    if (pipe_on && !ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
        basalt::log::error("basalt serve", "pipelined windows: %s", err.c_str());
        return 1;
    }
    // each stage's verifier gets its stage's slot sessions
    if (o.batch > 0) {
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            std::vector<basalt::core::SessionState*> ptrs;
            for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
            basalt::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            if (!vk.init_slots(ptrs, err)) {
                basalt::log::error("basalt serve", "--batch, stage %zu: %s", k + 1, err.c_str());
                return 1;
            }
        }
    }
    for (int st = 0; st < n_stages && n_stages > 1; ++st) {
        split_drive.plan[st] = stage_ver(st).plan_sink();
        if (pipe_on) split_drive_b.plan[st] = stage_verb(st).plan_sink();
    }
    // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
    win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
    win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
    return 0;
}

int Runtime::capture_windows() {

    // every window width's graph is captured before READY, not by the first request that needs it
    if (o.serve && stages.empty() && !ver.warm_graphs(err)) {
        basalt::log::error("basalt serve", "capturing the verify windows failed: %s", err.c_str());
        return 1;
    }
    // the solo windows of every slot session, so a lone request in any slot runs the solo path in place
    if (o.serve && slot0_is_solo && o.batch > 1) {
        const auto tw = Clock::now();
        size_t free0 = 0, total0 = 0;
        cudaMemGetInfo(&free0, &total0);
        for (int b = 1; b < o.batch; ++b) {
            make_view(rviews[(size_t) b], b);
            if (!ver.warm_session(&rviews[(size_t) b], err)) {
                basalt::log::error("basalt serve", "capturing slot %d's solo windows failed: %s", b, err.c_str());
                return 1;
            }
        }
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total0);
        basalt::log::info("basalt serve", "solo windows of %d more slot sessions captured in %.0f ms (%.1f MiB of VRAM)",
                     o.batch - 1, std::chrono::duration<double, std::milli>(Clock::now() - tw).count(),
                     (double) ((int64_t) free0 - (int64_t) free1) / 1048576.0);
    }
    S = o.spec;
    S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
    if (mtp_on && (pipe_want || S_mtp < S)) mtp.set_max_drafts((pipe_on ? basalt::core::kMtpChainMax : S_mtp) - 1);
    if (mtp_on && !mtp.prewarm(err)) {
        basalt::log::error("basalt serve", "capturing the draft graphs failed: %s", err.c_str());
        return 1;
    }
    ple_table.rearm_keepalive();   // a long load must not lapse the SSD keep-alive window before the first request
    // the tiers' placement at READY, which PROBE adapt puts back
    res_at_start = host_res;
    peer_at_start = peer.valid() ? peer.residency() : std::vector<int32_t>();
    prof_rank_start = prof_rank;
    std::printf("INFO file_experts_bytes=%llu\n", (unsigned long long) tier_file_bytes);   // experts only the file holds
    std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
    std::fflush(stdout);

    rounds = 0;
    sfx.emplace(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
    policy.emplace(S);   // MTP or lookup window, learned over the whole process
    // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
    // or more basalt-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
    // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
    // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
    mrope_identity = true;

    // ---- VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache until that
    // much VRAM is free for other programs, or grows it back towards its full size when more than that is free.
    // Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back the cache's
    // LAST segments: the experts in their slots become CPU misses, as any expert outside the cache is (no output
    // changes beyond what a smaller cache decodes; in the resident RAM mode each is first copied to a RAM home),
    // the prompt path's loan moves down to the end of what is left
    // (a smaller chunk when it no longer fits).  A grow maps them again and puts each slot's expert back (or, when
    // the adaptive tier already brought that one back, the most-routed missing expert of the same layer); the
    // adaptive tier carries on from there.
    chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache

    // ---- the elastic KV (kKvElasticFloor): a session's K/V pools back their cells on demand, out of the expert
    // cache's tail.  `kv_fit` backs `cells` of a session before anything writes them: the work in flight lands,
    // the cache gives back its last segments (their experts become CPU misses, read from their RAM copies), and the
    // pools map that VRAM, zeroed.  `kv_release` gives a session's cells past `cells` back and the cache maps its
    // tail again, each returned slot refilled from RAM.  Both run between windows, never inside one.
    // a request's room past its prompt, and each later step (batch slots: a smaller step, as their base)
    kKvGrowCells = o.batch > 0 ? 4096 : 16384;
    kv_moves = 0;

    kv_short = false;   // the last kv_fit refused: the cells would take the cache below its floor
    kv_keep_routed = false;
    return 0;
}

}  // namespace basalt::program::serve
