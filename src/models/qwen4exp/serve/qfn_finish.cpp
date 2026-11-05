// src/models/qwen4exp/serve/qfn_finish.cpp - what the end of a request does to Flash-Next's verifiers, drafter and tiers.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"

namespace basalt::models::qwen4exp::serve {

Flow QfnRequest::complete_begin() {
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
    return Flow::proceed;
}

Request::DoneCounts QfnRequest::done_counts() {
    req_hits = runtime_.drive.d.cache_hits - decode_hits0;
    req_look = (runtime_.drive.d.cache_hits + runtime_.drive.d.cache_admitted + runtime_.drive.d.cache_refused) - decode_look0;
    req_offload = runtime_.drive.d.offload_entries - offload0;
    DoneCounts dc;
    dc.hits = req_hits;
    dc.lookups = req_look;
    dc.ram_blobs = runtime_.src.ram_reads() - ram0;
    dc.file_blobs = runtime_.src.file_reads() - files0;
    dc.file_mb = (double) (runtime_.src.file_read_bytes() - file_bytes0) / 1e6;
    dc.offloaded = req_offload;
    return dc;
}

bool QfnRequest::tiers_settle() {
    if (!runtime_.xq_drain()) return false;
    runtime_.xq_saturated = false;   // the next request starts at the base cap
    return true;
}

// the draft layer's cell for the prompt's last token (it pairs with the first generated one)
bool QfnRequest::admit_draft() {
    return !runtime_.batch_spec || runtime_.mtp.prefill(runtime_.ver.final_R_all(), &x, 1, p - 1, runtime_.err);
}

void QfnRequest::slot_state_back(int slot) {
    basalt::core::SessionState& t = *runtime_.bslot_ss[0][(size_t) slot];
    t.ple_prev[0] = (*rview).ple_prev[0];
    t.ple_prev[1] = (*rview).ple_prev[1];
    t.ple_token = (*rview).ple_token;
}

bool QfnRequest::slot_draft_keep(int slot) {
    return !runtime_.batch_spec || runtime_.mtp.copy_store(runtime_.mtp.main_store(), slot, (int64_t) runtime_.live.size(), runtime_.err);
}

void QfnRequest::slot_set_sampling(int slot) { runtime_.ver.set_slot_sampling(slot, req_sp); }

void QfnRequest::slot_init(BSlot& sl) {
    sl.script = std::move(runtime_.bscript[(size_t) runtime_.admit_slot].first);
    sl.rng.seed(runtime_.bscript[(size_t) runtime_.admit_slot].second);
    runtime_.bscript[(size_t) runtime_.admit_slot].first.clear();
    if (runtime_.batch_spec && runtime_.o.suffix_draft > 0) {
        sl.sfx = std::make_unique<basalt::spec::SuffixDrafter>(std::max(1, runtime_.o.suffix_draft), 64,
                                                               (size_t) runtime_.o.max_context + 4096);
        for (int32_t t : runtime_.live) sl.sfx->append(t);
        sl.sfx->append(x);
    }
}

bool QfnRequest::slot_leave() {
    if (solo_req && !runtime_.ver.bind_session(&runtime_.ss, runtime_.err)) return false;
    runtime_.rsp = &runtime_.ss;
    runtime_.sp.set_session(runtime_.ss);
    if (runtime_.mtp_on) runtime_.mtp.set_ple_session(&runtime_.ss);
    return true;
}

void QfnRequest::report_request(const DoneCounts& dc) {
    if (runtime_.drive.routing != nullptr) std::fflush(runtime_.drive.routing);   // the routing trace survives a crash and is watchable mid-session
    if (relax_on)
        basalt::log::info("basalt serve", "relaxed drafting %.2f: %lld of the %lld accepted drafts were kept by it",
                     (double) req_relax, (long long) relaxed_kept, (long long) draft_accepted);
    report_expert_counts(dc);
    // the adaptive swaps this request queued: the resident RAM line's exchanges count only swaps through the
    // exchange buffers, not those whose evicted expert takes a free place in the RAM copy
    if (runtime_.adapt_fast && !runtime_.drive.d.usage.empty())
        basalt::log::info("basalt serve", "adaptive swaps: %lld into the VRAM tier this request, %lld deferred "
                          "(no free RAM place)", (long long) (runtime_.xq_swapped - swapped0),
                          (long long) (runtime_.xq_noplace - noplace0));
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
}

}  // namespace basalt::models::qwen4exp::serve
