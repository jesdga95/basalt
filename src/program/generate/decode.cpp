#include "context.hpp"

namespace basalt::program::generate {

int decode(Context& context) {
    auto& o = *context.o;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& arena = *context.arena;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& sbuf = *context.sbuf;
    auto& db = *context.db;
    auto& ple_table = *context.ple_table;
    auto& d_parts = *context.d_parts;
    auto& mtp = *context.mtp;
    auto& n_vocab = *context.n_vocab;
    auto& native_head = *context.native_head;
    auto& d_logits = *context.d_logits;
    auto& src = *context.src;
    auto& srcp = *context.srcp;
    auto& xcache = *context.xcache;
    auto& remote_experts = *context.remote_experts;
    auto& drive = *context.drive;
    auto& routing = *context.routing;
    auto& gr = *context.gr;
    auto& d_emb = *context.d_emb;
    auto& d_next = *context.d_next;
    auto& sp = *context.sp;
    auto& produced = *context.produced;
    auto& total_ms = *context.total_ms;
    auto& prefill_ms = *context.prefill_ms;
    auto& t_start = *context.t_start;
    auto& ttft_ms = *context.ttft_ms;
    auto& n_prompt = *context.n_prompt;
    auto& tok = *context.tok;
    auto& tgraph = *context.tgraph;
    auto& host_res = *context.host_res;
    auto& res_put = context.res_put;
    auto& d_res = *context.d_res;
    auto& thits = *context.thits;
    auto& ms_ple = *context.ms_ple;
    auto& ms_embed = *context.ms_embed;
    auto& ms_layers = *context.ms_layers;
    auto& ms_head = *context.ms_head;
    auto& ms_readback = *context.ms_readback;
    auto& ms_sample = *context.ms_sample;
    auto& phase_tokens = *context.phase_tokens;
    auto& pool = *context.pool;
    auto& spec_pos = *context.spec_pos;

    // ================================ SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t is
    // exactly what greedy decode would emit there, so the first draft that differs ends the round and the round emits
    // (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos >= 0 && (int64_t) produced.size() < o.max_new && !ended) {
        std::vector<int64_t> oracle;
        if (const auto code = cli::load_spec_oracle(o.spec_oracle, oracle)) return *code;
        if (thits.d_res == nullptr) {
            basalt::log::error("basalt generate", "--spec needs the device residency table (an expert profile, "
                                 "--expert-cache and the token graph)");
            return 2;
        }
        basalt::core::Verifier ver;
        basalt::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
        vh.n_slots = xcache.slots();
        ver.set_cpu_warm(drive.d.pool, drive.d.src);
        ver.set_route_tail_skip(o.route_tail_skip);
        ver.set_route_prior(o.route_prior);
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            return fatal_start_error(err);
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            return fatal_start_error(err);
        }
        ver.set_sampling(sp);   // the CLI's own sampling; no penalties here
        if (use_mtp) mtp.begin_request(o.tokens.data(), (int64_t) o.tokens.size());
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        // CS-T: the file tier since the decode began (the prompt path's copies are before this)
        const int64_t files0 = src.file_reads(), ram0 = src.ram_reads();
        const uint64_t fall0 = src.file_read_bytes();
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        cudaStream_t adapt_stream = nullptr;
        if (!drive.d.usage.empty() && cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            basalt::log::error("basalt generate", "cannot create the refill stream");
            return 1;
        }
        // swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        int64_t adapt_rounds = 0;   // counted here: `rounds` is declared below the adapt lambda
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) {
                return;
            }
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            pending.clear();
            if (d_res != nullptr)
                res_put(d_res);
        };
        // The VRAM tier follows the conversation.  Candidates are missing experts routed at least twice (decayed); each
        // is paired with its layer's least-routed resident expert and swapped when it was routed clearly more often.
        // Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            ++adapt_rounds;
            if (!pending.empty()) {
                return true;   // the previous swaps are still in flight
            }
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            const auto copy_back = [&](int32_t layer, int32_t slot, uint8_t* dst) -> bool {
                return cudaMemcpyAsync(dst, xcache.device_slot(slot),
                                       (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                                       cudaMemcpyDeviceToHost, adapt_stream) == cudaSuccess;
            };
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, copy_back,
                                      [&]() { return cudaStreamSynchronize(adapt_stream) == cudaSuccess; })) {
                basalt::log::error("basalt generate", "an adaptive refill failed (copying evicted experts back)");
                return false;
            }
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b, (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                    basalt::log::error("basalt generate", "an adaptive refill failed");
                    return false;
                }
                host_res[out] = basalt::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        basalt::core::DraftCumAdapt cum_adapt;
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // the first window is the last prompt token alone (it produces the first generated token and the MTP's
        // first cell)
        bool first_window = true;
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        if (use_mtp && !mtp.prewarm(err)) {
            basalt::log::error("basalt", "mtp: capturing the draft graphs failed: %s", err.c_str());
            return 1;
        }
        basalt::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        basalt::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        if (o.suffix_draft > 0) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        while ((int64_t) produced.size() < o.max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (use_mtp) {
                T = 1;
                while (T < S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const basalt::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                basalt::log::error("basalt generate", "ran out of context at position %lld", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert ran
            // on the GPU or the CPU (they round differently) would depend on the copy's timing
            apply_pending(true);
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                return fatal_start_error(err);
            }
            if (drive.d.failed) {
                basalt::log::error("basalt generate", "the expert pool failed at layer %lld expert %lld: %s",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            // the adaptive tier's host work (ranking, copy submission) runs on its own thread while the GPU commits and
            // drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                return fatal_start_error(err);
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < o.max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            if (use_mtp && !from_sfx) cum_adapt.observe(dprob.data(), T - 1, a);
            if (use_mtp) mtp.set_cum_th(cum_adapt.th());
            const bool drafted = !use_mtp || (int64_t) produced.size() >= o.max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data());
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                return fatal_start_error(err);
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            if (rounds % 64 == 0)
                basalt::log::info("basalt generate", "position %lld, %lld tokens, %lld rounds", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        // the last commit (set_commit_async) before anything reads the session again
        if (!ver.wait_commit(err)) {
            return fatal_start_error(err);
        }
        cli::print_speculation({
            .o = o, .g = g, .drive = drive,
            .pool = pool, .src = src, .srcp = srcp,
            .ver = ver, .mtp = mtp, .rounds = rounds,
            .drafts_ok = drafts_ok, .drafts_total = drafts_total, .window_hist = window_hist,
            .accepted_hist = accepted_hist, .sfx_windows = sfx_windows, .sfx_ok = sfx_ok,
            .sfx_drafts = sfx_drafts, .misses0 = misses0, .entries0 = entries0,
            .swaps_total = swaps_total, .ram0 = ram0, .files0 = files0,
            .pcie0 = pcie0, .fall0 = fall0, .pool_ms0 = pool_ms0,
            .ms_adapt = ms_adapt, .use_mtp = use_mtp
        });
    }

    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        cli::print_routing_dump(o.dump_routing, drive.calls);
    }
    if (o.stage_timing) basalt::core::stage_timing_report(g.n_layers);

    const double decode_ms = cli::print_generation(o, produced, n_prompt, total_ms, prefill_ms, ttft_ms);
    if (const int rc = basalt::program::measure::write_host_dumps({&o, &g, &ss}, err); rc != 0) return rc;

    cli::print_stats({
        .o = o, .g = g, .drive = drive,
        .pool = pool, .ple_table = ple_table, .srcp = srcp,
        .remote_experts = remote_experts, .xcache = xcache, .tgraph = tgraph,
        .gr = gr, .decode_ms = decode_ms, .phase_tokens = phase_tokens,
        .ms_ple = ms_ple, .ms_embed = ms_embed, .ms_layers = ms_layers,
        .ms_head = ms_head, .ms_readback = ms_readback, .ms_sample = ms_sample
    });

    basalt::core::session_graphs_free(gr);
    basalt::core::doorbell_free(db);
    cudaFree(d_next);
    cudaFree(d_logits);
    cudaFree(d_emb);
    cudaFree(d_parts);
    cudaFree(sbuf);
    cudaFree(arena);
    return 0;
}

}
