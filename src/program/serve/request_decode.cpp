// src/program/serve/request_decode.cpp - request decode operations.
#include "request.hpp"

namespace basalt::program::serve {

Request::DecSnap Request::dec_snap() {
    return DecSnap{runtime_.ver.ms_wait, runtime_.ver.ms_pool, runtime_.ver.ms_host, runtime_.drive.d.ms_plan, runtime_.drive.d.ms_actq, runtime_.drive.d.ms_jobs,
                   runtime_.drive.d.ms_run, runtime_.drive.d.peer != nullptr ? runtime_.drive.d.peer->ms_wait : 0.0,
                   runtime_.drive.d.multi_misses, runtime_.drive.d.multi_entries, runtime_.drive.d.cache_hits, runtime_.drive.d.pcie_experts,
                   runtime_.drive.d.peer_entries};
}

// the verify windows: the first holds the last prompt token, with the rest of the last window
Flow Request::decode() {
    if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }   // the windows between prompt chunks swapped it
    p = fold_p0;
    x = (int32_t) ids[(size_t) fold_p0];
    {   // a request decoded here starts its adaptation from the experts its own prompt routed to
        std::vector<uint32_t> hist;
        int64_t hist_tokens = 0;
        const bool seeded = !runtime_.drive.d.usage.empty() && runtime_.sp.route_hist_take(hist, hist_tokens) && (!in_slot || solo_req);
        // a split read counted its stages' layers there
        if (seeded && runtime_.split_hist.size() == hist.size())
            for (size_t i = 0; i < hist.size(); ++i) hist[i] += runtime_.split_hist[i];
        runtime_.split_hist.clear();
        if (seeded) seed_usage_from_prompt(runtime_.drive.d.usage, hist, hist_tokens, runtime_.host_res, runtime_.g.n_expert, runtime_.peer);
    }
    drafts = std::vector<int32_t>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0);
    window = std::vector<int32_t>((size_t) runtime_.S);
    outv = std::vector<int32_t>((size_t) runtime_.S);
    dprob = std::vector<float>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0.0f);
    sbuf = std::vector<int32_t>((size_t) runtime_.S, 0);
    if (runtime_.o.suffix_draft > 0) {
        runtime_.sfx->reset();
        for (int64_t t : ids) runtime_.sfx->append((int32_t) t);
    }
    first_window = true;
    produced_n = 0;
    sfx_windows = 0;
    sfx_drafts = 0;
    sfx_ok = 0;
    draft_offered = 0;
    draft_accepted = 0;

    consumed.reserve((size_t) (n + max_new + runtime_.S));
    for (int64_t i = 0; i < fold_p0; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
    finish = "length";
    // pipelined windows: `cv` runs the next window, `ov` is the other chain; a guess that held leaves the next
    // window's verdict in `held`
    cv = &runtime_.ver;
    ov = &runtime_.ver_b;

    held.win.assign((size_t) runtime_.S, 0);
    held.out.assign((size_t) runtime_.S, 0);
    win_b = std::vector<int32_t>((size_t) runtime_.S, 0);
    n_chain = 0;                       // drafts the last chain produced
    full_ema = 0.5;                 // how often windows are accepted whole
    pair_hint = false;                // windows are accepted whole often enough to run the chain on for a guess
    pair_req = runtime_.pipe_on && ppl == nullptr && !relax_on && gguard == nullptr && hist_n == 0 && !in_slot &&
                          runtime_.kv_floor == 0;
    if (runtime_.pipe_on) runtime_.mtp.set_max_drafts((pair_req ? basalt::core::kMtpChainMax : runtime_.S_mtp) - 1);
    pair_runs = 0;
    pair_holds = 0;
    d0 = Clock::now();
    // --decode-timing: where a request's decode time goes (one line per request)
    dec_timing = basalt::core::diagnostics().decode_timing;

    ds0 = dec_snap();
    dt_run = 0;
    dt_commit = 0;
    dt_draft = 0;
    dec_windows = 0;
    dec_T = 0;
    decode_hits0 = runtime_.drive.d.cache_hits;
    swapped0 = runtime_.xq_swapped;
    noplace0 = runtime_.xq_noplace;
    // the RAM and file tiers of this request (the mmap source; 0 with the arena)
    ram0 = runtime_.src.ram_reads();
    files0 = runtime_.src.file_reads();
    file_bytes0 = runtime_.src.file_read_bytes();
    decode_look0 = runtime_.drive.d.cache_hits + runtime_.drive.d.cache_admitted + runtime_.drive.d.cache_refused;
    offload0 = runtime_.drive.d.offload_entries;
    static const bool chain = basalt::core::Verifier::chain_on();
    adapt_age = 2;   // loop tops since the last adaptive round
    applied_early = false;

    copy_live = false;
    copy_done = false;   // exchanges applied early whose RAM copies are under way
    if (cancelled) finish = "cancel";
    // --draft-log FILE: one line per window (T, accepted, draft probabilities, ms) for draft-policy study
    static FILE* const draft_log = basalt::core::diagnostics().draft_log.empty() ? nullptr
        : std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
    if (draft_log != nullptr) {
        std::fprintf(draft_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp);
        std::fflush(draft_log);
    }
    // spec_log: per window, the target's sampling distribution at each verified row beside the draft head's top-4
    // of the draft that row checks, for offline tree-draft studies (written next to --draft-log as FILE.tk)
    static FILE* const tk_log = basalt::core::exp_flag("spec_log", 0) != 0 && !basalt::core::diagnostics().draft_log.empty()
        ? std::fopen((basalt::core::diagnostics().draft_log + ".tk").c_str(), "a") : nullptr;

    tk_valid = false;
    if (tk_log != nullptr) { std::fprintf(tk_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp); std::fflush(tk_log); }
    while (!cancelled && produced_n < max_new && (!in_slot || solo_req)) {   // (a batch slot's first token comes from the batch window)
        int T = 1;
        if (held.on) T = held.T;
        else
        while (T < runtime_.S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
        if (first_window) T = (int) (n - fold_p0);
        const int e0 = first_window ? T - 1 : 0;   // rows before it are prompt tokens: forced, not emitted
        if (ppl != nullptr) {
            if (p >= ppl->end()) break;
            T = (int) std::min<int64_t>(req_ppl_t, ppl->end() - p);
        }
        // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
        // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
        bool from_sfx = false;
        int sfx_match = 0;
        if (runtime_.o.suffix_draft > 0 && !first_window && ppl == nullptr && !held.on) {
            const int k = runtime_.sfx->propose(runtime_.S - 1, sbuf.data());
            sfx_match = runtime_.sfx->last_match();
            if (k > 0 && sbuf[0] == drafts[0]) {
                const basalt::spec::DraftPolicy::Pick pk = runtime_.policy->choose(T, k, sfx_match);
                if (pk.lookup) { T = pk.t; from_sfx = true; }
            }
        }
        const bool timed_round = !first_window;
        const Clock::time_point round0 = Clock::now();
        if (p + T > runtime_.o.max_context) break;
        window[0] = x;
        for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
        if (held.on)
            for (int i = 0; i < T; ++i) window[(size_t) i] = held.win[(size_t) i];
        if (e0 > 0)
            for (int i = 1; i < T; ++i) window[(size_t) i] = (int32_t) ids[(size_t) (p + i)];
        if (ppl != nullptr)
            for (int i = 0; i < T; ++i) window[(size_t) i] = ppl->token(p + i);
        // the guess: this window is accepted whole and the target's next token is the chain's draft T - 1; the next
        // window then holds that draft and the ones the chain made after it
        int TB = 0;
        const bool try_pair = pair_req && pair_hint && !held.on && !first_window && !from_sfx && e0 == 0 &&
                              p + T + runtime_.S_mtp + 64 < runtime_.o.max_context;
        const std::function<int(int32_t*)> guess = [&](int32_t* wb) -> int {
            n_chain = runtime_.mtp.draft_more(n_chain, drafts.data(), dprob.data());   // made beside this window's stage 0
            if (n_chain <= T || !basalt::core::draft_keep(dprob.data(), T - 1, kPairTheta)) return 0;
            int tb = 1;
            while (tb < runtime_.S_mtp && T - 1 + tb < n_chain && basalt::core::draft_keep(dprob.data() + T, tb - 1, cum_adapt.th()))
                ++tb;
            if (tb < 2) return 0;
            wb[0] = drafts[(size_t) T - 1];
            for (int i = 1; i < tb; ++i) wb[i] = drafts[(size_t) T - 1 + (size_t) i];
            return tb;
        };
        runtime_.drive.d.layers = 0;
        runtime_.drive.d.experts = 0;
        runtime_.drive.d.failed = false;
        // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert
        // ran on the GPU or the CPU (they round differently) would depend on the copy's timing
        basalt::core::decode_gaps().mark(17);
        // an adaptive round's swaps land before the second window after it (a fixed window, so reproducible),
        // applied beside the draft before it (apply_thr) when their copies are done; what that moved to the
        // peer lands one window later
        if (apply_thr.joinable()) apply_thr.join();
        if (copy_live && copy_done) {   // the RAM copies are done: their bookkeeping, between windows
            basalt::core::decode_events().tag(1u << 3);
            runtime_.src.commit_exchanges();
            copy_live = false;
        }
        ++adapt_age;
        if (!chain || (adapt_age >= 2 && !(applied_early && adapt_age == 2))) {
            if (!runtime_.pending.empty()) basalt::core::decode_events().tag(1u << 1);
            runtime_.apply_pending(true);
        }
        // the elastic KV: the next window's cells backed before it runs (between windows, rare)
        if (runtime_.kv_floor > 0 && p + T + 64 > basalt::core::session_kv_cells(*runtime_.rsp)) {
            if (adapt_thr.joinable()) adapt_thr.join();
            if (copy_live) { runtime_.src.commit_exchanges(); copy_live = false; }   // as at the request's end
            if (!runtime_.kv_fit(*runtime_.rsp, p + runtime_.kKvGrowCells, runtime_.err, true, p + runtime_.kKvWindowCells)) {
                if (runtime_.kv_short) {
                    kv_stop = runtime_.err;
                    break;
                }
                std::printf("ERR %s\n", runtime_.err.c_str());
                return Flow::error;
            }
        }
        basalt::core::decode_gaps().mark(18);
        if (hist_n > 0) {
            // the tails the penalties count over, one per row: the tokens the state has consumed, the fed-back
            // head `x` (it joins `consumed` only after this window commits), then the drafts before that row -
            // what plain decode would have counted there
            basalt::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                          hist_n, runtime_.hist_stage.data());
            const basalt::core::OnDevice on_h(runtime_.hist_dev);
            cudaMemcpy(runtime_.d_hist, runtime_.hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
        }
        basalt::core::decode_gaps().mark(19);
        basalt::core::decode_events().stamp(3);
        const Clock::time_point tw0 = Clock::now();
        void* const user_c = cv == &runtime_.ver ? runtime_.win_pool_user : (void*) &runtime_.split_drive_b;
        void* const user_o = cv == &runtime_.ver ? (void*) &runtime_.split_drive_b : runtime_.win_pool_user;
        const bool ran = held.on ? true
            : try_pair ? cv->run_pair(*ov, T, window.data(), p, guess, win_b.data(), TB, runtime_.win_pool_fn, user_c, user_o,
                                      outv.data(), runtime_.err)
                       : cv->run(T, window.data(), p, runtime_.win_pool_fn, user_c, outv.data(), runtime_.err);
        const bool paired = TB > 0;
        const bool held_run = held.on;
        if (held.on) {
            for (int i = 0; i < T; ++i) outv[(size_t) i] = held.out[(size_t) i];
            held.on = false;
        }
        if (!ran || runtime_.drive.d.failed) {
            std::printf("ERR %s\n", runtime_.drive.d.failed && runtime_.drive.d.fail ? runtime_.drive.d.fail : runtime_.err.c_str());
            return Flow::error;
        }
        int a = 0;
        if (e0 > 0) {
            for (int i = 0; i < e0; ++i) outv[(size_t) i] = window[(size_t) i + 1];
            a = T - 1;
            if (relax_on) {
                if (outv[(size_t) a] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) a] == runtime_.o.relax_call_close) in_call = false;
            }
        } else if (relax_on) {
            // a draft the target's draw disagrees with is kept when the target gives it probability >= P,
            // outside tool calls and never for an added token; the row then emits the draft
            bool call = in_call;
            while (a < T - 1) {
                const int32_t d = window[(size_t) a + 1];
                if (d != outv[(size_t) a]) {
                    if (call || d >= runtime_.o.relax_special_from || cv->draft_prob(a) < req_relax) break;
                    outv[(size_t) a] = d;
                    ++relaxed_kept;
                }
                if (d == runtime_.o.relax_call_open) call = true;
                else if (d == runtime_.o.relax_call_close) call = false;
                ++a;
            }
            for (int i = 0; i <= a; ++i)
                if (outv[(size_t) i] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) i] == runtime_.o.relax_call_close) in_call = false;
        } else
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
        if (gguard != nullptr && gguard->on()) {
            // the first sampled token the grammar refuses is drawn again under the grammar's mask, and the window ends there
            const int miss = gguard->scan(outv.data(), a + 1);
            if (miss >= 0) {
                gguard->mask_bias(req_bias, grammar_bias);
                int32_t gtok = 0;
                if (!cv->resample_row(miss, p, grammar_bias, gtok, runtime_.err)) {
                    std::printf("ERR %s\n", runtime_.err.c_str());
                    return Flow::error;
                }
                outv[(size_t) miss] = gtok;
                a = miss;
                if (!gguard->take(gtok))
                    basalt::log::info("basalt grammar", "the masked draw at row %d of %d was token %d", miss, T, (int) gtok);
                ++grammar_resamples;
            }
        }
        if (ppl != nullptr) {   // every row scored and committed
            a = T - 1;
            for (int t = 0; t < T; ++t) {
                std::vector<float> lg((size_t) runtime_.n_vocab);
                if (!cv->copy_logits(t, lg.data()) || !ppl->score(p + t, std::move(lg), runtime_.err)) {
                    std::printf("ERR ppl: %s\n", runtime_.err.empty() ? "the head logits copy failed" : runtime_.err.c_str());
                    return Flow::error;
                }
                outv[(size_t) t] = ppl->token(p + t + 1);   // as if every draft were accepted: x, the MTP
            }
        }
        // the guess held when the window was accepted whole, its last pick is the next window's first token, and
        // the request goes on past it
        if (!from_sfx && e0 == 0 && !held_run) full_ema = 0.9 * full_ema + (a == T - 1 ? 0.1 : 0.0);
        bool hold = false;
        if (paired) {
            ++pair_runs;
            int n_emit_h = 0;
            bool eos_h = false;
            for (int i = e0; i <= a && produced_n + n_emit_h < max_new && !eos_h; ++i) {
                ++n_emit_h;
                eos_h = std::find(runtime_.o.eos_ids.begin(), runtime_.o.eos_ids.end(), (int64_t) outv[(size_t) i]) != runtime_.o.eos_ids.end();
            }
            hold = a == T - 1 && outv[(size_t) a] == win_b[0] && !eos_h && produced_n + n_emit_h < max_new && !runtime_.halted();
            if (!hold && !cv->pair_miss(*ov, runtime_.win_pool_fn, user_o, runtime_.err)) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                return Flow::error;
            }
        }
        if (runtime_.logit_hash)   // the accepted rows' logits, per position (equality of the batch rows with these)
            for (int t = 0; t <= a; ++t) basalt::log::raw("LH -1 %lld %016llx\n", (long long) (p + t), (unsigned long long) runtime_.row_hash(t));
        if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
        if (tk_log != nullptr && tk_valid && !from_sfx && ppl == nullptr && req_temperature > 0.0) {
            std::vector<float> lg((size_t) runtime_.n_vocab);
            std::fprintf(tk_log, "W %d %d\n", T, a);
            for (int j = 0; j < T; ++j) {
                if (j > a) {   // past the first miss: only the draft side
                    std::fprintf(tk_log, "R %d - - T D", j);
                    for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                    std::fputc('\n', tk_log);
                    continue;
                }
                if (!cv->copy_logits(j, lg.data())) break;
                // top_k, top_p, then temperature over the survivors: the sampler chain without penalties
                std::vector<int> ord((size_t) runtime_.n_vocab);
                for (int i = 0; i < runtime_.n_vocab; ++i) ord[(size_t) i] = i;
                const int K = basalt::kernels::sampler_list_len(req_top_k, (int) runtime_.n_vocab);
                std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) {
                    return lg[(size_t) x] > lg[(size_t) y] || (lg[(size_t) x] == lg[(size_t) y] && x < y);
                });
                const double mx = lg[(size_t) ord[0]];
                std::vector<double> e((size_t) K);
                double s1 = 0.0;
                for (int i = 0; i < K; ++i) s1 += (e[(size_t) i] = std::exp((double) lg[(size_t) ord[(size_t) i]] - mx));
                int keep = K;
                double cum = 0.0;
                for (int i = 0; i < K; ++i) { cum += e[(size_t) i] / s1; if (cum >= (double) req_top_p) { keep = i + 1; break; } }
                double s2 = 0.0;
                for (int i = 0; i < keep; ++i) s2 += (e[(size_t) i] = std::exp(((double) lg[(size_t) ord[(size_t) i]] - mx) / (double) req_temperature));
                std::fprintf(tk_log, "R %d %d %d", j, j + 1 < T ? (int) window[(size_t) j + 1] : -1, (int) outv[(size_t) j]);
                std::fprintf(tk_log, " T");
                for (int i = 0; i < std::min(keep, 8); ++i) std::fprintf(tk_log, " %d:%.5f", ord[(size_t) i], e[(size_t) i] / s2);
                std::fprintf(tk_log, " D");
                for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                std::fputc('\n', tk_log);
            }
            std::fflush(tk_log);
        }
        tk_valid = false;
        const Clock::time_point tw1 = Clock::now();
        bool adapt_ok = true;
        const bool do_adapt = !runtime_.drive.d.usage.empty() && ((runtime_.rounds + 1) % runtime_.o.adapt_every) == 0 &&
                              (runtime_.adapt_fast ? runtime_.xq_stage == 0
                                          : (!chain || (runtime_.pending.empty() && !copy_live && !runtime_.src.exchanges_staged())));
        auto spawn_adapt = [&] {
            if (runtime_.adapt_fast) {   // a round, or the next step of the one in flight, beside the draft
                if (!do_adapt && runtime_.xq_stage == 0) return;
                if (do_adapt) adapt_age = 0;
                basalt::core::decode_events().tag(1u << 0);
                basalt::core::decode_events().tag(1u << (6 + runtime_.xq_stage));
                adapt_thr.run([&] {
                    const auto ta = Clock::now();
                    basalt::core::decode_events().stamp(8);
                    adapt_ok = runtime_.xq_stage != 0 ? runtime_.xq_advance(false) : runtime_.xq_round();
                    basalt::core::decode_events().stamp(6);
                    std::string perr;
                    if (adapt_ok && !runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    basalt::core::decode_events().stamp(7);
                    basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                });
                return;
            }
            if (!do_adapt) return;
            adapt_age = 0;
            adapt_thr.run([&] {
                const auto ta = Clock::now();
                adapt_ok = runtime_.adapt();
                if (chain && adapt_ok) {   // the evictions' residency, uploaded beside the draft
                    std::string perr;
                    if (!runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                }
                basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            });
        };
        if (chain) {
            // the draft launches first; the commit, the adaptive round and the output go in its shadow
            int n_emit = 0;
            bool eos_pre = false;
            for (int i = e0; i <= a && produced_n + n_emit < max_new && !eos_pre; ++i) {
                ++n_emit;
                eos_pre = ppl == nullptr &&
                          std::find(runtime_.o.eos_ids.begin(), runtime_.o.eos_ids.end(), (int64_t) outv[(size_t) i]) != runtime_.o.eos_ids.end();
            }
            const bool will_draft = !eos_pre && produced_n + n_emit < max_new;
            float wprob[16];
            const int wprob_n = std::min(runtime_.S_mtp, 16);
            if (draft_log != nullptr)
                for (int i = 0; i < wprob_n; ++i) wprob[i] = first_window || from_sfx ? -1.0f : dprob[(size_t) i];
            // the adaptive round and the swaps' residency, beside the commit and the draft
            auto adapt_side = [&] {
                spawn_adapt();
                // the swaps' residency at the next window, then their RAM copies a share per draft
                applied_early = adapt_age == 1 && !do_adapt && !runtime_.pending.empty();
                if (applied_early || copy_live) {
                    const bool boundary = applied_early, copies = applied_early || copy_live;
                    if (copies) {
                        copy_live = true;
                        copy_done = false;
                    }
                    basalt::core::decode_events().tag(1u << 2);
                    apply_thr.run([&, boundary, copies] {
                        if (boundary) {
                            runtime_.apply_pending(true, false);
                            std::string perr;
                            if (!runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                        }
                        if (copies) copy_done = !runtime_.src.exchanges_staged() || runtime_.src.copy_exchanges(8ull << 20);
                    });
                }
            };
            basalt::core::decode_events().stamp(9);
            if (!hold && !cv->commit(a + 1, runtime_.err, 2)) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                return Flow::error;
            }
            for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
            draft_offered += T - 1 - e0;
            draft_accepted += a - e0;
            first_window = false;
            bool commit_ok = true, after_ran = false, eos = false;
            std::string cerr;
            auto after_launch = [&] {
                after_ran = true;
                commit_ok = hold ? true : cv->commit(a + 1, cerr, 1);
                if (!hold) adapt_side();
                for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                    std::printf("T %d\n", (int) outv[(size_t) i]);
                    basalt::core::progress_beat();
                    ++produced_n;
                    if (runtime_.o.suffix_draft > 0) runtime_.sfx->append(outv[(size_t) i]);
                    eos = ppl == nullptr &&
                          std::find(runtime_.o.eos_ids.begin(), runtime_.o.eos_ids.end(), (int64_t) outv[(size_t) i]) != runtime_.o.eos_ids.end();
                }
                std::fflush(stdout);
            };
            ++runtime_.rounds;
            const Clock::time_point tw2 = Clock::now();
            basalt::core::decode_gaps().mark(5);
            if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
            runtime_.mtp.set_cum_th(cum_adapt.th());
            bool drafted = true;
            if (hold) {
                // the tokens go out; the next window's last stage runs behind this window's commit while its first
                // stage finishes, and the drafter (on the first card) takes this window's rows once that card is
                // free
                ++pair_holds;
                if (!cv->pair_hold_commit(T, runtime_.err)) {
                    std::printf("ERR %s\n", runtime_.err.c_str());
                    return Flow::error;
                }
                after_launch();
                const float* staged = nullptr;
                drafted = runtime_.mtp.stage_rows(cv->final_R_all(), T, &staged, runtime_.err);   // before the last card is busy again
                if (!drafted || !cv->pair_hold_begin(*ov, T, p, TB, win_b.data(), runtime_.err) ||
                    !cv->pair_hold_finish_first(*ov, runtime_.win_pool_fn, user_o, runtime_.err) ||
                    !(drafted = runtime_.mtp.prefill(staged, outv.data(), T, p, runtime_.err))) {
                    std::printf("ERR %s\n", runtime_.err.c_str());
                    return Flow::error;
                }
                if (!cv->pair_hold_finish(*ov, runtime_.win_pool_fn, user_o, held.out.data(), runtime_.err) || runtime_.drive.d.failed) {
                    std::printf("ERR %s\n", runtime_.drive.d.failed && runtime_.drive.d.fail ? runtime_.drive.d.fail : runtime_.err.c_str());
                    return Flow::error;
                }
                held.on = true;
                held.T = TB;
                for (int i = 0; i < TB; ++i) held.win[(size_t) i] = win_b[(size_t) i];
                std::swap(cv, ov);
            } else if (will_draft && runtime_.mtp_on) {
                runtime_.mtp.set_launch_hook(after_launch);
                pair_hint = pair_req && full_ema >= kPairFull;
                runtime_.mtp.set_cum_th(pair_hint ? cum_adapt.th() * kPairChainScale : cum_adapt.th());
                runtime_.mtp.set_host_th(cum_adapt.th());
                if (const int32_t* top = cv->top_ids())
                    runtime_.mtp.note_candidates(top, (a + 1) * basalt::core::Verifier::kTopIds);
                drafted = runtime_.mtp.draft_from(cv->final_R_all(), T, outv.data(), p, a, drafts.data(), runtime_.err, dprob.data(), &n_chain);
                runtime_.mtp.set_launch_hook(nullptr);
                if (drafted && runtime_.pipe_on) {   // the next window's first stage starts after the chain's kernels
                    const cudaEvent_t ev = runtime_.mtp.done_event();
                    runtime_.ver.set_wait_event(ev);
                    runtime_.ver_b.set_wait_event(ev);
                }
                if (tk_log != nullptr && drafted) {
                    for (int j = 0; j < 16 && j < runtime_.S_mtp; ++j) {
                        tk_prev[j].n = runtime_.mtp.diag_top(j, 4, tk_prev[j].id, tk_prev[j].p);
                    }
                    tk_valid = true;
                }
            }
            if (!after_ran) after_launch();
            basalt::core::decode_events().stamp(0);
            {
                const Clock::time_point tw3 = Clock::now();
                auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                ++dec_windows; dec_T += T;
                if (draft_log != nullptr) {
                    std::fprintf(draft_log, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0),
                                 msd(tw0, tw1), msd(tw1, tw3));
                    for (int i = 0; i < wprob_n; ++i) std::fprintf(draft_log, " %.4f", wprob[i]);
                    std::fputc('\n', draft_log);
                }
            }
            if (adapt_thr.joinable()) adapt_thr.join();
            basalt::core::decode_events().stamp(1);
            basalt::core::decode_gaps().mark(16);
            if (!commit_ok) {
                std::printf("ERR %s\n", cerr.c_str());
                return Flow::error;
            }
            if (!adapt_ok) {
                std::printf("ERR an adaptive refill failed\n");
                return Flow::error;
            }
            if (!drafted) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                return Flow::error;
            }
            if (timed_round && !eos)
                runtime_.policy->observe(from_sfx, T, a, sfx_match,
                               std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
            basalt::core::decode_gaps().mark(17);
            if (eos) { finish = "stop"; break; }
            runtime_.shed_if_low();
            if (runtime_.halted()) { finish = "cancel"; break; }
            x = outv[(size_t) a];
            p += a + 1;
            basalt::core::decode_events().stamp(2);
            continue;
        }
        spawn_adapt();
        if (!runtime_.ver.commit(a + 1, runtime_.err)) {
            if (adapt_thr.joinable()) adapt_thr.join();
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
        for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
        draft_offered += T - 1 - e0;
        draft_accepted += a - e0;
        first_window = false;
        bool eos = false;
        for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
            std::printf("T %d\n", (int) outv[(size_t) i]);
            basalt::core::progress_beat();
            ++produced_n;
            if (runtime_.o.suffix_draft > 0) runtime_.sfx->append(outv[(size_t) i]);
            eos = ppl == nullptr &&
                  std::find(runtime_.o.eos_ids.begin(), runtime_.o.eos_ids.end(), (int64_t) outv[(size_t) i]) != runtime_.o.eos_ids.end();
        }
        std::fflush(stdout);
        ++runtime_.rounds;
        const Clock::time_point tw2 = Clock::now();
        basalt::core::decode_gaps().mark(5);
        if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
        runtime_.mtp.set_cum_th(cum_adapt.th());
        const bool drafted = eos || produced_n >= max_new || !runtime_.mtp_on ||
                             runtime_.mtp.draft(T, outv.data(), p, a, drafts.data(), runtime_.err, dprob.data());
        {
            const Clock::time_point tw3 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
            ++dec_windows; dec_T += T;
        }
        if (adapt_thr.joinable()) adapt_thr.join();
        basalt::core::decode_gaps().mark(16);
        if (!adapt_ok) {
            std::printf("ERR an adaptive refill failed\n");
            return Flow::error;
        }
        if (!drafted) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        if (timed_round && !eos)
            runtime_.policy->observe(from_sfx, T, a, sfx_match,
                           std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
        basalt::core::decode_gaps().mark(17);
        if (eos) { finish = "stop"; break; }
        runtime_.shed_if_low();
        if (runtime_.halted()) { finish = "cancel"; break; }
        x = outv[(size_t) a];
        p += a + 1;
    }
    if (gguard != nullptr)
        basalt::log::info("basalt grammar", "%lld windows checked, %lld masked resamples, %.3f ms in grammar code (%.2f us/window)",
                     (long long) gguard->windows(), (long long) grammar_resamples, gguard->ms(),
                     gguard->windows() > 0 ? gguard->ms() * 1000.0 / (double) gguard->windows() : 0.0);
    if (apply_thr.joinable()) apply_thr.join();
    if (copy_live) runtime_.src.commit_exchanges();   // swaps applied early: the rest of their copies
    copy_live = false;
    if (ppl != nullptr && !ppl->finish(runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }

    return Flow::proceed;
}

}  // namespace basalt::program::serve
