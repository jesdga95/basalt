// src/program/serve/decode_loop.hpp - the window loop of a request, one instance per model backend.
//
// `B` is the model's request class: every call on it is inline, so a window pays no indirect call.  `B::kSpeculates`
// adds drafted windows, prompt lookup and pipelined windows; without it every window holds one token.
#pragma once

#include "request.hpp"
#include "reply_rows.hpp"

namespace basalt::program::serve {

template <class B>
Flow Request::decode_loop(B& b) {
    constexpr bool kSpec = B::kSpeculates;
    if (const Flow flow = b.decode_begin(); flow != Flow::proceed) return flow;
    p = fold_p0;
    x = (int32_t) ids[(size_t) fold_p0];
    b.seed_tiers();
    first_window = true;
    produced_n = 0;
    draft_offered = 0;
    draft_accepted = 0;
    window = std::vector<int32_t>((size_t) b.max_window());
    outv = std::vector<int32_t>((size_t) b.max_window());
    consumed.reserve((size_t) (n + max_new + b.max_window()));
    for (int64_t i = 0; i < fold_p0; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
    finish = "length";
    b.decode_state_begin();
    d0 = Clock::now();
    b.decode_stats_begin();
    if (cancelled) finish = "cancel";

    const std::vector<int64_t>& eos_ids = runtime_.o.eos_ids;
    while (!cancelled && produced_n < max_new && (!in_slot || solo_req)) {   // (a batch slot's first token comes from the batch window)
        int T = 1;
        if constexpr (kSpec) T = b.draft_window();
        if (first_window) T = (int) (n - fold_p0);
        const int e0 = first_window ? T - 1 : 0;   // rows before it are prompt tokens: forced, not emitted
        if (ppl != nullptr) {
            if (p >= ppl->end()) break;
            T = (int) std::min<int64_t>(req_ppl_t, ppl->end() - p);
        }
        bool from_sfx = false;
        int sfx_match = 0;
        if constexpr (kSpec) b.lookup_window(T, from_sfx, sfx_match);
        const bool timed_round = !first_window;
        const Clock::time_point round0 = Clock::now();
        if (p + T > runtime_.o.max_context) break;
        window[0] = x;
        if constexpr (kSpec) b.fill_drafts(T, from_sfx);
        if (e0 > 0)
            for (int i = 1; i < T; ++i) window[(size_t) i] = (int32_t) ids[(size_t) (p + i)];
        if (ppl != nullptr)
            for (int i = 0; i < T; ++i) window[(size_t) i] = ppl->token(p + i);
        int TB = 0;
        const int room_verdict = b.before_window(T, p);   // 0 go on, 1 the reply ends here (kv_stop), 2 failed
        if (room_verdict == 1) break;
        if (room_verdict == 2) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
        if (hist_n > 0) {
            // the tails the penalties count over, one per row: the tokens the state has consumed, the fed-back
            // head `x` (it joins `consumed` only after this window commits), then the drafts before that row -
            // what plain decode would have counted there
            basalt::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                          hist_n, b.history_stage());
            b.upload_history(T);
        }
        b.mark_before_run();
        typename B::TierStep tier(b);
        tier.before_window(T, e0, from_sfx);
        const Clock::time_point tw0 = Clock::now();
        const bool ran = b.run_window(T, e0, TB);
        const bool paired = TB > 0;
        bool held_run = false;
        if constexpr (kSpec) held_run = b.take_held(T);
        if (!ran || b.window_failed()) {
            std::printf("ERR %s\n", b.window_error());
            return Flow::error;
        }
        int a = 0;
        if (e0 > 0) {
            for (int i = 0; i < e0; ++i) outv[(size_t) i] = window[(size_t) i + 1];
            a = T - 1;
            if constexpr (kSpec) b.relax_note_forced(a);
        } else if constexpr (kSpec) {
            a = b.accept_drafts(T);
        }
        if (gguard != nullptr && gguard->on()) {
            // the first sampled token the grammar refuses is drawn again under the grammar's mask, and the window ends there
            const int miss = gguard->scan(outv.data(), a + 1);
            if (miss >= 0) {
                gguard->mask_bias(req_bias, grammar_bias);
                int32_t gtok = 0;
                if (!b.resample_row(miss, p, grammar_bias, gtok, runtime_.err)) {
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
                if (!b.copy_logits(t, lg.data()) || !ppl->score(p + t, std::move(lg), runtime_.err)) {
                    std::printf("ERR ppl: %s\n", runtime_.err.empty() ? "the head logits copy failed" : runtime_.err.c_str());
                    return Flow::error;
                }
                outv[(size_t) t] = ppl->token(p + t + 1);   // as if every draft were accepted: x, the MTP
            }
        }
        bool hold = false;
        if constexpr (kSpec) {
            if (!b.after_accept(T, a, e0, from_sfx, held_run, paired, hold)) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                return Flow::error;
            }
        }
        const Clock::time_point tw1 = Clock::now();
        // the picks this window sends and the rows it keeps: a window that ends the reply keeps no row past its
        // last pick, and that pick is the next token, as at any window's end
        const ReplyRows rr = reply_rows(outv.data(), e0, a, produced_n, max_new, eos_ids, ppl != nullptr);
        const int keep = rr.keep;
        // the draft launches first; the commit, the adaptive round and the output go in its shadow
        const bool will_draft = !rr.ends && produced_n + rr.emit < max_new;
        if constexpr (kSpec) b.note_draft_probs(from_sfx);
        b.stamp_before_commit();
        if (!hold && !b.commit(keep, runtime_.err, 2)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        for (int i = 0; i < keep; ++i) consumed.push_back(window[(size_t) i]);
        draft_offered += T - 1 - e0;
        draft_accepted += a - e0;
        first_window = false;
        bool commit_ok = true, after_ran = false, eos = false;
        std::string cerr;
        auto after_launch = [&] {
            after_ran = true;
            commit_ok = hold ? true : b.commit(keep, cerr, 1);
            if (!hold) tier.after_commit();
            for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                std::printf("T %d\n", (int) outv[(size_t) i]);
                basalt::core::progress_beat();
                ++produced_n;
                b.on_token(outv[(size_t) i]);
                eos = ppl == nullptr &&
                      std::find(eos_ids.begin(), eos_ids.end(), (int64_t) outv[(size_t) i]) != eos_ids.end();
            }
            std::fflush(stdout);
        };
        b.count_round();
        const Clock::time_point tw2 = Clock::now();
        bool drafted = true;
        if constexpr (kSpec) {
            drafted = b.draft_next(T, a, e0, TB, from_sfx, hold, will_draft, after_launch);
            if (!drafted && b.draft_fatal()) return Flow::error;
        }
        if (!after_ran) after_launch();
        b.window_timing(T, a, from_sfx, round0, tw0, tw1, tw2);
        tier.finish();
        b.mark_after_window();
        if (!commit_ok) {
            std::printf("ERR %s\n", cerr.c_str());
            return Flow::error;
        }
        if (!tier.ok()) {
            std::printf("ERR an adaptive refill failed\n");
            return Flow::error;
        }
        if (!drafted) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        if constexpr (kSpec) {
            if (timed_round && !eos) b.observe_policy(from_sfx, T, a, sfx_match, round0);
        }
        b.mark_loop_end();
        if (eos) { finish = "stop"; break; }
        b.shed_low();
        if (runtime_.halted()) { finish = "cancel"; break; }
        x = outv[(size_t) keep - 1];
        p += keep;
        b.stamp_loop_end();
    }
    if (gguard != nullptr)
        basalt::log::info("basalt grammar", "%lld windows checked, %lld masked resamples, %.3f ms in grammar code (%.2f us/window)",
                     (long long) gguard->windows(), (long long) grammar_resamples, gguard->ms(),
                     gguard->windows() > 0 ? gguard->ms() * 1000.0 / (double) gguard->windows() : 0.0);
    b.decode_end();
    if (ppl != nullptr && !ppl->finish(runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    return Flow::proceed;
}

}  // namespace basalt::program::serve
