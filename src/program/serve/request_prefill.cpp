// src/program/serve/request_prefill.cpp - request read_prompt operations.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::read_prompt() {
    ov_copied = false;

    batch_fatal = false;
    il_parts = 0;   // the slots' windows run between this prompt's chunks
    il_ms = 0;
    yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
    yielded_slot = -1;

    runtime_.apply_pending(true);
    if (!runtime_.xq_drain()) {
        std::printf("ERR an adaptive refill failed\n");
        return Flow::error;
    }

    req_sp.greedy = req_temperature <= 0.0f;
    req_sp.temperature = req_temperature;
    req_sp.top_p = req_top_p;
    req_sp.top_k = req_top_k;
    req_sp.seed = req_seed ? req_seed
                           : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
    req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
    req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
    req_sp.penalty_repeat = req_penalty_repeat;
    req_sp.penalty_freq = req_penalty_freq;
    req_sp.penalty_present = req_penalty_present;
    req_sp.counter = 0;
    if (!runtime_.ver.set_logit_bias(req_bias, runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::next;
    }
    runtime_.ver.set_sampling(req_sp);
    // relaxed acceptance: sampled text requests without a logit_bias; never the quality gate
    relax_on = req_relax > 0.0f && !req_sp.greedy && !req_bias_present && ppl == nullptr && gguard == nullptr;
    runtime_.ver.set_draft_probe(relax_on);
    in_call = false;   // inside a tool call (its tags seen last in the prompt or the output): exact
    for (size_t i = ids.size(); i-- > 0;)
        if (ids[i] == runtime_.o.relax_call_open || ids[i] == runtime_.o.relax_call_close) { in_call = ids[i] == runtime_.o.relax_call_open; break; }
    relaxed_kept = 0;
    runtime_.drive.d.pcie_num = std::max(0, std::min(256, (int) (runtime_.o.pcie_frac * 256.0 + 0.5)));
    // a layer split: CUDA0's share; a later GPU keeps its own (its link)
    for (int st = 0; st < runtime_.split_drive.n; ++st)
        runtime_.split_drive.pcie_num[st] = (st == 0 || runtime_.split_same)
                                       ? runtime_.drive.d.pcie_num : runtime_.pcie_num_of(runtime_.stages[(size_t) st - 1]->pcie_frac);
    hist_n = std::min(req_sp.penalty_last_n, runtime_.kPenaltyWindowCap);
    runtime_.ver.set_history(hist_n > 0 ? runtime_.d_hist : nullptr, hist_n);
    if (runtime_.pipe_on) {   // the second chain draws the same way
        if (!runtime_.ver_b.set_logit_bias(req_bias, runtime_.err)) {
            std::printf("ERR %s\n", runtime_.err.c_str());
            return Flow::next;
        }
        runtime_.ver_b.set_sampling(req_sp);
        runtime_.ver_b.set_draft_probe(relax_on);
        runtime_.ver_b.set_history(hist_n > 0 ? runtime_.d_hist : nullptr, hist_n);
        for (int st = 0; st < runtime_.split_drive_b.n; ++st) runtime_.split_drive_b.pcie_num[st] = runtime_.split_drive.pcie_num[st];
    }
    cancelled = false;
    // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
    // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
    // chat renders the same history - but not always the same header or the thinking of this reply - so that
    // checkpoint is the one it reuses.
    turn_at = -1;
    if (runtime_.o.prompt_cache > 0 && runtime_.o.turn_token >= 0)
        for (int64_t i = n - 1; i > resume; --i)
            if (ids[(size_t) i] == runtime_.o.turn_token) { turn_at = i; break; }
    // (opt-in) a short turn of --tail-role-token's role right before the new assistant turn (the server's
    // trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another effort, or the
    // next turn of the chat - finds the conversation without it
    if (turn_at > 0 && runtime_.o.tail_role_token >= 0)
        for (int64_t i = turn_at - 1; i > resume; --i)
            if (ids[(size_t) i] == runtime_.o.turn_token) {
                if (ids[(size_t) i + 1] == runtime_.o.tail_role_token) turn_at = i;
                break;
            }
    // A prompt read from token 0 also stops at its first turn boundary: the end of the system prompt (with the
    // tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root, which
    // the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.  Only
    // for a system prompt of --prompt-cache-root tokens or more: a small one is cheaper to read again than the
    // extra part costs.
    root_at = -1;
    if (runtime_.o.prompt_cache > 0 && runtime_.o.turn_token >= 0 && runtime_.o.prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
        for (int64_t i = 1; i < turn_at; ++i)
            if (ids[(size_t) i] == runtime_.o.turn_token) {
                if (i >= runtime_.o.prompt_cache_root) root_at = i;
                break;
            }
    // A short read into a slot beside decoding slots is not split: each part pays the lend, refill and
    // side-thread costs again, and a checkpoint stalls the windows.  One that runs alone stops at its root,
    // and at its turn boundary only past --prompt-cache-root tokens since the root: a short history is
    // cheaper to read again than the extra part costs.  The rest of a read that gave way keeps its parts.
    if (in_slot && runtime_.batch_on() && resumed_chunk <= 0 && n - 1 - read_from <= std::max<int64_t>(runtime_.sp.chunk(), 1)) {
        if (n - 1 - read_from > runtime_.kReadAlone) {
            turn_at = -1;
            root_at = -1;
        } else if (turn_at - std::max(read_from, root_at) < runtime_.o.prompt_cache_root) {
            turn_at = -1;
        }
    }
    // the parts after the first read their n-gram rows from memory: once the first part has its own, the rest
    // are read into the row cache beside it
    {
        int64_t first_to = 0;
        for (const int64_t to : {root_at, turn_at, n - 1})
            if (to > read_from) { first_to = to; break; }
        runtime_.sp.set_ple_warm(ids.data(), read_from, first_to, n - 1);
    }
    at = read_from;
    // the last window of the last windowed part carries the last prompt token too: the first generated
    // token is sampled from it, and the draft layer takes its rows like any accepted window
    fold_p0 = n - 1;
    for (const int64_t to : {root_at, turn_at, n - 1}) {
        if (to <= at) continue;
        runtime_.err.clear();
        const bool win = windows_ok(at, to);
        int64_t rto = to;
        if (win && to == n - 1 && to != turn_at && to != root_at && ppl == nullptr && gguard == nullptr) {
            const int64_t p0 = at + (int64_t) runtime_.S * ((to - at) / runtime_.S);
            if (!runtime_.mtp_on || n - p0 <= runtime_.S_mtp) { rto = p0; fold_p0 = p0; }
        }
        if (win && !refill(runtime_.err)) {
            std::printf("ERR refilling a lent slot failed: %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        const SplitPlan split = split_plan(at, to, win ? "a windowed read" : resumed_chunk > 0 ? "a resumed chunked read" : nullptr);
        const int64_t split_c = split.chunk;
        if (!win && !lend(to - at, runtime_.err, split_c, split.head)) {
            std::printf("ERR lending the prompt path its slots failed: %s\n", runtime_.err.c_str());
            return Flow::error;
        }
        // beside slots: a window between the serial steps of a short read keeps them from stopping
        auto ov_window = [&]() -> bool {
            if (!runtime_.batch_on()) return true;
            if (!runtime_.batch_step()) { batch_fatal = true; runtime_.err = "a batch window failed"; return false; }
            if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) return false;
            return true;
        };
        if (win && !ov_window()) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
        runtime_.ov_post_done = false;
        runtime_.ov_post = [&, to]() -> bool {
            if ((to == turn_at || to == root_at) && !runtime_.checkpoint_at(to)) return false;
            // the lent slots get their experts back here, beside the windows; the table follows on the main thread
            for (PfPart& p : runtime_.pf_parts) {
                if (p.lent.empty()) continue;
                std::string re;
                const basalt::core::OnDevice on(p.dev);
                if (!refill_issue(p, re, false) || !p.cache->sync_queued(re)) { runtime_.ckpt_why = ": " + re; return false; }
            }
            ov_copied = true;
            return true;
        };
        const auto tsp = Clock::now();
        runtime_.pp_part_ckpt = (to == turn_at || to == root_at) ? to : -1;
        bool split_ran = true;
        bool sp_ok = win ? (rto <= at || read_windows(at, rto, runtime_.err))
                   : split_c > 0 ? read_split(at, to, split, split_ran, runtime_.err)
                                 : read_part(at, to, runtime_.err);
        if (!sp_ok && !split_ran) {   // the split could not start: the main card reads the part
            basalt::log::warn("basalt serve", "%s; read on CUDA0", runtime_.err.c_str());
            runtime_.err.clear();
            sp_ok = read_part(at, to, runtime_.err);
        }
        runtime_.pp_part_ckpt = -1;
        if (sp_ok && win && !ov_window()) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
        if (!sp_ok) {
            if (yielded_at < 0 && (batch_fatal || !(runtime_.stop_req.load() || runtime_.stdin_lost.load() || runtime_.ram_stop.load()))) {
                basalt::log::error("basalt serve", "%s", runtime_.err.c_str());
                std::printf("ERR %s\n", runtime_.err.c_str());
                // a CUDA fault (an illegal address) poisons the context for the whole process, and unwinding
                // the destructors on it could hang until the watchdog: leave at once
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::fflush(stdout);
                    std::_Exit(1);
                }
                return Flow::error;
            }
            cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
            break;
        }
        at = to;
        resumed_chunk = 0;   // the later parts are planned as the read they continue planned them
        runtime_.ov_post = nullptr;
        if ((to == turn_at || to == root_at) && !runtime_.ov_post_done && !runtime_.checkpoint_at(to, nullptr, !in_slot && !runtime_.batch_on())) {
            std::printf("ERR saving a conversation checkpoint failed%s\n", runtime_.ckpt_why.c_str());
            return Flow::error;
        }
    }
    if (!refill(runtime_.err)) {
        std::printf("ERR refilling a lent slot failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if (il_parts > 0)
        basalt::log::info("basalt batch", "the prompt was read in %lld parts, the slots decoding %.0f ms between "
                             "them", (long long) il_parts + 1, il_ms);
    prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
    std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
    std::fflush(stdout);

    return Flow::proceed;
}

}  // namespace basalt::program::serve
