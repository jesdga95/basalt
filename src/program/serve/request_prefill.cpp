// src/program/serve/request_prefill.cpp - request read_prompt operations.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::read_prompt() {
    yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
    yielded_slot = -1;

    if (!prompt_begin()) {
        std::printf("ERR an adaptive refill failed\n");
        return Flow::error;
    }

    req_sp.greedy = req_temperature <= 0.0f;
    req_sp.temperature = req_temperature;
    req_sp.top_p = req_top_p;
    req_sp.top_k = req_top_k;
    req_sp.seed = req_seed_given ? req_seed
                                 : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
    req_sp.min_p = req_min_p;
    req_sp.penalty_last_n = req_penalty_last_n;   // 0..kPenaltyWindowCap: request_parse refuses a wider window
    req_sp.penalty_repeat = req_penalty_repeat;
    req_sp.penalty_freq = req_penalty_freq;
    req_sp.penalty_present = req_penalty_present;
    req_sp.counter = 0;
    if (const Flow flow = set_sampling(); flow != Flow::proceed) return flow;
    cancelled = false;
    // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
    // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
    // chat renders the same history - but not always the same header or the thinking of this reply - so that
    // checkpoint is the one it reuses.
    turn_at = -1;
    if (runtime_.o.prompt_cache > 0 && runtime_.o.turn_token >= 0)
        for (int64_t i = n - 1; i > resume; --i)
            if (ids[(size_t) i] == runtime_.o.turn_token) { turn_at = i; break; }
    if (runtime_.o.prompt_cache > 0 && turn_checkpoint_at_last_token() && n - 1 > resume) turn_at = n - 1;
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
    limit_parts();
    at = read_from;
    // the last window of the last windowed part carries the last prompt token too: the first generated
    // token is sampled from it, and the draft layer takes its rows like any accepted window
    fold_p0 = n - 1;
    prompt_parts_begin();
    for (const int64_t to : {root_at, turn_at, n - 1}) {
        if (to <= at) continue;
        const Flow flow = read_to(to);
        if (flow == Flow::quit) break;   // stopped while reading: the rest of the request runs, and ends as cancelled
        if (flow != Flow::proceed) return flow;
    }
    if (const Flow flow = prompt_end(); flow != Flow::proceed) return flow;
    prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
    std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
    std::fflush(stdout);

    return Flow::proceed;
}

}  // namespace basalt::program::serve
