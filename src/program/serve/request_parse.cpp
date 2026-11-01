// src/program/serve/request_parse.cpp - request parse operations.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::parse() {
    runtime_.stop_req.store(false);   // a STOP that arrived between requests is stale
    runtime_.ram_stop.store(false);
    runtime_.err.clear();
    if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }   // (batch windows swap the draft K/V)
    geni = runtime_.line.rfind("GENI ", 0) == 0;
    if (!geni && runtime_.line.rfind("GEN ", 0) != 0) {
        std::printf("ERR expected: GEN <max_new> <id,id,...>, GENI <max_new> <file> <id,id,...> or GENT <max_new> <nbytes> + text\n");
        return Flow::next;
    }
    endp = nullptr;
    max_new = std::strtoll(runtime_.line.c_str() + (geni ? 5 : 4), &endp, 10);
    // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
    // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
    // only).  Absent keys keep today's behavior: greedy, no penalties.
    req_temperature = 0.0f;
    req_top_p = 1.0f;
    req_top_k = 0;    // absent: no top-k limit (the whole vocabulary)
    req_seed = 0;
    req_seed_given = false;   // absent: a seed from the clock
    req_min_p = 0.0f;
    req_penalty_repeat = 1.0f;
    req_penalty_freq = 0.0f;
    req_penalty_present = 0.0f;
    req_penalty_last_n = 0;
    req_bias_present = false;
    req_gprompt = -1;       // gprompt=N: the first N ids are the prompt, the rest tokens the request has already generated
    req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
    req_relax = runtime_.o.relaxed_drafting;   // relax=P: this request's --relaxed-drafting (0 = exact)
    // ppl=K [ppl_t=T]: quality gate (basalt/program/ppl_gate.hpp) - chunk K of --ppl-base, the rest of
    // the chunk teacher-forced through verify windows of T tokens, every token committed
    req_ppl = -1;
    req_ppl_t = 3;
    req_clip = 0;   // clip=1 (GENT): max_new is a cap, cut to the room the context leaves after the prompt
    if (endp != nullptr) {   // GENI takes the same keys (image requests were always greedy); its
                             // embedding file path is the first token without an =
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "logit_bias") { req_bias_text = tok.substr(eq + 1); req_bias_present = true; }
            // ---- Jev: private protocol keys; never sampling parameters ----
            else if (key == "score" || key == "score_at") {
                auto& dst = key == "score" ? score_ids : score_frontiers;
                if (!runtime_.o.jev || !cli::parse_i64_list(tok.c_str() + eq + 1, dst, pe)) {
                    std::printf("ERR bad request: Jev is disabled or its token list is invalid\n");
                    return Flow::next;
                }
            }
            else if (key == "score_fresh") score_fresh = tok.substr(eq + 1) == "1";
            else if (key == "score_branches") {
                std::istringstream parts(tok.substr(eq + 1));
                for (std::string part; std::getline(parts, part, '/');) {
                    const size_t colon = part.find(':');
                    std::vector<int64_t> tokens, candidates;
                    if (!runtime_.o.jev || colon == std::string::npos ||
                        !cli::parse_i64_list(part.substr(0, colon).c_str(), tokens, pe) ||
                        !cli::parse_i64_list(part.substr(colon + 1).c_str(), candidates, pe) ||
                        tokens.empty() || candidates.empty() || candidates.size() > 255 || score_branches.size() >= 8) {
                        std::printf("ERR bad request: invalid Jev branch\n"); return Flow::next;
                    }
                    basalt::core::Verifier::ScoreBranch b;
                    b.tokens.assign(tokens.begin(), tokens.end()); b.candidates.assign(candidates.begin(), candidates.end());
                    if (std::any_of(tokens.begin(), tokens.end(), [&](int64_t t) { return t < 0 || t >= runtime_.n_vocab; }) ||
                        std::any_of(candidates.begin(), candidates.end(), [&](int64_t t) { return t < 0 || t >= runtime_.n_vocab; })) {
                        std::printf("ERR bad request: Jev branch token outside vocabulary\n"); return Flow::next;
                    }
                    score_branches.push_back(std::move(b));
                }
            }
            // ---- end Jev ----
            else if (key == "grammar") req_grammar_text = tok.substr(eq + 1);
            else if (key == "gprompt") req_gprompt = std::atoll(tok.c_str() + eq + 1);
            else if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
            else if (key == "temperature") req_temperature = fv;
            else if (key == "top_p") req_top_p = fv;
            else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") req_min_p = fv;
            else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
            else if (key == "penalty_repeat") req_penalty_repeat = fv;
            else if (key == "penalty_freq") req_penalty_freq = fv;
            else if (key == "penalty_present") req_penalty_present = fv;
            else if (key == "seed") { req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10); req_seed_given = true; }
            else if (key == "relax") req_relax = fv;
            else if (key == "clip") req_clip = std::atoi(tok.c_str() + eq + 1);
            else if (key == "ppl") req_ppl = std::atoi(tok.c_str() + eq + 1);
            else if (key == "ppl_t") req_ppl_t = std::atoi(tok.c_str() + eq + 1);
            // unknown keys are skipped: the ids start at the first token without '='
        }
    }

    // a sampling value out of range is refused, never changed (the server refuses it first: config.checked_sampling)
    const char* bad = nullptr;
    if (!std::isfinite(req_temperature) || req_temperature < 0.0f) bad = "temperature: a number >= 0";
    else if (!(req_top_p > 0.0f && req_top_p <= 1.0f)) bad = "top_p: 0 < top_p <= 1";
    else if (req_top_k < -1) bad = "top_k: an integer >= 0 (0 or -1 = off)";
    else if (!(req_min_p >= 0.0f && req_min_p <= 1.0f)) bad = "min_p: 0 <= min_p <= 1";
    else if (req_penalty_last_n < 0 || req_penalty_last_n > runtime_.kPenaltyWindowCap)
        bad = "penalty_last_n: 0 to the engine's penalty window (4096)";
    else if (!std::isfinite(req_penalty_repeat) || req_penalty_repeat <= 0.0f) bad = "penalty_repeat: a number > 0";
    else if (!std::isfinite(req_penalty_freq) || !std::isfinite(req_penalty_present))
        bad = "penalty_freq, penalty_present: numbers";
    else if (!(req_relax >= 0.0f && req_relax <= 1.0f)) bad = "relax: 0..1";
    if (bad == nullptr && runtime_.o.batch > 0 && runtime_.piped && req_penalty_last_n > 0 &&
        (req_penalty_repeat != 1.0f || req_penalty_freq != 0.0f || req_penalty_present != 0.0f))
        bad = "penalties: pipelined batch groups are not supported";
    if (bad != nullptr) {
        std::printf("ERR sampling %s\n", bad);
        return Flow::next;
    }

    if (req_bias_present) {
        std::string bias_error;
        if (runtime_.o.batch > 0) bias_error = "continuous batching is not supported";
        const bool ok = runtime_.o.batch <= 0 && basalt::core::parse_logit_bias(req_bias_text, (int) runtime_.n_vocab, req_bias, bias_error);
        if (!ok) { std::printf("ERR logit_bias: %s\n", bias_error.c_str()); return Flow::next; }
    }

    if (geni && endp != nullptr) {
        while (*endp == ' ') ++endp;
        char* gap = std::strchr(endp, ' ');
        if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
    }

    ids_ok = false;
    from_text = false;   // the ids came from a GENT request's text
    if (endp != nullptr) {
        while (*endp == ' ') ++endp;
        if (*endp == '@') {
            std::lock_guard<std::mutex> lk(runtime_.gent_mu);
            const auto it = runtime_.gent_ids.find(std::strtoull(endp + 1, nullptr, 10));
            if (it != runtime_.gent_ids.end()) {
                ids.assign(it->second.begin(), it->second.end());
                runtime_.gent_ids.erase(it);
                ids_ok = from_text = true;
            } else {
                pe = "the tokenized prompt is gone";
            }
        } else {
            ids_ok = cli::parse_i64_list(endp, ids, pe);
        }
    }
    if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !ids_ok) {
        std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
        return Flow::next;
    }
    n = (int64_t) ids.size();
    // ---- Jev: validate before the request can overwrite session state ----
    if (!score_ids.empty()) {
        if (geni || runtime_.batch_on() || runtime_.admit_slot >= 0 || runtime_.solo_slot >= 0 || score_ids.size() > 255 ||
            req_bias_present || !req_grammar_text.empty() || req_ppl >= 0 ||
            std::any_of(score_ids.begin(), score_ids.end(), [&](int64_t t) { return t < 0 || t >= runtime_.n_vocab; }) ||
            score_frontiers.size() > 32 || std::any_of(score_frontiers.begin(), score_frontiers.end(),
                [&](int64_t t) { return t < 1 || t >= n; })) {
            std::printf("ERR bad request: invalid Jev candidates or checkpoint frontiers\n");
            return Flow::next;
        }
        req_relax = 0;
        req_temperature = 0;
        req_penalty_last_n = 0;
        size_t longest = 0;
        for (const auto& b : score_branches) longest = std::max(longest, b.tokens.size());
        if (!score_branches.empty() && (score_branches.size() < 2 || score_branches.size() > (size_t) runtime_.o.jev ||
            score_branches.size() > (size_t) runtime_.ver.max_t() || n - 1 + (int64_t) longest + 9 > runtime_.o.max_context)) {
            std::printf("ERR bad request: Jev branch wave exceeds its lane or context budget\n"); return Flow::next;
        }
    } else if (score_fresh || !score_frontiers.empty() || !score_branches.empty()) {
        std::printf("ERR bad request: Jev checkpoint keys require candidates\n");
        return Flow::next;
    }
    // ---- end Jev ----

    if (!req_grammar_text.empty() && runtime_.grammar_vocab) {
        std::string gerr;
        if (runtime_.o.batch > 0 && runtime_.piped) gerr = "pipelined batch groups are not supported";
        else {
            gguard = std::make_unique<basalt::program::grammar::Guard>(*runtime_.grammar_vocab);
            const size_t gp = req_gprompt > 0 && (size_t) req_gprompt < ids.size() ? (size_t) req_gprompt : ids.size();
            if (!gguard->begin(req_grammar_text, ids.data(), gp, gerr)) gguard.reset();
            else if (gp < ids.size() && gguard->on()) {
                const std::vector<int32_t> done(ids.begin() + (std::ptrdiff_t) gp, ids.end());
                if (gguard->scan(done.data(), (int) done.size()) >= 0) {
                    gerr = "the tokens already generated do not satisfy the grammar";
                    gguard.reset();
                }
            }
            else if (!gguard->on()) gguard.reset();
        }
        if (!gerr.empty()) { std::printf("ERR grammar: %s\n", gerr.c_str()); return Flow::next; }
    } else if (!req_grammar_text.empty()) {
        basalt::log::warn("basalt grammar", "the request has a grammar but this engine runs with --grammar off; ignored");
    }

    grammar_resamples = 0;
    runtime_.req_imgs.clear();
    if (geni && !runtime_.o.vision) { std::printf("ERR this engine was started without --vision\n"); return Flow::next; }

    return Flow::proceed;
}

}  // namespace basalt::program::serve
