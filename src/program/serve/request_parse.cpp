// src/program/serve/request_parse.cpp - request parse operations.
#include "request.hpp"
#include "basalt/core/progress.hpp"
#include "../cli/tokens.hpp"
#include "logprobs.hpp"

#include <algorithm>

namespace basalt::program::serve {

Flow Request::parse() {
    runtime_.stop_req.store(false);   // a STOP that arrived between requests is stale
    runtime_.ram_stop.store(false);
    runtime_.err.clear();
    if (!parse_begin()) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }   // (batch windows swap the draft K/V)
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
    // the horde/koboldcpp filters (tfs/typical/top_a/xtc) and the temperature stage (dynatemp + smoothing):
    // every one off by default; dynatemp is off while dyna_low == dyna_high
    req_tfs = 0.0f;
    req_typical = 0.0f;
    req_top_a = 0.0f;
    req_xtc_prob = 0.0f;
    req_xtc_thresh = 0.1f;
    req_dyna_low = 0.0f;
    req_dyna_high = 0.0f;
    req_dyna_exp = 1.0f;
    req_smooth_factor = 0.0f;
    req_smooth_curve = 1.0f;
    req_bias_present = false;
    req_reply = -1;         // reply=N: the ids past the first N are the reply this request continues, the server's close tokens included
    req_gprompt = -1;       // gprompt=N: the first N ids are the prompt, the rest tokens the request has already generated
    req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
    req_relax = default_relax();   // relax=P: this request's --relaxed-drafting (0 = exact)
    // ppl=K [ppl_t=T]: quality gate (basalt/program/ppl_gate.hpp) - chunk K of --ppl-base, the rest of
    // the chunk teacher-forced through verify windows of T tokens, every token committed
    req_ppl = -1;
    req_ppl_t = 3;
    // score=N [score_top=K]: the ids are a token stream scored from position N on (basalt/program/ppl_gate.hpp)
    req_score = -1;
    req_score_top = 20;
    req_logprobs = -1;   // logprobs=K (0..20): each sent pick's log probability and the row's K likeliest tokens
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
            else if (key == "grammar") req_grammar_text = tok.substr(eq + 1);
            else if (key == "reply") req_reply = std::atoll(tok.c_str() + eq + 1);
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
            else if (key == "tfs") req_tfs = fv;
            else if (key == "typ") req_typical = fv;
            else if (key == "top_a") req_top_a = fv;
            else if (key == "xtc_p") req_xtc_prob = fv;
            else if (key == "xtc_t") req_xtc_thresh = fv;
            else if (key == "dyna_lo") req_dyna_low = fv;
            else if (key == "dyna_hi") req_dyna_high = fv;
            else if (key == "dyna_exp") req_dyna_exp = fv;
            else if (key == "sm_f") req_smooth_factor = fv;
            else if (key == "sm_c") req_smooth_curve = fv;
            else if (key == "seed") { req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10); req_seed_given = true; }
            else if (key == "relax") req_relax = fv;
            else if (key == "logprobs") req_logprobs = std::atoi(tok.c_str() + eq + 1);
            else if (key == "clip") req_clip = std::atoi(tok.c_str() + eq + 1);
            else if (key == "ppl") req_ppl = std::atoi(tok.c_str() + eq + 1);
            else if (key == "ppl_t") req_ppl_t = std::atoi(tok.c_str() + eq + 1);
            else if (key == "score") req_score = std::atoll(tok.c_str() + eq + 1);
            else if (key == "score_top") req_score_top = std::atoi(tok.c_str() + eq + 1);
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
    else if (!std::isfinite(req_tfs) || req_tfs < 0.0f || req_tfs > 1.0f) bad = "tfs: 0 <= tfs <= 1 (0 or 1 = off)";
    else if (!std::isfinite(req_typical) || req_typical < 0.0f || req_typical > 1.0f)
        bad = "typ: 0 <= typical <= 1 (0 or 1 = off)";
    else if (!std::isfinite(req_top_a) || req_top_a < 0.0f || req_top_a > 1.0f) bad = "top_a: 0 <= top_a <= 1";
    else if (!std::isfinite(req_xtc_prob) || req_xtc_prob < 0.0f || req_xtc_prob > 1.0f)
        bad = "xtc_p: 0 <= xtc_p <= 1";
    else if (!std::isfinite(req_xtc_thresh) || req_xtc_thresh < 0.0f || req_xtc_thresh > 0.5f)
        bad = "xtc_t: 0 <= xtc_t <= 0.5";
    else if (!std::isfinite(req_dyna_low) || !std::isfinite(req_dyna_high) || req_dyna_low < 0.0f || req_dyna_high < 0.0f)
        bad = "dyna_lo, dyna_hi: numbers >= 0";
    else if (!std::isfinite(req_dyna_exp) || req_dyna_exp < 0.0f) bad = "dyna_exp: a number >= 0";
    else if (!std::isfinite(req_smooth_factor) || req_smooth_factor < 0.0f) bad = "sm_f: a number >= 0";
    else if (!std::isfinite(req_smooth_curve)) bad = "sm_c: a number";
    else if (!runtime_.o.more_samplers &&
             ((req_tfs > 0.0f && req_tfs < 1.0f) || (req_typical > 0.0f && req_typical < 1.0f) || req_top_a > 0.0f ||
              req_xtc_prob > 0.0f || req_dyna_low != req_dyna_high || req_smooth_factor > 0.0f))
        bad = "tfs, typ, top_a, xtc_p, dyna_*, sm_f: start the engine with --more-samplers on";
    else if ((req_tfs > 0.0f && req_tfs < 1.0f) || (req_typical > 0.0f && req_typical < 1.0f) || req_top_a > 0.0f ||
             req_xtc_prob > 0.0f) {
        // the list filters run on the top_k survivors (koboldcpp's chain applies them after top_k too); the
        // wide path (top_k off or over 64) does not carry them
        if (req_top_k <= 0 || req_top_k > basalt::kernels::kSamplerNarrowMax)
            bad = "tfs, typ, top_a, xtc_p: the list filters need top_k 1..64";
    } else if (req_dyna_low != req_dyna_high || req_smooth_factor > 0.0f) {
        // dynatemp and smoothing reshape the list the pick softmaxes; the engine implements them on the
        // narrow path only (the same list scope), so they are refused on the wide path too
        if (req_top_k <= 0 || req_top_k > basalt::kernels::kSamplerNarrowMax)
            bad = "dyna_lo/dyna_hi, sm_f: the dynamic temperature needs top_k 1..64";
    }
    else if (!(req_relax >= 0.0f && req_relax <= 1.0f)) bad = "relax: 0..1";
    else if (req_logprobs < -1 || req_logprobs > kMaxTopLogprobs) bad = "logprobs: 0 to 20 alternatives";
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
        } else if (*endp == '#') {   // a prompt held by handle, then the ids this request adds after it
            char* after = nullptr;
            std::lock_guard<std::mutex> lk(runtime_.gent_mu);
            const auto it = runtime_.prompts.find(std::strtoull(endp + 1, &after, 10));
            if (it == runtime_.prompts.end()) {
                pe = "the held prompt is gone";
            } else {
                ids.assign(it->second.ids.begin(), it->second.ids.end());
                std::vector<int64_t> more;
                ids_ok = *after != '+' || cli::parse_i64_list(after + 1, more, pe);
                if (!more.empty()) req_reply = (int64_t) it->second.ids.size();   // what follows the held prompt is the reply so far
                ids.insert(ids.end(), more.begin(), more.end());
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

    if (!req_grammar_text.empty() && runtime_.grammar_vocab) {
        std::string gerr;
        gguard = std::make_unique<basalt::program::grammar::Guard>(*runtime_.grammar_vocab);
        const size_t gp = req_gprompt > 0 && (size_t) req_gprompt < ids.size() ? (size_t) req_gprompt : ids.size();
        if (!gguard->begin(req_grammar_text, ids.data(), gp, gerr)) gguard.reset();
        else if (gp < ids.size() && gguard->on()) {
            const std::vector<int32_t> done(ids.begin() + (std::ptrdiff_t) gp, ids.end());
            constexpr int kScanChunk = 4096;
            bool bad = false;
            for (int off = 0; off < (int) done.size(); off += kScanChunk) {
                const int n_chunk = (std::min)(kScanChunk, (int) done.size() - off);
                if (gguard->scan(done.data() + off, n_chunk, 0) >= 0) {
                    bad = true;
                    break;
                }
                basalt::core::progress_beat();
            }
            if (bad) {
                gerr = "the tokens already generated do not satisfy the grammar";
                gguard.reset();
            }
        }
        else if (!gguard->on()) gguard.reset();
        if (!gerr.empty()) { std::printf("ERR grammar: %s\n", gerr.c_str()); return Flow::next; }
    } else if (!req_grammar_text.empty()) {
        basalt::log::warn("basalt grammar", "the request has a grammar but this engine runs with --grammar off; ignored");
    }

    grammar_resamples = 0;
    grammar_dead = false;
    runtime_.req_imgs.clear();
    if (geni && !runtime_.o.vision) { std::printf("ERR this engine was started without --vision or --vision-external\n"); return Flow::next; }

    return Flow::proceed;
}

}  // namespace basalt::program::serve
