// basalt/program/serve/logprobs.hpp - a sent pick's log probability and the row's likeliest tokens.
//
// A request with `logprobs=K` gets one line per sent pick ahead of the pick itself: `P <id> <lp> [<id>:<lp> ...]` (solo)
// or `BP <slot> <id> <lp> [...]` (batch slots, ahead of the slot's BT line). `lp` is the natural log of the pick's
// probability under the model's own distribution: the row's raw head logits, before temperature, penalties, bias,
// top-k/top-p or a grammar mask, as OpenAI reports it. The K alternatives (at most kMaxTopLogprobs) follow, likeliest
// first, ties to the lower id. Plain values in, the line's fields out; no engine state.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace basalt::program::serve {

constexpr int kMaxTopLogprobs = 20;

/// " <tok> <lp>[ <id>:<lp> ...]" for pick `tok` of a row with `n_vocab` logits, with the `top` likeliest tokens.
inline std::string logprob_fields(const float* logits, int n_vocab, int32_t tok, int top) {
    if (top > kMaxTopLogprobs) top = kMaxTopLogprobs;
    if (top < 0) top = 0;
    double mx = -INFINITY;
    for (int i = 0; i < n_vocab; ++i)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - mx);
    const double lse = mx + std::log(sum);
    // the `top` likeliest ids, kept sorted (likeliest first, ties to the lower id) by insertion
    int best[kMaxTopLogprobs] = {};
    int nb = 0;
    for (int i = 0; i < n_vocab && top > 0; ++i) {
        const float v = logits[i];
        if (nb == top && !(v > logits[best[nb - 1]])) continue;
        int j = nb < top ? nb++ : nb - 1;
        for (; j > 0 && v > logits[best[j - 1]]; --j) best[j] = best[j - 1];
        best[j] = i;
    }
    auto lp_of = [&](int id) {
        const double lp = (double) logits[id] - lse;
        return std::isfinite(lp) ? lp : -9999.0;   // a masked (-inf) logit: the floor OpenAI clients expect
    };
    char buf[48];
    std::snprintf(buf, sizeof buf, " %d %.6f", (int) tok, tok >= 0 && tok < n_vocab ? lp_of(tok) : -9999.0);
    std::string out = buf;
    for (int k = 0; k < nb; ++k) {
        std::snprintf(buf, sizeof buf, " %d:%.6f", best[k], lp_of(best[k]));
        out += buf;
    }
    return out;
}

}  // namespace basalt::program::serve
