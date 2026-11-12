// basalt/program/serve/logprobs.hpp - the protocol fields of a sent pick's log probability.
//
// A request with `logprobs=K` gets one line per sent pick ahead of the pick itself: `P <id> <lp> [<id>:<lp> ...]` (solo)
// or `BP <slot> <id> <lp> [...]` (batch slots, ahead of the slot's BT line). The model computes the values on the GPU
// (kernels::row_logprobs): `lp` is the natural log of the pick's probability under the row's raw head logits, before
// temperature, penalties, bias, top-k/top-p or a grammar mask, as OpenAI reports it; the K alternatives follow,
// likeliest first, ties to the lower id. Plain values in, the line's fields out; no engine state.
#pragma once

#include "basalt/kernels/logprobs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace basalt::program::serve {

constexpr int kMaxTopLogprobs = basalt::kernels::kLogprobMaxTop;

/// " <tok> <lp>[ <id>:<lp> ...]" from one row of kernels::row_logprobs (kLogprobStride floats) for pick `tok`.
inline std::string logprob_fields(const float* row, int32_t tok, int top) {
    if (top > kMaxTopLogprobs) top = kMaxTopLogprobs;
    auto lp = [](float v) { return std::isfinite(v) ? (double) v : -9999.0; };   // a masked logit: OpenAI's floor
    char buf[48];
    std::snprintf(buf, sizeof buf, " %d %.6f", (int) tok, lp(row[0]));
    std::string out = buf;
    for (int k = 0; k < top; ++k) {
        int32_t id = 0;
        std::memcpy(&id, &row[2 + 2 * k], sizeof id);
        if (id < 0) break;   // a row with fewer tokens than asked
        std::snprintf(buf, sizeof buf, " %d:%.6f", (int) id, lp(row[3 + 2 * k]));
        out += buf;
    }
    return out;
}

}  // namespace basalt::program::serve
