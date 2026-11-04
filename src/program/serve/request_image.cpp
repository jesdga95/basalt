// src/program/serve/request_image.cpp - what every model checks once the prompt's images are placed.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::prepare_prompt() {
    if (const Flow flow = prepare_image(); flow != Flow::proceed) return flow;
    if (from_text && req_clip && runtime_.admit_slot < 0) {
        const long long room = (long long) (runtime_.o.max_context - 8 - n);
        if (room < 1) {
            std::printf("ERR prompt (%lld tokens) leaves no room to answer in the context (%lld)\n", (long long) n,
                        (long long) runtime_.o.max_context);
            std::fflush(stdout);
            return Flow::next;
        }
        max_new = std::min(max_new, room);
    }
    {
        const int64_t ctx = runtime_.admit_slot >= 0 ? runtime_.slot_context(runtime_.admit_slot) : runtime_.o.max_context;
        const long long want = runtime_.admit_slot >= 0 ? runtime_.admit_max_new : (long long) max_new;   // a slot's request: its own max_new
        if (n + want + 8 > ctx) {
            if (runtime_.admit_slot >= 0)
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds slot %d's context (%lld)\n", (long long) n,
                            want, runtime_.admit_slot, (long long) ctx);
            else
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            want, (long long) ctx);
            std::fflush(stdout);
            runtime_.admit_slot = -1;
            return Flow::next;
        }
    }
    if (from_text) { std::printf("PT %lld %lld\n", (long long) n, (long long) max_new); std::fflush(stdout); }
    bad = false;
    for (int64_t t : ids) bad = bad || t < 0 || t >= runtime_.n_vocab;
    if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); return Flow::next; }

    return Flow::proceed;
}

}  // namespace basalt::program::serve
