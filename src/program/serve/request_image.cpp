// src/program/serve/request_image.cpp - what every model checks once the prompt's images are placed.
#include "basalt/protocol.hpp"
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::begin_quality_gate(const std::string& base, const std::string& out, int most_rows) {
    ppl = nullptr;
    if (req_ppl < 0 && req_score < 0) return Flow::proceed;
    std::string pe2;
    PplGate::configure(base, out);
    ppl = &PplGate::get();
    const bool begun = req_score >= 0 ? ppl->begin_stream(ids.data(), n, req_score, req_score_top, runtime_.n_vocab, pe2)
                                      : ppl->begin(req_ppl, ids.data(), n, runtime_.n_vocab, pe2);
    if (geni || !begun) {
        std::printf("ERR %s\n", geni ? "ppl: text only" : pe2.c_str());
        ppl = nullptr;
        return Flow::next;
    }
    if (req_score >= 0) {   // the prompt is the stream up to the first scored row; the rest is forced
        n = (decltype(n)) (req_score + 1);
        max_new = (decltype(max_new)) (ppl->end() - ppl->first());
    }
    req_temperature = 0.0f;
    req_ppl_t = std::clamp(req_ppl_t, 1, most_rows);
    return Flow::proceed;
}

Flow Request::prepare_prompt() {
    if (const Flow flow = prepare_image(); flow != Flow::proceed) return flow;
    if (from_text && req_clip && runtime_.admit_slot < 0) {
        const long long room = (long long) (runtime_.o.max_context - 8 - n);
        if (room < 1) {
            basalt::protocol::emit_text(basalt::protocol::context_no_room_line((long long) n, (long long) runtime_.o.max_context));
            return Flow::next;
        }
        max_new = std::min(max_new, room);
    }
    {
        const int64_t ctx = runtime_.admit_slot >= 0 ? runtime_.slot_context(runtime_.admit_slot) : runtime_.o.max_context;
        const long long want = runtime_.admit_slot >= 0 ? runtime_.admit_max_new : (long long) max_new;   // a slot's request: its own max_new
        if (n + want + 8 > ctx) {
            basalt::protocol::emit_text(basalt::protocol::context_overflow_line((long long) n, want, (long long) ctx,
                                                                                runtime_.admit_slot));
            runtime_.admit_slot = -1;
            return Flow::next;
        }
    }
    if (from_text) { std::printf("PT %lld %lld\n", (long long) n, (long long) max_new); std::fflush(stdout); }
    bad = false;
    for (int64_t t : ids) bad = bad || t < 0 || t >= runtime_.n_vocab;
    if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); return Flow::next; }
    return admit_room();
}

}  // namespace basalt::program::serve
