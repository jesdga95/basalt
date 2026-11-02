// src/program/serve/request.cpp - ordered execution of one request.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::execute() {
    struct Scope {   // what tells the idle conversations' parking where this request works, until it ends
        Runtime& r;
        ~Scope() {
            if (r.jev_scoring && r.jev_cache.armed) r.jev_cache.touch();
            r.solo_out = nullptr; r.solo_busy = false; r.park_keep = -1; r.jev_scoring = false;
        }
    } scope{runtime_};
    if (const Flow flow = parse(); flow != Flow::proceed) return flow;
    // ---- Jev: discard only checkpoints whose draft state was never prepared ----
    runtime_.jev_scoring = !score_ids.empty();
    if (!runtime_.expire_jev(runtime_.err, !runtime_.jev_scoring)) {
        std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error;
    }
    if (!runtime_.jev_scoring) runtime_.ver.release_score_workspace();
    if (!runtime_.jev_scoring && runtime_.jev_draft_prefix >= 0) {
        runtime_.ckd_finish();
        std::erase_if(runtime_.checks, [&](const ConvCheckpoint& c) {
            return (int64_t) c.ids.size() > runtime_.jev_draft_prefix;
        });
        runtime_.jev_draft_prefix = -1;
    }
    // ---- end Jev ----
    if (const Flow flow = prepare_image(); flow != Flow::proceed) return flow;
    if (const Flow flow = restore(); flow != Flow::proceed) return flow;
    if (runtime_.jev_scoring) runtime_.jev_cache.touch();
    if (const Flow flow = read_prompt(); flow != Flow::proceed) return flow;
    if (const Flow flow = decode(); flow != Flow::proceed) return flow;
    return complete();
}

}  // namespace basalt::program::serve
