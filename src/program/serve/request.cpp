// src/program/serve/request.cpp - ordered execution of one request.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::execute() {
    struct Scope {   // what tells the idle conversations' parking where this request works, until it ends
        Runtime& r;
        ~Scope() { r.solo_out = nullptr; r.solo_busy = false; r.park_keep = -1; }
    } scope{runtime_};
    auto finish_request = [&](Flow flow) -> Flow {
        if (flow == Flow::next) {
            runtime_.admit_slot = -1;
            if (!leave_in_slot()) {
                std::printf("ERR %s\n", runtime_.err.c_str());
                std::fflush(stdout);
                return Flow::error;
            }
            std::fflush(stdout);
        } else if (flow == Flow::error) {
            std::fflush(stdout);
        }
        return flow;
    };
    if (const Flow flow = parse(); flow != Flow::proceed) return finish_request(flow);
    if (const Flow flow = prepare_prompt(); flow != Flow::proceed) return finish_request(flow);
    if (const Flow flow = restore(); flow != Flow::proceed) return finish_request(flow);
    if (const Flow flow = read_prompt(); flow != Flow::proceed) return finish_request(flow);
    if (const Flow flow = decode(); flow != Flow::proceed) return finish_request(flow);
    return finish_request(complete());
}

}  // namespace basalt::program::serve
