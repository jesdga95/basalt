// src/program/serve/request.cpp - ordered execution of one request.
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::execute() {
    if (const Flow flow = parse(); flow != Flow::proceed) return flow;
    if (const Flow flow = prepare_image(); flow != Flow::proceed) return flow;
    if (const Flow flow = restore(); flow != Flow::proceed) return flow;
    if (const Flow flow = read_prompt(); flow != Flow::proceed) return flow;
    if (const Flow flow = decode(); flow != Flow::proceed) return flow;
    return complete();
}

}  // namespace basalt::program::serve
