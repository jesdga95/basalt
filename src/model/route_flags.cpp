// src/model/route_flags.cpp - see basalt/model/route_flags.hpp.
#include "basalt/model/route_flags.hpp"

#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cstdlib>

namespace basalt::model {
namespace {
constexpr int kRouteTailSkipMaxK = 16;   // the most routed experts per token of any model's router
}  // namespace

bool parse_route_flag(const std::string& a, FlagArgs& args, RouteFlags& r, int& code) {
    if (a == "--route-tail-skip") {   // the value is optional: an integer when one follows
        r.tail_skip = RouteFlags::kTailSkipBare;
        if (args.peek() != nullptr) {
            char* e = nullptr;
            const long v = std::strtol(args.peek(), &e, 10);
            if (e != args.peek() && *e == '\0') {
                args.take();
                if (v < 1 || v >= kRouteTailSkipMaxK) {
                    basalt::log::error("basalt", "--route-tail-skip: R must be 1..k-1 (k: the model's routed experts per token)");
                    code = 2;
                    return true;
                }
                r.tail_skip = (int) v;
            }
        }
    } else if (a == "--route-prior") {   // the value is optional: a number in (0, 16] when one follows
        r.prior = RouteFlags::kPriorBare;
        if (args.peek() != nullptr) {
            char* e = nullptr;
            const float v = std::strtof(args.peek(), &e);
            if (e != args.peek() && *e == '\0') {
                args.take();
                if (!(v > 0.0f && v <= 16.0f)) {
                    basalt::log::error("basalt", "--route-prior: LAMBDA must be in (0, 16]");
                    code = 2;
                    return true;
                }
                r.prior = v;
            }
        }
    } else {
        return false;
    }
    return true;
}

std::optional<int> finish_route_flags(program::CommonOptions& o, int top_k, float prior_bare, const char* area) {
    RouteFlags& r = o.route;
    if (r.prior == RouteFlags::kPriorBare) r.prior = prior_bare;
    if (r.tail_skip != 0) {
        if (top_k < 2) {
            basalt::log::error(area, "--route-tail-skip needs the model's routed experts per token");
            return 2;
        }
        if (r.tail_skip == RouteFlags::kTailSkipBare) r.tail_skip = top_k - 1;
        if (r.tail_skip >= top_k) {
            basalt::log::error(area, "--route-tail-skip: R must be 1..%d (the model routes %d experts per token)", top_k - 1, top_k);
            return 2;
        }
    }
    char effect[128];
    if (r.prior > 0.0f) {
        std::snprintf(effect, sizeof effect, "--route-prior %g changes expert routing", (double) r.prior);
        o.lossy.push_back({"--route-prior", effect});
    }
    if (r.tail_skip > 0) {
        std::snprintf(effect, sizeof effect, "--route-tail-skip %d drops low-ranked missed experts", r.tail_skip);
        o.lossy.push_back({"--route-tail-skip", effect});
    }
    return std::nullopt;
}

}  // namespace basalt::model
