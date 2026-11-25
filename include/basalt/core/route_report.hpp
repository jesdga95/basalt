// include/basalt/core/route_report.hpp - the lines a run with the lossy routing flags ends with: what they changed, from
// the counters route_tail_skip and a residency-biased router add to (basalt/kernels/route_lossy.hpp).
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace basalt::core {

/// c: [0] entries the tail skip saw, [1] of them no GPU held, [2] dropped; [3] entries a counting biased router picked,
/// [4] picked only because of the bias, [5] held by no GPU.  A flag that is off, or saw nothing, gives no line.
inline std::vector<std::string> route_report(const unsigned long long c[6], int tail_skip, float prior) {
    std::vector<std::string> out;
    char b[240];
    if (tail_skip > 0 && c[0] > 0) {
        std::snprintf(b, sizeof b, "route-tail-skip %d: %llu of %llu routed entries no GPU held dropped (%.1f%%); %.1f%% of all %llu entries",
                      tail_skip, c[2], c[1], c[1] > 0 ? 100.0 * (double) c[2] / (double) c[1] : 0.0,
                      100.0 * (double) c[2] / (double) c[0], c[0]);
        out.push_back(b);
    }
    if (prior > 0.0f && c[3] > 0) {
        std::snprintf(b, sizeof b, "route-prior %g: %.1f%% of %llu routed entries picked because of the bias; %.1f%% of entries held by no GPU",
                      (double) prior, 100.0 * (double) c[4] / (double) c[3], c[3], 100.0 * (double) c[5] / (double) c[3]);
        out.push_back(b);
    }
    return out;
}

}  // namespace basalt::core
