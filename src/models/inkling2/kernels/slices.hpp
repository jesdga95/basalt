// src/models/inkling2/kernels/slices.hpp - how many slices a full-cache row's keys split into on a card (attn.cu): the
// count whose launches of `heads` blocks a slice end in the fewest partial waves, at every residency the card's attention
// plans run at, per key of work.
#pragma once

#include <vector>

namespace basalt::models::inkling2 {

/// Of the counts lo..hi, the one with the least sum over `resident` (blocks a multiprocessor, one per plan) of the waves
/// a launch of heads x count blocks takes on `sms` multiprocessors, per slice; the larger count on a tie. One count for
/// every plan, so a row's slices do not depend on its window's width.
inline int slice_count(int sms, const std::vector<int>& resident, int heads, int lo, int hi) {
    int best = hi;
    double best_cost = -1.;
    for (int count = hi; count >= lo; --count) {
        double cost = 0.;
        for (const int r : resident) {
            const long long slots = (long long) sms * r, blocks = (long long) heads * count;
            if (slots > 0) cost += (double) ((blocks + slots - 1) / slots) / count;
        }
        if (best_cost < 0. || cost < best_cost - 1e-12) {
            best = count;
            best_cost = cost;
        }
    }
    return best;
}

}  // namespace basalt::models::inkling2
