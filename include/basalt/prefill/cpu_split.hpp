// include/basalt/prefill/cpu_split.hpp - how many of a layer's RAM experts the CPU computes while the main card stages the
// rest over its bus: the count at which the two priced times end together.
#pragma once

#include <algorithm>
#include <vector>

namespace basalt::prefill {

/// The prices of the split, fixed for a run (no timing during a request, so the same request takes the same split every
/// run). Both sides read the same pinned memory: each rate is the one measured with the other side running.
struct CpuBusRates {
    double cpu_bytes_per_ms = 0;    ///< the CPU pool's reads of expert weights
    double cpu_ms_per_row = 0;      ///< the CPU's work per (token, expert) row beyond the read
    double card_bytes_per_ms = 0;   ///< the main card's staging of experts from RAM and their products
};

/// Of `n` RAM experts of `blob` bytes each, how many the CPU takes: it takes candidates in the order of `cand_rows` (their
/// rows, the fewest first), as many as make the longer of its time and the card's time for the rest the shortest. 0 when a
/// rate is missing.
inline int cpu_take(const CpuBusRates& r, double blob, int n, const std::vector<int>& cand_rows) {
    if (r.cpu_bytes_per_ms <= 0 || r.card_bytes_per_ms <= 0 || n <= 0) return 0;
    const double cpu_exp = blob / r.cpu_bytes_per_ms, card_exp = blob / r.card_bytes_per_ms;
    int best = 0;
    double best_t = n * card_exp, cpu_t = 0;
    const int m = std::min<int>((int) cand_rows.size(), n);
    for (int k = 1; k <= m; ++k) {
        cpu_t += cpu_exp + cand_rows[(size_t) k - 1] * r.cpu_ms_per_row;
        const double t = std::max(cpu_t, (n - k) * card_exp);
        if (t < best_t) {
            best_t = t;
            best = k;
        }
    }
    return best;
}

}  // namespace basalt::prefill
