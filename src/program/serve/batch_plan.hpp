// src/program/serve/batch_plan.hpp - how a round's windows share a batch's rows (header-only, no GPU).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::program::serve {

/// The drafts to ask for the next window: up to one more than the window just verified used (at least 2), so a chained
/// drafter's long window grows a row at a time where its drafts pay.  A block drafter makes its whole block in one pass and
/// offers every draft of it; the drafts read back are as many as the session made.
inline int drafts_ahead(int T, int most) { return std::clamp(std::max(T, 2), 1, most - 1); }

/// The most rows each active slot's window may take in a batch of `cap` rows. joint[i][k] is the chance slot i's drafts
/// 0..k are all kept. With no more slots than rows, one pass serves them all: each slot takes one row, then each further
/// row goes to the slot whose next draft is likeliest kept (the earlier slot on a tie), and rows no draft wants go round
/// the slots in turn (a lookup proposal may still take them). With more slots than rows each may take the cap.
/// Tile-cliff-aware cost model of a multi-slot decode_batch(M) pass over M total batch rows (1..kMaxM) on sm_120:
/// - Within M <= 8 (CTILES = 1 Q8 MMA), marginal cost is light since each slot's policy already filtered low-yield drafts.
/// - Crossing M = 8 -> 9 jumps Q8 projections (w.proj, w.out, lm_head) from CTILES = 1 to CTILES = 2.
/// - Crossing M = 12 -> 13 switches routed MoE (10 * M >= 128) to grouped MMA.
struct BatchRowCost {
    static constexpr int kMaxM = 128;

    static double cost(int m, int64_t max_pos) {
        m = std::clamp(m, 1, kMaxM);
        const double slope = max_pos >= 16384 ? 0.060 : max_pos >= 4096 ? 0.070 : 0.080;
        double c = 1.0 + slope * (double) (m - 1);
        if (m > 8) c += (max_pos >= 8192 ? 0.06 : 0.13);
        if (m > 12) c += (max_pos >= 8192 ? 0.04 : 0.06);
        if (m > 16) c += 0.08;
        return c;
    }
};

/// Global expected-throughput-maximizing row allocation across active batch slots: starts at 1 anchor row per slot
/// (M = n, E = n) and greedily adds the highest-joint-survival pre-filtered draft across slots up to `per_slot_cap` / `cap`,
/// picking the prefix M* that maximizes E(M) / BatchRowCost::cost(M, max_pos) (looking across M = 8->12+ tile cliffs when
/// multiple strong code/struct drafts pay together, while clamping weak prose/mix tails at M <= 8).
inline std::vector<int> batch_ev_shares(const std::vector<std::vector<double>>& joint, int per_slot_cap, int cap,
                                        int64_t max_pos = 0) {
    const size_t n = joint.size();
    if (n == 0) return {};
    if ((int) n >= cap) return std::vector<int>(n, 1);
    std::vector<int> rows(n, 1);
    int cur_m = (int) n;
    double cur_e = (double) n;
    double best_rate = cur_e / BatchRowCost::cost(cur_m, max_pos);
    std::vector<int> best_rows = rows;
    while (cur_m < cap) {
        size_t best_i = n;
        double best_j = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (rows[i] >= per_slot_cap) continue;
            const size_t k = (size_t) rows[i] - 1;
            if (k < joint[i].size() && joint[i][k] > best_j) {
                best_j = joint[i][k];
                best_i = i;
            }
        }
        if (best_i == n || best_j <= 0.05) break;
        // Step near-tied slots (e.g. synchronized multi-sample / shared-prefix slots) together when they fit within `cap`
        // so breaking a tie mid-cliff never desynchronizes identical trajectories or fragments shared MoE expert coalescing.
        std::vector<size_t> group;
        for (size_t i = 0; i < n; ++i) {
            if (rows[i] >= per_slot_cap) continue;
            const size_t k = (size_t) rows[i] - 1;
            if (k < joint[i].size() && best_j - joint[i][k] <= 1e-4) group.push_back(i);
        }
        if ((int) group.size() > 1 && cur_m + (int) group.size() <= cap) {
            for (const size_t i : group) {
                const size_t k = (size_t) rows[i] - 1;
                ++rows[i];
                ++cur_m;
                cur_e += joint[i][k];
            }
        } else {
            ++rows[best_i];
            ++cur_m;
            cur_e += best_j;
        }
        const double rate = cur_e / BatchRowCost::cost(cur_m, max_pos);
        if (rate > best_rate) {
            best_rate = rate;
            best_rows = rows;
        }
    }
    return best_rows;
}

inline std::vector<int> batch_shares(const std::vector<std::vector<double>>& joint, int cap) {
    const size_t n = joint.size();
    if (n == 0) return {};
    if ((int) n > cap) return std::vector<int>(n, cap);
    std::vector<int> rows(n, 1);
    int left = cap - (int) n;
    for (; left > 0; --left) {
        size_t best = n;
        for (size_t i = 0; i < n; ++i) {
            const size_t k = (size_t) rows[i] - 1;   // the draft its next row would verify
            if (k < joint[i].size() && joint[i][k] > 0 && (best == n || joint[i][k] > joint[best][(size_t) rows[best] - 1])) best = i;
        }
        if (best == n) break;
        ++rows[best];
    }
    for (size_t i = 0; left > 0; i = (i + 1) % n, --left) ++rows[i];
    return rows;
}

/// The same for drafters that set their own windows' length (no chances to weigh): from the rows each window would take
/// alone (`want`, at least 1 each). With no more slots than rows one pass serves them all: windows that fit together keep their rows, else the
/// rows are shared max-min fairly (the windows wanting least are served in full, the others split what is left evenly, a
/// remainder row going to the window wanting less, then the earlier slot), so no slot is held to one row while another
/// takes the pass. With more slots than rows each may take the cap.
inline std::vector<int> batch_fair_shares(const std::vector<int>& want, int cap) {
    const size_t n = want.size();
    if (n == 0) return {};
    if ((int) n > cap) return std::vector<int>(n, cap);
    std::vector<size_t> ord(n);
    for (size_t i = 0; i < n; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return want[a] < want[b]; });
    std::vector<int> rows(n, 1);
    int left = cap;
    for (size_t k = 0; k < n; ++k) {
        const int rest = (int) (n - k);
        const int share = std::max(1, (left + rest - 1) / rest);
        rows[ord[k]] = std::max(1, std::min(want[ord[k]], share));
        left -= rows[ord[k]];
    }
    return rows;
}

/// The windows of `rows` packed into groups of at most `cap` rows, first fit in slot order: each group starts at the
/// first window not yet taken and takes every later one that still fits.
inline std::vector<std::vector<size_t>> batch_groups(const std::vector<int>& rows, int cap) {
    std::vector<std::vector<size_t>> out;
    std::vector<bool> taken(rows.size(), false);
    for (size_t first = 0; first < rows.size(); ++first) {
        if (taken[first]) continue;
        std::vector<size_t> group;
        int n = 0;
        for (size_t i = first; i < rows.size(); ++i)
            if (!taken[i] && n + rows[i] <= cap) {
                group.push_back(i);
                taken[i] = true;
                n += rows[i];
            }
        out.push_back(std::move(group));
    }
    return out;
}

}  // namespace basalt::program::serve
