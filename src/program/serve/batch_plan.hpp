// src/program/serve/batch_plan.hpp - how a round's windows share a batch's rows (header-only, no GPU).
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace basalt::program::serve {

/// The most rows each active slot's window may take in a batch of `cap` rows. joint[i][k] is the chance slot i's drafts
/// 0..k are all kept. With no more slots than rows, one pass serves them all: each slot takes one row, then each further
/// row goes to the slot whose next draft is likeliest kept (the earlier slot on a tie), and rows no draft wants go round
/// the slots in turn (a lookup proposal may still take them). With more slots than rows each may take the cap.
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
