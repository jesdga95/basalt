// src/program/serve/batch_plan.hpp - how a round's windows share a batch's rows (header-only, no GPU).
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace basalt::program::serve {

/// The most rows the i-th of n active slots' window may take in a batch of `cap` rows: with no more slots than rows each
/// takes its share (the first cap % n one more), so one pass serves them all; with more slots each may take the cap.
inline int batch_share(int i, int n, int cap) {
    return n <= cap ? cap / n + (i < cap % n ? 1 : 0) : cap;
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
