// src/program/serve/batch_plan.hpp - how a round's windows share a batch's rows (header-only, no GPU).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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

/// Drafts with stated chances in a batch round: each slot verifies its leading drafts while the chance that all of them
/// are kept stays above kConfident x the busy share of the slots (with fewer slots busy a round's tokens fall while a row
/// costs about the same), at least kSpecMin and at most kSpecMax of them (as many as it has). After ingine-batch's
/// batch_engine.cpp (Apache-2.0).
constexpr double kConfident = 0.5;
constexpr int kSpecMin = 2, kSpecMax = 4;

/// The rows of each slot's window by that rule (joint[i][k]: the chance slot i's drafts 0..k are all kept; `slots` the
/// batch slots there are), within `cap`: past it the least likely drafts beyond each slot's first kSpecMin come off first,
/// then a row at a time from the slot with most, the later slot on a tie, down to one each. With more slots than rows
/// each may take the cap.
inline std::vector<int> batch_confident_shares(const std::vector<std::vector<double>>& joint, int slots, int cap) {
    const size_t n = joint.size();
    if (n == 0) return {};
    if ((int) n > cap) return std::vector<int>(n, cap);
    const double theta = std::max(0.05, kConfident * (double) n / (double) std::max<size_t>(n, (size_t) slots));
    std::vector<int> rows(n, 1);
    int total = 0;
    for (size_t i = 0; i < n; ++i) {
        int k = 0;
        while (k < kSpecMax && k < (int) joint[i].size() && (joint[i][(size_t) k] > theta || k < kSpecMin)) ++k;
        rows[i] = 1 + k;
        total += rows[i];
    }
    while (total > cap) {   // the least likely draft past the first kSpecMin of its slot, the later slot on a tie
        size_t worst = n;
        for (size_t i = 0; i < n; ++i)
            if (rows[i] - 1 > kSpecMin && (worst == n || joint[i][(size_t) rows[i] - 2] <= joint[worst][(size_t) rows[worst] - 2])) worst = i;
        if (worst == n) break;
        --rows[worst];
        --total;
    }
    while (total > cap) {
        size_t most = 0;
        for (size_t i = 1; i < n; ++i)
            if (rows[i] >= rows[most]) most = i;
        --rows[most];
        --total;
    }
    return rows;
}

/// The rows of a batch round for drafters that set their own windows' length (no chances to weigh): from the rows each window would take
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
