// tests/program/batch_plan_test.cpp - a round's windows over a batch's rows: each slot's rows of the cap by its drafts'
// chances or max-min fair over its own window, and the first-fit groups that run as one batch (CPU).
#include "program/serve/batch_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
using Groups = std::vector<std::vector<size_t>>;

// every window in exactly one group, and no group over the cap
bool sound(const std::vector<int>& rows, const Groups& g, int cap) {
    std::vector<int> seen(rows.size(), 0);
    for (const auto& group : g) {
        int n = 0;
        for (const size_t i : group) {
            ++seen[i];
            n += rows[i];
        }
        if (group.empty() || n > cap) return false;
    }
    for (const int s : seen)
        if (s != 1) return false;
    return true;
}
}  // namespace

int main() {
    using basalt::program::serve::batch_groups;
    using basalt::program::serve::batch_fair_shares;
    using basalt::program::serve::batch_shares;
    using Joint = std::vector<std::vector<double>>;
    using Rows = std::vector<int>;

    check(batch_shares({{0.9, 0.8, 0.7}}, 8) == Rows{8}, "one slot: its drafts' rows, then the rest of the cap");
    check(batch_shares({{0.9, 0.8, 0.7}, {0.3, 0.1}}, 8) == Rows{5, 3},
          "the likelier drafts take the rows first, then the leftover rows go round the slots");
    check(batch_shares({{0.9, 0.2}, {0.5, 0.4, 0.3}}, 5) == Rows{2, 3}, "each row to the slot whose next draft is likeliest kept");
    check(batch_shares({{}, {}}, 8) == Rows{4, 4}, "without drafts the rows go round in turn");
    check(batch_shares({{0.5}, {0.5}}, 3) == Rows{2, 1}, "a tie goes to the earlier slot");
    bool whole = true;
    for (int n = 1; n <= 16; ++n) {
        Joint j((size_t) n);
        for (int i = 0; i < n; ++i) j[(size_t) i] = {0.9 / (i + 1), 0.5 / (i + 1)};
        int sum = 0;
        for (const int r : batch_shares(j, 16)) sum += r;
        whole = whole && sum == 16;
    }
    check(whole, "with no more slots than rows the shares add up to the cap, so one pass serves all");
    check(batch_shares(Joint(20), 16) == Rows(20, 16), "more slots than rows: each may take the cap");

    using basalt::program::serve::batch_confident_shares;
    check(batch_confident_shares({{0.9, 0.8, 0.7, 0.6, 0.5}, {0.4, 0.2, 0.1}}, 2, 16) == Rows{5, 3},
          "drafts while all of them stay above half, at least two, at most four");
    check(batch_confident_shares({{0.9, 0.8, 0.3}, {0.9, 0.8, 0.3}}, 8, 16) == Rows{4, 4},
          "a quarter of the slots busy: the bar falls to an eighth");
    check(batch_confident_shares({{0.9}, {}}, 2, 16) == Rows{2, 1}, "no more drafts than the slot has");
    check(batch_confident_shares({{0.9, 0.8, 0.7, 0.6}, {0.9, 0.8, 0.65, 0.55}}, 2, 8) == Rows{4, 4},
          "past the cap the least likely drafts beyond the first two come off first");
    check(batch_confident_shares({{0.9, 0.8, 0.7}, {0.9, 0.8, 0.7}, {0.9, 0.8, 0.7}}, 3, 7) == Rows{3, 2, 2},
          "then a row at a time from the slot with most, the later slot first");
    check(batch_confident_shares(Joint(20), 20, 16) == Rows(20, 16), "more slots than rows: each may take the cap");

    check(batch_fair_shares({3}, 8) == Rows{3}, "one slot keeps its window");
    check(batch_fair_shares({3, 5}, 8) == Rows{3, 5}, "windows that fit together keep their rows");
    check(batch_fair_shares({3, 7}, 8) == Rows{3, 5}, "the window wanting less is served in full, the other takes the rest");
    check(batch_fair_shares({8, 8}, 8) == Rows{4, 4}, "equal wants split evenly");
    check(batch_fair_shares({8, 8, 8}, 8) == Rows{3, 3, 2}, "a remainder row goes to the earlier slot");
    check(batch_fair_shares({1, 8, 2, 8}, 8) == Rows{1, 3, 2, 2}, "small windows in full, the rest split what is left");
    bool filled = true, fair = true;
    for (int n = 1; n <= 16; ++n) {
        Rows want((size_t) n);
        for (int i = 0; i < n; ++i) want[(size_t) i] = 1 + (i * 7) % 16;
        const Rows r = batch_fair_shares(want, 16);
        int sum = 0, wsum = 0, lo = 99, hi = 0;
        for (int i = 0; i < n; ++i) {
            sum += r[(size_t) i];
            wsum += want[(size_t) i];
            if (r[(size_t) i] < want[(size_t) i]) lo = std::min(lo, r[(size_t) i]);
            hi = std::max(hi, r[(size_t) i]);
            fair = fair && r[(size_t) i] >= 1 && r[(size_t) i] <= want[(size_t) i];
        }
        filled = filled && sum == std::min(16, wsum);
        fair = fair && (lo == 99 || hi <= lo + 1);   // a window held short is within a row of the largest share
    }
    check(filled, "with no more slots than rows the shares fill the pass or every window, so one pass serves all");
    check(fair, "every share is 1 to the window's rows, a held window within a row of the largest share");
    check(batch_fair_shares(Rows(20, 3), 16) == Rows(20, 16), "more slots than rows: each may take the cap");

    check(batch_groups({8, 8}, 16) == Groups{{0, 1}}, "two halves run as one batch");
    check(batch_groups({6, 5, 5}, 16) == Groups{{0, 1, 2}}, "three shares run as one batch");
    check(batch_groups({16, 16}, 16) == Groups{{0}, {1}}, "two full windows run one after the other");
    check(batch_groups({10, 8, 6}, 16) == Groups{{0, 2}, {1}}, "first fit: a later window fills the first group");
    check(batch_groups({1, 16, 1}, 16) == Groups{{0, 2}, {1}}, "first fit skips a window that does not fit");
    check(batch_groups({}, 16).empty(), "no windows, no groups");

    bool all = true;
    for (int cap : {4, 9, 16})
        for (int a = 1; a <= cap; ++a)
            for (int b = 1; b <= cap; ++b)
                for (int c = 1; c <= cap; ++c) {
                    const std::vector<int> rows = {a, b, c};
                    all = all && sound(rows, batch_groups(rows, cap), cap);
                }
    check(all, "every window lands in exactly one group within the cap");

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
