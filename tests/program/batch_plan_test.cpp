// tests/program/batch_plan_test.cpp - a round's windows over a batch's rows: each slot's rows of the cap by its drafts'
// chances and the first-fit groups that run as one batch (CPU).
#include "program/serve/batch_plan.hpp"

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
