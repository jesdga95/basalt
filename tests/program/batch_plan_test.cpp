// tests/program/batch_plan_test.cpp - a round's windows over a batch's rows: each slot's share of the cap and the
// first-fit groups that run as one batch (CPU).
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
    using basalt::program::serve::batch_share;

    check(batch_share(0, 1, 16) == 16, "one slot takes the whole cap");
    check(batch_share(0, 2, 16) == 8 && batch_share(1, 2, 16) == 8, "two slots take half each");
    check(batch_share(0, 3, 16) == 6 && batch_share(1, 3, 16) == 5 && batch_share(2, 3, 16) == 5,
          "three slots: the first takes the remainder row");
    bool whole = true;
    for (int n = 1; n <= 16; ++n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) sum += batch_share(i, n, 16);
        whole = whole && sum == 16;
    }
    check(whole, "with no more slots than rows the shares add up to the cap, so one pass serves all");
    check(batch_share(0, 20, 16) == 16 && batch_share(19, 20, 16) == 16, "more slots than rows: each may take the cap");

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
