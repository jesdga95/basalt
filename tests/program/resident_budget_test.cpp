// --resident-budget-gib and --resident-budget-floor-override: the override raises
// the allowance and leaves a budget that was already set.
#include "basalt/program/resident_budget.hpp"

#include <cstdio>
#include <cstdint>

namespace {

int failures = 0;

constexpr uint64_t GiB = 1073741824ull;

void check(bool ok, const char* what) {
    std::printf("resident budget %-62s %s\n", what, ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) ++failures;
}

using basalt::program::ResidentBudgetChoice;
using basalt::program::resident_budget_limit;
using basalt::program::set_resident_budget_floor_override;
using basalt::program::set_resident_budget_gib;

}  // namespace

int main() {
    {
        ResidentBudgetChoice c;
        set_resident_budget_gib(c, 38 * GiB);
        set_resident_budget_floor_override(c, 39 * GiB);
        check(c.budget == 38 * GiB && c.floor_override == 39 * GiB && c.base,
              "budget 38 then floor override 39 keeps 38");
    }
    {
        ResidentBudgetChoice c;
        set_resident_budget_floor_override(c, 39 * GiB);
        set_resident_budget_gib(c, 38 * GiB);
        check(c.budget == 38 * GiB && c.floor_override == 39 * GiB && c.base,
              "floor override 39 then budget 38 keeps 38");
    }
    {
        ResidentBudgetChoice c;
        set_resident_budget_floor_override(c, 39 * GiB);
        check(c.budget == 39 * GiB && c.floor_override == 39 * GiB && !c.base,
              "floor override alone is the budget");
    }
    {
        ResidentBudgetChoice c;
        set_resident_budget_gib(c, 38 * GiB);
        check(c.budget == 38 * GiB && c.floor_override == 0 && c.base, "budget alone sets no floor override");
    }
    {
        ResidentBudgetChoice c;
        set_resident_budget_floor_override(c, 38 * GiB);
        set_resident_budget_floor_override(c, 39 * GiB);
        check(c.floor_override == 39 * GiB && c.budget == 38 * GiB, "a later floor override only raises");
    }
    {
        ResidentBudgetChoice c;
        set_resident_budget_floor_override(c, 39 * GiB);
        set_resident_budget_floor_override(c, 38 * GiB);
        check(c.floor_override == 39 * GiB && c.budget == 39 * GiB, "a lower floor override does not lower it");
    }

    const uint64_t safe = 37 * GiB + GiB / 2;
    {
        const auto r = resident_budget_limit(38 * GiB, safe, 0);
        check(r.refused && !r.above_safe && r.allowed == safe, "budget 38 with no override is refused");
    }
    {
        const auto r = resident_budget_limit(38 * GiB, safe, 39 * GiB);
        check(!r.refused && r.above_safe && r.allowed == 39 * GiB && r.copy_cap == 38 * GiB,
              "budget 38 within a floor override of 39 is kept");
    }
    {
        const auto r = resident_budget_limit(40 * GiB, safe, 39 * GiB);
        check(r.refused && r.allowed == 39 * GiB, "a budget above the floor override is refused");
    }
    {
        const auto r = resident_budget_limit(36 * GiB, safe, 39 * GiB);
        check(!r.refused && !r.above_safe && r.copy_cap == 36 * GiB, "a budget under the safe max is unchanged");
    }
    {
        const auto r = resident_budget_limit(38 * GiB, safe, 30 * GiB);
        check(r.refused && r.allowed == safe, "a floor override below the safe max does not lower it");
    }
    {
        const auto r = resident_budget_limit(39 * GiB, safe, 39 * GiB);
        check(!r.refused && r.above_safe && r.copy_cap == 39 * GiB, "floor override alone above the safe max is kept");
    }

    std::printf("resident_budget_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
