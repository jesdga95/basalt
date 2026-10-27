// basalt/program/resident_budget.hpp - how --resident-budget-gib and
// --resident-budget-floor-override combine. The floor override raises the largest
// budget the RAM floor will accept. It does not replace a budget already set.
#pragma once

#include <cstdint>

namespace basalt::program {

struct ResidentBudgetChoice {
    uint64_t budget = 0;          ///< the pin, in bytes. 0 = not set yet
    uint64_t floor_override = 0;  ///< raised allowance, in bytes. 0 = not set
    bool base = false;            ///< --resident-budget-gib was given
};

/// `--resident-budget-gib N` is the pin, whichever order it appears in.
inline void set_resident_budget_gib(ResidentBudgetChoice& c, uint64_t bytes) {
    c.budget = bytes;
    c.base = true;
}

/// `--resident-budget-floor-override N` raises the allowance to N when N is higher.
/// With no pin yet, N is also the pin. A pin already set is left as it is.
inline void set_resident_budget_floor_override(ResidentBudgetChoice& c, uint64_t bytes) {
    if (bytes > c.floor_override) c.floor_override = bytes;
    if (!c.base && c.budget == 0) c.budget = bytes;
}

/// The startup decision. `safe` is the RAM floor's largest copy. A floor override
/// only raises that allowance; a pin above the raised allowance is still refused.
/// `above_safe` means the pin is kept past `safe` and the file-cache floor is not held.
struct ResidentBudgetLimit {
    bool refused = false;
    bool above_safe = false;
    uint64_t allowed = 0;
    uint64_t copy_cap = 0;
};

inline ResidentBudgetLimit resident_budget_limit(uint64_t budget, uint64_t safe, uint64_t floor_override) {
    ResidentBudgetLimit r;
    r.allowed = floor_override > safe ? floor_override : safe;
    const bool over_safe = budget > 0 && budget > safe;
    r.refused = budget > 0 && budget > r.allowed;
    r.above_safe = over_safe && !r.refused;
    if (budget == 0) r.copy_cap = safe;
    else if (over_safe) r.copy_cap = budget;
    else r.copy_cap = budget < safe ? budget : safe;
    return r;
}

}  // namespace basalt::program
