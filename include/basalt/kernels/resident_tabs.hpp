// include/basalt/kernels/resident_tabs.hpp - which experts the GPUs hold, as the router kernels read it.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// One layer's residency tables, one per GPU tier ([n_expert]: the expert's slot, or -1). A GPU holds an expert
/// when any table has a slot for it.
struct ResidentTabs {
    static constexpr int kMax = 4;
    const int32_t* t[kMax] = {};
    int n = 0;
    void add(const int32_t* tab) { if (tab != nullptr && n < kMax) t[n++] = tab; }
};

}  // namespace basalt::kernels
