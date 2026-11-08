// include/basalt/program/expert_spill.hpp - the routed experts no GPU and no RAM tier holds, which decode reads from
// the SSD: one warning line and one set of INFO words for every model.
#pragma once

#include <cstdint>
#include <string>

#include "basalt/log.hpp"

namespace basalt::program {

struct ExpertSpill {
    int64_t experts = 0;   ///< (layer, expert) pairs only the file holds
    int64_t total = 0;     ///< all routed (layer, expert) pairs
    uint64_t bytes = 0;    ///< their bytes
};

/// The start's warning when any expert stays on the SSD. `levers` names the model's own fixes beside freeing RAM ("" for
/// none), as ", or a, b".
inline void warn_spill(const ExpertSpill& s, const std::string& levers) {
    if (s.experts <= 0) return;
    const double gib = (double) s.bytes / 1073741824.0;
    basalt::log::warn("basalt", "%lld experts on SSD (%.1f GiB) \xc2\xb7 free %.1f GiB RAM%s", (long long) s.experts, gib, gib,
                      levers.c_str());
}

/// `spill_experts=N spill_total=N spill_bytes=N` for the engine's INFO line.
inline std::string spill_words(const ExpertSpill& s) {
    return "spill_experts=" + std::to_string(s.experts) + " spill_total=" + std::to_string(s.total) +
           " spill_bytes=" + std::to_string(s.bytes);
}

}  // namespace basalt::program
