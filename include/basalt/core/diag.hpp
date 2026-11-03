// include/basalt/core/diag.hpp - the engine's diagnostic switches, set from the command line before any work starts.
#pragma once

#include <map>
#include <string>

namespace basalt::core {

struct Diag {
    bool profile = false;      ///< --profile: per-stage GPU stamps of the verify window (turns PDL off; not for A/B)
    bool wait_stats = false;   ///< --wait-stats: the MoE tail's wait counters, one report per 200 windows
    bool decode_timing = false;   ///< --decode-timing: the per-request decode timing line
    std::string draft_log;     ///< --draft-log FILE: one line per verify window (width, accepted, ms)
    bool zero_doorbell = false;   ///< --force-zero-doorbell: every window on the zero-doorbell graph (a scheduling test)
    std::map<std::string, double> exp;   ///< --exp NAME=V: short-lived experiment gates, removed when they resolve
};

inline Diag& diagnostics() {
    static Diag d;
    return d;
}

// The value of experiment gate `name` (--exp name=V), or `def` when it was not given.
inline double exp_flag(const char* name, double def = 0.0) {
    const auto& m = diagnostics().exp;
    const auto it = m.find(name);
    return it == m.end() ? def : it->second;
}

}  // namespace basalt::core
