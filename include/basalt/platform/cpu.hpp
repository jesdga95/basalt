// include/basalt/platform/cpu.hpp - which processors there are, and putting a thread on one.
//
// A processor is named by one int: its OS number on Linux, `group * 64 + index` on Windows, where a machine with more
// than 64 logical processors has several processor groups.
#pragma once

#include <string>
#include <vector>

namespace basalt::platform {

/// How worker threads are allocated across physical and logical processors.
enum class CpuAffinity {
    Auto,      ///< Hybrid: prioritize physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Restrict workers strictly to Performance cores and their SMT siblings
    All,       ///< The default: one worker per physical core in the OS's order, without hybrid distinction
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Total logical threads on performance cores
    int e_cores = 0;                ///< Efficient cores
    std::vector<int> worker_cores;  ///< Ordered CPU IDs, one per physical core unless the affinity adds siblings
    int host_core = -1;             ///< Logical core reserved for host thread
};

/// The processors this process may run on, one per physical core (SMT siblings share a core's load/store bandwidth),
/// ordered for `affinity`. `skip_first` takes the first one out of `worker_cores` and names it `host_core`.
CpuTopology detect_cpu_topology(bool skip_first, CpuAffinity affinity = CpuAffinity::All);

/// What the machine's processor is called and how many physical cores it has, for a benchmark row. `name` is the
/// vendor string as the OS gives it (empty when it gives none); `physical_cores` counts every core the OS lists,
/// whether or not this process may run on it (0 when unknown).
struct CpuInfo {
    std::string name;
    int physical_cores = 0;
};
CpuInfo cpu_info();

}  // namespace basalt::platform
