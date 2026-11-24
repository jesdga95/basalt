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

/// An Arm processor's name from /proc/cpuinfo text, which has no "model name" there: each core type's count and name from
/// its implementer and part ids, the most numerous first ("10x Cortex-X925 + 10x Cortex-A725"). A part this table does
/// not know is named by its ids ("Arm 0x41:0xd99"). Empty when the text lists no Arm parts.
std::string arm_cpu_name(const std::string& cpuinfo);

/// The logical processors that share each last-level (L3) cache, one group per cache in the OS's order: an AMD CCD, or an
/// Intel die. Processors are named as above. Empty when the OS does not say.
std::vector<std::vector<int>> cpu_l3_groups();

/// Arm64: whether the CPU, as the OS reports it, has the Armv8.2 dot product (sdot / udot), which every NEON kernel
/// of the engine uses. False on x86-64.
bool cpu_has_arm_dotprod();
/// Arm64: whether it also has the int8 matrix multiply extension (I8MM: smmla, usdot). False on x86-64.
bool cpu_has_arm_i8mm();

}  // namespace basalt::platform
