// src/platform/cpu.cpp - see include/basalt/platform/cpu.hpp.
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "affinity_linux.hpp"
#endif

namespace basalt::platform {

#if defined(_WIN32)
// --------------------------------------------------------------------------------------------- Windows

CpuTopology detect_cpu_topology(bool skip_first, CpuAffinity affinity) {
    CpuTopology topo;
    // Ask the OS rather than assuming a layout: hardware_concurrency() counts LOGICAL processors, and on an SMT
    // machine half of them are siblings, so one worker on each of the first N would put two workers on one core.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        // No detailed topology available: keep the group identity and treat every processor as its own core.
        const WORD groups = GetActiveProcessorGroupCount();
        for (WORD group = 0; group < groups; ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            if (count == 0 || count == (DWORD) -1 || count > 64) continue;
            for (DWORD i = 0; i < count; ++i) topo.worker_cores.push_back((int) group * 64 + (int) i);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        topo.p_threads = (int) topo.worker_cores.size();
        topo.p_cores = topo.p_threads;
        return topo;
    }
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        // The size probe succeeded but the read did not (the layout changed underneath, say). Same answer as no
        // topology at all: one worker per logical processor is never wrong, just not always optimal.
        const DWORD n = (DWORD) std::thread::hardware_concurrency();
        for (DWORD i = 0; i < n; ++i) topo.worker_cores.push_back((int) i);
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        topo.p_threads = (int) topo.worker_cores.size();
        topo.p_cores = topo.p_threads;
        return topo;
    }
    struct CoreDesc {
        uint8_t efficiency = 0;
        bool has_smt = false;
        std::vector<int> lps;
    };
    std::vector<CoreDesc> descs;
    for (const char *p = buf.data(), *end = buf.data() + len; p < end;) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship == RelationProcessorCore) {
            CoreDesc cd;
            cd.efficiency = e->Processor.EfficiencyClass;
            cd.has_smt = (e->Processor.Flags & LTP_PC_SMT) != 0;
            for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY& ga = e->Processor.GroupMask[g];
                for (int bit = 0; bit < 64; ++bit)
                    if (ga.Mask & (KAFFINITY(1) << bit)) cd.lps.push_back((int) (ga.Group * 64 + bit));
            }
            if (!cd.lps.empty()) descs.push_back(std::move(cd));
        }
        p += e->Size;
    }
    uint8_t min_eff = 255, max_eff = 0;
    for (const auto& c : descs) {
        min_eff = (std::min)(min_eff, c.efficiency);
        max_eff = (std::max)(max_eff, c.efficiency);
    }
    topo.is_hybrid = (max_eff > min_eff);
    if (topo.is_hybrid) {
        for (const auto& c : descs) {
            if (c.efficiency == max_eff) {
                topo.p_cores++;
                topo.p_threads += (int) c.lps.size();
            } else {
                topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) descs.size();
        for (const auto& c : descs) topo.p_threads += (int) c.lps.size();
    }
    if (affinity == CpuAffinity::All || !topo.is_hybrid) {
        // On a hybrid CPU the P-cores come first, so a pool smaller than the core count runs on the P-cores rather
        // than on whatever the OS happened to number first. All cores alike: the order is unchanged.
        if (topo.is_hybrid)
            std::stable_sort(descs.begin(), descs.end(),
                             [](const CoreDesc& x, const CoreDesc& y) { return x.efficiency > y.efficiency; });
        for (const auto& c : descs) topo.worker_cores.push_back(c.lps[0]);
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }
    // Hybrid with Auto or PCores affinity: one logical processor per P-core first (an SMT sibling would contend for
    // the same core's bandwidth), then the P-core siblings, then the E-cores as overflow.
    std::vector<int> p_primaries, p_siblings, e_cores;
    for (const auto& c : descs) {
        if (c.efficiency == max_eff) {
            p_primaries.push_back(c.lps[0]);
            for (size_t s = 1; s < c.lps.size(); ++s) p_siblings.push_back(c.lps[s]);
        } else {
            for (int lp : c.lps) e_cores.push_back(lp);
        }
    }
    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }
    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != CpuAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
}

CpuInfo cpu_info() {
    CpuInfo info;
    char name[256] = {};
    DWORD size = sizeof name;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, name, &size) == ERROR_SUCCESS)
        info.name = name;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            for (const char *p = buf.data(), *end = buf.data() + len; p < end;) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
                if (e->Relationship == RelationProcessorCore) ++info.physical_cores;
                p += e->Size;
            }
        }
    }
    return info;
}

#else
// --------------------------------------------------------------------------------------------- Linux

CpuTopology detect_cpu_topology(bool skip_first, CpuAffinity affinity) {
    CpuTopology topo;
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE: SMT siblings share a core's load/store
    // bandwidth, so a worker on each would put two workers on one core.  sysfs names each CPU's (package, core); the
    // first allowed CPU of each pair is kept, so a taskset that leaves out the first sibling still gets its core.
    // Without sysfs every allowed CPU counts.
    auto topo_read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    auto cap_read = [](int cpu) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };

    std::vector<int> allowed;
    std::vector<unsigned long> allowed_mask;
    if (detail::get_thread_affinity(allowed_mask, &allowed) != 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }

    struct CoreLinux {
        int cpu = -1;
        long pkg = -1;
        long core = -1;
        long cap = -1;
        bool is_sibling = false;
    };
    std::vector<CoreLinux> all_cpus;
    std::vector<std::pair<long, long>> seen_phys;
    long max_cap = 0, min_cap = 1000000;

    for (int cpu : allowed) {
        CoreLinux cl;
        cl.cpu = cpu;
        cl.pkg = topo_read(cpu, "physical_package_id");
        cl.core = topo_read(cpu, "core_id");
        cl.cap = cap_read(cpu);
        if (cl.cap > 0) {
            max_cap = (std::max)(max_cap, cl.cap);
            min_cap = (std::min)(min_cap, cl.cap);
        }
        if (cl.pkg >= 0 && cl.core >= 0) {
            const std::pair<long, long> key{cl.pkg, cl.core};
            if (std::find(seen_phys.begin(), seen_phys.end(), key) != seen_phys.end()) {
                cl.is_sibling = true;
            } else {
                seen_phys.push_back(key);
            }
        }
        all_cpus.push_back(cl);
    }

    // A core class is a capacity band, not one value: cores of one class report different capacities.
    // Cores above the midpoint of the extremes are the fast class; a gap under 10% is one class.
    const long mid_cap = (max_cap + min_cap) / 2;
    const bool cap_gap = max_cap > 0 && (long long) max_cap * 10 > (long long) min_cap * 11;
    topo.is_hybrid = cap_gap;
    if (topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (cl.cap > mid_cap) {
                if (!cl.is_sibling) topo.p_cores++;
                topo.p_threads++;
            } else {
                if (!cl.is_sibling) topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) seen_phys.size();
        topo.p_threads = (int) all_cpus.size();
    }

    if (affinity == CpuAffinity::All || !topo.is_hybrid) {
        if (topo.is_hybrid)
            std::stable_sort(all_cpus.begin(), all_cpus.end(),
                             [](const CoreLinux& x, const CoreLinux& y) { return x.cap > y.cap; });
        for (const auto& cl : all_cpus) {
            if (!cl.is_sibling) topo.worker_cores.push_back(cl.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }

    // Hybrid CPU:
    std::vector<int> p_primaries;
    std::vector<int> p_siblings;
    std::vector<int> e_cores;

    for (const auto& cl : all_cpus) {
        if (cl.cap > mid_cap) {
            if (!cl.is_sibling) p_primaries.push_back(cl.cpu);
            else p_siblings.push_back(cl.cpu);
        } else {
            e_cores.push_back(cl.cpu);
        }
    }

    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }

    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != CpuAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
}

CpuInfo cpu_info() {
    CpuInfo info;
    // /proc/cpuinfo lists every logical CPU; a core is one (physical id, core id) pair
    std::ifstream f("/proc/cpuinfo");
    std::set<std::pair<std::string, std::string>> cores;
    std::string line, pkg, core;
    const auto value = [](const std::string& l) {
        const size_t c = l.find(':');
        if (c == std::string::npos) return std::string();
        const size_t b = l.find_first_not_of(" \t", c + 1);
        return b == std::string::npos ? std::string() : l.substr(b);
    };
    const auto flush = [&] {
        if (!core.empty()) cores.insert({pkg, core});
        pkg.clear();
        core.clear();
    };
    while (std::getline(f, line)) {
        if (line.empty()) flush();
        else if (line.rfind("model name", 0) == 0 && info.name.empty()) info.name = value(line);
        else if (line.rfind("physical id", 0) == 0) pkg = value(line);
        else if (line.rfind("core id", 0) == 0) core = value(line);
    }
    flush();
    info.physical_cores = (int) cores.size();
    if (info.physical_cores == 0) info.physical_cores = (int) detect_cpu_topology(false).worker_cores.size();
    return info;
}

#endif

}  // namespace basalt::platform
