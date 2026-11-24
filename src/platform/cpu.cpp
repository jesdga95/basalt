// src/platform/cpu.cpp - see include/basalt/platform/cpu.hpp.
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "affinity_linux.hpp"
#if defined(__aarch64__)
#include <sys/auxv.h>
#endif
#endif

namespace basalt::platform {

std::string arm_cpu_name(const std::string& cpuinfo) {
    struct Part { unsigned impl, part; const char* name; };
    static const Part kParts[] = {
        {0x41, 0xd0c, "Neoverse-N1"}, {0x41, 0xd40, "Neoverse-V1"}, {0x41, 0xd49, "Neoverse-N2"}, {0x41, 0xd4f, "Neoverse-V2"},
        {0x41, 0xd80, "Cortex-A520"}, {0x41, 0xd81, "Cortex-A720"}, {0x41, 0xd82, "Cortex-X4"},   {0x41, 0xd84, "Neoverse-V3"},
        {0x41, 0xd85, "Cortex-X925"}, {0x41, 0xd87, "Cortex-A725"}, {0x41, 0xd8e, "Neoverse-N3"}, {0x4e, 0x004, "Carmel"},
        {0x4e, 0x010, "Olympus"},
    };
    std::vector<std::pair<std::string, int>> counts;   // in first-seen order, so equal counts keep the OS's order
    const auto add = [&](long impl, long part) {
        if (impl < 0 || part < 0) return;
        std::string name;
        for (const Part& p : kParts)
            if (p.impl == (unsigned) impl && p.part == (unsigned) part) name = p.name;
        if (name.empty()) {
            char buf[48];
            std::snprintf(buf, sizeof buf, "Arm 0x%lx:0x%lx", impl, part);
            name = buf;
        }
        for (auto& c : counts)
            if (c.first == name) { ++c.second; return; }
        counts.push_back({name, 1});
    };
    std::istringstream in(cpuinfo);
    std::string line;
    long impl = -1, part = -1;
    const auto hex = [](const std::string& l) {
        const size_t c = l.find(':');
        return c == std::string::npos ? -1L : std::strtol(l.c_str() + c + 1, nullptr, 0);
    };
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) { add(impl, part); impl = part = -1; }
        else if (line.rfind("CPU implementer", 0) == 0) impl = hex(line);
        else if (line.rfind("CPU part", 0) == 0) part = hex(line);
    }
    add(impl, part);
    std::stable_sort(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string out;
    for (const auto& [name, n] : counts) out += (out.empty() ? "" : " + ") + std::to_string(n) + "x " + name;
    return out;
}

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

std::vector<std::vector<int>> cpu_l3_groups() {
    std::vector<std::vector<int>> groups;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) return groups;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationCache, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len))
        return groups;
    for (DWORD off = 0; off < len;) {
        const auto* info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + off);
        if (info->Relationship == RelationCache && info->Cache.Level == 3 &&
            (info->Cache.Type == CacheUnified || info->Cache.Type == CacheData)) {
            const GROUP_AFFINITY& a = info->Cache.GroupMask;
            std::vector<int> g;
            for (int i = 0; i < 64; ++i)
                if (a.Mask & ((KAFFINITY) 1 << i)) g.push_back((int) a.Group * 64 + i);
            if (!g.empty()) groups.push_back(std::move(g));
        }
        off += info->Size;
    }
    return groups;
}

#else
// --------------------------------------------------------------------------------------------- Linux

namespace {
// A kernel CPU list file ("0-7", "0-3,8-11", "5"): mark[cpu] = 1 for each CPU it names. False when the file is
// missing, empty or not a list.
bool read_cpu_list(const char* path, std::vector<char>& mark) {
    mark.clear();
    std::FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[1024];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char* s = buf;
    bool any = false;
    while (*s) {
        char* end = nullptr;
        const long lo = std::strtol(s, &end, 10);
        if (end == s || lo < 0) return false;
        long hi = lo;
        s = end;
        if (*s == '-') {
            hi = std::strtol(s + 1, &end, 10);
            if (end == s + 1 || hi < lo) return false;
            s = end;
        }
        if (hi >= 1 << 20) return false;
        if ((size_t) hi >= mark.size()) mark.resize((size_t) hi + 1, 0);
        for (long c = lo; c <= hi; ++c) mark[(size_t) c] = 1;
        any = true;
        if (*s == ',') ++s;
        else if (*s == '\n' || *s == 0) break;
        else return false;
    }
    return any;
}
}  // namespace

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

    // x86 kernels publish no cpu_capacity (the file comes from the arch_topology driver, which x86 does not build), so
    // a hybrid Intel processor read as one class of cores. Its classes are the CPU lists of the kernel's two hybrid
    // PMUs instead (Linux 5.13 and later): cpu_core names the performance cores' CPUs, cpu_atom the efficient ones.
    // They stand in as two capacities for the rule below. A CPU in neither list keeps none and counts as efficient.
    if (max_cap == 0) {
        std::vector<char> in_core, in_atom;
        if (read_cpu_list("/sys/devices/cpu_core/cpus", in_core) && read_cpu_list("/sys/devices/cpu_atom/cpus", in_atom)) {
            const auto has = [](const std::vector<char>& mark, int cpu) {
                return cpu >= 0 && (size_t) cpu < mark.size() && mark[(size_t) cpu] != 0;
            };
            for (auto& cl : all_cpus) {
                if (has(in_core, cl.cpu)) cl.cap = 1024;
                else if (has(in_atom, cl.cpu)) cl.cap = 512;
                if (cl.cap > 0) {
                    max_cap = (std::max)(max_cap, cl.cap);
                    min_cap = (std::min)(min_cap, cl.cap);
                }
            }
        }
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
    std::string text;
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
        text += line;
        text += '\n';
        if (line.empty()) flush();
        else if (line.rfind("model name", 0) == 0 && info.name.empty()) info.name = value(line);
        else if (line.rfind("physical id", 0) == 0) pkg = value(line);
        else if (line.rfind("core id", 0) == 0) core = value(line);
    }
    flush();
    if (info.name.empty()) info.name = arm_cpu_name(text);   // Arm's /proc/cpuinfo has no model name
    info.physical_cores = (int) cores.size();
    if (info.physical_cores == 0) info.physical_cores = (int) detect_cpu_topology(false).worker_cores.size();
    return info;
}

std::vector<std::vector<int>> cpu_l3_groups() {
    std::vector<std::vector<int>> groups;
    std::set<std::vector<int>> seen;
    std::vector<char> online, mark;
    if (!read_cpu_list("/sys/devices/system/cpu/online", online)) return groups;
    for (size_t c = 0; c < online.size(); ++c) {
        if (!online[c]) continue;
        // the cache whose level is 3 among the CPU's index<N> entries
        for (int idx = 0; idx < 8; ++idx) {
            const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cache/index" + std::to_string(idx);
            std::ifstream lv(dir + "/level");
            int level = 0;
            if (!(lv >> level)) break;
            if (level != 3 || !read_cpu_list((dir + "/shared_cpu_list").c_str(), mark)) continue;
            std::vector<int> g;
            for (size_t k = 0; k < mark.size(); ++k)
                if (mark[k]) g.push_back((int) k);
            if (seen.insert(g).second) groups.push_back(std::move(g));
            break;
        }
    }
    return groups;
}

#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_WIN32)
bool cpu_has_arm_dotprod() { return IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0; }
bool cpu_has_arm_i8mm() {
#if defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
    return IsProcessorFeaturePresent(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return false;   // an SDK from before the constant: the baseline kernels run
#endif
}
#else
bool cpu_has_arm_dotprod() { return (getauxval(AT_HWCAP) & (1ul << 20)) != 0; }    // HWCAP_ASIMDDP
bool cpu_has_arm_i8mm() { return (getauxval(AT_HWCAP2) & (1ul << 13)) != 0; }      // HWCAP2_I8MM
#endif
#else
bool cpu_has_arm_dotprod() { return false; }
bool cpu_has_arm_i8mm() { return false; }
#endif

}  // namespace basalt::platform
