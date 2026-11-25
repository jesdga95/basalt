// tests/platform/cpu_test.cpp - platform::detect_cpu_topology and the thread placement calls. CPU only. Where the OS
// refuses a placement (a restricted container) the placement cases are skipped with a line saying so.
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("cpu %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    {   // an Arm /proc/cpuinfo has no model name: the core types by implementer and part, the most numerous first
        std::string gb10;
        for (int c = 0; c < 20; ++c)
            gb10 += "processor\t: " + std::to_string(c) + "\nCPU implementer\t: 0x41\nCPU part\t: " +
                    (c < 10 ? "0xd87" : "0xd85") + "\n\n";
        check(arm_cpu_name(gb10) == "10x Cortex-A725 + 10x Cortex-X925", "an Arm cpuinfo names its core types");
        check(arm_cpu_name("processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd99\n") == "1x Arm 0x41:0xd99",
              "an unknown Arm part is named by its ids");
        check(arm_cpu_name("processor\t: 0\nmodel name\t: AMD Ryzen 9 9950X\n").empty(), "an x86 cpuinfo gives no Arm name");
    }
    const CpuTopology all = detect_cpu_topology(false, CpuAffinity::All);
    check(!all.worker_cores.empty(), "at least one worker core is found");
    check(all.host_core == -1, "without skip_first no core is reserved");
    check(all.p_cores >= 1, "at least one performance core is counted");
    bool unique = true;
    for (size_t i = 0; i < all.worker_cores.size(); ++i)
        for (size_t j = i + 1; j < all.worker_cores.size(); ++j) unique &= all.worker_cores[i] != all.worker_cores[j];
    check(unique, "no core is listed twice");

    const CpuTopology skip = detect_cpu_topology(true, CpuAffinity::All);
    if (all.worker_cores.size() > 1) {
        check(skip.host_core == all.worker_cores.front(), "skip_first reserves the first core for the host");
        check(skip.worker_cores.size() + 1 == all.worker_cores.size(), "and takes it out of the workers");
    }
    for (CpuAffinity a : {CpuAffinity::Auto, CpuAffinity::PCores}) {
        const CpuTopology t = detect_cpu_topology(false, a);
        check(!t.worker_cores.empty(), "a hybrid-aware ordering still finds cores");
    }

#if defined(__linux__)
    if (all.is_hybrid) {
        check(all.p_cores > 0 && all.e_cores > 0, "a hybrid processor has cores of both classes");
        check(all.p_cores + all.e_cores == (int) all.worker_cores.size(), "and the two classes add up to its cores");
    }
    // An x86 kernel names a hybrid processor's classes in its two hybrid PMUs' CPU lists (it has no cpu_capacity):
    // where this process may run on CPUs of both lists, the topology must say hybrid.
    {
        const auto listed = [](const char* path, const std::vector<int>& cpus) {
            std::FILE* f = std::fopen(path, "r");
            if (!f) return -1;
            int lo = 0, hi = 0, n = 0;
            for (;;) {
                const int got = std::fscanf(f, "%d-%d", &lo, &hi);
                if (got < 1) break;
                if (got == 1) hi = lo;
                for (int c : cpus) n += c >= lo && c <= hi;
                if (std::fgetc(f) != ',') break;
            }
            std::fclose(f);
            return n;
        };
        const int n_core = listed("/sys/devices/cpu_core/cpus", all.worker_cores);
        const int n_atom = listed("/sys/devices/cpu_atom/cpus", all.worker_cores);
        if (n_core > 0 && n_atom > 0) {
            check(all.is_hybrid, "the kernel's hybrid CPU lists make the topology hybrid");
            check(all.p_cores == n_core, "the performance cores are the cpu_core list's");
            check(all.e_cores == n_atom, "the efficient cores are the cpu_atom list's");
        } else {
            std::printf("cpu the kernel lists no hybrid classes here: the hybrid cases are skipped\n");
        }
    }
#endif

    return failures ? 1 : 0;
}
