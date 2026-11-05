// tests/platform/cpu_test.cpp - platform::detect_cpu_topology and platform::current_cpu. CPU only.
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <sched.h>
#endif

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("cpu %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
// Whether the calling thread may run on `cpu`, read from the OS in the topology's numbering (true when it cannot say).
bool may_run_on(int cpu) {
#if defined(_WIN32)
    GROUP_AFFINITY ga{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga)) return true;
    return (int) ga.Group == cpu / 64 && ((ga.Mask >> (cpu % 64)) & 1) != 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0 || cpu >= CPU_SETSIZE) return true;
    return CPU_ISSET(cpu, &set);
#endif
}
// Puts the calling thread on `cpu` alone; false when the OS refuses.
bool pin_to(int cpu) {
#if defined(_WIN32)
    GROUP_AFFINITY ga{};
    ga.Group = (WORD) (cpu / 64);
    ga.Mask = KAFFINITY(1) << (cpu % 64);
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
#else
    if (cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0;
#endif
}
}  // namespace

int main() {
    using namespace basalt::platform;
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
    check(all.is_hybrid == !all.efficient.empty(), "efficient processors are listed exactly on a hybrid CPU");
    check((int) all.efficient.size() >= all.e_cores, "every efficient core has a listed processor");

    // current_cpu numbers processors as the topology does: on a thread put on a listed processor it names that one, so
    // the pool can tell a task's core class by it
    const int here = current_cpu();
    check(here >= -1, "current_cpu names a processor or says it cannot");
    bool allowed = true;
    for (int i = 0; i < 1000 && here >= 0; ++i) {
        const int c = current_cpu();
        allowed &= c >= 0 && may_run_on(c);
    }
    check(allowed, "each processor it names is one the thread may run on");
    std::vector<int> probe{all.worker_cores.front(), all.worker_cores.back()};
    if (!all.efficient.empty()) probe.push_back(all.efficient.front());
    for (int c : probe) {
        int seen = -2;
        bool pinned = false;
        std::thread([&] {
            pinned = pin_to(c);
            if (pinned) seen = current_cpu();
        }).join();
        if (!pinned) {
            std::printf("cpu the OS refused to put a thread on processor %d: skipped\n", c);
            continue;
        }
        check(seen == c || (seen == -1 && here == -1),
              "on processor " + std::to_string(c) + " it names that processor (" + std::to_string(seen) + ")");
    }

    return failures ? 1 : 0;
}
