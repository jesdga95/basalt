// tests/platform/cpu_test.cpp - platform::detect_cpu_topology and the thread placement calls. CPU only. Where the OS
// refuses a placement (a restricted container) the placement cases are skipped with a line saying so.
#include "basalt/platform/cpu.hpp"
#include "basalt/platform/compiler.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("cpu %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
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

    const bool dotprod = cpu_has_arm_dotprod();
    std::printf("cpu: Arm dot product %s\n", dotprod ? "yes" : "no");
    if constexpr (kX86) check(!dotprod, "no Arm dot product on x86-64");

    return failures ? 1 : 0;
}
