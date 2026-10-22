// tests/platform/system_test.cpp - the host-system questions in basalt/platform/system.hpp, the DynLib loader and the
// memory-accounting paths. CPU only. Each answer is checked for being usable; none depends on a particular machine.
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"

#include <cstdio>
#include <fstream>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("system %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    check(process_id() > 0, "process_id is positive");
    (void) stderr_is_terminal();   // either answer is valid; the call must not fail

    const std::string mi = meminfo_path();
    if (!mi.empty()) {
        std::ifstream f(mi);
        check((bool) f, "the memory accounting file named is readable");
        check(!self_cgroup_path().empty() && !cgroup_root().empty(), "and the cgroup paths come with it");
    } else {
        check(self_cgroup_path().empty() && cgroup_root().empty(), "no accounting file means no cgroup paths either");
    }
    check(available_memory() > 0, "available_memory reads");

    const ProcessMemory pm = process_memory();
    if (!mi.empty()) check(pm.rss_mib > 0 && pm.avail_mib > 0, "process_memory reads this process's resident set");

    std::string dev;
    const int rot = disk_rotational(temp_dir(), dev);
    check(rot >= -1 && rot <= 1, "disk_rotational answers 1, 0 or unknown");
    const ThreadWaits w = threads_in_disk_wait();
    check(w.waiting <= w.threads, "threads_in_disk_wait counts waiting within all");
    if (!mi.empty()) check(w.threads >= 1, "and lists this process's threads");

    (void) gpu_peer_access();
    const DynLib nvml = DynLib::open_nvml();
    if (nvml) {
        check(nvml.sym<void*>("nvmlInit_v2") != nullptr, "the management library, when present, exports nvmlInit_v2");
        check(nvml.sym<void*>("basalt_no_such_symbol") == nullptr, "and a missing symbol is null");
    } else {
        std::printf("system: no GPU management library here; the loader returned empty\n");
    }
    DynLib moved = DynLib();
    check(!moved, "an empty DynLib is false");
    return failures ? 1 : 0;
}
