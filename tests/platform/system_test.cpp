// tests/platform/system_test.cpp - the host-system questions in basalt/platform/system.hpp, the DynLib loader and the
// memory-accounting paths. CPU only. Each answer is checked for being usable; none depends on a particular machine.
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#endif

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

    // the root is two path components above the executable: what is left of the path is "/<dir>/<name>"
    const std::string self = module_path(), root = install_root();
    const std::string rest = self.compare(0, root.size(), root) == 0 ? self.substr(root.size()) : std::string();
    int seps = 0;
    for (char c : rest) seps += c == '/' || c == '\\';
    check(!root.empty() && seps == 2 && (rest[0] == '/' || rest[0] == '\\'), "install_root is the directory above the executable's own");
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
    // the commit limit: Windows keeps one, and a page-locked allocation's budget leaves the reserve of its headroom (the
    // two reads are moments apart, so within 256 MiB); Linux keeps none
    {
        uint64_t cavail = 0, climit = 0;
        const bool has = commit_headroom(cavail, climit);
#if defined(_WIN32)
        check(has && climit > 0 && cavail <= climit, "commit_headroom reads the commit limit on Windows");
        const uint64_t room = cavail > ram_reserve() ? cavail - ram_reserve() : 0;
        check(ram_budget() <= room + (256ull << 20), "and ram_budget leaves the reserve of the commit headroom");
        check(climit >= total_physical_memory() / 2, "the commit limit is of the RAM's order at least");
#else
        check(!has && cavail == 0 && climit == 0, "commit_headroom: no commit limit off Windows");
#endif
    }

    const ProcessMemory pm = process_memory();
    check(pm.rss_mib > 0 && pm.avail_mib > 0, "process_memory reads this process's resident set");

    std::string dev;
    const int rot = disk_rotational(temp_dir(), dev);
    check(rot >= -1 && rot <= 1, "disk_rotational answers 1, 0 or unknown");
    const ThreadWaits w = threads_in_disk_wait();
    check(w.waiting <= w.threads, "threads_in_disk_wait counts waiting within all");
    if (!mi.empty()) check(w.threads >= 1, "and lists this process's threads");

    // keep_awake asks something of a Modern Standby machine only, holds what it says it holds, and is idempotent
    if (modern_standby()) {
        check(keep_awake(true), "keep_awake(true) holds a power request on a Modern Standby machine");
        check(keep_awake(true), "and holds it when asked again");
        check(!keep_awake(false), "keep_awake(false) withdraws it");
        check(!keep_awake(false), "and again");
        check(keep_awake(true) && !keep_awake(false), "it can be held and withdrawn again");
    } else {
        check(!keep_awake(true) && !keep_awake(false), "keep_awake asks nothing where the machine has no Modern Standby");
    }
    check(keep_full_speed(), "keep_full_speed is accepted");
    check(keep_full_speed(), "and accepted again");
#if defined(_WIN32)
    // older Windows builds accept the state but cannot report it back
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    if (GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof pt) != 0)
        check((pt.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) != 0 &&
                  (pt.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) == 0,
              "and the OS reports execution speed as never lowered");
#endif

    const std::string os = host_os_name();
    check(gpu_memory_pages() == (os == "Windows" || os == "WSL2"), "gpu_memory_pages: only the Windows driver pages");
    const DynLib nvml = DynLib::open_nvml();
    if (nvml) {
        check(nvml.sym<void*>("nvmlInit_v2") != nullptr, "the management library, when present, exports nvmlInit_v2");
        check(nvml.sym<void*>("basalt_no_such_symbol") == nullptr, "and a missing symbol is null");
    } else {
        std::printf("system: no GPU management library here; the loader returned empty\n");
    }
    DynLib moved = DynLib();
    check(!moved, "an empty DynLib is false");

    // application libraries: a name nobody ships loads nothing; the C runtime's own does load, and names its file
    check(!DynLib::open_app(DynLib::versioned_name("basalt_no_such_library", 99)), "open_app: a missing library is empty");
#if defined(_WIN32)
    check(DynLib::versioned_name("avformat", 62) == "avformat-62.dll", "versioned_name follows the platform's spelling");
    const DynLib crt = DynLib::open_app("ucrtbase.dll");
#else
    check(DynLib::versioned_name("avformat", 62) == "libavformat.so.62", "versioned_name follows the platform's spelling");
    const DynLib crt = DynLib::open_app("libc.so.6");
#endif
    check((bool) crt && !crt.path().empty(), "open_app: the C runtime loads and path() names its file");
    return failures ? 1 : 0;
}
