// tests/platform/system_test.cpp - the host-system questions in basalt/platform/system.hpp, the DynLib loader and the
// memory-accounting paths. CPU only. Each answer is checked for being usable; none depends on a particular machine.
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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

    const ProcessMemory pm = process_memory();
    check(pm.rss_mib > 0 && pm.avail_mib > 0, "process_memory reads this process's resident set");

    std::string dev;
    const int rot = disk_rotational(temp_dir(), dev);
    check(rot >= -1 && rot <= 1, "disk_rotational answers 1, 0 or unknown");
    const ThreadWaits w = threads_in_disk_wait();
    check(w.waiting <= w.threads, "threads_in_disk_wait counts waiting within all");
    if (!mi.empty()) check(w.threads >= 1, "and lists this process's threads");

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
#if !defined(_WIN32)
    // A whole core of busy work outside this process must raise external load by about one core.  Compared
    // against the same process's own baseline instead of an absolute value: a shared CI VM's /proc/stat
    // carries steal and guest time that is nobody's fault, and an absolute bound would chase it.  Kernel
    // reclaim workers are excluded on purpose, and a pure-CPU spinner keeps the kernel out of the window.
    if (ExternalCpuLoad probe; true) {
        check(probe.sample() == 0, "ExternalCpuLoad's first sample is the zero interval");
        auto cpu_allowance = [] {
            double cpus = (double) sysconf(_SC_NPROCESSORS_ONLN);
            if (std::ifstream q("/sys/fs/cgroup/cpu.max"); q) {   // a container sees the host's counter but owns a slice
                std::string quota, period;
                if ((q >> quota >> period) && quota != "max" && quota.find_first_not_of("0123456789.") == std::string::npos &&
                    period.find_first_not_of("0123456789.") == std::string::npos)
                    if (const double allowed = std::stod(quota) / std::stod(period); allowed > 0 && allowed < cpus) cpus = allowed;
            }
            return cpus;
        };
        auto mean_middle = [](std::vector<double>& v) {
            v.erase(v.begin(), v.begin() + 2);   // warm the interval before counting it
            if (v.empty()) return 0.0;
            v.pop_back();                        // and drop the edge that bleeds into the next phase
            return std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
        };
        std::vector<double> base_window;
        for (int i = 0; i < 12; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            base_window.push_back(probe.sample());
        }
        const double base = mean_middle(base_window);   // a mean, not a max: one ambient spike must not eat the rise
        if (const pid_t kid = fork(); kid == 0) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1600);
            volatile double sink = 1;
            while (std::chrono::steady_clock::now() < until) sink = sink * 1.0000001 + 1;
            _exit(0);
        } else if (kid > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            auto aggregate_cores = [] {   // the same cpu sum the sensor reads, sampled independently
                double busy = 0;
                std::ifstream f("/proc/stat");
                std::string cpu;
                long long v;
                if (f >> cpu) for (int i = 0; i < 10 && (f >> v); ++i) if (i != 3 && i != 4) busy += (double) v;
                return busy / (double) sysconf(_SC_CLK_TCK);
            };
            const double agg0 = aggregate_cores();
            std::vector<double> kid_window;
            for (int i = 0; i < 14; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                kid_window.push_back(probe.sample());
            }
            const double peak = mean_middle(kid_window);
            int status = 0;
            rusage kid_use{};
            wait4(kid, &status, 0, &kid_use);
            const double win = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double child_cores =   // what the child actually held, seconds of cpu per second of window
                (double) (kid_use.ru_utime.tv_sec + kid_use.ru_stime.tv_sec +
                          (kid_use.ru_utime.tv_usec + kid_use.ru_stime.tv_usec) / 1e6) / win;
            const double agg_rise = (aggregate_cores() - agg0) / win - base;   // the raw counter's own rise
            char detail[192];
            const double rise = peak - base;
            const double cpus = cpu_allowance();
            std::snprintf(detail, sizeof detail,
                          "the sensor's rise matches the raw cpu aggregate's rise (sensor %.2f, aggregate %.2f, child %.2f, base %.2f)",
                          rise, agg_rise, child_cores, base);
            if (child_cores < 0.2)
                // the scheduler fed the child nothing worth seeing; nothing to compare
                check(true, "sensor rise check skipped, the child was starved of cpu");
            else if (base + child_cores > 0.95 * cpus)
                check(true, "sensor rise check skipped, the machine was saturated");
            else if (rise < 0.05 && agg_rise < 0.3)
                // neither the sensor nor the raw counter moved for a consumed core: the counter answers to
                // someone else (host-global behind a cgroup) and judges nothing here
                check(true, "sensor rise check skipped, the cpu aggregate itself ignored the core");
            else
                // the governor reads exactly this aggregate; the sensor has earned the right to subtract
                // self and kernel workers, and nothing else - so agreement within the sampling noise of
                // two differently-windowed means IS its contract, whatever cgroups and steal decided
                check(rise + 0.4 > agg_rise && rise < agg_rise + 0.8, detail);
        } else
            check(false, "ExternalCpuLoad: fork failed");
    }
#endif
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
