// Parity of cgroup_cpu_quota against fake cgroup trees (v2, v1, hybrid, hidden paths).
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/platform/system.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using basalt::kernels::cpu::cgroup_cpu_quota;

static int g_fail = 0;
static void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("cpu_quota_test_" + std::to_string(basalt::platform::process_id()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto put = [&](const std::string& rel, const std::string& text) {
        fs::create_directories((dir / rel).parent_path());
        std::ofstream(dir / rel) << text;
    };
    const std::string root = (dir / "fs").string(), cg = (dir / "cgroup").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        return cgroup_cpu_quota(cg, root);
    };
    auto is = [](double a, double b) { return std::fabs(a - b) < 1e-9; };

    require(is(probe(""), 0), "no cgroup line: no quota");
    require(is(cgroup_cpu_quota((dir / "missing").string(), root), 0), "no cgroup file");
    // v2
    put("fs/box/cpu.max", "400000 100000\n");
    require(is(probe("0::/box\n"), 4), "v2 quota 4 CPUs");
    put("fs/box/cpu.max", "max 100000\n");
    require(is(probe("0::/box\n"), 0), "v2 max");
    put("fs/box/cpu.max", "garbage 100000\n");
    require(is(probe("0::/box\n"), 0), "v2 malformed");
    // the tightest of the group and its ancestors
    put("fs/outer/inner/cpu.max", "max 100000\n");
    put("fs/outer/cpu.max", "250000 100000\n");
    require(is(probe("0::/outer/inner\n"), 2.5), "v2 ancestor quota");
    put("fs/outer/inner/cpu.max", "100000 100000\n");
    require(is(probe("0::/outer/inner\n"), 1), "v2 group tighter than ancestor");
    // v2 group path hidden by a cgroup namespace: read at the mount's root
    put("fs/cpu.max", "800000 100000\n");
    require(is(probe("0::/docker/4f3c2a\n"), 8), "v2 hidden path");
    // v1, cpu,cpuacct mount
    put("fs/cpu,cpuacct/docker/abc/cpu.cfs_quota_us", "300000\n");
    put("fs/cpu,cpuacct/docker/abc/cpu.cfs_period_us", "100000\n");
    put("fs/cpu,cpuacct/cpu.cfs_quota_us", "-1\n");
    put("fs/cpu,cpuacct/cpu.cfs_period_us", "100000\n");
    require(is(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n"), 3), "v1 quota");
    put("fs/cpu,cpuacct/docker/abc/cpu.cfs_quota_us", "-1\n");
    require(is(probe("12:cpu,cpuacct:/docker/abc\n"), 0), "v1 unlimited");
    // hybrid: the v1 cpu controller wins over the unified line
    put("fs/cpu,cpuacct/docker/abc/cpu.cfs_quota_us", "200000\n");
    require(is(probe("12:cpu,cpuacct:/docker/abc\n0::/docker/abc\n"), 2), "hybrid layout");
    // v1, path hidden, plain "cpu" mount
    fs::remove_all(dir / "fs/cpu,cpuacct");
    put("fs/cpu/cpu.cfs_quota_us", "3200000\n");
    put("fs/cpu/cpu.cfs_period_us", "100000\n");
    require(is(probe("12:cpu,cpuacct:/docker/4f3c2a\n0::/docker/4f3c2a\n"), 32), "v1 hidden path, 32 CPUs");

    fs::remove_all(dir);
    if (g_fail == 0) std::printf("cpu_quota_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
