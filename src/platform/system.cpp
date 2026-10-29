// src/platform/system.cpp - see include/basalt/platform/system.hpp.
#include "basalt/platform/system.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <fcntl.h>   // _O_BINARY
#include <io.h>      // _fileno, _setmode
#else
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace basalt::platform {

namespace {
/// An argument for a shell's command line: wrapped in quotes the shell reads literally.
std::string shell_quote(const std::string& a) {
#if defined(_WIN32)
    return quote_arg(a);
#else
    std::string out = "'";
    for (char c : a) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += '\'';
    return out;
#endif
}
}  // namespace

std::optional<std::string> capture_output(const std::vector<std::string>& argv, bool include_stderr) {
    if (argv.empty()) return std::nullopt;
    std::string cmd;
    for (const std::string& a : argv) {
        if (!cmd.empty()) cmd += ' ';
        cmd += shell_quote(a);
    }
#if defined(_WIN32)
    cmd += include_stderr ? " 2>&1" : " 2>nul";
    // cmd.exe drops the first and last quote of a command line that holds more than one quoted word or a redirection,
    // so the whole line is wrapped in one more pair
    cmd = "\"" + cmd + "\"";
    std::FILE* pf = _popen(cmd.c_str(), "r");
#else
    cmd += include_stderr ? " 2>&1" : " 2>/dev/null";
    std::FILE* pf = popen(cmd.c_str(), "r");
#endif
    if (pf == nullptr) return std::nullopt;
    std::string out;
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, pf)) > 0;) out.append(buf, n);
#if defined(_WIN32)
    _pclose(pf);
#else
    pclose(pf);
#endif
    return out;
}

#if defined(_WIN32)
// --------------------------------------------------------------------------------------------- Windows

int64_t process_id() { return (int64_t) GetCurrentProcessId(); }

std::string host_os_name() { return "Windows"; }

std::string python_interpreter(const std::string& src_dir) {
    const std::string venv = src_dir + "/.venv/Scripts/python.exe";
    return path_executable(venv) ? venv : "python";
}

void prefer_oom_victim() {}

bool keep_full_speed() {
    PROCESS_POWER_THROTTLING_STATE s{};
    s.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;   // the process sets its execution speed itself
    s.StateMask = 0;                                            // and it is never lowered
    return SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof s) != 0;
}

bool gpu_peer_access() { return false; }

bool stderr_is_terminal() {
    if (!_isatty(_fileno(stderr))) return false;
    const HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode = 0;
    if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) return false;
    return (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0 || SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
}

void set_stdio_binary() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
}

std::string meminfo_path() { return std::string(); }
std::string self_cgroup_path() { return std::string(); }
std::string cgroup_root() { return std::string(); }

ProcessMemory process_memory() { return ProcessMemory{}; }   // the stall report names the Linux readings only

int disk_rotational(const std::string&, std::string&) {
    return -1;   // sysfs names the queue on Linux; unknown means the caller does not warn
}

ThreadWaits threads_in_disk_wait() {
    return ThreadWaits{};   // state D and wchan are Linux readings; there is no per-thread wait address to list here
}

#else
// --------------------------------------------------------------------------------------------- Linux

int64_t process_id() { return (int64_t) getpid(); }

std::string host_os_name() {
    std::ifstream in("/proc/sys/kernel/osrelease");
    std::string rel;
    std::getline(in, rel);
    for (char& c : rel) c = (char) std::tolower((unsigned char) c);
    return rel.find("microsoft") != std::string::npos || rel.find("wsl") != std::string::npos ? "WSL2" : "Linux";
}

std::string python_interpreter(const std::string& src_dir) {
    const std::string venv = src_dir + "/.venv/bin/python";
    return path_executable(venv) ? venv : "python3";
}

void prefer_oom_victim() {
    if (std::FILE* f = std::fopen("/proc/self/oom_score_adj", "w")) { std::fputs("1000", f); std::fclose(f); }
}

bool keep_full_speed() { return true; }   // the scheduler does not slow a process for its window's state

bool gpu_peer_access() { return true; }

bool stderr_is_terminal() { return isatty(STDERR_FILENO) != 0; }

void set_stdio_binary() {}

std::string meminfo_path() { return "/proc/meminfo"; }
std::string self_cgroup_path() { return "/proc/self/cgroup"; }
std::string cgroup_root() { return "/sys/fs/cgroup"; }

ProcessMemory process_memory() {
    ProcessMemory m;
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
    return m;
}

int disk_rotational(const std::string& path, std::string& dev) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return -1;
    char sys[64];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
    std::error_code ec;
    std::filesystem::path p = std::filesystem::canonical(sys, ec);
    for (int up = 0; !ec && up < 2; ++up, p = p.parent_path()) {   // a partition has its disk's queue
        std::ifstream q(p / "queue" / "rotational");
        int r = -1;
        if (q >> r) {
            dev = p.filename().string();
            return r;
        }
    }
    return -1;
}

ThreadWaits threads_in_disk_wait() {
    ThreadWaits w;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        ++w.threads;
        std::ifstream st(e.path() / "stat");
        std::string line;
        std::getline(st, line);
        const size_t rp = line.rfind(')');
        if (rp == std::string::npos || rp + 2 >= line.size() || line[rp + 2] != 'D') continue;
        ++w.waiting;
        std::ifstream wc(e.path() / "wchan");
        std::string name;
        std::getline(wc, name);
        ++w.where[name.empty() || name == "0" ? "?" : name];
    }
    return w;
}

#endif

}  // namespace basalt::platform
