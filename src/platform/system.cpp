// src/platform/system.cpp - see include/basalt/platform/system.hpp.
#include "basalt/platform/system.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"

#include <cctype>
#include <cerrno>
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
#include <psapi.h>   // K32GetProcessMemoryInfo
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" { extern char** environ; }
#endif

namespace basalt::platform {

std::optional<std::string> capture_output(const std::vector<std::string>& argv, bool include_stderr) {
    if (argv.empty()) return std::nullopt;
#if defined(_WIN32)
    // Start the program directly with CreateProcessW rather than cmd.exe (_popen), so UTF-8 paths and shell
    // metacharacters (&, |, <, >, ^, %) pass literally without shell interpretation.
    std::string cmd;
    for (const std::string& a : argv) {
        if (!cmd.empty()) cmd += ' ';
        cmd += quote_arg(a);
    }
    const std::wstring wcmd = to_wide(cmd);
    std::vector<wchar_t> line(wcmd.begin(), wcmd.end());
    line.push_back(L'\0');

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return std::nullopt;
    (void) SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (nul == INVALID_HANDLE_VALUE) {
        CloseHandle(rd);
        CloseHandle(wr);
        return std::nullopt;
    }

    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = include_stderr ? wr : nul;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        return std::nullopt;
    }
    CloseHandle(pi.hThread);

    std::string raw;
    char buf[4096];
    for (DWORD got = 0; ReadFile(rd, buf, (DWORD) sizeof buf, &got, nullptr) && got > 0;) raw.append(buf, got);
    CloseHandle(rd);
    (void) WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);

    // Match text-mode line endings (\r\n -> \n) without exposing output to CRT EOF translation.
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n') continue;
        out.push_back(raw[i]);
    }
    return out;
#else
    int pfd[2] = {-1, -1};
    if (pipe2(pfd, O_CLOEXEC) != 0) return std::nullopt;
    const int devnull = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (devnull < 0) {
        close(pfd[0]);
        close(pfd[1]);
        return std::nullopt;
    }
    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) {
        close(pfd[0]);
        close(pfd[1]);
        close(devnull);
        return std::nullopt;
    }
    if (posix_spawn_file_actions_adddup2(&fa, devnull, STDIN_FILENO) != 0 ||
        posix_spawn_file_actions_adddup2(&fa, pfd[1], STDOUT_FILENO) != 0 ||
        posix_spawn_file_actions_adddup2(&fa, include_stderr ? pfd[1] : devnull, STDERR_FILENO) != 0) {
        (void) posix_spawn_file_actions_destroy(&fa);
        close(pfd[0]);
        close(pfd[1]);
        close(devnull);
        return std::nullopt;
    }

    std::vector<std::string> args = argv;
    std::vector<char*> av;
    av.reserve(args.size() + 1);
    for (std::string& s : args) av.push_back(s.data());
    av.push_back(nullptr);

    pid_t pid = -1;
    const int err = posix_spawnp(&pid, args[0].c_str(), &fa, nullptr, av.data(), environ);
    (void) posix_spawn_file_actions_destroy(&fa);
    close(pfd[1]);
    close(devnull);
    if (err != 0) {
        close(pfd[0]);
        return std::nullopt;
    }

    std::string out;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(pfd[0], buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        out.append(buf, (size_t) n);
    }
    close(pfd[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return out;
#endif
}

#if defined(_WIN32)
// --------------------------------------------------------------------------------------------- Windows

int64_t process_id() { return (int64_t) GetCurrentProcessId(); }

std::string host_os_name() { return "Windows"; }

std::string python_interpreter(const std::string& root) {
    const std::string venv = root + "/.venv/Scripts/python.exe";
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

bool gpu_memory_pages() { return true; }

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

ProcessMemory process_memory() {
    ProcessMemory m;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PrivateUsage > pmc.WorkingSetSize ? (pmc.PrivateUsage - pmc.WorkingSetSize) >> 20 : 0;
    }
    return m;
}

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

std::string python_interpreter(const std::string& root) {
    const std::string venv = root + "/.venv/bin/python";
    return path_executable(venv) ? venv : "python3";
}

void prefer_oom_victim() {
    if (std::FILE* f = std::fopen("/proc/self/oom_score_adj", "w")) { std::fputs("1000", f); std::fclose(f); }
}

bool keep_full_speed() { return true; }   // the scheduler does not slow a process for its window's state

// WSL2 reaches the GPU through the Windows driver, which pages
bool gpu_memory_pages() { return host_os_name() == "WSL2"; }

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
