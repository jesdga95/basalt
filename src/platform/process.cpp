// src/platform/process.cpp - see include/basalt/platform/process.hpp.
#include "basalt/platform/process.hpp"

#include "basalt/platform/file.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#endif

namespace basalt::platform {

std::string quote_arg(const std::string& s) {
    // An argument list is one string here, so an argument that holds a space or a quote has to be quoted and its
    // backslashes kept meaningful: they escape quotes, so a run of them before a quote - or before the closing
    // quote - has to be doubled.
    if (s.empty()) return "\"\"";
    if (s.find_first_of(" \t\"") == std::string::npos) return s;
    std::string out = "\"";
    size_t backslashes = 0;
    for (char c : s) {
        if (c == '\\') { ++backslashes; out += c; continue; }
        if (c == '"') { out.append(backslashes + 1, '\\'); out += '"'; backslashes = 0; continue; }
        backslashes = 0;
        out += c;
    }
    out.append(backslashes, '\\');
    out += '"';
    return out;
}

namespace {
#if defined(_WIN32)
/// The command line CreateProcessW wants: UTF-16, in a buffer it is allowed to write into.
std::vector<wchar_t> command_line(const std::vector<std::string>& argv) {
    std::string cmd;
    for (const std::string& a : argv) {
        if (!cmd.empty()) cmd += ' ';
        cmd += quote_arg(a);
    }
    const std::wstring w = to_wide(cmd);
    std::vector<wchar_t> out;
    out.reserve(w.size() + 1);
    for (wchar_t c : w) out.push_back(c);
    out.push_back(0);
    return out;
}

/// Starts the child and hands back its handle. nullptr when it could not start, with errno set for the caller.
HANDLE start(const std::vector<std::string>& argv) {
    std::vector<wchar_t> line = command_line(argv);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    // The handles the caller marked inheritable go to the child: that is how a caller wires a child's standard
    // descriptors to pipes it owns.
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        const DWORD e = GetLastError();
        errno = (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) ? ENOENT : EINVAL;
        return nullptr;
    }
    CloseHandle(pi.hThread);
    return pi.hProcess;
}
#else
/// argv as the exec and spawn calls want it: writable elements, terminated by a null.
std::vector<char*> c_argv(std::vector<std::string>& args) {
    std::vector<char*> av;
    av.reserve(args.size() + 1);
    for (std::string& s : args) av.push_back(s.data());
    av.push_back(nullptr);
    return av;
}

int exited(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}
#endif
}  // namespace

int child_wait(Child& c, int timeout_ms) {
#if defined(_WIN32)
    if (c.handle == nullptr) return -1;
    const HANDLE h = (HANDLE) c.handle;
    const DWORD until = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (WaitForSingleObject(h, until) != WAIT_OBJECT_0) return -1;
    DWORD code = 0;
    return GetExitCodeProcess(h, &code) ? (int) code : -1;
#else
    if (c.id <= 0) return -1;
    const bool block = timeout_ms < 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        int status = 0;
        const pid_t r = waitpid((pid_t) c.id, &status, block ? 0 : WNOHANG);
        if (r == (pid_t) c.id) {
            c.id = -1;   // reaped: a second wait would report the child as gone
            return exited(status);
        }
        if (r == 0) {   // still running, and the call was not told to wait for it
            if (std::chrono::steady_clock::now() >= deadline) return -1;
            const timespec every{0, 20 * 1000 * 1000};   // a look every 20 ms; a signal interrupts it, and it is looked at again
            (void) nanosleep(&every, nullptr);
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        return -1;   // ECHILD: there is nothing left to wait for
    }
#endif
}

void child_terminate(Child& c) {
#if defined(_WIN32)
    if (c.handle != nullptr) (void) TerminateProcess((HANDLE) c.handle, 1);
#else
    if (c.id > 0) (void) kill((pid_t) c.id, SIGTERM);
#endif
}

void child_close(Child& c) {
#if defined(_WIN32)
    if (c.handle != nullptr) CloseHandle((HANDLE) c.handle);
#endif
    c.handle = nullptr;
    c.id = -1;
}

namespace {
Child g_stop_child;   // the child a stop signal ends: the run's engine, while it is running

void on_stop_signal(int) noexcept {
    if (g_stop_child.id > 0 || g_stop_child.handle != nullptr) child_terminate(g_stop_child);
    // The child has been told; this process does not wait for it to finish writing.
#if defined(_WIN32)
    _exit(130);
#else
    ::_exit(130);
#endif
}
}  // namespace

void set_stop_child(const Child& c) { g_stop_child = c; }
void clear_stop_child() { g_stop_child = Child{}; }

void install_child_stop_signals() {
    (void) std::signal(SIGINT, on_stop_signal);
    (void) std::signal(SIGTERM, on_stop_signal);
}

void ignore_broken_pipe() {
#if !defined(_WIN32)
    (void) std::signal(SIGPIPE, SIG_IGN);
#endif
}

void set_env(const std::string& name, const std::string& value) {
#if defined(_WIN32)
    _putenv((name + "=" + value).c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

int spawn_and_wait(const std::vector<std::string>& argv) {
    if (argv.empty()) { errno = EINVAL; return -1; }
#if defined(_WIN32)
    const HANDLE h = start(argv);
    if (h == nullptr) return -1;
    Child c;
    c.handle = h;
    c.id = (int64_t) GetProcessId(h);
    const int code = child_wait(c, -1);
    child_close(c);
    return code;
#else
    // The child's own exec failure has to reach the caller, and a spawn reports it the way the platform does:
    // posix_spawnp answers with the error the exec would have returned.
    std::vector<std::string> args = argv;
    std::vector<char*> av = c_argv(args);
    pid_t pid = -1;
    const int e = posix_spawnp(&pid, args[0].c_str(), nullptr, nullptr, av.data(), environ);
    if (e != 0) { errno = e; return -1; }
    Child c;
    c.id = (int64_t) pid;
    return child_wait(c, -1);
#endif
}

int exec_replacing(const std::vector<std::string>& argv) {
    if (argv.empty()) { errno = EINVAL; return -1; }
#if defined(_WIN32)
    // No exec here. The child runs in the process's place as far as the caller and its supervisor can see: this
    // process stays until the child ends, then ends with the child's exit code.
    const int code = spawn_and_wait(argv);
    if (code >= 0) std::exit(code);
    return -1;
#else
    std::vector<std::string> args = argv;
    std::vector<char*> av = c_argv(args);
    execvp(args[0].c_str(), av.data());
    return -1;   // errno says why, and it is the caller's message
#endif
}

}  // namespace basalt::platform
