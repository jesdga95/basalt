// src/platform/process.cpp - see include/basalt/platform/process.hpp.
#include "basalt/platform/process.hpp"

#include "basalt/core/progress.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <fcntl.h>   // _O_BINARY, _O_WRONLY, _O_RDONLY
#include <io.h>      // _open_osfhandle, _read, _write, _close
#else
#include <execinfo.h>
#include <fcntl.h>
#include <poll.h>
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
/// With `std_handles`, the child's three standard handles are those (stdin, stdout, stderr).
HANDLE start(const std::vector<std::string>& argv, const HANDLE* std_handles = nullptr) {
    std::vector<wchar_t> line = command_line(argv);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    if (std_handles != nullptr) {
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = std_handles[0];
        si.hStdOutput = std_handles[1];
        si.hStdError = std_handles[2];
    }
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

bool spawn_piped(const std::vector<std::string>& argv, const std::string& stderr_path, PipedChild& out,
                 std::string& err) {
    out = PipedChild{};
    if (argv.empty()) { err = "no program to start"; return false; }
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE rd_in = nullptr, wr_in = nullptr, rd_out = nullptr, wr_out = nullptr;
    if (!CreatePipe(&rd_in, &wr_in, &sa, 0) || !CreatePipe(&rd_out, &wr_out, &sa, 0)) {
        err = "pipe failed";
        return false;
    }
    // The parent keeps only the ends it uses, un-inherited: if the child held the read end of the request pipe
    // or the write end of the reply pipe, neither would ever report the other side gone.
    (void) SetHandleInformation(wr_in, HANDLE_FLAG_INHERIT, 0);
    (void) SetHandleInformation(rd_out, HANDLE_FLAG_INHERIT, 0);
    // The share mode is what lets the log be read while the run is going on; a share mode of 0 keeps it closed
    // to any other process until the child exits. `sa` (bInheritHandle) is what hands the handle to the child.
    HANDLE log = CreateFileW(to_wide_path(stderr_path).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, &sa,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    const HANDLE std_handles[3] = {rd_in, wr_out, log != INVALID_HANDLE_VALUE ? log : wr_out};
    const HANDLE proc = start(argv, std_handles);
    if (proc == nullptr) {
        err = "CreateProcess failed (error " + std::to_string(GetLastError()) + ")";
        CloseHandle(rd_in); CloseHandle(wr_in); CloseHandle(rd_out); CloseHandle(wr_out);
        if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
        return false;
    }
    out.child.handle = proc;
    out.child.id = (int64_t) GetProcessId(proc);
    // The parent's ends become CRT descriptors, so the request and reply loops are the same code either way.
    out.in_fd = _open_osfhandle((intptr_t) wr_in, _O_BINARY | _O_WRONLY);
    out.out_fd = _open_osfhandle((intptr_t) rd_out, _O_BINARY | _O_RDONLY);
    CloseHandle(rd_in);              // the child owns the request pipe's read end
    CloseHandle(wr_out);             // the child owns the reply pipe's write end
    if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
    return true;
#else
    int in[2], outp[2];
    if (pipe(in) != 0 || pipe(outp) != 0) { err = "pipe failed"; return false; }
    std::vector<std::string> store = argv;
    std::vector<char*> av = c_argv(store);
    const pid_t pid = fork();
    if (pid < 0) { err = "fork failed"; return false; }
    if (pid == 0) {
        dup2(in[0], 0);
        dup2(outp[1], 1);
        const int lf = open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (lf >= 0) dup2(lf, 2);
        close(in[0]); close(in[1]); close(outp[0]); close(outp[1]);
        execv(store[0].c_str(), av.data());
        _exit(127);
    }
    close(in[0]);
    close(outp[1]);
    out.in_fd = in[1];
    out.out_fd = outp[0];
    out.child.id = (int64_t) pid;
    return true;
#endif
}

bool pipe_write_all(int fd, const void* data, size_t n) {
    const char* p = (const char*) data;
    size_t off = 0;
    while (off < n) {
#if defined(_WIN32)
        const int w = _write(fd, p + off, (unsigned) (n - off));
#else
        const ssize_t w = ::write(fd, p + off, n - off);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += (size_t) w;
    }
    return true;
}

int64_t pipe_read(int fd, void* dst, size_t n) {
    for (;;) {
#if defined(_WIN32)
        const int r = _read(fd, dst, (unsigned) n);
#else
        const ssize_t r = ::read(fd, dst, n);
#endif
        if (r < 0 && errno == EINTR) continue;
        return (int64_t) r;
    }
}

bool pipe_wait_readable(int fd, int timeout_ms) {
#if defined(_WIN32)
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || GetFileType(h) != FILE_TYPE_PIPE) return true;
    for (int waited = 0;; waited += 10) {
        DWORD avail = 0;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr) || avail > 0) return true;   // data, or the writer is gone
        if (waited >= timeout_ms) return false;
        Sleep(10);
    }
#else
    pollfd p{fd, POLLIN, 0};
    const int r = ::poll(&p, 1, timeout_ms);
    return r != 0;   // an interrupted or failed poll lets the read report it
#endif
}

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

namespace {
std::atomic<bool> g_crash_logged{false};

struct CrashBuf {
    char data[512];
    size_t n = 0;

    void putc(char c) noexcept {
        if (n + 1 < sizeof data) data[n++] = c;
    }
    void puts(const char* s) noexcept {
        if (s == nullptr) return;
        for (; *s != '\0' && n + 1 < sizeof data; ++s) data[n++] = *s;
    }
    void put_u64(uint64_t v) noexcept {
        char tmp[24];
        int k = 0;
        do {
            tmp[k++] = (char) ('0' + (v % 10));
            v /= 10;
        } while (v != 0);
        while (k > 0) putc(tmp[--k]);
    }
    void put_i64(int64_t v) noexcept {
        if (v < 0) {
            putc('-');
            put_u64((uint64_t) (-(v + 1)) + 1u);
        } else {
            put_u64((uint64_t) v);
        }
    }
    void put_hex(uint64_t v, int min_digits = 1) noexcept {
        static constexpr char kHex[] = "0123456789ABCDEF";
        char tmp[16];
        int k = 0;
        do {
            tmp[k++] = kHex[v & 0xf];
            v >>= 4;
        } while (v != 0 || k < min_digits);
        puts("0x");
        while (k > 0) putc(tmp[--k]);
    }
};

// `with_memory` is false in a POSIX signal handler: the Linux reading opens and parses /proc files, which is not
// async-signal-safe.
void append_crash_stage(CrashBuf& b, bool with_memory) noexcept {
    const auto& p = basalt::core::progress();
    const char* where = p.where.load(std::memory_order_relaxed);
    const int64_t detail = p.detail.load(std::memory_order_relaxed);
    const int64_t chunk = p.chunk.load(std::memory_order_relaxed);
    b.puts(" in ");
    b.puts(where != nullptr ? where : "unknown");
    if (detail >= 0) {
        b.puts(" (");
        b.put_i64(detail);
        b.putc(')');
    }
    if (chunk >= 0) {
        b.puts(" chunk ");
        b.put_i64(chunk);
    }
    if (!with_memory) return;
    const ProcessMemory pm = process_memory();
    if (pm.rss_mib > 0 || pm.avail_mib > 0) {
        b.puts(", ");
        b.put_u64(pm.rss_mib);
        b.puts(" MiB RSS, ");
        b.put_u64(pm.avail_mib);
        b.puts(" MiB RAM free");
    }
}

void write_crash_line(const CrashBuf& b) noexcept {
#if defined(_WIN32)
    const HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (h != nullptr && h != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        (void) WriteFile(h, b.data, (DWORD) b.n, &written, nullptr);
    }
#else
    const ssize_t r = ::write(STDERR_FILENO, b.data, b.n);
    (void) r;
#endif
}

void write_crash_backtrace() noexcept {
    void* frames[24]{};
#if defined(_WIN32)
    const USHORT n = CaptureStackBackTrace(0, 24, frames, nullptr);
    if (n > 0) {
        CrashBuf b;
        b.puts("basalt crash: error: backtrace:");
        for (USHORT i = 0; i < n; ++i) {
            b.putc(' ');
            b.put_hex((uintptr_t) frames[i]);
        }
        b.putc('\n');
        write_crash_line(b);
    }
#else
    const int n = ::backtrace(frames, 24);
    if (n > 0) ::backtrace_symbols_fd(frames, n, STDERR_FILENO);
#endif
}

void on_terminate() noexcept {
    if (!g_crash_logged.exchange(true, std::memory_order_relaxed)) {
        const char* what = "unknown exception";
        try {
            if (const std::exception_ptr ep = std::current_exception()) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        CrashBuf b;
        b.puts("basalt crash: error: unhandled exception (");
        b.puts(what);
        b.putc(')');
        append_crash_stage(b, true);
        b.putc('\n');
        write_crash_line(b);
        write_crash_backtrace();
    }
    std::abort();
}

#if defined(_WIN32)
const char* win_exception_name(DWORD code) noexcept {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION: return "EXCEPTION_ACCESS_VIOLATION";
        case EXCEPTION_IN_PAGE_ERROR: return "EXCEPTION_IN_PAGE_ERROR";
        case EXCEPTION_STACK_OVERFLOW: return "EXCEPTION_STACK_OVERFLOW";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "EXCEPTION_ILLEGAL_INSTRUCTION";
        case EXCEPTION_INT_DIVIDE_BY_ZERO: return "EXCEPTION_INT_DIVIDE_BY_ZERO";
        case 0xC0000374u: return "STATUS_HEAP_CORRUPTION";
        case 0xC0000409u: return "STATUS_STACK_BUFFER_OVERRUN";
        case 0xE06D7363u: return "unhandled SEH exception";
        default: return "unhandled exception";
    }
}

LONG WINAPI on_win_exception(EXCEPTION_POINTERS* ep) noexcept {
    if (!g_crash_logged.exchange(true, std::memory_order_relaxed) && ep != nullptr && ep->ExceptionRecord != nullptr) {
        const EXCEPTION_RECORD* r = ep->ExceptionRecord;
        CrashBuf b;
        b.puts("basalt crash: error: ");
        b.puts(win_exception_name(r->ExceptionCode));
        b.puts(" (");
        b.put_hex((uint32_t) r->ExceptionCode, 8);
        b.puts(") at ");
        b.put_hex((uintptr_t) r->ExceptionAddress);
        if ((r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
            r->NumberParameters >= 2) {
            const ULONG_PTR op = r->ExceptionInformation[0];
            b.puts(op == 0 ? " (read from " : op == 1 ? " (write to " : op == 8 ? " (execute at " : " (address ");
            b.put_hex((uint64_t) r->ExceptionInformation[1]);
            b.putc(')');
        }
        append_crash_stage(b, true);
        b.putc('\n');
        write_crash_line(b);
        write_crash_backtrace();
    }
    return EXCEPTION_CONTINUE_SEARCH;   // the system's own crash handling and the exit status stay as they were
}
#else
const char* posix_signal_name(int sig) noexcept {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGBUS: return "SIGBUS";
        case SIGILL: return "SIGILL";
        case SIGFPE: return "SIGFPE";
        default: return "signal";
    }
}

void on_crash_signal(int sig, siginfo_t* info, void*) noexcept {
    if (!g_crash_logged.exchange(true, std::memory_order_relaxed)) {
        CrashBuf b;
        b.puts("basalt crash: error: ");
        b.puts(posix_signal_name(sig));
        if (info != nullptr && info->si_code > 0 && info->si_addr != nullptr) {   // a fault, not a sent signal
            b.puts(" at ");
            b.put_hex((uintptr_t) info->si_addr);
        }
        append_crash_stage(b, false);
        b.putc('\n');
        write_crash_line(b);
        write_crash_backtrace();
    }
    (void) ::raise(sig);   // blocked until this returns, then the default action (SA_RESETHAND) ends the process
}
#endif
}  // namespace

void install_crash_handler() {
    (void) std::set_terminate(on_terminate);
#if defined(_WIN32)
    (void) SetUnhandledExceptionFilter(on_win_exception);
#else
    // backtrace() loads its unwinder on first use, which allocates: do it now, not in the handler
    void* warm[1];
    (void) ::backtrace(warm, 1);
    // a stack overflow of the calling thread faults with no stack left to run a handler on
    static char alt_stack[64 * 1024];
    stack_t ss{};
    ss.ss_sp = alt_stack;
    ss.ss_size = sizeof alt_stack;
    (void) ::sigaltstack(&ss, nullptr);
    struct sigaction sa{};
    sa.sa_sigaction = on_crash_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_ONSTACK;
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) (void) sigaction(sig, &sa, nullptr);
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
