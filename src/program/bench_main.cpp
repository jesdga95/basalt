// basalt-bench: the standard benchmark. It starts the engine as a child process and drives it over its
// stdin/stdout protocol (GEN requests, T / DONE lines) with pre-tokenized fixtures, so the tokenizer and the
// chat template cannot move the numbers. Decode fixtures run in a fixed order (draft state carries across
// requests), prefill runs on scaled, cache-busted prompts, and an optional quality session reports PPL / KLD.
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
// The CRT spells read, write and close with a leading underscore; the inline aliases let this file keep one spelling.
#pragma warning(disable : 4996)
inline long getpid() { return (long) GetCurrentProcessId(); }
inline FILE* popen(const char* command, const char* mode) { return _popen(command, mode); }
inline int pclose(FILE* stream) { return _pclose(stream); }
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <errno.h>

#ifndef BASALT_VERSION
#define BASALT_VERSION "unknown"
#endif
#ifndef BASALT_BENCH_ENGINE
#define BASALT_BENCH_ENGINE ""
#endif
#ifndef BASALT_BENCH_FIXTURES
#define BASALT_BENCH_FIXTURES "data/bench"
#endif
#if __has_include("basalt_git.h")
#include "basalt_git.h"
#endif
#ifndef BASALT_BENCH_GIT
#define BASALT_BENCH_GIT "unknown"
#endif
#ifndef BASALT_BENCH_BUILD_TYPE
#define BASALT_BENCH_BUILD_TYPE "unknown"
#endif
#ifndef BASALT_BENCH_CUDA
#define BASALT_BENCH_CUDA "unknown"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------------------------------------- small helpers

std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    std::string s((size_t) std::max(n, 0), '\0');
    std::vsnprintf(s.data(), s.size() + 1, f, ap2);
    va_end(ap2);
    return s;
}

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::string json_quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            case '\r': o += "\\r"; break;
            default:
                if (c < 0x20) o += fmt("\\u%04x", (unsigned) c);
                else o += (char) c;
        }
    }
    return o + "\"";
}

// Streaming JSON writer: commas and key/value pairing are tracked here.
class JW {
public:
    void begin_obj() { pre(); s_ += '{'; first_.push_back(true); }
    void end_obj() { s_ += '}'; first_.pop_back(); }
    void begin_arr() { pre(); s_ += '['; first_.push_back(true); }
    void end_arr() { s_ += ']'; first_.pop_back(); }
    void key(const std::string& k) { pre(); s_ += json_quote(k); s_ += ':'; key_ = true; }
    void str(const std::string& v) { pre(); s_ += json_quote(v); }
    void num(double v) {
        pre();
        s_ += std::isfinite(v) ? fmt("%.6g", v) : std::string("null");
    }
    void integer(long long v) { pre(); s_ += std::to_string(v); }
    void boolean(bool v) { pre(); s_ += v ? "true" : "false"; }
    const std::string& text() const { return s_; }

private:
    void pre() {
        if (key_) { key_ = false; return; }
        if (!first_.empty()) {
            if (!first_.back()) s_ += ',';
            first_.back() = false;
        }
    }
    std::string s_;
    std::vector<bool> first_;
    bool key_ = false;
};

// ---------------------------------------------------------------------------------------------- NVML (dlopen)

class Nvml {
public:
    struct Snap {
        std::string name;
        double power_limit_w = -1, power_max_w = -1;
        int sm_mhz = -1, mem_mhz = -1;
    };

    bool open() {
#if defined(_WIN32)
        // A bare name is looked up in System32 only: the application and current directories are not searched, so
        // an nvml.dll lying next to the executable cannot stand in for the driver's.
        HMODULE m = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (m == nullptr) m = load_nvsmi();
        h_ = (void*) m;
#else
        h_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (h_ == nullptr) return false;
        init_ = sym<int (*)()>("nvmlInit_v2");
        shut_ = sym<int (*)()>("nvmlShutdown");
        drv_ = sym<int (*)(char*, unsigned)>("nvmlSystemGetDriverVersion");
        cuda_ = sym<int (*)(int*)>("nvmlSystemGetCudaDriverVersion");
        count_ = sym<int (*)(unsigned*)>("nvmlDeviceGetCount_v2");
        handle_ = sym<int (*)(unsigned, void**)>("nvmlDeviceGetHandleByIndex_v2");
        name_ = sym<int (*)(void*, char*, unsigned)>("nvmlDeviceGetName");
        plim_ = sym<int (*)(void*, unsigned*)>("nvmlDeviceGetPowerManagementLimit");
        pcon_ = sym<int (*)(void*, unsigned*, unsigned*)>("nvmlDeviceGetPowerManagementLimitConstraints");
        clk_ = sym<int (*)(void*, int, unsigned*)>("nvmlDeviceGetClockInfo");
        if (!init_ || !count_ || !handle_ || init_() != 0) return false;
        unsigned n = 0;
        if (count_(&n) != 0) return false;
        for (unsigned i = 0; i < n; ++i) {
            void* d = nullptr;
            if (handle_(i, &d) == 0) dev_.push_back(d);
        }
        ok_ = true;
        return true;
    }
    ~Nvml() {
        if (ok_ && shut_) shut_();
    }
    bool ok() const { return ok_; }
    size_t count() const { return dev_.size(); }
    std::string driver() const {
        char b[96] = {0};
        if (ok_ && drv_ && drv_(b, sizeof b) == 0) return b;
        return "unknown";
    }
    std::string cuda_driver() const {
        int v = 0;
        if (ok_ && cuda_ && cuda_(&v) == 0) return fmt("%d.%d", v / 1000, (v % 1000) / 10);
        return "unknown";
    }
    Snap snap(size_t i) const {
        Snap s;
        if (!ok_ || i >= dev_.size()) return s;
        char nm[128] = {0};
        if (name_ && name_(dev_[i], nm, sizeof nm) == 0) s.name = nm;
        unsigned mw = 0, lo = 0, hi = 0;
        if (plim_ && plim_(dev_[i], &mw) == 0) s.power_limit_w = mw / 1000.0;
        if (pcon_ && pcon_(dev_[i], &lo, &hi) == 0) s.power_max_w = hi / 1000.0;
        unsigned c = 0;
        if (clk_ && clk_(dev_[i], 1, &c) == 0) s.sm_mhz = (int) c;    // NVML_CLOCK_SM
        if (clk_ && clk_(dev_[i], 2, &c) == 0) s.mem_mhz = (int) c;   // NVML_CLOCK_MEM
        return s;
    }

private:
#if defined(_WIN32)
    /// A driver that keeps nvml.dll outside System32: its NVSMI directory under one of the program-file roots. The
    /// root comes from the environment as UTF-16, so the path never goes through the ANSI code page. The DLL's own
    /// dependencies are looked up beside it.
    static HMODULE load_nvsmi() {
        static const wchar_t* const roots[] = { L"ProgramFiles", L"ProgramFiles(x86)" };
        wchar_t dir[4096];
        for (const wchar_t* root : roots) {
            const DWORD n = GetEnvironmentVariableW(root, dir, (DWORD) (sizeof dir / sizeof dir[0]));
            if (n == 0 || n >= sizeof dir / sizeof dir[0]) continue;
            const std::wstring path = std::wstring(dir, n) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
            const HMODULE m = LoadLibraryExW(path.c_str(), nullptr,
                                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (m != nullptr) return m;
        }
        return nullptr;
    }
#endif
    template <class F>
#if defined(_WIN32)
    F sym(const char* n) const { return reinterpret_cast<F>(GetProcAddress((HMODULE) h_, n)); }
#else
    F sym(const char* n) const { return reinterpret_cast<F>(dlsym(h_, n)); }
#endif
    void* h_ = nullptr;
    bool ok_ = false;
    std::vector<void*> dev_;
    int (*init_)() = nullptr;
    int (*shut_)() = nullptr;
    int (*drv_)(char*, unsigned) = nullptr;
    int (*cuda_)(int*) = nullptr;
    int (*count_)(unsigned*) = nullptr;
    int (*handle_)(unsigned, void**) = nullptr;
    int (*name_)(void*, char*, unsigned) = nullptr;
    int (*plim_)(void*, unsigned*) = nullptr;
    int (*pcon_)(void*, unsigned*, unsigned*) = nullptr;
    int (*clk_)(void*, int, unsigned*) = nullptr;
};

// ---------------------------------------------------------------------------------------------- engine child

volatile sig_atomic_t g_child = -1;
#if defined(_WIN32)
HANDLE g_child_handle = nullptr;

// Argument quoting for CreateProcessA, which takes one command line rather than a vector: backslashes before a
// quote are doubled, a run of backslashes before the closing quote is doubled, and anything with a space or a
// quote is wrapped.
std::string quote_arg(const std::string& s) {
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
#endif

void on_signal(int) {
    if (g_child > 0) {
#if defined(_WIN32)
        if (g_child_handle != nullptr) (void) TerminateProcess(g_child_handle, 1);
#else
        kill((pid_t) g_child, SIGTERM);
#endif
    }
    _exit(130);
}

struct Done {
    bool ok = false;
    std::string err;
    long long generated = 0, prompt = 0, accepted = 0, offered = 0, reused = 0, read_n = -1;
    double prompt_ms = 0, decode_ms = 0, wall_s = 0, first_token_s = -1;
    std::string finish;
};

std::string engine_failure(const std::string& log_path);

class Engine {
public:
    ~Engine() { stop(); }

    bool start(const std::string& engine_bin, const std::vector<std::string>& args, const std::string& log_path,
               std::string& err) {
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
        HANDLE log = CreateFileA(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, &sa,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        STARTUPINFOA si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = rd_in;
        si.hStdOutput = wr_out;
        si.hStdError = log != INVALID_HANDLE_VALUE ? log : wr_out;
        std::string cmd = quote_arg(engine_bin) + " --serve";
        for (const auto& a : args) cmd += " " + quote_arg(a);
        PROCESS_INFORMATION pi{};
        if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
            err = "CreateProcess failed (error " + std::to_string(GetLastError()) + ")";
            CloseHandle(rd_in); CloseHandle(wr_in); CloseHandle(rd_out); CloseHandle(wr_out);
            if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
            return false;
        }
        CloseHandle(pi.hThread);
        proc_ = pi.hProcess;
        g_child = 1;
        g_child_handle = proc_;
        // The parent's ends become CRT descriptors, so the request and reply loops are the same code either way.
        in_ = _open_osfhandle((intptr_t) wr_in, _O_BINARY | _O_WRONLY);
        out_ = _open_osfhandle((intptr_t) rd_out, _O_BINARY | _O_RDONLY);
        CloseHandle(rd_in);              // the child owns the request pipe's read end
        CloseHandle(wr_out);             // the child owns the reply pipe's write end
        if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
#else
        int in[2], out[2];
        if (pipe(in) != 0 || pipe(out) != 0) { err = "pipe failed"; return false; }
        const pid_t pid = fork();
        if (pid < 0) { err = "fork failed"; return false; }
        if (pid == 0) {
            dup2(in[0], 0);
            dup2(out[1], 1);
            const int lf = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (lf >= 0) dup2(lf, 2);
            close(in[0]); close(in[1]); close(out[0]); close(out[1]);
            std::vector<std::string> store;
            store.push_back(engine_bin);
            store.push_back("--serve");
            for (const auto& a : args) store.push_back(a);
            std::vector<char*> av;
            for (auto& s : store) av.push_back(s.data());
            av.push_back(nullptr);
            execv(engine_bin.c_str(), av.data());
            _exit(127);
        }
        close(in[0]);
        close(out[1]);
        in_ = in[1];
        out_ = out[0];
        pid_ = pid;
        g_child = pid;
#endif
        log_path_ = log_path;
        std::string line;
        while (read_line(line)) {
            if (line.rfind("INFO ", 0) == 0) {
                std::istringstream is(line.substr(5));
                std::string kv;
                while (is >> kv) {
                    const size_t eq = kv.find('=');
                    if (eq != std::string::npos) info[kv.substr(0, eq)] = kv.substr(eq + 1);
                }
            } else if (line.rfind("READY", 0) == 0) {
                return true;
            }
        }
        err = "the engine exited before READY" + engine_failure(log_path);
        reap();
        return false;
    }

    // One GEN request. `keys` are the sampling keys, each with a leading space.
    bool request(const std::string& keys, long long max_new, const std::vector<int64_t>& ids, Done& d) {
        d = Done{};
        std::string msg = "GEN " + std::to_string(max_new) + keys + " ";
        msg.reserve(msg.size() + ids.size() * 7 + 2);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) msg += ',';
            msg += std::to_string(ids[i]);
        }
        msg += '\n';
        const auto t0 = Clock::now();
        if (!write_all(msg)) { d.err = "the engine closed its input"; return false; }
        std::string line;
        while (read_line(line)) {
            if (line.rfind("T ", 0) == 0) {
                if (d.first_token_s < 0) d.first_token_s = secs(t0, Clock::now());
            } else if (line.rfind("DONE", 0) == 0) {
                d.wall_s = secs(t0, Clock::now());
                std::istringstream is(line);
                std::string tag;
                std::vector<std::string> f;
                while (is >> tag) f.push_back(tag);
                if (f.size() < 8) { d.err = "short DONE line: " + line; return false; }
                d.generated = std::atoll(f[1].c_str());
                d.prompt = std::atoll(f[2].c_str());
                d.prompt_ms = std::atof(f[3].c_str());
                d.decode_ms = std::atof(f[4].c_str());
                d.finish = f[5];
                d.accepted = std::atoll(f[6].c_str());
                d.offered = std::atoll(f[7].c_str());
                if (f.size() > 8) d.reused = std::atoll(f[8].c_str());
                if (f.size() > 14) d.read_n = std::atoll(f[14].c_str());
                d.ok = true;
                return true;
            } else if (line.rfind("ERR", 0) == 0) {
                d.err = line;
                return false;
            }
        }
        d.err = "the engine exited during the request (see " + log_path_ + ")";
        return false;
    }

    // QUIT and wait: the engine flushes its logs on a normal exit.
    void stop() {
#if defined(_WIN32)
        if (proc_ == nullptr) return;
#else
        if (pid_ <= 0) return;
#endif
        const std::string q = "QUIT\n";
        if (in_ >= 0) { (void) !write(in_, q.data(), q.size()); close(in_); in_ = -1; }
        for (int i = 0; i < 600; ++i) {
#if defined(_WIN32)
            if (WaitForSingleObject(proc_, 0) == WAIT_OBJECT_0) { reap(); return; }
            Sleep(100);
#else
            int st = 0;
            if (waitpid(pid_, &st, WNOHANG) == pid_) { finish_reap(); return; }
            usleep(100000);
#endif
        }
#if defined(_WIN32)
        (void) TerminateProcess(proc_, 1);
#else
        kill(pid_, SIGTERM);
#endif
        reap();
    }

    std::map<std::string, std::string> info;

private:
    bool write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = write(in_, s.data() + off, s.size() - off);
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            off += (size_t) n;
        }
        return true;
    }
    bool read_line(std::string& line) {
        for (;;) {
            const size_t nl = buf_.find('\n', rpos_);
            if (nl != std::string::npos) {
                line.assign(buf_, rpos_, nl - rpos_);
                rpos_ = nl + 1;
                if (rpos_ > (1u << 20)) { buf_.erase(0, rpos_); rpos_ = 0; }
                return true;
            }
            char tmp[65536];
            const ssize_t n = read(out_, tmp, sizeof tmp);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            buf_.append(tmp, (size_t) n);
        }
    }
    void reap() {
#if defined(_WIN32)
        if (proc_ != nullptr) (void) WaitForSingleObject(proc_, 10000);
#else
        if (pid_ > 0) {
            int st = 0;
            waitpid(pid_, &st, 0);
        }
#endif
        finish_reap();
    }
    void finish_reap() {
        if (in_ >= 0) close(in_);
        if (out_ >= 0) close(out_);
        in_ = out_ = -1;
#if defined(_WIN32)
        if (proc_ != nullptr) { CloseHandle(proc_); proc_ = nullptr; }
        g_child_handle = nullptr;
#else
        pid_ = -1;
#endif
        g_child = -1;
    }
#if defined(_WIN32)
    HANDLE proc_ = nullptr;
#else
    pid_t pid_ = -1;
#endif
    int in_ = -1, out_ = -1;
    std::string buf_, log_path_;
    size_t rpos_ = 0;
};

// ---------------------------------------------------------------------------------------------- options

struct Options {
    std::string engine, model, fixtures = BASALT_BENCH_FIXTURES, format = "md", logs, kld_base, power_label;
    std::vector<std::pair<std::string, std::string>> sets;
    std::vector<std::string> passthrough;
    std::string spec, spec_min_p;        // --spec N, --spec-min-p P: engine arguments of the same names
    std::vector<std::string> decode;     // -t: conv, struct, prose
    std::vector<long long> depths;       // -p: prefill depths in tokens
    bool sel_decode = false, sel_prefill = false;
    int reps = 1;
    bool quality = false, no_windows = false, print_command = false;
    int quality_chunks = 20, quality_t = 3;
};

void usage(const std::string& engine) {
    std::printf(
        "basalt-bench - the standard benchmark: decode fixtures, prefill depths, optional quality gate\n"
        "\n"
        "usage: basalt-bench --model FILE.basalt [options] [-- ENGINE ARGS...]\n"
        "\n"
        "what to run (default: -p 16384 -t conv,struct,prose; giving one of -p / -t runs only what is given)\n"
        "  -t LIST            decode fixtures, any of conv,struct,prose; they always run in that order, because\n"
        "                     draft state carries across requests: one fixture alone does not match a full run\n"
        "  -p LIST            prefill depths in tokens (16384,32768,65536,131072; 32k style also accepted)\n"
        "  -r N               repetitions of the whole set (default 1); with N > 1 the table shows mean +- stddev\n"
        "                     and the first repetition is the cold one\n"
        "  --quality [N]      the quality gate on N chunks (default 20): PPL / KLD / top-1 against --kld-base,\n"
        "                     through the decode path, adaptation off, --spec 5 --spec-min-p 0.7. Alone it runs\n"
        "                     only the gate; with -p / -t it runs after them\n"
        "  --kld-base FILE    llama.cpp KL-divergence base for the quality gate\n"
        "  --quality-t T      tokens per verify window in the quality gate (default 3)\n"
        "\n"
        "engine\n"
        "  --model FILE       a .basalt model file; --peer-device 1 is added when a second GPU is visible. Every\n"
        "                     other --flag is an engine flag, passed on as given (same names as `basalt`; the engine\n"
        "                     rejects a bad one at startup); --print-command shows the command and exits\n"
        "  --engine BIN       engine binary (default: the one built with this program)\n"
        "  --set \"F V ...\"    FLAG VALUE pairs that replace or add engine arguments (repeatable)\n"
        "  --spec N           speculation window for the engine (--spec N); without it, and without --spec after --,\n"
        "                     the run has no speculation (--spec 0): a clean baseline\n"
        "  --spec-min-p P     the draft cutoff (--spec-min-p P); with N >= 2 and no cutoff given, 0.5\n"
        "                     MTP: --spec 7 --spec-min-p 0.5 (or -- --spec 7 --spec-min-p 0.5)\n"
        "  -- ARGS...         the same engine arguments, after --: replacing and adding flags\n"
        "  built in (only what the measurement needs, each replaced by giving the flag): --spec 0, --max-context sized\n"
        "  to the fixtures, --spec-min-p 0.5 when --spec is 2 or more\n"
        "\n"
        "output\n"
        "  -o md|json         a markdown table (default), or the full detail as JSON on stdout: per-repetition\n"
        "                     values, tokens, drafted, accepted, ms/window, verify ms by window width, GPU names,\n"
        "                     driver, clocks, model, expert tiers, engine arguments\n"
        "  --power-label W    the power limit to print, instead of the one read from the GPU\n"
        "  --logs DIR         keep the engine log, the draft log and the quality rows there\n"
        "  --no-windows       do not ask the engine for the per-window log (no window statistics)\n"
        "  progress goes to stderr\n"
        "\n"
        "fixtures (--fixtures DIR, default %s)\n"
        "  conv.ids struct.ids prose.ids   one line each: the comma-separated token ids of the chat-templated\n"
        "                                  prompt (thinking off, generation prompt added)\n"
        "  prefill.ids                     three lines: the ids before the body, the body, the ids after the body;\n"
        "                                  the body is the 16384-token depth, cut or repeated to the asked depth,\n"
        "                                  and 4 random ids are inserted before it so the prompt cache cannot match\n"
        "  Prompts: conv  \"Explain how a B-tree differs from an LSM tree, with a worked example of\n"
        "  each. About 600 words.\" (1200 tokens, temperature 0.6); struct  the JSON-array prompt (8192 tokens,\n"
        "  temperature 0, 7230 tokens expected); prose  \"Write an essay on the Odyssey.\" (1200 tokens,\n"
        "  temperature 0.6); prefill  40000 characters of filler as the body, then \"In one sentence, what is this\n"
        "  document?\".\n"
        "  Make the files once, on a machine with the model, with the server's own tokenizer and template:\n"
        "    python3 -c \"import sys; sys.path[:0]=['.','tools']; from tokenizer import Tokenizer;\n"
        "      from serve.frontend import ChatTemplate; t=Tokenizer.from_gguf(GGUF_SHARD_1);\n"
        "      c=ChatTemplate('serve/chat_template.jinja'); r=lambda p: t.encode(c.render(\n"
        "      [{'role':'user','content':p}], enable_thinking=False), parse_special=True)\n"
        "      print(','.join(map(str, r(PROMPT))))\"\n"
        "  (for prefill, render with a marker in place of the body and split the ids around it)\n",
        BASALT_BENCH_FIXTURES);
    // the engine's flags are defined once, in the engine: print its own help
    std::string bin = engine.empty() ? std::string(BASALT_BENCH_ENGINE) : engine;
    if (!bin.empty() && fs::exists(bin)) {
        std::printf("\nEngine flags (passed to basalt):\n");
        std::fflush(stdout);
        if (FILE* pf = popen(("'" + bin + "' --help 2>&1").c_str(), "r")) {
            char buf[4096];
            while (std::fgets(buf, sizeof buf, pf) != nullptr) std::fputs(buf, stdout);
            pclose(pf);
        }
    }
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> r;
    std::string item;
    std::istringstream is(s);
    while (std::getline(is, item, sep))
        if (!item.empty()) r.push_back(item);
    return r;
}

bool all_digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

bool parse_depth(const std::string& s, long long& tokens) {
    std::string d = s;
    std::transform(d.begin(), d.end(), d.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (d.empty()) return false;
    char* e = nullptr;
    if (d.back() == 'k') {
        const double v = std::strtod(d.c_str(), &e);
        if (e != d.c_str() + d.size() - 1 || v <= 0) return false;
        tokens = (long long) (v * 1024);
    } else {
        if (!all_digits(d)) return false;
        tokens = std::atoll(d.c_str());
    }
    return tokens > 0;
}

// FLAG VALUE: replace the value when the flag is there, else append both.
void set_arg(std::vector<std::string>& args, const std::string& flag, const std::string& value) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) { args[i + 1] = value; return; }
    args.push_back(flag);
    args.push_back(value);
}

bool has_arg(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

std::string arg_value(const std::vector<std::string>& args, const std::string& flag) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) return args[i + 1];
    return "";
}

// The GPUs this process can see: CUDA_VISIBLE_DEVICES when set, else what NVML lists.
size_t visible_gpus() {
    if (const char* v = std::getenv("CUDA_VISIBLE_DEVICES")) {
        size_t n = 0;
        for (const auto& d : split(v, ','))
            if (d != "-1") ++n;
        return n;
    }
    Nvml nvml;
    return nvml.open() ? nvml.count() : 0;
}

// FLAG [VALUE] words added to `args`: a flag already there takes the new value, else both are appended.  A word
// that does not begin with "--" after a flag is its value.
void merge_args(std::vector<std::string>& args, const std::vector<std::string>& add) {
    for (size_t i = 0; i < add.size(); ++i) {
        const bool value = i + 1 < add.size() && add[i + 1].rfind("--", 0) != 0;
        if (add[i].rfind("--", 0) != 0) { args.push_back(add[i]); continue; }
        if (value) { set_arg(args, add[i], add[i + 1]); ++i; }
        else if (!has_arg(args, add[i])) args.push_back(add[i]);
    }
}

// The engine's own `basalt error:` lines from its log, else the last few lines.
std::string engine_failure(const std::string& log_path) {
    std::ifstream f(log_path);
    std::vector<std::string> lines, errors;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        lines.push_back(line);
        if (line.rfind("basalt error: ", 0) == 0) errors.push_back(line);
    }
    if (errors.empty() && lines.size() > 8) lines.erase(lines.begin(), lines.end() - 8);
    std::string r;
    for (const auto& l : errors.empty() ? lines : errors) r += "\n  " + l;
    return r;
}

bool parse_args(int argc, char** argv, Options& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { err = std::string(what) + " needs a value"; return ""; }
            return argv[++i];
        };
        if (a == "--") {
            for (++i; i < argc; ++i) o.passthrough.push_back(argv[i]);
        } else if (a == "--help" || a == "-h") {
            std::string engine;
            for (int k = 1; k + 1 < argc; ++k)
                if (std::string(argv[k]) == "--engine") engine = argv[k + 1];
            usage(engine);
            std::exit(0);
        } else if (a == "--model") o.model = next("--model");
        else if (a == "--engine") o.engine = next("--engine");
        else if (a == "--spec") o.spec = next("--spec");
        else if (a == "--spec-min-p") o.spec_min_p = next("--spec-min-p");
        else if (a == "--fixtures") o.fixtures = next("--fixtures");
        else if (a == "-o") o.format = next("-o");
        else if (a == "--power-label") o.power_label = next("--power-label");
        else if (a == "--logs") o.logs = next("--logs");
        else if (a == "--kld-base") o.kld_base = next("--kld-base");
        else if (a == "--no-windows") o.no_windows = true;
        else if (a == "-r") o.reps = std::atoi(next("-r").c_str());
        else if (a == "--quality-t") o.quality_t = std::atoi(next("--quality-t").c_str());
        else if (a == "-p") {
            o.sel_prefill = true;
            for (const auto& d : split(next("-p"), ',')) {
                long long t = 0;
                if (!parse_depth(d, t)) { err = "-p: bad depth " + d; return false; }
                o.depths.push_back(t);
            }
        } else if (a == "-t") {
            o.sel_decode = true;
            o.decode = split(next("-t"), ',');
        } else if (a == "--quality") {
            o.quality = true;
            if (i + 1 < argc && all_digits(argv[i + 1])) o.quality_chunks = std::atoi(argv[++i]);
        } else if (a == "--set") {
            const std::vector<std::string> w = split(next("--set"), ' ');
            if (w.size() % 2 != 0) { err = "--set takes FLAG VALUE pairs"; return false; }
            for (size_t k = 0; k < w.size(); k += 2) o.sets.emplace_back(w[k], w[k + 1]);
        } else if (a == "--print-command") o.print_command = true;
        else if (a.rfind("--", 0) == 0) {
            // any other long flag is the engine's, with its value when one follows; a following option
            // (anything starting with '-' that is not a negative number) means the flag is a switch
            o.passthrough.push_back(a);
            if (i + 1 < argc) {
                const std::string v = argv[i + 1];
                const bool option = !v.empty() && v[0] == '-' &&
                                    !(v.size() > 1 && (std::isdigit(static_cast<unsigned char>(v[1])) || v[1] == '.'));
                if (!option) o.passthrough.push_back(argv[++i]);
            }
        } else {
            err = "unknown argument " + a + " (see --help)";
        }
        if (!err.empty()) return false;
    }
    if (o.reps < 1) { err = "-r needs N >= 1"; return false; }
    if (o.format != "md" && o.format != "json") { err = "-o takes md or json"; return false; }
    if (o.quality_chunks < 1 || o.quality_t < 1) { err = "--quality and --quality-t need values >= 1"; return false; }
    for (const auto& s : o.decode)
        if (s != "conv" && s != "struct" && s != "prose") {
            err = "-t: unknown fixture " + s + " (conv, struct, prose)";
            return false;
        }
    if (o.model.empty() && o.passthrough.empty()) {
        err = "give --model FILE or engine arguments after --";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- fixtures

struct Fixture {
    const char* name;
    const char* file;
    long long max_new;
    double temperature;   // 0: greedy
};

// Same content and order as the standard decode set; sampling keys as the daily server config sends them.
const Fixture kDecode[] = {
    {"conv", "conv.ids", 1200, 0.6},
    {"struct", "struct.ids", 8192, 0.0},
    {"prose", "prose.ids", 1200, 0.6},
};
constexpr long long kStructTokens = 7230;
constexpr long long kPrefillBase = 16384;   // the base body's depth in tokens
constexpr long long kPrefillBaseDecode = 32;

bool read_id_lines(const std::string& path, std::vector<std::vector<int64_t>>& lines, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot read " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        std::vector<int64_t> v;
        const char* p = line.c_str();
        char* e = nullptr;
        for (;;) {
            while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r') ++p;
            if (*p == '\0') break;
            const long long x = std::strtoll(p, &e, 10);
            if (e == p) { err = path + ": not a list of token ids"; return false; }
            v.push_back(x);
            p = e;
        }
        if (!v.empty()) lines.push_back(std::move(v));
    }
    return true;
}

std::string sampling_keys(double temperature) {
    std::string k;
    if (temperature > 0.0) k += fmt(" temperature=%.17g", temperature);
    k += " top_p=0.95 top_k=20";
    return k;
}

// ---------------------------------------------------------------------------------------------- window log

struct WinStats {
    bool present = false;
    long long windows = 0;
    double mean_t = 0, tokens_per_window = 0;
    std::array<double, 17> verify_ms{};   // median verify ms by window width T (0: none)
    std::array<double, 17> share{};       // share of the windows by T
    double mix = 0;                       // sum over T of share * median verify ms
};

using LogBlock = std::vector<std::array<double, 6>>;   // T, accepted, from-suffix, pre ms, verify ms, post ms

std::vector<LogBlock> read_draft_log(const std::string& path) {
    std::vector<LogBlock> blocks;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("R ", 0) == 0) {
            blocks.emplace_back();
        } else if (line.rfind("W ", 0) == 0 && !blocks.empty()) {
            int t = 0, a = 0, s = 0;
            double x = 0, y = 0, z = 0;
            if (std::sscanf(line.c_str(), "W %d %d %d %lf %lf %lf", &t, &a, &s, &x, &y, &z) == 6)
                blocks.back().push_back({(double) t, (double) a, (double) s, x, y, z});
        }
    }
    return blocks;
}

WinStats window_stats(const LogBlock& b) {
    WinStats s;
    if (b.empty()) return s;
    s.present = true;
    s.windows = (long long) b.size();
    std::array<std::vector<double>, 17> by;
    double sum_t = 0, toks = 0;
    for (const auto& w : b) {
        sum_t += w[0];
        toks += w[1] + 1;
        const int t = std::clamp((int) w[0], 0, 16);
        if (w[2] == 0) by[(size_t) t].push_back(w[4]);
    }
    s.mean_t = sum_t / (double) b.size();
    s.tokens_per_window = toks / (double) b.size();
    size_t n_all = 0;
    for (const auto& v : by) n_all += v.size();
    for (size_t t = 0; t < by.size(); ++t) {
        if (by[t].empty()) continue;
        std::sort(by[t].begin(), by[t].end());
        s.verify_ms[t] = by[t][by[t].size() / 2];
        s.share[t] = (double) by[t].size() / (double) std::max<size_t>(n_all, 1);
        s.mix += s.share[t] * s.verify_ms[t];
    }
    return s;
}

// ---------------------------------------------------------------------------------------------- results

struct Result {
    std::string name;   // conv, struct, prose, prefill
    int run = 0;
    long long depth = 0;   // prefill: requested depth in tokens
    Done d;
    int sm_mhz = -1, mem_mhz = -1;
    WinStats win;
    bool checksum_bad = false;
};

double tok_s(const Done& d) { return d.decode_ms > 0 ? (double) d.generated / (d.decode_ms / 1000.0) : 0.0; }
double prefill_engine_ts(const Done& d) {
    const long long n = d.prompt - d.reused;
    return d.prompt_ms > 0 && n > 0 ? (double) n / (d.prompt_ms / 1000.0) : 0.0;
}
double prefill_wall_ts(const Done& d) {
    const double w = d.first_token_s > 0 ? d.first_token_s : d.wall_s;
    return w > 0 ? (double) d.prompt / w : 0.0;
}
// windows the draft log did not show are estimated as tokens minus accepted drafts
double window_count(const Result& r) {
    return r.win.present ? (double) r.win.windows : (double) std::max<long long>(r.d.generated - r.d.accepted, 1);
}

struct Quality {
    bool ran = false;
    std::string err;
    long long rows = 0, chunks = 0;
    double ppl = 0, ppl_err = 0, base_ppl = 0, ln_ratio = 0, kld_mean = 0, kld_median = 0, kld_p99 = 0, kld_max = 0;
    double top1_pct = 0, dlogp = 0, load_s = 0, run_s = 0, prompt_ms_chunk = 0, decode_ms_chunk = 0;
    int t = 3, n_chunks_asked = 0;
    std::string args;
};

void summarize_rows(const std::string& path, Quality& q) {
    std::ifstream f(path);
    std::string line;
    std::vector<double> nll, nllb, kld, dlp;
    long long same = 0;
    std::vector<long long> chunk_ids;
    while (std::getline(f, line)) {
        std::vector<std::string> c = split(line, '\t');
        if (c.size() < 10) continue;
        chunk_ids.push_back(std::atoll(c[0].c_str()));
        nll.push_back(std::atof(c[3].c_str()));
        nllb.push_back(std::atof(c[4].c_str()));
        kld.push_back(std::atof(c[5].c_str()));
        same += std::atoll(c[6].c_str());
        dlp.push_back(std::atof(c[9].c_str()));
    }
    const size_t n = nll.size();
    if (n == 0) { q.err = "no rows in " + path; return; }
    std::sort(chunk_ids.begin(), chunk_ids.end());
    q.chunks = (long long) (std::unique(chunk_ids.begin(), chunk_ids.end()) - chunk_ids.begin());
    q.rows = (long long) n;
    double mn = 0, mb = 0, sq = 0, mk = 0, md = 0;
    for (size_t i = 0; i < n; ++i) { mn += nll[i]; mb += nllb[i]; sq += nll[i] * nll[i]; mk += kld[i]; md += dlp[i]; }
    mn /= (double) n; mb /= (double) n; mk /= (double) n; md /= (double) n;
    const double se = std::sqrt(std::max(0.0, sq / (double) n - mn * mn) / (double) std::max<size_t>(1, n - 1));
    q.ppl = std::exp(mn);
    q.ppl_err = q.ppl * se;
    q.base_ppl = std::exp(mb);
    q.ln_ratio = mn - mb;
    std::sort(kld.begin(), kld.end());
    auto pct = [&](double p) { return kld[std::min(n - 1, (size_t) std::llround(p * (double) (n - 1)))]; };
    q.kld_mean = mk;
    q.kld_median = pct(0.5);
    q.kld_p99 = pct(0.99);
    q.kld_max = kld.back();
    q.top1_pct = 100.0 * (double) same / (double) n;
    q.dlogp = md;
    q.ran = true;
}

bool read_kld_base(const std::string& path, uint32_t& n_ctx, int32_t& n_vocab, int32_t& n_chunk,
                   std::vector<std::vector<int64_t>>& chunks, long long want, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, "_logits_", 8) != 0) { err = path + ": not a KL-divergence base"; return false; }
    if (!f.read((char*) &n_ctx, 4) || !f.read((char*) &n_vocab, 4) || !f.read((char*) &n_chunk, 4) || n_ctx < 4 || n_chunk < 1) {
        err = path + ": truncated header";
        return false;
    }
    const long long take = std::min<long long>(want, n_chunk);
    for (long long k = 0; k < take; ++k) {
        std::vector<int32_t> t(n_ctx);
        if (!f.read((char*) t.data(), (std::streamsize) (4ull * n_ctx))) { err = path + ": truncated tokens"; return false; }
        chunks.emplace_back(t.begin(), t.end());
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- main

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string err;
    if (!parse_args(argc, argv, o, err)) {
        std::fprintf(stderr, "basalt-bench: %s\n", err.c_str());
        return 2;
    }
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);   // no SIGPIPE here: a closed pipe surfaces as a failed write
#endif
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // what to run
    const bool quality_only = o.quality && !o.sel_decode && !o.sel_prefill;
    if (!o.sel_decode && !o.sel_prefill && !quality_only) {   // one of -t / -p runs only what it names
        o.decode = {"conv", "struct", "prose"};
        o.depths = {kPrefillBase};
    }
    auto wants = [&](const std::string& s) { return std::find(o.decode.begin(), o.decode.end(), s) != o.decode.end(); };
    const bool do_quality = o.quality;
    const bool do_decode = !o.decode.empty();
    const bool do_prefill = !o.depths.empty();
    const bool do_perf = do_decode || do_prefill;
    if (do_quality && o.kld_base.empty()) {
        std::fprintf(stderr, "basalt-bench: the quality gate needs --kld-base FILE\n");
        return 2;
    }

    // engine command from the model and the overrides
    std::string engine_bin = o.engine;
    std::vector<std::string> args;
    if (!o.model.empty()) {
        args = {"--model", fs::absolute(o.model).string()};
        // A second GPU can hold an expert tier only where a process is given GPU-to-GPU access between its
        // devices; the engine refuses --peer-device without it, so the bench does not offer one.
#if !defined(_WIN32)
        if (!has_arg(args, "--peer-device") && visible_gpus() > 1) set_arg(args, "--peer-device", "1");
#endif
    }
    if (engine_bin.empty()) engine_bin = BASALT_BENCH_ENGINE;
    if (o.engine.empty() && std::string(BASALT_BENCH_ENGINE).size() > 0 && fs::exists(BASALT_BENCH_ENGINE)) engine_bin = BASALT_BENCH_ENGINE;
    if (engine_bin.empty() || !fs::exists(engine_bin)) {
        std::fprintf(stderr, "basalt-bench: the engine binary %s does not exist (--engine BIN)\n", engine_bin.c_str());
        return 2;
    }
    for (const auto& kv : o.sets) set_arg(args, kv.first, kv.second);
    if (!o.spec.empty()) set_arg(args, "--spec", o.spec);
    if (!o.spec_min_p.empty()) set_arg(args, "--spec-min-p", o.spec_min_p);
    if (o.model.empty()) for (const auto& a : o.passthrough) args.push_back(a);
    else merge_args(args, o.passthrough);
    if (!has_arg(args, "--spec")) set_arg(args, "--spec", "0");
    if (!has_arg(args, "--spec-min-p") && std::atoi(arg_value(args, "--spec").c_str()) >= 2)
        set_arg(args, "--spec-min-p", "0.5");

    // fixtures, read before anything is started
    std::vector<std::vector<int64_t>> decode_ids(3);
    std::vector<std::vector<int64_t>> prefill_parts;
    const std::vector<long long>& prefill_runs = o.depths;
    if (do_decode)
        for (size_t i = 0; i < 3; ++i) {
            if (!wants(kDecode[i].name)) continue;
            std::vector<std::vector<int64_t>> lines;
            if (!read_id_lines((fs::path(o.fixtures) / kDecode[i].file).string(), lines, err) || lines.empty()) {
                std::fprintf(stderr, "basalt-bench: %s\n", err.empty() ? (std::string(kDecode[i].file) + " is empty").c_str() : err.c_str());
                return 2;
            }
            for (auto& l : lines) decode_ids[i].insert(decode_ids[i].end(), l.begin(), l.end());
        }
    if (do_prefill) {
        if (!read_id_lines((fs::path(o.fixtures) / "prefill.ids").string(), prefill_parts, err) || prefill_parts.size() != 3) {
            std::fprintf(stderr, "basalt-bench: %s\n", err.empty() ? "prefill.ids needs three lines (before, body, after)" : err.c_str());
            return 2;
        }
    }

    // the context the fixtures need, unless the command line gives one
    if (!has_arg(args, "--max-context")) {
        long long need = 4096;
        for (size_t i = 0; i < 3; ++i)
            if (!decode_ids[i].empty()) need = std::max(need, (long long) decode_ids[i].size() + kDecode[i].max_new + 16);
        if (do_prefill)
            for (long long depth : o.depths) {
                const double scale = (double) depth / (double) kPrefillBase;
                need = std::max(need, (long long) ((double) prefill_parts[1].size() * scale) + (long long) prefill_parts[0].size() +
                                          (long long) prefill_parts[2].size() + kPrefillBaseDecode + 16);
            }
        set_arg(args, "--max-context", std::to_string(need));
    }
    if (o.print_command) {
        std::printf("%s --serve", engine_bin.c_str());
        for (const auto& a : args) std::printf(" %s", a.c_str());
        std::printf("\n");
        return 0;
    }

    // scratch files
    std::string work = o.logs;
    bool temp_work = false;
    if (work.empty()) {
#if defined(_WIN32)
        char tmp[MAX_PATH] = {};
        if (GetTempPathA(sizeof tmp, tmp) == 0) tmp[0] = '.';
        work = fmt("%sbasalt-bench-%d", tmp, (int) getpid());
#else
        work = fmt("/tmp/basalt-bench-%d", (int) getpid());
#endif
        temp_work = true;
    }
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { std::fprintf(stderr, "basalt-bench: cannot create %s\n", work.c_str()); return 2; }
    struct Cleanup {
        std::string dir;
        bool on;
        ~Cleanup() {
            if (on) {
                std::error_code e;
                fs::remove_all(dir, e);
            }
        }
    } cleanup{work, temp_work};

    Nvml nvml;
    const bool have_nvml = nvml.open();
    if (!have_nvml) std::fprintf(stderr, "basalt-bench: NVML is unavailable: power limits and clocks are NOT recorded\n");
    std::vector<Nvml::Snap> gpus_start;
    for (size_t i = 0; i < nvml.count(); ++i) gpus_start.push_back(nvml.snap(i));

    std::vector<Result> results;
    std::map<std::string, std::string> info;
    std::string peer_line;
    std::string engine_args_text;
    Quality quality;

    // ---- performance session
    std::vector<std::string> perf_args = args;
    const std::string draft_log = (fs::path(work) / "draft.log").string();
    const bool windows_on = !o.no_windows;
    if (do_perf) {
        if (windows_on) {
            std::remove(draft_log.c_str());
            set_arg(perf_args, "--draft-log", draft_log);
        }
        for (const auto& a : perf_args) engine_args_text += (engine_args_text.empty() ? "" : " ") + a;
        const std::string log_path = (fs::path(work) / "engine.log").string();
        Engine eng;
        std::fprintf(stderr, "basalt-bench: starting %s\n", engine_bin.c_str());
        if (!eng.start(engine_bin, perf_args, log_path, err)) {
            std::fprintf(stderr, "basalt-bench: %s\n", err.c_str());
            return 1;
        }
        info = eng.info;
        {
            std::ifstream lg(log_path);
            std::string l;
            while (std::getline(lg, l))
                if (l.find("peer GPU") != std::string::npos && l.find("experts") != std::string::npos) peer_line = l;
        }
        std::mt19937_64 rng((uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() ^ ((uint64_t) getpid() << 32));
        int n_requests = 0, n_decode_requests = 0;
        bool failed = false;
        for (int run = 1; run <= o.reps && !failed; ++run) {
            for (size_t i = 0; i < 3 && !failed; ++i) {
                if (!wants(kDecode[i].name)) continue;
                Result r;
                r.name = kDecode[i].name;
                r.run = run;
                if (!eng.request(sampling_keys(kDecode[i].temperature), kDecode[i].max_new, decode_ids[i], r.d)) {
                    std::fprintf(stderr, "basalt-bench: %s (run %d): %s\n", r.name.c_str(), run, r.d.err.c_str());
                    failed = true;
                    break;
                }
                const Nvml::Snap s = nvml.snap(0);
                r.sm_mhz = s.sm_mhz;
                r.mem_mhz = s.mem_mhz;
                r.checksum_bad = r.name == "struct" && r.d.generated != kStructTokens;
                ++n_requests;
                ++n_decode_requests;
                std::fprintf(stderr, "  run %d %-7s %.1f tok/s, %lld tokens\n", run, r.name.c_str(), tok_s(r.d), r.d.generated);
                results.push_back(std::move(r));
            }
            for (size_t pi = 0; pi < prefill_runs.size() && !failed; ++pi) {
                const long long depth = prefill_runs[pi];
                const auto& pre = prefill_parts[0];
                const auto& body = prefill_parts[1];
                const auto& post = prefill_parts[2];
                const size_t n_body = std::max<size_t>(1, (size_t) std::llround((double) body.size() * (double) depth / (double) kPrefillBase));
                std::vector<int64_t> ids(pre);
                std::uniform_int_distribution<int64_t> nonce(1000, 150000);
                for (int k = 0; k < 4; ++k) ids.push_back(nonce(rng));
                ids.reserve(ids.size() + n_body + post.size());
                for (size_t k = 0; k < n_body; ++k) ids.push_back(body[k % body.size()]);
                ids.insert(ids.end(), post.begin(), post.end());
                Result r;
                r.name = "prefill";
                r.run = run;
                r.depth = depth;
                if (!eng.request(" top_p=0.95 top_k=20", depth == kPrefillBase ? kPrefillBaseDecode : 1, ids, r.d)) {
                    std::fprintf(stderr, "basalt-bench: pp%lld (run %d): %s\n", depth, run, r.d.err.c_str());
                    failed = true;
                    break;
                }
                const Nvml::Snap s = nvml.snap(0);
                r.sm_mhz = s.sm_mhz;
                r.mem_mhz = s.mem_mhz;
                ++n_requests;
                std::fprintf(stderr, "  run %d pp%lld %lld tokens, %.0f t/s engine, %.0f t/s wall\n", run, depth, r.d.prompt,
                             prefill_engine_ts(r.d), prefill_wall_ts(r.d));
                results.push_back(std::move(r));
            }
        }
        std::vector<Nvml::Snap> gpus_end;
        for (size_t i = 0; i < nvml.count(); ++i) gpus_end.push_back(nvml.snap(i));
        eng.stop();
        for (size_t i = 0; i < gpus_end.size() && i < gpus_start.size(); ++i)
            if (gpus_end[i].power_limit_w != gpus_start[i].power_limit_w)
                std::fprintf(stderr, "basalt-bench: WARNING: the power limit of GPU %zu changed during the run (%.0f W -> %.0f W)\n", i,
                             gpus_start[i].power_limit_w, gpus_end[i].power_limit_w);
        if (failed) return 1;

        // the draft log: one block per request, in order
        if (windows_on) {
            const std::vector<LogBlock> blocks = read_draft_log(draft_log);
            std::vector<size_t> order;   // indexes into results that own a block
            if ((int) blocks.size() == n_requests) {
                for (size_t i = 0; i < results.size(); ++i) order.push_back(i);
            } else if ((int) blocks.size() == n_decode_requests) {
                for (size_t i = 0; i < results.size(); ++i)
                    if (results[i].name != "prefill") order.push_back(i);
            } else {
                std::fprintf(stderr, "basalt-bench: the draft log has %zu request blocks for %d requests: window statistics skipped\n",
                             blocks.size(), n_requests);
            }
            for (size_t k = 0; k < order.size(); ++k) results[order[k]].win = window_stats(blocks[k]);
        }
    }

    // ---- quality session
    if (do_quality) {
        uint32_t n_ctx = 0;
        int32_t n_vocab = 0, n_chunk = 0;
        std::vector<std::vector<int64_t>> chunks;
        quality.t = o.quality_t;
        quality.n_chunks_asked = o.quality_chunks;
        if (!read_kld_base(o.kld_base, n_ctx, n_vocab, n_chunk, chunks, o.quality_chunks, quality.err)) {
            std::fprintf(stderr, "basalt-bench: %s\n", quality.err.c_str());
        } else {
            std::vector<std::string> qa = args;
            set_arg(qa, "--spec", "5");
            set_arg(qa, "--spec-min-p", "0.7");
            set_arg(qa, "--adapt-swaps", "0");
            set_arg(qa, "--peer-adapt-swaps", "0");
            // the gate must run on every pack: without this a pack larger than free RAM is refused
            if (!has_arg(qa, "--resident-experts") && !has_arg(qa, "--mmap-experts")) qa.push_back("--resident-experts");
            const std::string rows = (fs::path(work) / "rows.tsv").string();
            std::remove(rows.c_str());
            set_arg(qa, "--ppl-base", o.kld_base);
            set_arg(qa, "--ppl-out", rows);
            for (const auto& a : qa) quality.args += (quality.args.empty() ? "" : " ") + a;
            Engine eng;
            const auto t0 = Clock::now();
            std::fprintf(stderr, "basalt-bench: starting the engine for the quality gate\n");
            if (!eng.start(engine_bin, qa, (fs::path(work) / "engine-quality.log").string(), quality.err)) {
                std::fprintf(stderr, "basalt-bench: %s\n", quality.err.c_str());
            } else {
                quality.load_s = secs(t0, Clock::now());
                const auto t1 = Clock::now();
                const int64_t first = (int64_t) (n_ctx / 2);
                double pp_ms = 0, dec_ms = 0;
                bool ok = true;
                for (size_t k = 0; k < chunks.size(); ++k) {
                    std::vector<int64_t> ids(chunks[k].begin(), chunks[k].begin() + first + 1);
                    Done d;
                    const std::string keys = fmt(" ppl=%zu ppl_t=%d", k, o.quality_t);
                    if (!eng.request(keys, (long long) n_ctx - 1 - first, ids, d)) {
                        quality.err = fmt("chunk %zu: %s", k, d.err.c_str());
                        ok = false;
                        break;
                    }
                    pp_ms += d.prompt_ms;
                    dec_ms += d.decode_ms;
                    if ((k + 1) % 10 == 0) std::fprintf(stderr, "  quality %zu/%zu chunks, %.0f s\n", k + 1, chunks.size(), secs(t1, Clock::now()));
                }
                quality.run_s = secs(t1, Clock::now());
                eng.stop();
                if (ok) {
                    summarize_rows(rows, quality);
                    quality.prompt_ms_chunk = pp_ms / (double) chunks.size();
                    quality.decode_ms_chunk = dec_ms / (double) chunks.size();
                }
            }
        }
    }


    // ---- report
    const std::string model = arg_value(args, "--model");

    // the power limit of the primary GPU: as set now, or the label the caller gave
    std::string power = "unknown";
    if (!o.power_label.empty()) {
        power = o.power_label;
        if (all_digits(power)) power += " W";
    } else if (!gpus_start.empty() && gpus_start[0].power_limit_w > 0) {
        power = fmt("%.0f W", gpus_start[0].power_limit_w);
#if defined(_WIN32)
    } else if (FILE* pf = popen("nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits 2>nul", "r")) {
#else
    } else if (FILE* pf = popen("nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null", "r")) {
#endif
        char b[64] = {0};
        if (std::fgets(b, sizeof b, pf) != nullptr && std::atof(b) > 0) power = fmt("%.0f W", std::atof(b));
        pclose(pf);
    }
    if (power == "unknown") std::fprintf(stderr, "basalt-bench: WARNING: the power limit could not be read; pass --power-label W\n");

    // one table row per test: values over the repetitions
    struct Row {
        std::string test;
        bool decode = false;
        std::vector<double> ts, acc, dlen;
        bool spec = false;   // some request drafted
    };
    std::vector<Row> rows;
    for (long long depth : o.depths) {
        Row r;
        r.test = fmt("pp%lld", depth);
        for (const Result& x : results)
            if (x.name == "prefill" && x.depth == depth) r.ts.push_back(prefill_engine_ts(x.d));
        if (!r.ts.empty()) rows.push_back(std::move(r));
    }
    for (size_t i = 0; i < 3; ++i) {
        Row r;
        r.test = std::string("tg ") + kDecode[i].name;
        r.decode = true;
        for (const Result& x : results)
            if (x.name == kDecode[i].name) {
                r.ts.push_back(tok_s(x.d));
                r.acc.push_back(x.d.offered > 0 ? 100.0 * (double) x.d.accepted / (double) x.d.offered : 0.0);
                r.dlen.push_back(x.d.offered > 0 ? (double) x.d.offered / window_count(x) : 0.0);
                r.spec = r.spec || x.d.offered > 0;
            }
        if (!r.ts.empty()) rows.push_back(std::move(r));
    }
    auto mean = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / (double) v.size();
    };
    auto sd = [&](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        const double m = mean(v);
        double s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / (double) (v.size() - 1));
    };
    auto ulen = [](const std::string& s) {
        size_t n = 0;
        for (unsigned char c : s)
            if ((c & 0xC0) != 0x80) ++n;
        return n;
    };

    std::string rep;
    if (o.format == "md" && do_perf) {
        rep += fmt("basalt-bench  build %s  power limit %s\n\n", BASALT_BENCH_GIT, power.c_str());
        std::vector<std::array<std::string, 4>> cells;
        cells.push_back({"test", "t/s", "accept %", "draft len"});
        for (const Row& r : rows) {
            std::string ts = o.reps > 1 ? fmt("%.0f \xC2\xB1 %.0f", mean(r.ts), sd(r.ts)) : fmt("%.0f", mean(r.ts));
            cells.push_back({r.test, ts, r.decode && r.spec ? fmt("%.1f", mean(r.acc)) : "-",
                             r.decode && r.spec ? fmt("%.2f", mean(r.dlen)) : "-"});
        }
        size_t w[4] = {0, 0, 0, 0};
        for (const auto& c : cells)
            for (size_t k = 0; k < 4; ++k) w[k] = std::max(w[k], ulen(c[k]));
        auto cell = [&](const std::string& s, size_t k) {
            const std::string pad(w[k] - ulen(s), ' ');
            return k == 0 ? s + pad : pad + s;
        };
        for (size_t ri = 0; ri < cells.size(); ++ri) {
            std::string l = "|";
            for (size_t k = 0; k < 4; ++k) l += " " + cell(cells[ri][k], k) + " |";
            rep += l + "\n";
            if (ri == 0) {
                std::string s = "|";
                for (size_t k = 0; k < 4; ++k) s += " " + std::string(w[k] - (k == 0 ? 0 : 1), '-') + (k == 0 ? "" : ":") + " |";
                rep += s + "\n";
            }
        }
        for (const Result& x : results)
            if (x.checksum_bad && x.run == 1)
                rep += fmt("\nnote: struct produced %lld tokens, the expected count is %lld\n", x.d.generated, kStructTokens);
    }
    if (o.format == "md" && do_quality) {
        rep += fmt("\nquality gate: %d chunks asked, T=%d\n", quality.n_chunks_asked, quality.t);
        if (!quality.ran) {
            rep += "quality gate failed: " + quality.err + "\n";
        } else {
            rep += fmt("chunks %lld  rows %lld\n", quality.chunks, quality.rows);
            rep += fmt("PPL %.4f +/- %.4f   base PPL %.4f   ln(PPL/base) %.5f\n", quality.ppl, quality.ppl_err, quality.base_ppl, quality.ln_ratio);
            rep += fmt("KLD mean %.5f  median %.5f  p99 %.4f  max %.4f\n", quality.kld_mean, quality.kld_median, quality.kld_p99, quality.kld_max);
            rep += fmt("same top-1 %.2f%%   mean |dlogp| of the base top-1 %.5f\n", quality.top1_pct, quality.dlogp);
        }
    }

    if (o.format == "json") {
        JW j;
        j.begin_obj();
        j.key("engine_version"); j.str(BASALT_VERSION);
        j.key("commit"); j.str(BASALT_BENCH_GIT);
        j.key("build_type"); j.str(BASALT_BENCH_BUILD_TYPE);
        j.key("cuda_toolkit"); j.str(BASALT_BENCH_CUDA);
        j.key("power_limit"); j.str(power);
        j.key("repetitions"); j.integer(o.reps);
        j.key("driver"); j.str(nvml.driver());
        j.key("cuda_driver"); j.str(nvml.cuda_driver());
        j.key("gpus");
        j.begin_arr();
        for (const auto& g : gpus_start) {
            j.begin_obj();
            j.key("name"); j.str(g.name);
            j.key("power_limit_w"); j.num(g.power_limit_w);
            j.key("power_max_w"); j.num(g.power_max_w);
            j.key("sm_mhz"); j.integer(g.sm_mhz);
            j.key("mem_mhz"); j.integer(g.mem_mhz);
            j.end_obj();
        }
        j.end_arr();
        j.key("model"); j.str(model);
        j.key("engine_info");
        j.begin_obj();
        for (const auto& kv : info) { j.key(kv.first); j.str(kv.second); }
        j.end_obj();
        j.key("peer_tier"); j.str(peer_line);
        j.key("engine_args"); j.str(engine_args_text);
        j.key("summary");
        j.begin_arr();
        for (const Row& r : rows) {
            j.begin_obj();
            j.key("test"); j.str(r.test);
            j.key("tok_s_mean"); j.num(mean(r.ts));
            j.key("tok_s_stddev"); j.num(sd(r.ts));
            if (r.decode) {
                j.key("accept_pct_mean"); j.num(mean(r.acc));
                j.key("draft_len_mean"); j.num(mean(r.dlen));
            }
            j.end_obj();
        }
        j.end_arr();
        j.key("results");
        j.begin_arr();
        for (const Result& r : results) {
            j.begin_obj();
            j.key("repetition"); j.integer(r.run);
            j.key("test"); j.str(r.name == "prefill" ? fmt("pp%lld", r.depth) : "tg " + r.name);
            if (r.name == "prefill") {
                j.key("depth_tokens"); j.integer(r.depth);
                j.key("prompt_tokens"); j.integer(r.d.prompt);
                j.key("reused_tokens"); j.integer(r.d.reused);
                j.key("prompt_ms"); j.num(r.d.prompt_ms);
                j.key("wall_s"); j.num(r.d.wall_s);
                j.key("first_token_s"); j.num(r.d.first_token_s);
                j.key("wall_tok_s"); j.num(prefill_wall_ts(r.d));
                j.key("engine_tok_s"); j.num(prefill_engine_ts(r.d));
            } else {
                j.key("tok_s"); j.num(tok_s(r.d));
                j.key("tokens"); j.integer(r.d.generated);
                j.key("decode_ms"); j.num(r.d.decode_ms);
                j.key("drafted"); j.integer(r.d.offered);
                j.key("accepted"); j.integer(r.d.accepted);
                j.key("acceptance_pct"); j.num(r.d.offered > 0 ? 100.0 * (double) r.d.accepted / (double) r.d.offered : 0.0);
                j.key("tokens_per_pass"); j.num(r.d.generated > r.d.accepted ? (double) r.d.generated / (double) (r.d.generated - r.d.accepted) : 0.0);
                j.key("mean_draft_len"); j.num(r.d.offered > 0 ? (double) r.d.offered / window_count(r) : 0.0);
                j.key("finish"); j.str(r.d.finish);
                if (r.name == "struct") { j.key("checksum_ok"); j.boolean(!r.checksum_bad); }
                if (r.win.present) {
                    j.key("windows"); j.integer(r.win.windows);
                    j.key("mean_t"); j.num(r.win.mean_t);
                    j.key("tokens_per_window"); j.num(r.win.tokens_per_window);
                    j.key("ms_per_window"); j.num(r.d.decode_ms / (double) r.win.windows);
                    j.key("passes_per_s"); j.num(1000.0 * (double) r.win.windows / r.d.decode_ms);
                    j.key("mix_verify_ms"); j.num(r.win.mix);
                    j.key("verify_ms_by_t");
                    j.begin_obj();
                    for (size_t t = 1; t < r.win.verify_ms.size(); ++t)
                        if (r.win.verify_ms[t] > 0) { j.key(std::to_string(t)); j.num(r.win.verify_ms[t]); }
                    j.end_obj();
                    j.key("window_share_by_t");
                    j.begin_obj();
                    for (size_t t = 1; t < r.win.share.size(); ++t)
                        if (r.win.share[t] > 0) { j.key(std::to_string(t)); j.num(r.win.share[t]); }
                    j.end_obj();
                }
            }
            j.key("sm_mhz_after"); j.integer(r.sm_mhz);
            j.key("mem_mhz_after"); j.integer(r.mem_mhz);
            j.end_obj();
        }
        j.end_arr();
        if (do_quality) {
            j.key("quality");
            j.begin_obj();
            j.key("ran"); j.boolean(quality.ran);
            j.key("error"); j.str(quality.err);
            j.key("chunks"); j.integer(quality.chunks);
            j.key("rows"); j.integer(quality.rows);
            j.key("t"); j.integer(quality.t);
            j.key("ppl"); j.num(quality.ppl);
            j.key("ppl_err"); j.num(quality.ppl_err);
            j.key("base_ppl"); j.num(quality.base_ppl);
            j.key("kld_mean"); j.num(quality.kld_mean);
            j.key("kld_median"); j.num(quality.kld_median);
            j.key("kld_p99"); j.num(quality.kld_p99);
            j.key("kld_max"); j.num(quality.kld_max);
            j.key("top1_pct"); j.num(quality.top1_pct);
            j.key("prompt_ms_per_chunk"); j.num(quality.prompt_ms_chunk);
            j.key("decode_ms_per_chunk"); j.num(quality.decode_ms_chunk);
            j.key("engine_args"); j.str(quality.args);
            j.end_obj();
        }
        j.end_obj();
        rep = j.text() + "\n";
    }
    std::fputs(rep.c_str(), stdout);
    return do_quality && !quality.ran ? 1 : 0;
}
