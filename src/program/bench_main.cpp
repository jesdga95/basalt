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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/program/engine_session.hpp"

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
#include "basalt/log.hpp"
#endif
#ifndef BASALT_BENCH_GIT
#define BASALT_BENCH_GIT "unknown"
#endif
#ifndef BASALT_BENCH_DIRTY
#define BASALT_BENCH_DIRTY 0
#endif
#ifndef BASALT_BENCH_BUILD_TYPE
#define BASALT_BENCH_BUILD_TYPE "unknown"
#endif
#ifndef BASALT_BENCH_CUDA
#define BASALT_BENCH_CUDA "unknown"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace basalt::program;

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

using basalt::program::json_quote;

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
        double power_enforced_w = -1;   // the limit the driver applies now (the management limit when it cannot say)
        double vram_gib = -1;
        int sm_mhz = -1, mem_mhz = -1;
    };

    bool open() {
        lib_ = basalt::platform::DynLib::open_nvml();
        if (!lib_) return false;
        init_ = lib_.sym<int (*)()>("nvmlInit_v2");
        shut_ = lib_.sym<int (*)()>("nvmlShutdown");
        drv_ = lib_.sym<int (*)(char*, unsigned)>("nvmlSystemGetDriverVersion");
        cuda_ = lib_.sym<int (*)(int*)>("nvmlSystemGetCudaDriverVersion");
        count_ = lib_.sym<int (*)(unsigned*)>("nvmlDeviceGetCount_v2");
        handle_ = lib_.sym<int (*)(unsigned, void**)>("nvmlDeviceGetHandleByIndex_v2");
        name_ = lib_.sym<int (*)(void*, char*, unsigned)>("nvmlDeviceGetName");
        plim_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetPowerManagementLimit");
        pcon_ = lib_.sym<int (*)(void*, unsigned*, unsigned*)>("nvmlDeviceGetPowerManagementLimitConstraints");
        clk_ = lib_.sym<int (*)(void*, int, unsigned*)>("nvmlDeviceGetClockInfo");
        enf_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetEnforcedPowerLimit");
        mem_ = lib_.sym<int (*)(void*, MemInfo*)>("nvmlDeviceGetMemoryInfo");
        bus_ = lib_.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
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
    Snap snap(size_t i) const { return i < dev_.size() ? snap_of(dev_[i]) : Snap{}; }
    /// The card at a PCI bus id (as the CUDA runtime prints it); false when NVML does not know it.
    bool snap_bus(const std::string& id, Snap& out) const {
        void* h = nullptr;
        if (!ok_ || !bus_ || bus_(id.c_str(), &h) != 0) return false;
        out = snap_of(h);
        return true;
    }

private:
    struct MemInfo {
        unsigned long long total, free, used;
    };
    Snap snap_of(void* h) const {
        Snap s;
        if (!ok_ || h == nullptr) return s;
        char nm[128] = {0};
        if (name_ && name_(h, nm, sizeof nm) == 0) s.name = nm;
        unsigned mw = 0, lo = 0, hi = 0;
        if (plim_ && plim_(h, &mw) == 0) s.power_limit_w = mw / 1000.0;
        if (pcon_ && pcon_(h, &lo, &hi) == 0) s.power_max_w = hi / 1000.0;
        s.power_enforced_w = s.power_limit_w;
        if (enf_ && enf_(h, &mw) == 0) s.power_enforced_w = mw / 1000.0;
        MemInfo mi{};
        if (mem_ && mem_(h, &mi) == 0) s.vram_gib = (double) mi.total / (double) (1ull << 30);
        unsigned c = 0;
        if (clk_ && clk_(h, 1, &c) == 0) s.sm_mhz = (int) c;    // NVML_CLOCK_SM
        if (clk_ && clk_(h, 2, &c) == 0) s.mem_mhz = (int) c;   // NVML_CLOCK_MEM
        return s;
    }
    basalt::platform::DynLib lib_;
    bool ok_ = false;
    std::vector<void*> dev_;
    int (*enf_)(void*, unsigned*) = nullptr;
    int (*mem_)(void*, MemInfo*) = nullptr;
    int (*bus_)(const char*, void**) = nullptr;
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

// ---------------------------------------------------------------------------------------------- options

struct Options {
    std::string engine, model, fixtures = BASALT_BENCH_FIXTURES, format = "md", logs, kld_base, power_label;
    std::vector<std::pair<std::string, std::string>> sets;
    std::vector<std::string> passthrough;
    std::string spec;                    // --spec N: the engine argument of the same name
    std::vector<std::string> decode;     // -t: conv, struct, prose
    std::vector<long long> depths;       // -p: prefill depths in tokens
    bool sel_decode = false, sel_prefill = false;
    int reps = 1;
    bool quality = false, no_windows = false, print_command = false;
    bool community = false;
    std::string submitter;
    std::string community_file;          // --community-file: the table the row is added to
    int quality_chunks = 20, quality_t = 3;
    unsigned long long seed = 0;         // --seed N: the sampling seed of every decode request (0: the engine seeds from the clock)
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
        "                     through the decode path, adaptation off, --spec 5. Alone it runs\n"
        "                     only the gate; with -p / -t it runs after them\n"
        "  --kld-base FILE    llama.cpp KL-divergence base for the quality gate\n"
        "  --quality-t T      tokens per verify window in the quality gate (default 3)\n"
        "  --seed N           sampling seed (N >= 1) sent with every decode request, so the sampled fixtures\n"
        "                     (conv, prose) give the same text on every run; without it each request is seeded\n"
        "                     from the clock\n"
        "\n"
        "community row\n"
        "  --community        the fixed preset \"community-1\" for the public benchmark table: the three decode\n"
        "                     fixtures, a 16384-token prefill and a 65536-token prefill, once each, with int8 KV cache\n"
        "                     and the engine's own defaults otherwise (no warm-up, no repetition, drafting at its\n"
        "                     default). The decode fixtures and each prefill run in their own fresh engine process, so\n"
        "                     every number is cold. Prints the table, adds the row to benchmarks.json and prints it. prefill_16k_tps and\n"
        "                     prefill_64k_tps = engine tokens/s of those requests; decode_tps = tokens / decode time\n"
        "                     summed over conv, struct and prose (token-weighted). Needs NVML for the GPU names, VRAM and enforced power limits. Only\n"
        "                     --model, --submitter, --community-file, --fixtures, --logs and --print-command may be given with it:\n"
        "                     any other option or engine flag would make rows incomparable and is refused\n"
        "  --submitter NAME   the name shown in the table, written into the row (required with --community)\n"
        "  --community-file PATH  the benchmarks.json the row is added to (default: docs/community/benchmarks.json of the\n"
        "                     checkout this build came from); when it cannot be written the row is printed instead\n"
        "\n"
        "engine\n"
        "  --model FILE       a .basalt model file; --peer-device 1 is added when a second GPU is visible. Every\n"
        "                     other --flag is an engine flag, passed on as given (same names as `basalt`; the engine\n"
        "                     rejects a bad one at startup); --print-command shows the command and exits\n"
        "  --engine BIN       engine binary (default: the one built with this program)\n"
        "  --set \"F V ...\"    FLAG VALUE pairs that replace or add engine arguments (repeatable)\n"
        "  --spec N           speculation window for the engine (--spec N, e.g. 7; also --spec after --); without it\n"
        "                     the run has no speculation (--spec 0): a clean baseline\n"
        "  -- ARGS...         the same engine arguments, after --: replacing and adding flags\n"
        "  built in (only what the measurement needs, each replaced by giving the flag): --spec 0, --max-context sized\n"
        "  to the fixtures\n"
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
        if (const auto help = basalt::platform::capture_output({bin, "--help"}, true)) std::fputs(help->c_str(), stdout);
    }
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
        else if (a == "--community") o.community = true;
        else if (a == "--submitter") o.submitter = next("--submitter");
        else if (a == "--community-file") o.community_file = next("--community-file");
        else if (a == "--fixtures") o.fixtures = next("--fixtures");
        else if (a == "-o") o.format = next("-o");
        else if (a == "--power-label") o.power_label = next("--power-label");
        else if (a == "--logs") o.logs = next("--logs");
        else if (a == "--kld-base") o.kld_base = next("--kld-base");
        else if (a == "--no-windows") o.no_windows = true;
        else if (a == "-r") o.reps = std::atoi(next("-r").c_str());
        else if (a == "--seed") {
            const std::string v = next("--seed");
            if (!all_digits(v.c_str()) || std::strtoull(v.c_str(), nullptr, 10) == 0) { err = "--seed needs an integer >= 1"; return false; }
            o.seed = std::strtoull(v.c_str(), nullptr, 10);
        } else if (a == "--quality-t") o.quality_t = std::atoi(next("--quality-t").c_str());
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
    if (o.community) {
        const char* bad = o.sel_decode || o.sel_prefill ? "-t / -p" : o.reps != 1 ? "-r" : o.quality ? "--quality"
                        : !o.spec.empty() ? "--spec" : !o.sets.empty() ? "--set" : !o.engine.empty() ? "--engine"
                        : !o.power_label.empty() ? "--power-label" : o.seed ? "--seed" : o.no_windows ? "--no-windows"
                        : o.format != "md" ? "-o" : !o.passthrough.empty() ? o.passthrough[0].c_str() : nullptr;
        if (bad != nullptr) {
            err = std::string("--community runs the fixed preset: ") + bad + " would make the row incomparable";
            return false;
        }
        if (o.model.empty()) { err = "--community needs --model FILE"; return false; }
        std::string who = o.submitter;
        for (char& c : who) c = (char) std::toupper((unsigned char) c);
        if (who.find_first_not_of(" \t") == std::string::npos || who == "YOUR_NAME") {
            err = "--community needs --submitter NAME, the name shown in the table";
            return false;
        }
    }
    if (!o.community_file.empty() && !o.community) { err = "--community-file needs --community"; return false; }
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
constexpr long long kCommunityPrefillShort = 16384, kCommunityPrefillLong = 65536;   // the community preset's depths
constexpr long long kPrefillBaseDecode = 32;

std::string sampling_keys(double temperature, unsigned long long seed) {
    std::string k;
    if (temperature > 0.0) k += fmt(" temperature=%.17g", temperature);
    k += " top_p=0.95 top_k=20";
    if (seed) k += " seed=" + std::to_string(seed);
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
        basalt::log::error("basalt-bench", "%s", err.c_str());
        return 2;
    }
    basalt::platform::ignore_broken_pipe();   // a closed pipe surfaces as a failed write
    basalt::platform::install_child_stop_signals();

    if (o.community) o.no_windows = true;   // the draft log is analysis, not part of the preset

    // what to run
    const bool quality_only = o.quality && !o.sel_decode && !o.sel_prefill;
    if (!o.sel_decode && !o.sel_prefill && !quality_only) {   // one of -t / -p runs only what it names
        o.decode = {"conv", "struct", "prose"};
        o.depths = {kPrefillBase};
    }
    if (o.community) o.depths = {kCommunityPrefillShort, kCommunityPrefillLong};
    auto wants = [&](const std::string& s) { return std::find(o.decode.begin(), o.decode.end(), s) != o.decode.end(); };
    const bool do_quality = o.quality;
    const bool do_decode = !o.decode.empty();
    const bool do_prefill = !o.depths.empty();
    const bool do_perf = do_decode || do_prefill;
    if (do_quality && o.kld_base.empty()) {
        basalt::log::error("basalt-bench", "the quality gate needs --kld-base FILE");
        return 2;
    }

    // engine command from the model and the overrides
    std::string engine_bin = o.engine;
    std::vector<std::string> args;
    if (!o.model.empty()) {
        args = {"--model", fs::absolute(o.model).string()};
        // A second GPU can hold an expert tier only where a process is given GPU-to-GPU access between its
        // devices; the engine refuses --peer-device without it, so the bench does not offer one.
        if (basalt::platform::gpu_peer_access() && !has_arg(args, "--peer-device") && visible_gpus() > 1)
            set_arg(args, "--peer-device", "1");
    }
    if (engine_bin.empty()) engine_bin = BASALT_BENCH_ENGINE;
    if (o.engine.empty() && std::string(BASALT_BENCH_ENGINE).size() > 0 && fs::exists(BASALT_BENCH_ENGINE)) engine_bin = BASALT_BENCH_ENGINE;
    if (engine_bin.empty() || !fs::exists(engine_bin)) {
        basalt::log::error("basalt-bench", "the engine binary %s does not exist (--engine BIN)", engine_bin.c_str());
        return 2;
    }
    for (const auto& kv : o.sets) set_arg(args, kv.first, kv.second);
    if (!o.spec.empty()) set_arg(args, "--spec", o.spec);
    if (o.model.empty()) for (const auto& a : o.passthrough) args.push_back(a);
    else merge_args(args, o.passthrough);
    if (!o.community && !has_arg(args, "--spec")) set_arg(args, "--spec", "0");

    // fixtures, read before anything is started
    std::vector<std::vector<int64_t>> decode_ids(3);
    std::vector<std::vector<int64_t>> prefill_parts;
    if (do_decode)
        for (size_t i = 0; i < 3; ++i) {
            if (!wants(kDecode[i].name)) continue;
            std::vector<std::vector<int64_t>> lines;
            if (!read_id_lines((fs::path(o.fixtures) / kDecode[i].file).string(), lines, err) || lines.empty()) {
                basalt::log::error("basalt-bench", "%s", err.empty() ? (std::string(kDecode[i].file) + " is empty").c_str() : err.c_str());
                return 2;
            }
            for (auto& l : lines) decode_ids[i].insert(decode_ids[i].end(), l.begin(), l.end());
        }
    if (do_prefill) {
        if (!read_id_lines((fs::path(o.fixtures) / "prefill.ids").string(), prefill_parts, err) || prefill_parts.size() != 3) {
            basalt::log::error("basalt-bench", "%s", err.empty() ? "prefill.ids needs three lines (before, body, after)" : err.c_str());
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
    if (o.community) set_arg(args, "--kv", "int8");   // the cache format users run
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
        work = basalt::platform::temp_dir() + "/basalt-bench-" + std::to_string((long long) basalt::platform::process_id());
        temp_work = true;
    }
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { basalt::log::error("basalt-bench", "cannot create %s", work.c_str()); return 2; }
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
    if (!have_nvml) basalt::log::warn("basalt-bench", "NVML is unavailable: power limits and clocks are NOT recorded");
    if (o.community && (!have_nvml || nvml.count() == 0)) {
        basalt::log::error("basalt-bench", "--community needs NVML (the NVIDIA driver's management library) to read the GPU names, "
                             "VRAM and enforced power limits for the row; it could not be loaded");
        return 1;
    }
    if (o.community && std::string(BASALT_BENCH_GIT).find_first_not_of("0123456789abcdef") != std::string::npos) {
        basalt::log::error("basalt-bench", "--community needs a build from a git checkout: the row records the commit");
        return 1;
    }
    if (o.community && BASALT_BENCH_DIRTY)
        basalt::log::warn("basalt-bench", "tracked files differ from commit %s; the row names that commit", BASALT_BENCH_GIT);
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
        // Normally one engine process runs everything. The community preset runs the decode fixtures in one
        // fresh process and each prefill depth in its own, so every measurement is the first of its kind.
        struct Session {
            std::vector<std::string> decode;
            std::vector<long long> depths;
        };
        std::vector<Session> sessions;
        if (o.community) {
            sessions.push_back({o.decode, {}});
            for (long long d : o.depths) sessions.push_back({{}, {d}});
        } else {
            sessions.push_back({o.decode, o.depths});
        }
        std::mt19937_64 rng((uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() ^ ((uint64_t) basalt::platform::process_id() << 32));
        int n_requests = 0, n_decode_requests = 0;
        bool failed = false;
        for (size_t si = 0; si < sessions.size() && !failed; ++si) {
        const Session& sess = sessions[si];
        const std::string log_path = (fs::path(work) / (si == 0 ? "engine.log" : fmt("engine-%zu.log", si))).string();
        Engine eng;
        basalt::log::info("basalt-bench", "starting %s (log: %s)", engine_bin.c_str(), log_path.c_str());
        if (!eng.start(engine_bin, perf_args, log_path, err)) {
            basalt::log::error("basalt-bench", "%s", err.c_str());
            return 1;
        }
        if (o.community) {
            // every expert must sit in VRAM, the peer GPU or RAM: reads from the file would make the row incomparable
            const char* k = "file_experts_bytes";
            const double spill = eng.info.count(k) ? std::atof(eng.info[k].c_str()) / 1073741824.0 : -1.0;
            if (spill != 0.0) {
                eng.stop();
                if (spill < 0)
                    basalt::log::warn("basalt-bench", "the engine did not report where the experts live; cannot check for SSD spill");
                else
                    basalt::log::error("basalt-bench", "this run would read %.1f GiB of experts from the SSD. Community runs need every "
                                 "expert in VRAM, the peer GPU or RAM: free roughly %.0f GiB more RAM (close other programs) "
                                 "and run again. No row was written.", spill, std::ceil(spill));
                return 1;
            }
        }
        if (si == 0) {
            info = eng.info;
            std::ifstream lg(log_path);
            std::string l;
            while (std::getline(lg, l))
                if (l.find("peer GPU") != std::string::npos && l.find("experts") != std::string::npos) peer_line = l;
        }
        for (int run = 1; run <= o.reps && !failed; ++run) {
            for (size_t i = 0; i < 3 && !failed; ++i) {
                if (std::find(sess.decode.begin(), sess.decode.end(), kDecode[i].name) == sess.decode.end()) continue;
                Result r;
                r.name = kDecode[i].name;
                r.run = run;
                if (!eng.request(sampling_keys(kDecode[i].temperature, o.seed), kDecode[i].max_new, decode_ids[i], r.d)) {
                    basalt::log::error("basalt-bench", "%s (run %d): %s", r.name.c_str(), run, r.d.err.c_str());
                    failed = true;
                    break;
                }
                const Nvml::Snap s = nvml.snap(0);
                r.sm_mhz = s.sm_mhz;
                r.mem_mhz = s.mem_mhz;
                r.checksum_bad = r.name == "struct" && r.d.generated != kStructTokens;
                ++n_requests;
                ++n_decode_requests;
                basalt::log::raw("  run %d %-7s %.1f tok/s, %lld tokens\n", run, r.name.c_str(), tok_s(r.d), r.d.generated);
                results.push_back(std::move(r));
            }
            for (size_t pi = 0; pi < sess.depths.size() && !failed; ++pi) {
                const long long depth = sess.depths[pi];
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
                    basalt::log::error("basalt-bench", "pp%lld (run %d): %s", depth, run, r.d.err.c_str());
                    failed = true;
                    break;
                }
                const Nvml::Snap s = nvml.snap(0);
                r.sm_mhz = s.sm_mhz;
                r.mem_mhz = s.mem_mhz;
                ++n_requests;
                basalt::log::raw("  run %d pp%lld %lld tokens, %.0f t/s engine, %.0f t/s wall\n", run, depth, r.d.prompt,
                             prefill_engine_ts(r.d), prefill_wall_ts(r.d));
                results.push_back(std::move(r));
            }
        }
        std::vector<Nvml::Snap> gpus_end;
        for (size_t i = 0; i < nvml.count(); ++i) gpus_end.push_back(nvml.snap(i));
        eng.stop();
        for (size_t i = 0; i < gpus_end.size() && i < gpus_start.size(); ++i)
            if (gpus_end[i].power_limit_w != gpus_start[i].power_limit_w)
                basalt::log::warn("basalt-bench", "the power limit of GPU %zu changed during the run (%.0f W -> %.0f W)", i,
                             gpus_start[i].power_limit_w, gpus_end[i].power_limit_w);
        }
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
                basalt::log::warn("basalt-bench", "the draft log has %zu request blocks for %d requests: window statistics skipped",
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
            basalt::log::error("basalt-bench", "%s", quality.err.c_str());
        } else {
            std::vector<std::string> qa = args;
            set_arg(qa, "--spec", "5");
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
            basalt::log::info("basalt-bench", "starting the engine for the quality gate");
            if (!eng.start(engine_bin, qa, (fs::path(work) / "engine-quality.log").string(), quality.err)) {
                basalt::log::error("basalt-bench", "%s", quality.err.c_str());
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
                    if ((k + 1) % 10 == 0) basalt::log::raw("  quality %zu/%zu chunks, %.0f s\n", k + 1, chunks.size(), secs(t1, Clock::now()));
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

    // the power limit of the primary GPU (the first card the engine reports, by PCI bus id): as set now, or the label the caller gave
    std::string primary_bus;
    if (info.count("gpus")) primary_bus = info["gpus"].substr(0, info["gpus"].find(','));
    Nvml::Snap primary;
    const bool have_primary = !primary_bus.empty() && nvml.snap_bus(primary_bus, primary);
    std::string power = "unknown";
    if (!o.power_label.empty()) {
        power = o.power_label;
        if (all_digits(power)) power += " W";
    } else if (have_primary && primary.power_limit_w > 0) {
        power = fmt("%.0f W", primary.power_limit_w);
    } else if (primary_bus.empty() && !gpus_start.empty() && gpus_start[0].power_limit_w > 0) {
        power = fmt("%.0f W", gpus_start[0].power_limit_w);
    } else if (const auto smi = basalt::platform::capture_output(
                   {"nvidia-smi", "-i", primary_bus.empty() ? "0" : primary_bus, "--query-gpu=power.limit", "--format=csv,noheader,nounits"}, false);
               smi && std::atof(smi->c_str()) > 0) {
        power = fmt("%.0f W", std::atof(smi->c_str()));
    }
    if (power == "unknown") basalt::log::warn("basalt-bench", "the power limit could not be read; pass --power-label W");

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
    std::string community_row;
    basalt::program::CommunityRow crow;
    if (o.community) {
        // one aggregate per direction over the single cold pass: tokens over time summed across requests
        double pre_tps[2] = {0, 0}, pre_s[2] = {0, 0}, dec_tok = 0, dec_ms = 0;
        for (const Result& x : results) {
            if (x.name == "prefill") {
                const int k = x.depth == kCommunityPrefillLong ? 1 : 0;
                pre_tps[k] = prefill_engine_ts(x.d);
                pre_s[k] = x.d.prompt_ms / 1000.0;
            } else {
                dec_tok += (double) x.d.generated;
                dec_ms += x.d.decode_ms;
            }
        }
        std::string gpu_err;
        std::istringstream ids(info.count("gpus") ? info["gpus"] : "");
        for (std::string id; std::getline(ids, id, ',');) {
            Nvml::Snap g;
            if (!nvml.snap_bus(id, g) || g.name.empty() || g.vram_gib <= 0 || g.power_enforced_w <= 0) {
                gpu_err = "NVML could not read the GPU at " + id;
                break;
            }
            for (const char* v : {"NVIDIA ", "GeForce "})
                if (g.name.rfind(v, 0) == 0) g.name.erase(0, std::strlen(v));
            crow.gpus.push_back({g.name, g.vram_gib, g.power_enforced_w});
        }
        const uint64_t ram_b = basalt::platform::total_physical_memory();
        if (gpu_err.empty() && crow.gpus.empty()) gpu_err = "the engine did not report which GPUs it used";
        if (gpu_err.empty() && (pre_tps[0] <= 0 || pre_tps[1] <= 0 || dec_ms <= 0)) gpu_err = "a request reported no time";
        if (gpu_err.empty() && ram_b == 0) gpu_err = "the system RAM size could not be read";
        if (!gpu_err.empty()) {
            basalt::log::error("basalt-bench", "cannot write the community row: %s", gpu_err.c_str());
            std::fputs(rep.c_str(), stdout);
            return 1;
        }
        std::time_t now = std::time(nullptr);
        char day[16] = {};
        std::strftime(day, sizeof day, "%Y-%m-%d", std::localtime(&now));
        crow.submitter = o.submitter;
        crow.date = day;
        crow.commit = BASALT_BENCH_GIT;
        crow.preset = "community-1";
        crow.ram_gib = (double) ram_b / (double) (1ull << 30);
        crow.os = basalt::platform::host_os_name();
        crow.quant = info.count("quant") ? info["quant"] : "unknown";
        crow.prefill_16k_tps = pre_tps[0];
        crow.prefill_64k_tps = pre_tps[1];
        crow.decode_tps = dec_tok / (dec_ms / 1000.0);
        community_row = basalt::program::community_row_text(crow);
        rep += fmt("\ncommunity-1: prefill 16K %.1f t/s (%.1f s), prefill 64K %.1f t/s (%.1f s), decode %.1f t/s (%.0f tokens in %.1f s), "
                   "each measured cold\n",
                   pre_tps[0], pre_s[0], pre_tps[1], pre_s[1], dec_tok / (dec_ms / 1000.0), dec_tok, dec_ms / 1000.0);
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
    if (!community_row.empty()) {
        const std::string path = o.community_file.empty() ? std::string(BASALT_BENCH_SOURCE_DIR) + "/docs/community/benchmarks.json"
                                                          : o.community_file;
        std::string why;
        std::printf("\nCOMMUNITY ROW:\n%s\n", community_row.c_str());
        std::fflush(stdout);
        if (basalt::program::community_add_row_file(path, crow, why))
            basalt::log::info("basalt-bench", "the row was added to %s; review it, commit it and open a pull request (docs/community.md)",
                              path.c_str());
        else
            basalt::log::warn("basalt-bench", "the row was not added to %s (%s); paste it at the end of the runs list of docs/community/benchmarks.json, "
                              "run python3 tools/check_benchmarks.py --fix on it and open a pull request (docs/community.md)",
                              path.c_str(), why.c_str());
    }
    return do_quality && !quality.ran ? 1 : 0;
}
