// basalt-probe: Basalt's built-in tuner, first stage. It starts the engine as a child process (like basalt-bench),
// asks it for the machine's rates (PROBE rates), measures cold decode speed on the three standard fixtures at
// the engine's context, and prints a short report with the values it suggests. Manual only: it prints and saves
// nothing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/engine_session.hpp"
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/probe_derive.hpp"
#include "basalt/program/probe_report.hpp"
#include "basalt/log.hpp"

#ifndef BASALT_BENCH_ENGINE
#define BASALT_BENCH_ENGINE ""
#endif
#ifndef BASALT_BENCH_FIXTURES
#define BASALT_BENCH_FIXTURES "data/bench"
#endif

namespace fs = std::filesystem;
using namespace basalt::program;
using Clock = std::chrono::steady_clock;

namespace {

// Tokens generated per measured request. Cold speed depends on how long the engine has had to adapt, so this is part
// of what "cold" means; it is also what keeps the whole run short.
constexpr long long kTokens = 400;

struct Fixture {
    const char* name;
    const char* file;
};
const Fixture kFixtures[] = {{"conv", "conv.ids"}, {"struct", "struct.ids"}, {"prose", "prose.ids"}};

void usage() {
    std::printf(
        "basalt-probe - measures this machine and prints its rates, decode speeds and suggested values\n"
        "\n"
        "usage: basalt-probe --model FILE.basalt [-- ENGINE ARGS...]\n"
        "\n"
        "  --model FILE          a .basalt model file\n"
        "  --engine BIN          engine binary (default: the one built with this program)\n"
        "  --fixtures DIR        fixture ids (default %s)\n"
        "  --logs DIR            keep the engine log there\n"
        "  --gpu A,B,..|auto     the GPUs, as for the server and the engine (default: GPU 0 alone)\n"
        "  -- ARGS...            engine flags, as for basalt-bench (--kv, --spec, ...)\n"
        "\n"
        "It runs conv, struct and prose once each (cold: the first request after load counts), %lld tokens each,\n"
        "greedy, then the adaptive tier's settings, each from the start placement. Nothing is saved; a\n"
        "--expert-profile-save file is compared, not written.\n",
        BASALT_BENCH_FIXTURES, kTokens);
}

struct ProbeOptions {
    std::string model, engine, fixtures = BASALT_BENCH_FIXTURES, logs;
    std::vector<std::string> passthrough;
};

bool parse_args(int argc, char** argv, ProbeOptions& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { err = std::string(what) + " needs a value"; return ""; }
            return argv[++i];
        };
        if (a == "--") {
            for (++i; i < argc; ++i) o.passthrough.push_back(argv[i]);
        } else if (a == "--help" || a == "-h") {
            usage();
            std::exit(0);
        } else if (a == "--model") o.model = next("--model");
        else if (a == "--engine") o.engine = next("--engine");
        else if (a == "--fixtures") o.fixtures = next("--fixtures");
        else if (a == "--logs") o.logs = next("--logs");
        else if (a == "--gpu") o.passthrough.insert(o.passthrough.end(), {a, next("--gpu")});
        else err = "unknown argument " + a + " (engine flags go after --; see --help)";
        if (!err.empty()) return false;
    }
    if (o.model.empty()) { err = "give --model FILE"; return false; }
    return true;
}

// "peer GPU 1: 5704 experts, ..." from the engine log; -1 when there is none.
long long peer_experts_from_log(const std::string& log_path) {
    std::ifstream f(log_path);
    std::string l;
    long long n = -1;
    while (std::getline(f, l)) {
        const size_t p = l.find("peer GPU ");
        if (p == std::string::npos) continue;
        const size_t c = l.find(": ", p);
        if (c == std::string::npos) continue;
        char* e = nullptr;
        const long long v = std::strtoll(l.c_str() + c + 2, &e, 10);
        if (e != l.c_str() + c + 2 && std::string(e).rfind(" experts", 0) == 0) n = v;
    }
    return n;
}

// The adaptive tier's settings, each run from the tiers' start placement (PROBE adapt) on one sequence: prose, then
// struct (a cold start, then a topic switch). The first request is as long as a served reply, so the start's ramp
// weighs in as it does there. Fills `t.tps`; a setting left at 0 is the engine's and is read back.
constexpr long long kAdaptTokens[2] = {1200, 400};
constexpr double kAdaptBandFloor = 0.03;   // two passes of the same setting on this sequence spread up to this much
bool adapt_trial(Engine& eng, const std::vector<std::vector<int64_t>>& fixtures, const std::string& saved, AdaptTrial& t,
                 std::string& err) {
    std::string cmd = "PROBE adapt swaps=" + std::to_string(t.swaps);
    char b[64];
    if (t.every > 0) cmd += " every=" + std::to_string(t.every);
    if (t.decay > 0) {
        std::snprintf(b, sizeof b, " decay=%.4f", t.decay);
        cmd += b;
    }
    if (t.saved) cmd += " profile=" + saved;
    std::vector<std::string> blk;
    if (!eng.command(cmd, "PROBE end", blk, err)) return false;
    std::string ms = "?", moved = "?", peer_moved = "0", kept = "0";
    std::istringstream is(blk.back());
    for (std::string kv; is >> kv;) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "every") t.every = std::atoi(v.c_str());
        else if (k == "decay") t.decay = std::atof(v.c_str());
        else if (k == "ms") ms = v;
        else if (k == "moved") moved = v;
        else if (k == "peer_moved") peer_moved = v;
        else if (k == "kept") kept = v;
    }
    long long gen = 0;
    double dms = 0, rate[2] = {0, 0};
    const size_t order[2] = {2, 1};   // prose, struct
    for (int k = 0; k < 2; ++k) {
        Done d;
        if (!eng.request(" temperature=0 top_p=0.95 top_k=20", kAdaptTokens[k], fixtures[order[k]], d) || d.generated <= 0) {
            err = d.err.empty() ? "no tokens generated" : d.err;
            return false;
        }
        gen += d.generated;
        dms += d.decode_ms;
        rate[k] = d.decode_ms > 0 ? (double) d.generated * 1000.0 / d.decode_ms : 0.0;
    }
    t.tps = dms > 0 ? (double) gen * 1000.0 / dms : 0.0;
    if (t.saved) std::snprintf(b, sizeof b, "saved profile");
    else std::snprintf(b, sizeof b, "%d/%d/%.2f", t.every, t.swaps, t.decay);
    basalt::log::raw("  adapt %s: %.1f tok/s (prose %.1f, struct %.1f; reset %s ms, %s + %s slots moved, %s kept)\n", b,
                     t.tps, rate[0], rate[1], ms.c_str(), moved.c_str(), peer_moved.c_str(), kept.c_str());
    return true;
}

// The prefill CPU share, measured live: 8k-token chunks (the fixtures cycled, rotated so no chunk shares a prefix with
// another) timed at share 0 (all GPU) and 1 (all CPU), then at the share that balances the two and at the default,
// ABBA. Ends the engine's share at the last value set. Returns the balance share, or -1 when it cannot be measured.
constexpr long long kChunk = 8192;
constexpr double kWorkerTolerance = 0.02, kShareMinGain = 0.08;
double measure_share(Engine& eng, const std::vector<std::vector<int64_t>>& fixtures, double base, std::vector<ShareTrial>& trials) {
    std::vector<int64_t> pool;
    for (const auto& f : fixtures) pool.insert(pool.end(), f.begin(), f.end());
    if (pool.empty()) return -1;
    int idx = 0;
    std::string err;
    auto run = [&](double share, bool keep) {
        std::vector<std::string> blk;
        char cmd[64];
        std::snprintf(cmd, sizeof cmd, "PROBE set share=%.3f", share);
        if (!eng.command(cmd, "PROBE end", blk, err)) return false;
        std::vector<int64_t> ids((size_t) kChunk);
        const size_t off = (size_t) (1 + 1531 * idx++);
        for (size_t j = 0; j < ids.size(); ++j) ids[j] = pool[(off + j) % pool.size()];
        Done d;
        if (!eng.request(" temperature=0", 1, ids, d) || d.prompt_ms <= 0) { err = d.err; return false; }
        basalt::log::raw("  chunk share %.2f: %.0f ms (%.0f tok/s)%s%s\n", share, d.prompt_ms, 1000.0 * (double) d.prompt / d.prompt_ms,
                     d.reused > 0 ? ", prefix reused" : "", keep ? "" : " (warm-up)");
        if (keep) trials.push_back({share, d.prompt_ms});
        return d.reused == 0;
    };
    if (!run(base, false) || !run(0.0, true) || !run(1.0, true)) {
        basalt::log::error("basalt-probe", "prefill share: %s", err.c_str());
        return -1;
    }
    const double star = balance_share(mean_ms_at(trials, 0.0), mean_ms_at(trials, 1.0));
    if (std::fabs(star - base) > 1e-9 && run(base, true) && run(star, true) && run(star, true)) run(base, true);
    return star;
}

}  // namespace

int main(int argc, char** argv) {
    ProbeOptions o;
    std::string err;
    if (!parse_args(argc, argv, o, err)) {
        basalt::log::error("basalt-probe", "%s", err.c_str());
        return 2;
    }
    basalt::platform::ignore_broken_pipe();
    basalt::platform::install_child_stop_signals();

    std::string engine_bin = o.engine.empty() ? std::string(BASALT_BENCH_ENGINE) : o.engine;
    if (engine_bin.empty() || !fs::exists(engine_bin)) {
        basalt::log::error("basalt-probe", "the engine binary %s does not exist (--engine BIN)", engine_bin.c_str());
        return 2;
    }

    std::vector<std::string> args = {"--model", fs::absolute(o.model).string()};
    merge_args(args, o.passthrough);
    // the adaptive tier: a saved profile is compared, never written (the engine starts on the model's own); the RAM
    // copy's free places are sized at start for the largest swap count tried
    std::string saved_profile;
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--expert-profile-save") {
            saved_profile = args[i + 1];
            args.erase(args.begin() + (std::ptrdiff_t) i, args.begin() + (std::ptrdiff_t) i + 2);
            break;
        }
    {
        uint64_t bytes = 0;
        if (!saved_profile.empty() && !(basalt::platform::file_size(saved_profile, bytes) && bytes > 0)) saved_profile.clear();
    }
    const std::string swaps_given = arg_value(args, "--adapt-swaps"), every_given = arg_value(args, "--adapt-every");
    const int base_swaps = swaps_given.empty() ? basalt::program::Options().adapt_swaps : std::atoi(swaps_given.c_str());
    bool adapt_on = base_swaps > 0 && (every_given.empty() || std::atoi(every_given.c_str()) > 0);
    const AdaptTrial kAdaptTried[] = {{64, 256, 0.70, false, 0}, {1, 160, 0, false, 0}};
    if (adapt_on) {
        int most = base_swaps;
        for (const auto& t : kAdaptTried) most = std::max(most, t.swaps);
        set_arg(args, "--adapt-swaps", std::to_string(most));
    }
    // the GPUs, chosen here once as the server chooses them: the engine sees only those cards and says nothing again
    {
        std::string gpu_value;
        std::vector<GpuCard> used;
        if (!select_gpus(arg_value(args, "--gpu"), "basalt-probe", gpu_value, used, err)) {
            basalt::log::error("basalt-probe", "%s", err.c_str());
            return 2;
        }
        set_arg(args, "--gpu", gpu_value);
    }

    // fixtures
    std::vector<std::vector<int64_t>> ids(3);
    long long need = 4096;
    for (size_t i = 0; i < 3; ++i) {
        std::vector<std::vector<int64_t>> lines;
        if (!read_id_lines((fs::path(o.fixtures) / kFixtures[i].file).string(), lines, err) || lines.empty()) {
            basalt::log::error("basalt-probe", "%s", err.empty() ? (std::string(kFixtures[i].file) + " is empty").c_str() : err.c_str());
            return 2;
        }
        for (auto& l : lines) ids[i].insert(ids[i].end(), l.begin(), l.end());
        need = std::max(need, (long long) ids[i].size() + std::max(kTokens, kAdaptTokens[0]) + 16);
    }
    // the context fits the fixtures unless the engine arguments set one
    if (!has_arg(args, "--max-context")) set_arg(args, "--max-context", std::to_string(need));

    std::string work = o.logs;
    bool temp_work = false;
    if (work.empty()) {
        work = basalt::platform::temp_dir() + "/basalt-probe-" + std::to_string((long long) basalt::platform::process_id());
        temp_work = true;
    }
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { basalt::log::error("basalt-probe", "cannot create %s", work.c_str()); return 2; }
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
    const std::string log_path = (fs::path(work) / "engine.log").string();

    Engine eng;
    eng.label = "basalt-probe";
    basalt::log::info("basalt-probe", "starting the engine");
    if (!eng.start(engine_bin, args, log_path, err)) {
        basalt::log::error("basalt-probe", "%s", err.c_str());
        return 1;
    }
    const auto t_ready = Clock::now();
    // the user's own swap count (the start took the largest tried, for the free places), set while the tiers are as
    // loaded: nothing moves, so the measurements below and the first request stay cold and the user's
    if (adapt_on) {
        std::vector<std::string> blk;
        if (eng.command("PROBE adapt swaps=" + std::to_string(base_swaps), "PROBE end", blk, err)) {
            basalt::log::info("basalt-probe", "adaptive tier at the given swap count: %s", blk.back().c_str());
        } else {
            basalt::log::warn("basalt-probe", "no adapt test: %s", err.c_str());
            err.clear();
            adapt_on = false;
        }
    }

    // 1. the machine's rates
    std::vector<std::string> block;
    ProbeRates rates;
    if (!eng.command("PROBE rates", "PROBE end", block, err) || !parse_probe_rates(block, rates)) {
        basalt::log::error("basalt-probe", "PROBE rates failed: %s", err.empty() ? "no GPU in the reply" : err.c_str());
        return 1;
    }

    // 2. the tiers the engine placed experts in
    const auto info_num = [&](const char* k) -> long long {
        const auto it = eng.info.find(k);
        return it == eng.info.end() ? -1 : std::atoll(it->second.c_str());
    };
    long long primary = info_num("expert_slots_primary");
    if (primary < 0) primary = info_num("expert_slots");
    const long long peer = peer_experts_from_log(log_path);
    std::string experts = "experts " + rates.gpus[0].name + " " + std::to_string(std::max(primary, 0LL));
    if (peer >= 0 && rates.gpus.size() > 1) experts += " / " + rates.gpus[1].name + " " + std::to_string(peer);
    experts += " / RAM rest";
    const long long engine_ctx = info_num("context");
    const std::string kv = eng.info.count("kv") ? eng.info["kv"] : "?";

    // 3. the cold pass: the first request after load counts, nothing is discarded
    std::vector<FixtureSpeed> speeds(3);
    for (size_t i = 0; i < 3; ++i) {
        Done d;
        if (!eng.request(" temperature=0 top_p=0.95 top_k=20", kTokens, ids[i], d) || d.generated <= 0) {
            basalt::log::error("basalt-probe", "%s: %s", kFixtures[i].name, d.err.empty() ? "no tokens generated" : d.err.c_str());
            return 1;
        }
        const double ts = d.decode_ms > 0 ? (double) d.generated / (d.decode_ms / 1000.0) : 0.0;
        basalt::log::raw("  cold %-6s %.1f tok/s, %lld tokens\n", kFixtures[i].name, ts, d.generated);
        speeds[i].name = kFixtures[i].name;
        speeds[i].cold = ts;
    }
    // 4. values the measured rates say beat the defaults here (printed, never applied)
    std::vector<std::string> suggested;
    {
        WorkerPick wp;
        if (pick_workers(rates, 8, kWorkerTolerance, wp)) {
            const std::string s = suggest_workers(rates, wp);
            if (!s.empty()) {
                suggested.push_back(s);
                basalt::log::info("basalt-probe", "pool pick %d frees %d cores", wp.workers,
                             (rates.active_workers > 0 ? rates.active_workers : rates.pool_workers) - wp.workers);
            }
        }
        const std::string given = arg_value(args, "--prefill-cpu-share");
        const double base = given.empty() ? 0.6 : std::atof(given.c_str());
        if (rates.pool_workers > 0 && engine_ctx >= kChunk + 16) {
            std::vector<ShareTrial> trials;
            const double star = measure_share(eng, ids, base, trials);
            const double gain = share_gain(trials, base, star);
            basalt::log::info("basalt-probe", "prefill share: balance %.2f, %+.1f%% over %.2f", star, gain * 100.0, base);
            if (star >= 0 && gain > kShareMinGain) suggested.push_back(suggest_share(star, base, gain));
        }
    }
    // 5. the adaptive tier's settings from the start placement; the first one runs again last for the noise band
    std::string adapt_line;
    if (adapt_on) {
        const auto ta = Clock::now();
        std::vector<AdaptTrial> trials;
        AdaptTrial base_t{0, base_swaps, 0, false, 0};
        std::string aerr;
        bool ok = adapt_trial(eng, ids, saved_profile, base_t, aerr);
        if (ok) trials.push_back(base_t);
        for (AdaptTrial t : kAdaptTried) {
            if (!ok) break;
            if (t.decay <= 0) t.decay = base_t.decay;
            if (t.every == base_t.every && t.swaps == base_t.swaps && std::fabs(t.decay - base_t.decay) < 1e-6) continue;
            ok = adapt_trial(eng, ids, saved_profile, t, aerr);
            if (ok) trials.push_back(t);
        }
        AdaptTrial again = base_t;
        if (ok && (ok = adapt_trial(eng, ids, saved_profile, again, aerr))) trials.push_back(again);
        AdaptTrial saved_t = base_t;
        saved_t.saved = true;
        if (ok && !saved_profile.empty() && (ok = adapt_trial(eng, ids, saved_profile, saved_t, aerr))) trials.push_back(saved_t);
        if (!ok) {
            basalt::log::warn("basalt-probe", "adapt test: %s", aerr.c_str());
        } else {
            const double band = adapt_band(base_t.tps, again.tps, kAdaptBandFloor), base_tps = (base_t.tps + again.tps) / 2.0;
            const std::string sa = suggest_adapt(base_t, base_tps, band, trials);
            if (!sa.empty()) suggested.push_back(sa);
            if (!saved_profile.empty()) {
                const std::string sp = suggest_saved_profile(saved_profile, base_tps, saved_t.tps, band);
                if (!sp.empty()) suggested.push_back(sp);
            }
            adapt_line = format_adapt_line(trials, band);
            basalt::log::info("basalt-probe", "adapt test: %zu runs in %.1f s", trials.size(),
                              std::chrono::duration<double>(Clock::now() - ta).count());
        }
    }
    eng.stop();
    basalt::log::info("basalt-probe", "%.1f s after READY", std::chrono::duration<double>(Clock::now() - t_ready).count());

    ReportInput in;
    in.command_line = "basalt-probe";
    in.rates = rates;
    in.experts_line = experts;
    in.kv = kv;
    in.engine_context = engine_ctx;
    in.suggested = suggested;
    in.speeds = speeds;
    in.adapt_line = adapt_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--model") { ++i; continue; }
        in.config += (in.config.empty() ? "" : " ") + args[i];
    }
    std::fputs(format_report(in).c_str(), stdout);
    return 0;
}
