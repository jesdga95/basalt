// src/program/probe_derive.cpp - see probe_derive.hpp.
#include "basalt/program/probe_derive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace basalt::program {

namespace {
std::string fmt(const char* f, ...) {
    char b[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
}  // namespace

bool pick_workers(const ProbeRates& rates, int entries, double tolerance, WorkerPick& out) {
    out = WorkerPick{};
    double best = 0;
    for (const auto& r : rates.cpu_expert)
        if (r.entries == entries && r.per_entry_us > 0 && (best == 0 || r.per_entry_us < best)) best = r.per_entry_us;
    if (best <= 0) return false;
    for (const auto& r : rates.cpu_expert) {
        if (r.entries != entries || !(r.per_entry_us > 0) || r.per_entry_us > best * (1.0 + tolerance)) continue;
        if (out.workers == 0 || r.workers < out.workers) {
            out.workers = r.workers;
            out.us_pick = r.per_entry_us;
        }
    }
    out.us_best = best;
    return out.workers > 0;
}

double balance_share(double ms_gpu, double ms_cpu) {
    if (!(ms_gpu > 0) || !(ms_cpu > 0)) return 0.0;
    const double s = ms_gpu / (ms_gpu + ms_cpu);
    return std::min(0.95, std::max(0.05, std::round(s * 20.0) / 20.0));
}

double mean_ms_at(const std::vector<ShareTrial>& trials, double share) {
    double sum = 0;
    int n = 0;
    for (const auto& t : trials)
        if (std::fabs(t.share - share) < 1e-9) { sum += t.ms; ++n; }
    return n ? sum / n : 0.0;
}

double share_gain(const std::vector<ShareTrial>& trials, double base, double candidate) {
    const double a = mean_ms_at(trials, base), b = mean_ms_at(trials, candidate);
    return a > 0 && b > 0 ? a / b - 1.0 : 0.0;
}

std::string suggest_workers(const ProbeRates& rates, const WorkerPick& pick) {
    const int cur = rates.active_workers > 0 ? rates.active_workers : rates.pool_workers;
    if (pick.workers <= 0 || pick.workers >= cur) return "";
    return fmt("--pool-workers %d (%.0f us/expert, best %.0f us)", pick.workers, pick.us_pick, pick.us_best);
}

std::string suggest_share(double candidate, double base, double gain) {
    if (std::fabs(candidate - base) < 1e-9) return "";
    return fmt("--prefill-cpu-share %.2f (prefill %+.0f%%)", candidate, gain * 100.0);
}

double adapt_band(double a, double b, double floor) {
    const double m = (a + b) / 2.0;
    return std::max(floor, m > 0 ? std::fabs(a - b) / m : 0.0);
}

namespace {
bool same_setting(const AdaptTrial& a, const AdaptTrial& b) {
    return a.every == b.every && a.swaps == b.swaps && std::fabs(a.decay - b.decay) < 1e-6 && a.saved == b.saved;
}
}  // namespace

std::string suggest_adapt(const AdaptTrial& base, double base_tps, double band, const std::vector<AdaptTrial>& trials) {
    const AdaptTrial* best = nullptr;
    for (const auto& t : trials)
        if (!t.saved && !same_setting(t, base) && t.tps > 0 && (best == nullptr || t.tps > best->tps)) best = &t;
    if (best == nullptr || !(base_tps > 0) || best->tps <= base_tps * (1.0 + band)) return "";
    std::string s;
    if (best->every != base.every) s += fmt("--adapt-every %d ", best->every);
    if (best->swaps != base.swaps) s += fmt("--adapt-swaps %d ", best->swaps);
    if (std::fabs(best->decay - base.decay) > 1e-6) s += fmt("--adapt-decay %.2f ", best->decay);
    return s + fmt("(adapt test %+.0f%%)", (best->tps / base_tps - 1.0) * 100.0);
}

std::string suggest_saved_profile(const std::string& file, double base_tps, double saved_tps, double band) {
    if (!(base_tps > 0) || !(saved_tps > 0)) return "";
    if (saved_tps > base_tps * (1.0 + band))
        return fmt("--expert-profile-save %s (adapt test %+.0f%%)", file.c_str(), (saved_tps / base_tps - 1.0) * 100.0);
    if (base_tps > saved_tps * (1.0 + band))
        return fmt("no --expert-profile-save (adapt test %+.0f%% on the model's profile)", (base_tps / saved_tps - 1.0) * 100.0);
    return "";
}

std::string format_adapt_line(const std::vector<AdaptTrial>& trials, double band) {
    std::string s;
    std::vector<bool> done(trials.size(), false);
    for (size_t i = 0; i < trials.size(); ++i) {
        if (done[i]) continue;
        const AdaptTrial& t = trials[i];
        if (!s.empty()) s += " | ";
        s += t.saved ? std::string("saved profile") : fmt("%d/%d/%.2f", t.every, t.swaps, t.decay);
        for (size_t j = i; j < trials.size(); ++j)
            if (!done[j] && same_setting(trials[j], t)) {
                s += fmt(" %.0f", trials[j].tps);
                done[j] = true;
            }
    }
    return s + fmt(" tok/s (band %.1f%%)", band * 100.0);
}

}  // namespace basalt::program
