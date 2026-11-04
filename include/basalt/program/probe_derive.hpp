// include/basalt/program/probe_derive.hpp - values basalt-probe computes from what it measured and suggests when they
// beat the engine's built-in defaults on this machine. Pure functions, unit tested; the probe prints, it applies nothing.
#pragma once

#include "basalt/program/probe_report.hpp"

#include <string>
#include <vector>

namespace basalt::program {

/// The smallest worker count whose CPU expert time (entries per call = `entries`) is within `tolerance` of the best
/// measured count. False when the rates carry no such rows. `us_best` and `us_pick` are the per-entry times.
struct WorkerPick {
    int workers = 0;
    double us_best = 0, us_pick = 0;
};
bool pick_workers(const ProbeRates& rates, int entries, double tolerance, WorkerPick& out);

/// The prefill CPU share that balances the CPU and the GPU tier when both take part of a streamed prompt chunk:
/// `ms_gpu` is the chunk time at share 0, `ms_cpu` at share 1. Rounded to 0.05, within 0.05..0.95.
double balance_share(double ms_gpu, double ms_cpu);

/// One timed prompt chunk.
struct ShareTrial {
    double share = 0;
    double ms = 0;
};

/// Mean chunk time of the trials at `share`; 0 when there are none.
double mean_ms_at(const std::vector<ShareTrial>& trials, double share);

/// Speed gain of `candidate` over `base` (mean chunk times), as a fraction; 0 when either is missing.
double share_gain(const std::vector<ShareTrial>& trials, double base, double candidate);

/// The suggestion lines for the report. `workers` is empty when the pick is the pool's current count; the share one is
/// empty unless the candidate beat the default by more than `min_gain` in the live chunks.
std::string suggest_workers(const ProbeRates& rates, const WorkerPick& pick);
std::string suggest_share(double candidate, double base, double gain);

/// One adaptive-tier setting the probe ran from the start placement (PROBE adapt), and its sequence's decode rate.
struct AdaptTrial {
    int every = 0, swaps = 0;
    double decay = 0;
    bool saved = false;   ///< placed from the saved profile instead of the start's
    double tps = 0;
};

/// The probe's noise band from two passes of the same setting: their spread over their mean, at least `floor`.
double adapt_band(double a, double b, double floor);

/// The suggestion for the adaptive tier: the fastest trial other than `base` (not `saved`) when it beats `base_tps`
/// by more than `band`, as the flags that differ from `base` and the gain; empty otherwise.
std::string suggest_adapt(const AdaptTrial& base, double base_tps, double band, const std::vector<AdaptTrial>& trials);

/// The saved profile's line: `saved_tps` against `base_tps`, outside `band` either way; empty inside it.
std::string suggest_saved_profile(const std::string& file, double base_tps, double saved_tps, double band);

/// "1/96/0.85 251 252 | 64/256/0.70 190 | saved profile 255 tok/s (band 1.0%)", in trial order.
std::string format_adapt_line(const std::vector<AdaptTrial>& trials, double band);

}  // namespace basalt::program
