// include/basalt/program/probe_report.hpp - basalt-probe's report: the rates block parsed into the first line and the
// text layout. Pure functions of their inputs (no engine, no clock), so they are unit tested.
#pragma once

#include <string>
#include <vector>

namespace basalt::program {

/// The line printed under every report.
extern const char* const kProbeDisclaimer;

/// What the report quotes from the `PROBE rates` block.
struct ProbeRates {
    struct Gpu {
        std::string name;        ///< short form, e.g. "5090"
        double mem_gbps = 0;     ///< device-to-device copy rate
    };
    std::vector<Gpu> gpus;
    double pcie_gbps = 0;        ///< host-to-device rate of the first GPU
    double ram_gbps = 0;         ///< host read rate at the engine's active worker count
    int pool_workers = 0, active_workers = 0;
    struct CpuExpert {
        int workers = 0, entries = 0;
        double per_entry_us = 0;
    };
    std::vector<CpuExpert> cpu_expert;   ///< cold experts through the pool, by active worker count and entries per call
};

/// Parses the `PROBE ...` lines. False when no GPU line is found.
bool parse_probe_rates(const std::vector<std::string>& lines, ProbeRates& out);

/// "NVIDIA GeForce RTX 5060 Ti" -> "5060 Ti".
std::string short_gpu_name(const std::string& name);

/// "256k" for multiples of 1024, else the number.
std::string format_context(long long tokens);

/// One fixture's decode speed from the first request of its kind after load.
struct FixtureSpeed {
    std::string name;            ///< "prose", "conv", "struct"
    double cold = 0;             ///< tok/s
};

struct ReportInput {
    std::string command_line;    ///< the probe's own arguments, echoed
    ProbeRates rates;
    std::string experts_line;    ///< "experts 5090 9460 / 5060 Ti 5704 / RAM rest"
    std::string kv;
    long long engine_context = 0;
    std::vector<FixtureSpeed> speeds;
    std::string config;
    std::vector<std::string> suggested;   ///< values measured to beat the defaults here: a flag and its measured reason
    std::string adapt_line;      ///< the adaptive-tier settings' sequence rates (format_adapt_line); empty: not run
};

std::string format_report(const ReportInput& in);

}  // namespace basalt::program
