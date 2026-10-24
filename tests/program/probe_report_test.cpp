// Unit test for basalt-probe's report: parsing the rates block, the k-suffix depth parser and the layout.
#include "basalt/program/engine_session.hpp"
#include "basalt/program/probe_report.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace basalt::program;

static int failures = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } \
    } while (0)

static std::vector<FixtureSpeed> speeds(double conv, double st, double prose) {
    return {{"conv", conv}, {"struct", st}, {"prose", prose}};
}

int main() {
    // the rates block: pool facts and the CPU expert rows
    {
        ProbeRates pr;
        const std::vector<std::string> block = {
            "PROBE begin rates v1",
            "PROBE gpu dev=0 name=\"NVIDIA GeForce RTX 5090\" sm=170",
            "PROBE host pool_workers=15 active_workers=15",
            "PROBE cpu_expert workers=8 entries=8 median_us=400.0 per_entry_us=50.0 gbps=55.0",
            "PROBE cpu_expert workers=15 entries=1 median_us=53.5 per_entry_us=53.5 gbps=51.6",
            "PROBE end ms=500"};
        CHECK(parse_probe_rates(block, pr) && pr.pool_workers == 15 && pr.active_workers == 15);
        CHECK(pr.cpu_expert.size() == 2 && pr.cpu_expert[0].workers == 8 && pr.cpu_expert[0].entries == 8 &&
              pr.cpu_expert[0].per_entry_us == 50.0);
    }

    // depths: k is 1024, except a number that rounds in thousands to a power of two
    long long t = 0;
    CHECK(parse_depth("262k", t) && t == 262144 && parse_depth("262K", t) && t == 262144);
    CHECK(parse_depth("256k", t) && t == 262144 && parse_depth("8k", t) && t == 8192);
    CHECK(parse_depth("131k", t) && t == 131072 && parse_depth("33k", t) && t == 32768);
    CHECK(parse_depth("240k", t) && t == 245760 && parse_depth("100k", t) && t == 102400);
    CHECK(parse_depth("1049k", t) && t == 1048576 && parse_depth("1.5k", t) && t == 1536);
    CHECK(parse_depth("8192", t) && t == 8192);
    CHECK(!parse_depth("12x", t) && !parse_depth("", t) && !parse_depth("k", t));
    CHECK(format_context(262144) == "256k" && format_context(8192) == "8k" && format_context(1000) == "1000");

    // the rates block
    std::vector<std::string> block = {
        "PROBE begin rates v1",
        "PROBE gpu dev=0 name=\"NVIDIA GeForce RTX 5090\" sm=170 vram_total_mib=32146",
        "PROBE gpu dev=1 name=\"NVIDIA GeForce RTX 5060 Ti\" sm=36 vram_total_mib=16311",
        "PROBE host ram_total_mib=124114 pool_workers=15 active_workers=15",
        "PROBE link dev=0 h2d_gbps=28.9 d2h_gbps=28.7 burst_mib=64",
        "PROBE link dev=1 h2d_gbps=27.0 d2h_gbps=26.0 burst_mib=64",
        "PROBE gpu_mem dev=0 copy_gbps=1542 buffer_mib=160",
        "PROBE gpu_mem dev=1 copy_gbps=390 buffer_mib=160",
        "PROBE host_mem workers=1 gbps=49.9 bytes_mib=1023",
        "PROBE host_mem workers=15 gbps=56.4 bytes_mib=1023",
        "PROBE host_mem workers=8 gbps=50.3 bytes_mib=1023",
        "PROBE end ms=427 workers_restored=15"};
    ProbeRates r;
    CHECK(parse_probe_rates(block, r));
    CHECK(r.gpus.size() == 2 && r.gpus[0].name == "5090" && r.gpus[1].name == "5060 Ti");
    CHECK(r.gpus[0].mem_gbps == 1542 && r.gpus[1].mem_gbps == 390);
    CHECK(r.pcie_gbps == 28.9 && r.ram_gbps == 56.4);
    ProbeRates none;
    CHECK(!parse_probe_rates({"PROBE begin rates v1"}, none));

    // layout
    ReportInput in;
    in.command_line = "basalt-probe";
    in.rates = r;
    in.experts_line = "experts 5090 9460 / 5060 Ti 5704 / RAM rest";
    in.kv = "int8";
    in.engine_context = 8192;
    in.speeds = speeds(120, 130, 125);
    in.suggested = {"--pool-workers 8 (52 us/expert, best 51 us)"};
    in.config = "--gpu 0,1";
    const std::string rep = format_report(in);
    CHECK(rep.find("  5090 1542 GB/s | 5060 Ti 390 GB/s | PCIe 28.9 GB/s | RAM 56 GB/s\n") != std::string::npos);
    CHECK(rep.find("  experts 5090 9460 / 5060 Ti 5704 / RAM rest | KV int8 | ctx 8k\n") != std::string::npos);
    CHECK(rep.find("  decode cold  prose 125 conv 120 struct 130 tok/s (measured)\n") != std::string::npos);
    CHECK(rep.find("warm") == std::string::npos && rep.find("PASS") == std::string::npos && rep.find("predicted") == std::string::npos);
    CHECK(rep.find("  suggested: --pool-workers 8 (52 us/expert, best 51 us)\n  config: --gpu 0,1\n") != std::string::npos);
    CHECK(rep.find(kProbeDisclaimer) != std::string::npos);

    if (failures) return 1;
    std::printf("probe_report_test: ok\n");
    return 0;
}
