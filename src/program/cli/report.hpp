#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::program { struct Options; }
namespace basalt::program::engine_local { struct Drive; }
namespace basalt::core {
struct ModelGeometry;
struct TokenGraph;
struct SessionGraphs;
class ExpertSource;
class FileExpertSource;
class ExpertCache;
class Verifier;
class MtpDrafter;
}
namespace basalt::kernels {
class PleTable;
namespace cpu { class ExpertPool; }
}
namespace basalt::core { class RemoteExperts; }

namespace basalt::program::cli {

struct StatsReport {
    const Options& o;
    const basalt::core::ModelGeometry& g;
    const engine_local::Drive& drive;
    const basalt::kernels::cpu::ExpertPool& pool;
    const basalt::kernels::PleTable& ple_table;
    const basalt::core::ExpertSource* srcp;
    const std::array<basalt::core::RemoteExperts, 3>& remote_experts;
    const basalt::core::ExpertCache& xcache;
    const basalt::core::TokenGraph& tgraph;
    const basalt::core::SessionGraphs& gr;
    double decode_ms;
    int64_t phase_tokens;
    double ms_ple;
    double ms_embed;
    double ms_layers;
    double ms_head;
    double ms_readback;
    double ms_sample;
};

struct SpeculationReport {
    const Options& o;
    const basalt::core::ModelGeometry& g;
    const engine_local::Drive& drive;
    const basalt::kernels::cpu::ExpertPool& pool;
    const basalt::core::FileExpertSource& src;
    const basalt::core::ExpertSource* srcp;
    const basalt::core::Verifier& ver;
    const basalt::core::MtpDrafter& mtp;
    int64_t rounds;
    int64_t drafts_ok;
    int64_t drafts_total;
    const std::vector<int64_t>& window_hist;
    const std::vector<int64_t>& accepted_hist;
    int64_t sfx_windows;
    int64_t sfx_ok;
    int64_t sfx_drafts;
    int64_t misses0;
    int64_t entries0;
    int64_t swaps_total;
    int64_t ram0;
    int64_t files0;
    int64_t pcie0;
    uint64_t fall0;
    double pool_ms0;
    double ms_adapt;
    bool use_mtp;
};

void print_stats(const StatsReport& report);
void print_speculation(const SpeculationReport& report);
void print_routing_dump(const std::string& path, int64_t records);
double print_generation(const Options& o, const std::vector<int64_t>& produced, int64_t n_prompt,
                        double total_ms, double prefill_ms, double ttft_ms);

}  // namespace basalt::program::cli
