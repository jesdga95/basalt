// src/program/serve/runtime.hpp - the persistent driver over startup's loaded engine.
#pragma once

#include "../engine_local.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/platform/system.hpp"

#include <array>
#include <functional>
#include <optional>

namespace basalt::core {
class ModelFile;
class MtpDrafter;
class RemoteExperts;
class RemoteExpertOpt;
}  // namespace basalt::core
namespace basalt::kernels { class PleTable; }
namespace basalt::kernels::cpu { class ExpertPool; }
namespace basalt::vision { class CudaEncoder; }

namespace basalt::program {
struct Options;
namespace serve {

// Below this much free VRAM a later allocation (a first-used window's buffers, the desktop, another program) can make
// a driver that pages GPU memory do so, and a verify graph spinning on a host flag then never finishes.  A driver that
// fails such an allocation instead needs room only for what is allocated after loading (lazily loaded kernels, graphs,
// a streaming state's residency map).
inline constexpr int kVramStallMib = 256;
inline constexpr int kVramFloorMib = 128;
inline int vram_floor_mib() { return basalt::platform::gpu_memory_pages() ? kVramStallMib : kVramFloorMib; }

// All engine objects belong to startup and outlive run(). Callbacks borrow the same objects.
struct Context {
    Options& o;
    std::string& ppl_base;
    std::string& ppl_out;
    basalt::core::ModelFile& model_file;
    const bool& peer_streamed_share;
    std::vector<int64_t>& split_at;
    std::vector<int>& split_devs;
    bool& split_same;
    bool& multi_gpu;
    const bool& pipe_want;
    const bool& no_spec;
    std::string& err;
    double& main_link_gbps;
    basalt::core::ModelGeometry& g;
    int64_t& K;
    std::vector<std::pair<int32_t, int32_t>>& profile;
    bool& peer_hot_share;
    std::vector<std::pair<int32_t, int32_t>>& profile_loaded;
    const bool& pf_borrow;
    basalt::core::WeightTable& wt;
    int32_t*& d_mrope;
    std::vector<int32_t>& mrope_host;
    std::string& cvec_summary;
    basalt::core::SessionState& ss;
    cudaStream_t& main_stream;
    void* const& main_cs;
    basalt::kernels::PleTable& ple_table;
    basalt::vision::CudaEncoder& vis_enc;
    std::vector<std::unique_ptr<engine_local::GpuStage>>& stages;
    engine_local::GpuStage* const& last_st;
    std::function<int(int64_t)> stage_of;
    std::vector<std::pair<int32_t, int32_t>>& profile_all;
    const int64_t& kv_floor;
    std::vector<std::vector<engine_local::SlotSession>>& bslot_ss;
    std::vector<int64_t>& slot_cap;
    bool& slot0_is_solo;
    basalt::core::MtpDrafter& mtp;
    const bool& mtp_on;
    const basalt::core::QsaState* const& mtp_kv;
    const int64_t& n_vocab;
    std::optional<basalt::program::grammar::PackVocab>& grammar_vocab;
    basalt::core::NativeHead& native_head;
    basalt::core::NativeHead& mtp_head;
    basalt::core::FileExpertSource& src;
    basalt::core::ExpertSource*& srcp;
    basalt::core::ExpertCache& xcache;
    std::array<basalt::core::RemoteExperts, 3>& remote_experts;
    std::unique_ptr<basalt::core::RemoteExpertOpt>& remote_opt;
    basalt::core::PeerExperts& peer;
    std::vector<std::unique_ptr<basalt::core::PeerExperts>>& xpeers;
    engine_local::Drive& drive;
    std::vector<int32_t>& host_res;
    std::function<cudaError_t(int32_t*)> res_put;
    int32_t*& d_res;
    basalt::core::TokenHits& thits;
    const bool& graph_hits;
    int64_t& auto_lend_pct;
    int64_t& auto_ceiling;
    uint64_t& prompt_file_bytes;
    basalt::kernels::cpu::ExpertPool& pool;
    uint64_t& ram_floor_bytes;
    std::vector<std::unique_ptr<engine_local::PromptStage>>& pstages;
    std::function<int64_t()> read_cap;
    const int64_t& kSplitRefCells;
    int64_t& split_hand_rows;
    int64_t& split_peer_rows;
    std::function<engine_local::SplitPricing(int64_t, int64_t, int64_t, int)> split_price;
    std::function<std::string(int, const std::vector<int>&)> split_layers_text;
    std::function<std::string(const PsPrice&)> split_stages_text;
    int& split_stages;
    uint64_t& tier_file_bytes;
    std::function<void()> report_expert_tiers;
};

// Host RAM a server takes while it serves, beyond what it holds once ready: the conversation checkpoints (the solo
// session's and those a batch slot keeps, each the whole running state), the parked conversations, one image's
// host buffers at the largest canvas and a running request's own buffers.  The RAM copy of the experts leaves this
// free beside the margin, so serving stays above the RAM guard.
uint64_t serve_growth_bytes(const Options& o, const basalt::core::ModelGeometry& g);

// Serves until QUIT or stdin closes; returns the process exit code.
int run(Context& context);

}  // namespace serve
}  // namespace basalt::program
