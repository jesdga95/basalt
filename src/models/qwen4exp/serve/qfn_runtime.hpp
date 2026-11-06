// src/models/qwen4exp/serve/qfn_runtime.hpp - the persistent driver over startup's loaded engine.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include "../../../program/serve/state.hpp"
#include "../load/engine_local.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/platform/system.hpp"

#include <functional>
#include <optional>

namespace basalt::core {
class ModelFile;
class MtpDrafter;
}  // namespace basalt::core
namespace basalt::kernels { class PleTable; }
namespace basalt::kernels::cpu { class ExpertPool; }
namespace basalt::vision { class CudaEncoder; }

namespace basalt::models::qwen4exp {
struct Options;
namespace serve {

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
    basalt::kernels::cpu::ExpertPool& pool;
    std::vector<std::unique_ptr<engine_local::PromptStage>>& pstages;
    std::function<int64_t()> read_cap;
    const int64_t& kSplitRefCells;
    int64_t& split_hand_rows;
    int64_t& split_chunk_cap;
    int64_t& split_peer_rows;
    std::function<engine_local::SplitPricing(int64_t, int64_t, int64_t, int)> split_price;
    std::function<std::string(int, const std::vector<int>&)> split_layers_text;
    std::function<std::string(const PsPrice&)> split_stages_text;
    int& split_stages;
    basalt::program::ExpertSpill& tier_spill;
    std::function<void()> report_expert_tiers;
};

// Serves until QUIT or stdin closes; returns the process exit code.
int run(Context& context);

}  // namespace serve
}  // namespace basalt::models::qwen4exp
