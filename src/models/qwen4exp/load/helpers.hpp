#pragma once

#include "dependencies.hpp"

namespace basalt::program::generate {

using basalt::program::serve::vram_floor_mib;
using basalt::program::serve::serve_growth_bytes;
using basalt::program::engine_local::kSatBoost;


using Clock = std::chrono::steady_clock;

using basalt::program::Options;
namespace cli = basalt::program::cli;
namespace engine_local = basalt::program::engine_local;
namespace serve_loans = basalt::program::serve;
using engine_local::Drive;
using engine_local::GpuStage;
using engine_local::JobRunner;
using engine_local::PromptStage;
using engine_local::SlotDelete;
using engine_local::SlotSession;
using engine_local::SplitDrive;
using engine_local::drive_pool;
using engine_local::drive_pool_multi;
using engine_local::peer_hot_share_order;



std::string gib_text(uint64_t bytes);
int fatal_start_error(const std::string& text);
std::string vram_free_note();
bool for_each_index_name(const basalt::core::FileRange& index, const std::function<void(const std::string&)>& f);
int argmax(const std::vector<float>& v);
double probe_pcie_h2d_gbps(std::string* samples = nullptr);
double probe_h2d_together_gbps(const std::vector<int>& devs);
double pcie_frac_for_gbps(double gbps, double base);

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: `copy_back(layer, slot, dst)` queues the copy of a slot of the layer's own cache (a layer
// split keeps one cache per card) into the buffer, and `wait` waits for every queued copy before the refills are queued.
template <class Swap, class CopyBack, class Wait>
bool resident_stage_swaps(basalt::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, CopyBack&& copy_back, Wait&& wait) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        if (!copy_back(s.layer, slot, src.exchange_buffer(q))) return false;
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        if (!wait()) return false;
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}


}
