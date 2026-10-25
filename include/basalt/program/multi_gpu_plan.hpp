// include/basalt/program/multi_gpu_plan.hpp - what two or more chosen cards do together, priced at start.
//
// The cards after the main one can be expert tiers (the main card runs every layer, the others hold and compute experts),
// run blocks of layers (the layer split: stages one after another), or split each layer (tensor parallel). The engine
// prices each from the cards' memory bandwidth and VRAM and the model's size, as milliseconds per decode window, runs the
// cheapest mechanism it has and prints the figures. A pure function over a card table: no GPU calls.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "basalt/program/gpu_select.hpp"

namespace basalt::program {

enum class Mechanism { ExpertTier, LayerSplit, TensorParallel };
const char* mechanism_name(Mechanism m);

/// What the model costs a window.
struct MgModel {
    int layers = 48;
    int64_t pairs = 48 * 512;           ///< (layer, expert) pairs the experts section holds
    uint64_t blob_bytes = 2760000;      ///< one expert's bytes
    uint64_t main_overhead = 0;         ///< bytes the main card spends before any expert: dense weights, session, graphs, reserve
    uint64_t peer_overhead = 0;         ///< the same on a card after the main one
    int tp_exchanges_per_layer = 2;     ///< all-reduces a layer needs when each layer is split across the cards
    double tokens = 3.0;                ///< rows in a decode window
};

/// The mechanism's figure for a card set.
struct MgPrice {
    Mechanism mechanism = Mechanism::ExpertTier;
    bool runs = false;                  ///< the engine has it; a mechanism it does not have is priced but never picked
    double ms = 0;                      ///< per decode window
};

struct MgPlan {
    std::vector<MgPrice> prices;        ///< expert tier, layer split, tensor parallel
    Mechanism pick = Mechanism::ExpertTier;
    bool equal = false;                 ///< the cards match in bandwidth and VRAM
    std::string line;                   ///< the start line
};

/// Whether `a` and `b` match in memory bandwidth and VRAM within 10% (an unknown figure matches).
bool cards_alike(const GpuCard& a, const GpuCard& b);

/// Prices the mechanisms for `cards` (two or more, main first) and picks the cheapest one that runs. Only a set of alike
/// cards may pick another mechanism than the expert tier, and only when it is at least 5% cheaper.
MgPlan plan_multi_gpu(const std::vector<GpuCard>& cards, const MgModel& model);

}  // namespace basalt::program
