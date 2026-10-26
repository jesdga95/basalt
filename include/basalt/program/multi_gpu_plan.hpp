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

// ---- the prompt-time layer split (expert tier mode).  A long part of a prompt is read as a pipeline: the main card
// reads layers [0, lb) of each chunk and hands it on, the tier card reads layers [lb, layers) of it meanwhile, and the
// main card reads the last chunk whole once the tier card's state is back.  Priced per part against the main card alone.

/// One card's side of a prompt read.
struct PsCard {
    double mem_gbps = 0;                ///< memory bandwidth, the card's prompt compute rate; 0: unknown
    double link_gbps = 0;               ///< host->device bandwidth (probed)
    std::vector<double> stream_bytes;   ///< per layer: the expert bytes each chunk streams to it (its experts not in VRAM)
};

/// A part of a prompt and how each way reads it.
struct PsPart {
    int64_t cell0 = 0;                  ///< the context before it
    std::vector<int64_t> tier_chunks;   ///< the main card alone
    std::vector<int64_t> split_chunks;  ///< the split
    uint64_t row_bytes = 0;             ///< one token's hand-off rows
    int64_t draft_rows = 0;             ///< rows of the split's chunks before the last that the draft layer reads back
    double setup_ms = 0;                ///< the tier card's stage made ready before the first chunk
};

struct PsPrice {
    double tier_ms = 0;                 ///< the main card alone
    double split_ms = 0;                ///< the split at `lb`, the best first layer for the tier card
    int lb = 0;                         ///< 0: no split point fits
    bool split = false;                 ///< the split is enough cheaper to read the part with
};

/// The prices of `part`: `main_tier` and `main_split` are the main card with each way's loan of its slots, `peer` the
/// tier card with the split's; the tier card's first layer is searched in [lb_min, layers).
PsPrice price_prompt_split(const PsCard& main_tier, const PsCard& main_split, const PsCard& peer, const PsPart& part,
                           int lb_min, int layers);

}  // namespace basalt::program
