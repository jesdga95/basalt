// include/basalt/program/multi_gpu_plan.hpp - what two or more chosen cards do together, priced at start.
//
// The cards after the main one can be expert tiers (the main card runs every layer, the others hold and compute experts),
// run blocks of layers (the layer split: stages one after another), or split each layer (tensor parallel). The engine
// prices each from the cards' memory bandwidth and VRAM and the model's size, as milliseconds per decode window, runs the
// cheapest mechanism it has and prints the figures. A pure function over a card table: no GPU calls.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
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

// ---- the prompt-time layer split (expert tier mode).  A long part of a prompt is read as a pipeline of stages: the
// main card reads layers [0, b1) of each chunk and hands it on, the first tier card layers [b1, b2) of it meanwhile, and
// so on to the last layer; the main card reads the last chunk whole once the tier cards' state is back.  Priced per part
// against the main card alone, for every number of stages and every set of cuts.

/// One card's side of a prompt read.
struct PsCard {
    double mem_gbps = 0;                ///< memory bandwidth, the card's prompt compute rate; 0: unknown
    double link_gbps = 0;               ///< host->device bandwidth (probed)
    std::vector<double> stream_bytes;   ///< per layer: the expert bytes each chunk streams to it (its experts not in VRAM)
    double setup_ms = 0;                ///< a tier card: its stage made ready before the first chunk
    int layer_lo = 0, layer_hi = 0;     ///< a tier card: the layers its stage's weights hold, [layer_lo, layer_hi); 0: any
};

/// A part of a prompt and how each way reads it.
struct PsPart {
    int64_t cell0 = 0;                  ///< the context before it
    std::vector<int64_t> tier_chunks;   ///< the main card alone
    std::vector<int64_t> split_chunks;  ///< the split
    uint64_t row_bytes = 0;             ///< one token's hand-off rows
    int64_t draft_rows = 0;             ///< rows of the split's chunks before the last that the draft layer reads back
    double host_gbps = 0;               ///< host->device bandwidth with every card copying at once (probed); 0: no limit
};

/// The most stages a split is priced with, the main card's included.
constexpr int kPsMaxStages = 8;

struct PsPrice {
    double tier_ms = 0;                 ///< the main card alone
    double split_ms = 0;                ///< the cheapest split
    int stages = 0;                     ///< its stages, the main card's included; 0: no split fits
    std::vector<int> cuts;              ///< its cuts: stage k (1 .. stages - 1) reads [cuts[k - 1], cuts[k] or the last layer]
    std::vector<double> stage_ms;       ///< [s]: the cheapest split over s stages, 0 where none fits
    std::vector<std::vector<int>> stage_cuts;   ///< [s]: its cuts
    bool split = false;                 ///< the split is enough cheaper to read the part with
    int lb() const { return cuts.empty() ? 0 : cuts[0]; }   ///< the main card's last layer + 1
};

/// The prices of `part`: `main_tier` and `main_split` are the main card with each way's loan of its slots, `peers` the
/// tier cards with the split's, in the order they join: a split over s stages runs stage k (1 .. s - 1) on peers[s - 1 - k],
/// so peers[0] always reads the last layers.  Every stage count up to `max_stages` and every cut each card's layers allow
/// is priced; the cheapest wins, fewer stages on a tie.
PsPrice price_prompt_split(const PsCard& main_tier, const PsCard& main_split, const std::vector<PsCard>& peers,
                           const PsPart& part, int max_stages, int layers);

/// The order tier cards join a split (peers[0] first): the higher memory bandwidth first, the given order between equals.
std::vector<size_t> prompt_split_order(const std::vector<double>& mem_gbps);

/// The layers each of `n_peers` cards' stage weights must hold for every stage count `ref` priced at least 5% below the
/// main card alone, widened by `margin` layers on each side (a card's last stage ends at the last layer): [lo, hi) per
/// card in join order, {0, 0} for a card none of them uses.
std::vector<std::pair<int, int>> prompt_split_windows(const PsPrice& ref, size_t n_peers, int margin, int layers);

/// The rows of the last card's output a split read can hold for the draft layer before the kept last chunk: the last
/// two chunks handed on stay in the two hand-off buffers they came in, and earlier ones go past `chunk` rows in either
/// buffer of `hand_rows` rows, which no chunk writes.  `chunks`: the part's chunk sizes, the kept last one included.
int64_t prompt_split_draft_room(const std::vector<int64_t>& chunks, int64_t chunk, int64_t hand_rows);

/// How one card reads a part of `tokens` in chunks of at most `cap`: equal chunks on the 256-token grid for the fewest
/// chunks, and with `head_ok` a long part starts with a chunk of `head_chunk`.  The chunk, and the head chunk (0: none;
/// never one the chunk does not exceed, as the prompt path reads no first chunk longer than the rest).
std::pair<int64_t, int64_t> prompt_one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, int64_t head_chunk);

/// The sizes of the chunks the prompt path reads a part of `tokens` in, for a chunk and head chunk: a first chunk of
/// min(head, chunk) when the part is longer than one chunk, then chunks of `chunk`, the last one what is left.
std::vector<int64_t> prompt_chunk_list(int64_t tokens, int64_t chunk, int64_t head);

}  // namespace basalt::program
