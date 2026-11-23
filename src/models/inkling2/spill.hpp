// src/models/inkling2/spill.hpp - which experts leave the card when the requested context does not fit beside all of them.
//
// RAM holds a copy of every expert that leaves; the card keeps the rest. A layer that gives some up lays its experts out
// by slot in one reserved range: the ones it keeps first (their own memory, rounded up to the allocation granularity),
// then the ones in RAM, whose part of the range maps one staging allocation that every such layer shares. Decode reads a
// RAM expert where it is (its blob table entry is the mapped host address); a prompt chunk copies the layer's RAM experts
// into the shared staging first, so its products read every expert of the layer in place, by slot.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::inkling2 {

/// Zeroed bytes after a spilled layer's last slot: the prompt path's products read past the end of a weight matrix (a
/// tile's last K step runs over a down matrix whose rows are not a whole number of steps) and multiply what they find by
/// zero activations, so those bytes must hold finite values. A resident layer's are its allocation's neighbours; a
/// spilled layer's would be the shared staging, which holds whatever layer copied there last.
constexpr uint64_t kSlotGuard = 64 << 10;

struct SpillPlan {
    /// [layer][expert] the expert's slot: the kept ones first in expert order, then the ones in RAM in expert order.
    /// Empty for a layer that keeps every expert.
    std::vector<std::vector<int32_t>> slot;
    std::vector<int32_t> n_spill;   ///< [layer] experts in RAM
    uint64_t staging = 0;           ///< bytes of the shared staging allocation (0: nothing leaves)
    uint64_t device = 0;            ///< device bytes of every layer's experts, the staging included
    uint64_t full = 0;              ///< what they take with every expert on the card
    int64_t spilled = 0;            ///< experts in RAM, over all layers
    bool any() const { return spilled > 0; }
};

/// The experts that leave so that at least `need` device bytes are freed (`need` 0 and `keep` < 0: none leave), or so
/// that at most `keep` experts stay on the card (`keep` >= 0, a diagnostic; `need` must be met too). `rank`: every
/// (layer, expert) pair, the most routed first (the shipped profile); empty: each layer gives up its highest numbered
/// experts in turn. `stride`: [layer] an expert's bytes on the card; `gran`: the allocation granularity. False with `err`
/// when even every expert in RAM does not free `need`.
bool plan_spill(int n_layers, int n_expert, const std::vector<uint64_t>& stride, uint64_t gran,
                const std::vector<std::pair<int32_t, int32_t>>& rank, uint64_t need, int64_t keep, SpillPlan& out,
                std::string& err);

/// One step of a prompt chunk's staging of a spilled layer: `n` RAM slots from `slot` (0 = the layer's first RAM slot)
/// copied from their adjacent RAM blobs, or one slot copied from its card copy.
struct StageOp {
    enum Kind : uint8_t { Ram, Card } kind = Ram;
    int32_t slot = 0;
    int32_t n = 1;
};

/// The staging of a layer's `n_spill` RAM slots, `card[j]` whether slot j has a card copy: a slot with one is copied from
/// it, a run of adjacent slots without one in a single RAM copy.
std::vector<StageOp> stage_plan(int n_spill, const uint8_t* card);

}  // namespace basalt::models::inkling2
