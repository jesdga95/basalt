// src/models/inkling2/spill.hpp - which experts leave the card when the requested context does not fit beside all of them.
//
// RAM holds a copy of every expert that leaves; the card keeps the rest. A layer that gives some up lays its experts out
// by slot in one reserved range: the ones it keeps first (their own memory, rounded up to the allocation granularity),
// then the ones in RAM, whose part of the range maps one staging allocation that every such layer shares. Decode reads a
// RAM expert where it is (its blob table entry is the mapped host address, or its card copy's); a prompt chunk first copies
// the RAM experts it reads into the shared staging (each from its card copy when it has one; a short chunk only the ones
// its rows route to), so its products read every expert they need in place, by slot.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::inkling2 {

/// Zeroed bytes after a spilled layer's last slot, and at the start of a RAM slot a chunk does not stage when the slot
/// before it is read: the prompt path's products read past the end of a weight matrix (a tile's last K step runs over a
/// down matrix whose rows are not a whole number of steps) and multiply what they find by zero activations, so those bytes
/// must hold finite values. A resident layer's are its allocation's neighbours; a spilled layer's would be the shared
/// staging, which holds whatever layer copied there last.
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
/// copied from their adjacent RAM blobs, one slot copied from its card copy, or the zeroed guard at one slot's start.
struct StageOp {
    enum Kind : uint8_t { Ram, Card, Guard } kind = Ram;
    int32_t slot = 0;
    int32_t n = 1;
};

/// The staging of a layer's `n_spill` RAM slots: `card[j]` whether slot j has a card copy, `want[j]` whether the chunk
/// reads it (null: every one), `kept` whether kept slots precede them. A wanted slot is staged from its card copy, or with
/// its adjacent wanted slots without one in a single RAM copy; a slot left out right after one the products may read (a
/// wanted or kept one) gets the guard (kSlotGuard: the products read past a matrix's end into the next slot).
std::vector<StageOp> stage_plan(int n_spill, const uint8_t* card, const uint8_t* want, bool kept);

/// The RAM experts' card copies back to back in one address range, each 256-byte aligned: copy k spans [at[k], end[k]),
/// in the order they were placed (a demote packs the survivors to the front), so the first m need the range's first
/// pages(m) granules mapped.
struct CopyLayout {
    std::vector<uint64_t> at, end;
    uint64_t gran = 0;
    static uint64_t aligned(uint64_t bytes) { return (bytes + 255) / 256 * 256; }
    void append(uint64_t stride) {
        const uint64_t start = end.empty() ? 0 : aligned(end.back());
        at.push_back(start);
        end.push_back(start + stride);
    }
    uint64_t bytes() const { return end.empty() ? 0 : (end.back() + gran - 1) / gran * gran; }
    uint64_t pages(int64_t m) const { return m > 0 ? (end[(size_t) m - 1] + gran - 1) / gran : 0; }
    /// How many of the first n copies stay so that dropping the rest unmaps at least `bytes` of `mapped` granules; 0 when
    /// even dropping all of them unmaps less.
    int64_t keep_for(int64_t n, uint64_t mapped, uint64_t bytes) const;
};
CopyLayout copy_layout(const std::vector<uint64_t>& strides, uint64_t gran);

/// Sends copies back until `room()` counts `free` bytes: `demote(bytes, freed)` gives back at least `bytes`, the lowest
/// ranked copies first, and sets what it freed. Stops when `copies()` is 0 or a step frees nothing; false when a step
/// failed.
template <class Room, class Copies, class Demote>
bool give_back(uint64_t free, Room room, Copies copies, Demote demote) {
    while (copies() > 0 && room() < free) {
        uint64_t freed = 0;
        if (!demote(free - room(), freed)) return false;
        if (freed == 0) break;
    }
    return true;
}

/// The positions a conversation's elastic caches keep for a prompt that reads up to `need`: whole `step`s, never below
/// `base` nor past `context`.
inline int64_t kept_positions(int64_t need, int64_t step, int64_t base, int64_t context) {
    return std::min(context, std::max(base, (need + step - 1) / step * step));
}

/// Whether a prompt that resumes past position 0 trims caches backing `backed` positions down to `keep`: only when that
/// gives back at least half of them. A conversation's own turns only go deeper, so this is a compacted conversation or
/// another one on a shared prefix; a switch between conversations costs at most one trim and one regrow, each a move of
/// the memory freed, beside the switch's own read or restore of the deeper conversation.
inline bool resume_trims(int64_t keep, int64_t backed) { return keep > 0 && 2 * keep <= backed; }

}  // namespace basalt::models::inkling2
