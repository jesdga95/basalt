// include/basalt/model/peer_hooks.hpp - gpu-mode peer for any model (docs/architecture/multi-gpu.md): the main card runs every
// layer and the other cards help. Whatever runs a model's peers derives PeerHooks and writes the hooks below: each peer
// card's own buffers and the room they leave, what it divides (experts by layer block or one by one, hottest first; a
// long prompt's layers), its part on each card once dealt, and the main card's part for the start line. The driver
// does the rest, the same for every model: the deal by the shares within the rooms, the refusal of a peer that would hold
// nothing, and each card's part of the start line after the load. Which cards, their mode and their shares are the
// shared card layer's (model/multi_gpu.hpp); how a peer computes what it holds stays the model's.
#pragma once

#include "basalt/model/multi_gpu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::model {

/// How a model's experts are dealt to its peers.
enum class PeerDeal {
    None,       ///< no experts on the peers (prompt stages only)
    Blocks,     ///< by layer block: each peer holds a contiguous block of layers and the same number of experts in every
                ///< layer of it, every layer on one peer (core::policy::split_by_bytes)
    Experts,    ///< one by one, hottest first: each to the peer with the fewest for its share (deal_experts)
};

/// What a model divides among its peers, stated once each peer's own buffers are open.
struct PeerSpec {
    PeerDeal deal = PeerDeal::None;
    /// Blocks: per layer, one expert's bytes on a card. Experts: each expert the main card does not hold, hottest first,
    /// its bytes on a card (empty: the main card holds every expert).
    std::vector<uint64_t> unit;
    int block_most = 0;             ///< Blocks: the most experts a layer gives its peer (its experts)
    uint64_t reserve = 0;           ///< what each peer leaves free past what it is dealt
    bool main_holds_all = false;    ///< Blocks: the main card holds every expert, why a peer dealt nothing is refused
    /// A long prompt's chunk read in stages: a stage reads within layers [stage_lo, stage_layers); 0: no stages.
    int stage_layers = 0, stage_lo = 0;
};

/// What the driver dealt, indexed by card as GpuCards::cards is (0 = the main card).
struct PeerPlacement {
    std::vector<int> first, end;    ///< block deal: card i holds layers [first[i], end[i]) (the main card none)
    int homes = 0;                  ///< block deal: a peer's experts in every layer of its block
    std::vector<int> owner;         ///< expert deal: each expert's card, -1 none
    std::vector<int64_t> held;      ///< expert deal: per card, the experts it holds
    std::vector<uint64_t> bytes;    ///< expert deal: per card, their bytes
    std::vector<int> stage_first;   ///< stages: card i reads [stage_first[i], stage_first[i + 1]) of a chunk's layers
};

/// The deal of what `s` divides over the peers of `c`, within each peer's `room` (bytes, per card; the main card's unused)
/// less the reserve: blocks by core::policy::split_by_bytes over the rooms, each capped at its gpu-share of them when one is
/// given (else the rooms are the shares); experts by deal_experts over the Experts shares; stages by stage_cuts. `error`
/// stays empty, or is the refusal of a peer that would hold nothing.
PeerPlacement deal_peers(const GpuCards& c, const PeerSpec& s, const std::vector<uint64_t>& room, std::string& error);

/// Peer card `card`'s part of a placement for the start line: "5120 experts, layers 21-42", "9812 experts (21.3 GiB)",
/// "prompt layers 38-47" or both, over a chunk of `stage_layers` layers.
std::string peer_part(const PeerPlacement& p, size_t card, int stage_layers);

class PeerHooks {
public:
    virtual ~PeerHooks() = default;

    /// Opens the peers, the same for every model: each peer card's own buffers (peer_card), the deal of what the model
    /// divides (peer_spec) within the rooms they leave (peer_room), a peer that would hold nothing refused, then the
    /// model's part on each card (peer_place). False with `err`.
    bool peer_open(const GpuCards& c, std::string& err) {
        if (c.mode != program::GpuMode::Peer || c.cards.size() < 2) return true;
        for (size_t i = 1; i < c.cards.size(); ++i)
            if (!peer_card(i, c.cards[i].device, err)) return false;
        std::vector<uint64_t> room(c.cards.size(), 0);
        for (size_t i = 1; i < c.cards.size(); ++i) room[i] = peer_room(i);
        const PeerSpec s = peer_spec();
        std::string refused;
        placed_ = deal_peers(c, s, room, refused);
        if (!refused.empty()) {
            err = refused;
            return false;
        }
        stage_layers_ = s.stage_layers;
        open_ = true;
        return peer_place(placed_, err);
    }
    /// Card `card`'s part once open (0 = the main card), for the start line after the load; empty when the peers are not open.
    std::string peer_report(size_t card) const {
        if (!open_) return {};
        return card == 0 ? peer_main() : peer_part(placed_, card, stage_layers_);
    }
    /// What the driver dealt (empty until peer_open).
    const PeerPlacement& peer_placement() const { return placed_; }
    /// The prompt stages dealt are not read after all (their buffers do not fit): the start line shows none.
    void peer_drop_stages() { placed_.stage_first.clear(); }

protected:
    // ---- the hooks a model writes (docs/architecture/multi-gpu.md, "Peer")

    /// Opens peer card `card`'s own buffers on CUDA device `device`, whatever it is dealt (its prompt share's staging).
    virtual bool peer_card(size_t card, int device, std::string& err) = 0;
    /// The bytes peer card `card` has for what it is dealt, once every peer's own buffers are open.
    virtual uint64_t peer_room(size_t card) const = 0;
    /// What the model divides among the peers.
    virtual PeerSpec peer_spec() const = 0;
    /// The model's part on each card as dealt: its experts' slots, its stages.
    virtual bool peer_place(const PeerPlacement& p, std::string& err) = 0;
    /// The main card's part for the start line ("dense, KV, 40 of 256 experts per layer (21.3 GiB)").
    virtual std::string peer_main() const = 0;

private:
    PeerPlacement placed_;
    int stage_layers_ = 0;
    bool open_ = false;
};

}  // namespace basalt::model
