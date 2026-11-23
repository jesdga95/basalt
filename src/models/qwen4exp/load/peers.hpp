// src/models/qwen4exp/load/peers.hpp - Flash-Next's gpu-mode peer (model::PeerHooks, docs/design/multi-gpu.md): an expert
// tier on each card after the main one, holding the ranked experts the main card does not, and a prompt stage on each.
// The shared driver deals the experts hottest first by the cards' rooms and shares, refuses a peer that would hold
// nothing, cuts the prompt stages' layers and writes each card's part of the start line; this fills the tiers it dealt.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/model/peer_hooks.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace basalt::core {
class LoadPlan;
}

namespace basalt::models::qwen4exp::engine_local {

class TierPeers final : public basalt::model::PeerHooks {
public:
    /// What the tiers are opened over, once the main card's cache is filled.
    struct Open {
        const std::vector<std::pair<int32_t, int32_t>>* profile = nullptr;   ///< every expert, hottest first
        const basalt::core::ExpertCache* main = nullptr;                      ///< the main card's cache
        basalt::core::ExpertSource* src = nullptr;
        basalt::core::LoadPlan* reads = nullptr;                              ///< the start's pass the fills join
        std::vector<std::function<bool(std::string&)>>* after = nullptr;      ///< checks once the pass landed
        int64_t n_layers = 0, n_expert = 0, n_embd = 0, n_ff = 0;
        uint64_t reserve = 0;           ///< what every peer leaves free past its experts (--peer-reserve-mib, 128 MiB at least)
        std::function<uint64_t(size_t card)> keep;   ///< what a card keeps free besides: its prompt buffers and splits
        bool stages = false;            ///< a long prompt's chunk is read in stages over the cards
    };

    /// The tiers: card 1's is `first`, card i's after it `more[i - 2]` (made as the cards open).
    TierPeers(basalt::core::PeerExperts& first, std::vector<std::unique_ptr<basalt::core::PeerExperts>>& more)
        : first_(first), more_(more) {}

    /// Set before peer_open.
    void set_open(const Open& in);
    /// Card `card`'s tier (1 = the first peer).
    basalt::core::PeerExperts& tier(size_t card) const { return card == 1 ? first_ : *more_[card - 2]; }

protected:
    bool peer_card(size_t card, int device, std::string& err) override;
    uint64_t peer_room(size_t card) const override;
    basalt::model::PeerSpec peer_spec() const override;
    bool peer_place(const basalt::model::PeerPlacement& p, std::string& err) override;
    std::string peer_main() const override;

private:
    basalt::core::PeerExperts& first_;
    std::vector<std::unique_ptr<basalt::core::PeerExperts>>& more_;
    Open in_;
    std::vector<std::pair<int32_t, int32_t>> outside_;   ///< the ranked experts the main card does not hold
};

}  // namespace basalt::models::qwen4exp::engine_local
