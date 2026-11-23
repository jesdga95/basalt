// src/models/qwen4exp/load/peers.cpp - see peers.hpp.
#include "peers.hpp"

#include "basalt/log.hpp"

#include <algorithm>
#include <cstdio>

namespace basalt::models::qwen4exp::engine_local {

void TierPeers::set_open(const Open& in) {
    in_ = in;
    outside_.clear();
    for (const auto& pr : *in.profile)
        if (in.main->slot_of(pr.first, pr.second) < 0) outside_.push_back(pr);
}

bool TierPeers::peer_card(size_t card, int device, std::string& err) {
    while (more_.size() + 2 <= card) more_.push_back(std::make_unique<basalt::core::PeerExperts>());
    return tier(card).prepare(device, *in_.src, in_.n_layers, in_.n_expert, in_.n_embd, in_.n_ff, err);
}

uint64_t TierPeers::peer_room(size_t card) const {
    const uint64_t free = tier(card).room(), keep = in_.keep ? in_.keep(card) : 0;
    return free > keep ? free - keep : 0;
}

basalt::model::PeerSpec TierPeers::peer_spec() const {
    basalt::model::PeerSpec s;
    s.deal = basalt::model::PeerDeal::Experts;
    for (const auto& pr : outside_) s.unit.push_back(basalt::core::PeerExperts::slot_bytes(pr.first));
    s.reserve = in_.reserve;
    if (in_.stages) {
        s.stage_layers = (int) in_.n_layers;
        s.stage_lo = 2;   // the first two layers stay on the main card
    }
    return s;
}

bool TierPeers::peer_place(const basalt::model::PeerPlacement& p, std::string& err) {
    int64_t held = in_.main->slots();
    const size_t n = p.held.size();
    for (size_t card = 1; card < n; ++card) {
        std::vector<std::pair<int32_t, int32_t>> pick;
        for (size_t j = 0; j < outside_.size() && j < p.owner.size(); ++j)
            if (p.owner[j] == (int) card) pick.push_back(outside_[j]);
        basalt::core::PeerExperts& t = tier(card);
        if (!t.fill(pick, *in_.reads, err)) return false;
        in_.after->push_back([&t](std::string& e) { return t.verify_fill(e); });
        held += t.resident();
        basalt::log::info("basalt generate", "peer GPU %d: %lld experts, %.2f GiB (read in the start's pass)", t.device(),
                          (long long) t.resident(), t.gib());
    }
    basalt::log::info("basalt generate", "with the main card's %lld experts that is %lld of %lld on the GPUs",
                      (long long) in_.main->slots(), (long long) held, (long long) (in_.n_layers * in_.n_expert));
    return true;
}

std::string TierPeers::peer_main() const {
    char b[96];
    std::snprintf(b, sizeof b, "dense, KV, %lld experts (%.1f GiB)", (long long) in_.main->slots(), in_.main->gib());
    return b;
}

}  // namespace basalt::models::qwen4exp::engine_local
