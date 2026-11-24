// src/model/peer_hooks.cpp - see include/basalt/model/peer_hooks.hpp.
#include "basalt/model/peer_hooks.hpp"

#include "basalt/core/tier_policy.hpp"

#include <algorithm>
#include <cstdio>

namespace basalt::model {

PeerPlacement deal_peers(const GpuCards& c, const PeerSpec& s, const std::vector<uint64_t>& room, std::string& error) {
    PeerPlacement p;
    const size_t n = c.cards.size();
    if (n < 2) return p;
    std::vector<uint64_t> rooms;   // the peers'
    for (size_t i = 1; i < n; ++i) rooms.push_back(i < room.size() ? room[i] : 0);
    if (s.deal == PeerDeal::Blocks) {
        // with gpu-share, each peer's room is capped at its share of them all; without, the rooms are the shares
        if (!c.given.empty()) {
            const std::vector<double> sh = peer_shares(c, ShareUse::Experts);
            if (sh.size() == rooms.size() && rooms.size() > 1) {
                uint64_t all = 0;
                for (uint64_t r : rooms) all += r;
                for (size_t i = 0; i < rooms.size(); ++i) rooms[i] = std::min<uint64_t>(rooms[i], (uint64_t) (sh[i] * (double) all));
            }
        }
        int homes = 0;
        const core::policy::PeerBlocks b = core::policy::split_by_bytes(s.unit, rooms, s.reserve, s.block_most, homes);
        p.homes = homes;
        p.first.assign(n, 0);
        p.end.assign(n, 0);
        for (size_t i = 0; i < b.lo.size() && i + 1 < n; ++i) {
            p.first[i + 1] = b.lo[i];
            p.end[i + 1] = b.hi[i];
        }
        if (homes == 0) {
            // the first peer whose block does not take one expert a layer (the first peer when none)
            size_t card = 1;
            if (!s.main_holds_all)
                for (size_t i = n; i-- > 1;) {
                    uint64_t one = 0;
                    for (int l = p.first[i]; l < p.end[i]; ++l) one += s.unit[(size_t) l];
                    if (rooms[i - 1] < s.reserve + one) card = i;
                }
            error = peer_holds_nothing(c, card, s.main_holds_all ? "the main card holds every expert" : "no room for an expert");
            return p;
        }
    } else if (s.deal == PeerDeal::Experts) {
        std::vector<uint64_t> net;
        for (uint64_t r : rooms) net.push_back(r > s.reserve ? r - s.reserve : 0);
        const std::vector<int> owner = deal_experts(s.unit, net, peer_shares(c, ShareUse::Experts));
        p.owner.assign(owner.size(), -1);
        p.held.assign(n, 0);
        p.bytes.assign(n, 0);
        for (size_t e = 0; e < owner.size(); ++e) {
            if (owner[e] < 0) continue;
            const size_t card = (size_t) owner[e] + 1;
            p.owner[e] = (int) card;
            ++p.held[card];
            p.bytes[card] += s.unit[e];
        }
        for (size_t i = 1; i < n; ++i)
            if (p.held[i] == 0) {
                error = peer_holds_nothing(c, i, s.unit.empty() ? "the main card holds every expert" : "no room for an expert");
                return p;
            }
    }
    if (s.stage_layers > 0) p.stage_first = stage_cuts(c, s.stage_layers, s.stage_lo);
    return p;
}

std::string peer_part(const PeerPlacement& p, size_t card, int stage_layers) {
    std::string part;
    char b[96];
    if (card < p.first.size() && p.end[card] > p.first[card]) {
        std::snprintf(b, sizeof b, "%lld experts, layers %d-%d", (long long) (p.end[card] - p.first[card]) * p.homes, p.first[card],
                      p.end[card] - 1);
        part = b;
    } else if (card < p.held.size() && p.held[card] > 0) {
        std::snprintf(b, sizeof b, "%lld experts (%s)", (long long) p.held[card], gib(p.bytes[card]).c_str());
        part = b;
    }
    if (card > 0 && card < p.stage_first.size()) {
        const int e = card + 1 < p.stage_first.size() ? p.stage_first[card + 1] : stage_layers;
        part += (part.empty() ? "" : ", ") + std::string("prompt layers ") + std::to_string(p.stage_first[card]) + "-" + std::to_string(e - 1);
    }
    return part;
}

}  // namespace basalt::model
