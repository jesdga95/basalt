// src/models/inkling2/expert_map.hpp - the console's expert map of Inkling 2 (core/expert_map.hpp): every routed expert
// sits on the card, so each sampled one's home is VRAM, and its lookups are read from the card's running count of each
// (layer, expert) pair, which the experts' plan adds to on every visit (a looped layer's three visits each count).
#pragma once

#include "basalt/core/expert_map.hpp"

#include <cstdint>
#include <vector>

namespace basalt::models::inkling2 {

/// The card's running counts split by where each pair is served: `all` every lookup, `card` those of the experts on the card
/// (`on_card(layer, expert)`), `peer` those another card computes (`on_peer`, for the experts not on the card); the rest were
/// read from RAM. The end-of-session line and the server's hit rate both come from this.
struct CardShare {
    uint64_t card = 0, peer = 0, all = 0;
};
template <class OnCard, class OnPeer>
CardShare card_share(const std::vector<uint32_t>& counts, int n_expert, OnCard on_card, OnPeer on_peer) {
    CardShare s;
    for (size_t i = 0; i < counts.size(); ++i) {
        s.all += counts[i];
        const int l = (int) (i / (size_t) n_expert), e = (int) (i % (size_t) n_expert);
        if (on_card(l, e)) s.card += counts[i];
        else if (on_peer(l, e)) s.peer += counts[i];
    }
    return s;
}

/// Where a pair's lookups are served now.
enum class Served : uint8_t { Ram, Card, Loan, Peer };

/// The card's share of the running counts, banked at every placement change (settle): each lookup counts where its expert
/// was when it ran, so the share between two reads is a rate even when copies move between them. A loaned copy's lookups
/// count for the card, and `loan` counts them again. `where(layer, expert)`: the pair's Served now.
class SettledShare {
public:
    template <class Where>
    CardShare read(const std::vector<uint32_t>& counts, int n_expert, Where where, uint64_t& loan) const {
        CardShare s = bank_;
        loan = loan_;
        add(counts, n_expert, where, s, loan);
        return s;
    }
    template <class Where>
    void settle(const std::vector<uint32_t>& counts, int n_expert, Where where) {
        add(counts, n_expert, where, bank_, loan_);
        seen_ = counts;
    }

private:
    template <class Where>
    void add(const std::vector<uint32_t>& counts, int n_expert, Where where, CardShare& s, uint64_t& loan) const {
        for (size_t i = 0; i < counts.size(); ++i) {
            const uint64_t d = (uint32_t) (counts[i] - (i < seen_.size() ? seen_[i] : 0u));   // wraps
            if (d == 0) continue;
            s.all += d;
            switch (where((int) (i / (size_t) n_expert), (int) (i % (size_t) n_expert))) {
                case Served::Card: s.card += d; break;
                case Served::Loan: s.card += d; loan += d; break;
                case Served::Peer: s.peer += d; break;
                case Served::Ram: break;
            }
        }
    }
    std::vector<uint32_t> seen_;   // the counts at the last settle
    CardShare bank_;               // the lookups before it, by where they were served
    uint64_t loan_ = 0;
};

class ResidentExpertMap {
public:
    /// The map from `counts` (the card's running counts, [layer * n_expert + e]): the sample of all `counts.size()`
    /// pairs, all in VRAM, and each one's lookups since the last call; the first call starts the counting (all zero).
    void take(const std::vector<uint32_t>& counts, int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) {
        total = (int64_t) counts.size();
        if (pairs_.empty()) pairs_ = core::expert_map_pairs(total, core::ExpertMapTally::kCells);
        homes.assign(pairs_.size(), (uint8_t) core::ExpertHome::Vram);
        routed.assign(pairs_.size(), 0u);
        const bool first = last_.empty();
        last_.resize(pairs_.size());
        for (size_t i = 0; i < pairs_.size(); ++i) {
            const uint32_t now = counts[(size_t) pairs_[i]];
            if (!first) routed[i] = now - last_[i];   // the counts wrap at 2^32; the difference does not
            last_[i] = now;
        }
    }

private:
    std::vector<int32_t> pairs_;
    std::vector<uint32_t> last_;
};

}  // namespace basalt::models::inkling2
