// src/models/inkling2/expert_map_test.cpp - the expert map from the card's running counts: the sample, all in VRAM, and
// each sampled pair's lookups since the last take (the first take starts the counting; a wrapped count still differs right),
// and the card's share settled at each placement change.
#include "models/inkling2/expert_map.hpp"

#include <cstdio>
#include <utility>

using namespace basalt::models::inkling2;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    const int n_layers = 24, n_expert = 128;
    std::vector<uint32_t> counts((size_t) n_layers * n_expert, 0u);
    const std::vector<int32_t> pairs = basalt::core::expert_map_pairs((int64_t) counts.size(), basalt::core::ExpertMapTally::kCells);
    ResidentExpertMap map;
    int64_t total = 0;
    std::vector<uint8_t> homes;
    std::vector<uint32_t> routed;
    for (size_t i = 0; i < counts.size(); ++i) counts[i] = (uint32_t) (7 * i);
    map.take(counts, total, homes, routed);
    expect(total == 3072, "the total is every (layer, expert) pair");
    expect(homes.size() == pairs.size() && routed.size() == pairs.size() && pairs.size() == 640, "the sample's size");
    bool vram = true, zero = true;
    for (size_t i = 0; i < homes.size(); ++i) {
        vram = vram && homes[i] == (uint8_t) basalt::core::ExpertHome::Vram;
        zero = zero && routed[i] == 0;
    }
    expect(vram, "every sampled expert lives in VRAM");
    expect(zero, "the first take starts the counting");
    // some lookups: pair p looked up p % 5 times, the first sampled pair's count wrapping past 2^32
    counts[(size_t) pairs[0]] = 0xfffffffeu;
    map.take(counts, total, homes, routed);
    for (size_t i = 0; i < counts.size(); ++i) counts[i] += (uint32_t) (i % 5);
    counts[(size_t) pairs[0]] = 3u;   // 0xfffffffe + 5
    map.take(counts, total, homes, routed);
    bool right = routed[0] == 5;
    for (size_t i = 1; i < pairs.size(); ++i) right = right && routed[i] == (uint32_t) (pairs[i] % 5);
    expect(right, "each sampled pair's lookups since the last take");
    map.take(counts, total, homes, routed);
    zero = true;
    for (uint32_t r : routed) zero = zero && r == 0;
    expect(zero, "no lookups since the last take");
    // the card's share of the lookups: the experts a spill sent to RAM are in the lookups but no hits
    {
        std::vector<uint32_t> c((size_t) n_layers * n_expert, 1u);
        auto spilled = [&](int l, int e) { return l >= 20 && e >= 96; };   // layers 20-23 lose their top 32 experts
        const CardShare s = card_share(c, n_expert, [&](int l, int e) { return !spilled(l, e); }, [](int, int) { return false; });
        expect(s.all == 3072 && s.card == 3072 - 4 * 32, "a spill's lookups count, its RAM experts are no hits");
        const CardShare r = card_share(c, n_expert, [](int, int) { return true; }, [](int, int) { return false; });
        expect(r.card == r.all && r.all == 3072, "every expert on the card: every lookup a hit");
        c[(size_t) 21 * n_expert + 100] = 1001u;   // a hot RAM expert drags the share down
        const CardShare h = card_share(c, n_expert, [&](int l, int e) { return !spilled(l, e); }, [](int, int) { return false; });
        expect(h.all == 4072 && h.card == 2944 && h.card * 100 / h.all == 72, "the share follows where the lookups went");
        const CardShare p = card_share(c, n_expert, [&](int l, int e) { return !spilled(l, e); }, [&](int l, int e) { return l == 21 && e >= 100; });   // a second card holds the hot one and more
        expect(p.all == 4072 && p.card == 2944 && p.peer == 1001 + 27 && p.all - p.card - p.peer == 4 * 32 - 28, "a second card's experts count apart from the CPU's");
    }
    {   // settled shares: a copy moves between two requests; each request's share stays a rate
        const int ne = 4;
        std::vector<uint32_t> c(8, 0u);   // 2 layers x 4 experts
        std::vector<Served> at(8, Served::Card);
        at[1] = Served::Ram;   // pair 0 has a copy, pair 1 is in RAM
        const auto where = [&](int l, int e) { return at[(size_t) l * ne + e]; };
        SettledShare s;
        uint64_t loan = 0;
        const CardShare r0 = s.read(c, ne, where, loan);
        c[0] += 100;   // request 1: the copy takes 100, the RAM expert 50
        c[1] += 50;
        const CardShare r1 = s.read(c, ne, where, loan);
        expect(r1.card - r0.card == 100 && r1.all - r0.all == 150, "request 1: 100 of 150 on the card");
        s.settle(c, ne, where);   // the swap: pair 0 to RAM, pair 1 to the card
        expect(s.read(c, ne, where, loan).card == r1.card, "a settle moves no count");
        std::swap(at[0], at[1]);
        const CardShare r2s = s.read(c, ne, where, loan);
        expect(r2s.card == r1.card && r2s.all == r1.all, "the swap does not move past lookups");
        c[0] += 10;   // request 2: the RAM expert 10, the new copy 90
        c[1] += 90;
        const CardShare r2 = s.read(c, ne, where, loan);
        expect(r2.card - r2s.card == 90 && r2.all - r2s.all == 100, "request 2: 90 of 100 on the card, not 140 - 100");
        // a loaned copy counts for the card and for the loan; a second card's expert apart; a count that wrapped
        s.settle(c, ne, where);
        at[2] = Served::Loan;
        at[3] = Served::Peer;
        c[2] += 7;
        c[3] += 5;
        c[4] = 0xFFFFFFF0u;
        s.settle(c, ne, where);
        c[4] += 0x20u;   // wraps to 0x10
        const CardShare r3 = s.read(c, ne, where, loan);
        expect(r3.all - r2.all == 7 + 5 + 0xFFFFFFF0ull + 0x20 && loan == 7 && r3.peer == 5, "loan and peer counted apart");
        expect(r3.card - r2.card == 7 + 0xFFFFFFF0ull + 0x20, "a wrapped count still adds its lookups");
    }
    if (failures) return 1;
    std::printf("inkling2 expert map: ok\n");
    return 0;
}
