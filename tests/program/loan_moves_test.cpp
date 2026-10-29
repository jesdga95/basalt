// tests/program/loan_moves_test.cpp - a prompt loan's table work: the re-lend split and the routed experts' trades.
// The lend sequence runs on a model of the cache (each slot holds an expert's bytes or a loan's buffers) through the
// same helpers the engine calls: no trade reads a slot of a loan still out, and every table entry names a slot that
// holds its expert.
#include "loan_moves.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace basalt::program::serve;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {

uint64_t g_rng = 0;
uint64_t next_u64() {   // splitmix64: the same sequence on every compiler
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
int64_t below(int64_t n) { return (int64_t) (next_u64() % (uint64_t) n); }

constexpr int32_t kBuffers = -2;   // a slot that holds the prompt path's buffers, no expert

// The cache: `slot` per expert (negative: none), `bytes` per slot (the expert it holds, or kBuffers).
struct Cache {
    int32_t layers = 0, n = 0;
    std::vector<int32_t> slot, bytes;
    std::vector<float> use;
    std::vector<uint8_t> in_ram;
};

// the lend loop: every entry from `first` on is lent, its slot takes the loan's buffers
std::vector<LentEntry> lend_from(Cache& c, int64_t first) {
    std::vector<LentEntry> lent;
    for (size_t i = 0; i < c.slot.size(); ++i)
        if (c.slot[i] >= first) {
            lent.emplace_back((int32_t) i, c.slot[i]);
            c.bytes[(size_t) c.slot[i]] = kBuffers;
            c.slot[i] = -1;
        }
    return lent;
}

void refill(Cache& c, const std::vector<LentEntry>& lent) {
    for (const auto& [i, s] : lent) {
        c.bytes[(size_t) s] = i;
        c.slot[(size_t) i] = s;
    }
}

// every entry names a slot that holds its expert
bool table_true(const Cache& c) {
    for (size_t i = 0; i < c.slot.size(); ++i)
        if (c.slot[i] >= 0 && c.bytes[(size_t) c.slot[i]] != (int32_t) i) return false;
    return true;
}

struct Outcome {
    bool reads_lent_slot = false;   // a trade copied from a slot of the loan still out
    bool table_ok = true;           // after the trades and the new loan
    bool sides_ok = true;           // every trade from `first` on to below it
    bool refill_ok = true;          // after the new loan's refill
    int64_t trades = 0, from_ram = 0;
};

// A re-lend that does not cover the old loan (from `first1`), with the new loan from `first2`, then its trades and its
// lend, as the engine's lend runs them.  `mask`: the trades see which entries the old loan still holds.
Outcome relend(Cache c, int64_t first1, int64_t first2, bool mask) {
    Outcome o;
    std::vector<LentEntry> lent = lend_from(c, first1);
    const LoanSplit sp = split_loan(lent, first2);
    refill(c, sp.back);
    for (const auto& [i, s] : sp.again) c.slot[(size_t) i] = s;   // back in the table, the slot still the old buffers
    std::vector<uint8_t> out(c.slot.size(), 0), lent_slot(c.bytes.size(), 0);
    for (const auto& [i, s] : sp.again) {
        out[(size_t) i] = 1;
        lent_slot[(size_t) s] = 1;
    }
    for (int32_t l = 0; l < c.layers; ++l) {
        const size_t at = (size_t) l * (size_t) c.n;
        std::vector<RoutedTrade> tr;
        plan_routed_trades(c.use.data() + at, c.slot.data() + at, c.in_ram.data() + at, mask ? out.data() + at : nullptr,
                           c.n, first2, tr);
        for (const RoutedTrade& t : tr) {
            const int32_t hot = (int32_t) at + t.hot, cold = (int32_t) at + t.cold;
            o.sides_ok = o.sides_ok && t.from >= first2 && t.to < first2 && t.to >= 0 && c.slot[(size_t) hot] == t.from &&
                         c.slot[(size_t) cold] == t.to;
            if (t.from_ram) {
                o.sides_ok = o.sides_ok && c.in_ram[(size_t) hot] != 0;
                ++o.from_ram;
            } else if (lent_slot[(size_t) t.from]) {
                o.reads_lent_slot = true;
            }
            c.bytes[(size_t) t.to] = t.from_ram ? hot : c.bytes[(size_t) t.from];
            c.slot[(size_t) hot] = t.to;
            c.slot[(size_t) cold] = t.from;
            ++o.trades;
        }
    }
    const std::vector<LentEntry> lent2 = lend_from(c, first2);
    o.table_ok = table_true(c);
    refill(c, lent2);
    o.refill_ok = table_true(c);
    return o;
}

Cache random_cache(int32_t layers, int32_t n, int32_t slots) {
    Cache c;
    c.layers = layers;
    c.n = n;
    const size_t experts = (size_t) layers * (size_t) n;
    c.slot.assign(experts, -1);
    c.bytes.assign((size_t) slots, -1);
    c.use.assign(experts, 0.0f);
    c.in_ram.assign(experts, 0);
    std::vector<int32_t> order(experts);
    for (size_t i = 0; i < experts; ++i) order[i] = (int32_t) i;
    for (size_t i = experts; i > 1; --i) std::swap(order[i - 1], order[(size_t) below((int64_t) i)]);
    for (int32_t s = 0; s < slots && (size_t) s < experts; ++s) {
        if (below(10) == 0) continue;   // an empty slot
        c.slot[(size_t) order[(size_t) s]] = s;
        c.bytes[(size_t) s] = order[(size_t) s];
    }
    for (size_t i = 0; i < experts; ++i) {
        c.use[i] = below(3) == 0 ? 0.0f : (float) below(1000) / 64.0f;
        c.in_ram[i] = below(8) != 0;
    }
    return c;
}

}  // namespace

int main() {
    // the split: the entries the next loan takes again, and the rest, in their order
    {
        const std::vector<LentEntry> lent = {{3, 90}, {7, 40}, {1, 75}, {9, 60}};
        const LoanSplit s = split_loan(lent, 60);
        CHECK(s.again.size() == 3 && s.back.size() == 1);
        if (s.again.size() == 3) CHECK(s.again[0] == LentEntry(3, 90) && s.again[1] == LentEntry(1, 75) && s.again[2] == LentEntry(9, 60));
        if (s.back.size() == 1) CHECK(s.back[0] == LentEntry(7, 40));
        CHECK(split_loan({}, 10).again.empty() && split_loan({}, 10).back.empty());
    }
    // the pairing: most-routed tail expert with the least-routed RAM-backed one below, while it is routed more
    {
        //                        0     1     2     3     4     5     6     7
        const float use[8] = {5.0f, 0.0f, 9.0f, 1.0f, 2.0f, 0.5f, 3.0f, 4.0f};
        const int32_t slot[8] = {12, 13, 11, 2, 3, 4, 5, -1};
        const uint8_t ram[8] = {1, 1, 1, 1, 1, 0, 1, 1};
        std::vector<RoutedTrade> t;
        CHECK(plan_routed_trades(use, slot, ram, nullptr, 8, 10, t) == 2);   // 0 and 2 (1 is not routed)
        CHECK(t.size() == 2);
        if (t.size() == 2) {   // 2 (9.0) with 3 (1.0), then 0 (5.0) with 4 (2.0); 5 has no RAM copy, 7 no slot
            CHECK(t[0].hot == 2 && t[0].cold == 3 && t[0].from == 11 && t[0].to == 2 && !t[0].from_ram);
            CHECK(t[1].hot == 0 && t[1].cold == 4 && t[1].from == 12 && t[1].to == 3 && !t[1].from_ram);
        }
        // a cold expert routed as much as the hot one stays
        const float flat[3] = {2.0f, 2.0f, 2.0f};
        const int32_t s3[3] = {0, 1, 9};
        const uint8_t r3[3] = {1, 1, 1};
        t.clear();
        CHECK(plan_routed_trades(flat, s3, r3, nullptr, 3, 5, t) == 1 && t.empty());
        // equally routed cold experts: the one in the later slot goes first
        const float ties[3] = {1.0f, 1.0f, 3.0f};
        t.clear();
        CHECK(plan_routed_trades(ties, s3, r3, nullptr, 3, 5, t) == 1 && t.size() == 1);
        if (t.size() == 1) CHECK(t[0].hot == 2 && t[0].cold == 1 && t[0].to == 1);
    }
    // a tail slot a loan still holds: its expert trades from its RAM copy, or not at all without one
    {
        const float use[4] = {8.0f, 6.0f, 1.0f, 0.5f};
        const int32_t slot[4] = {20, 21, 3, 4};
        const uint8_t ram[4] = {1, 0, 1, 1};
        const uint8_t out[4] = {1, 1, 0, 0};
        std::vector<RoutedTrade> t;
        CHECK(plan_routed_trades(use, slot, ram, out, 4, 10, t) == 2);
        CHECK(t.size() == 1);
        if (t.size() == 1) CHECK(t[0].hot == 0 && t[0].cold == 3 && t[0].from == 20 && t[0].to == 4 && t[0].from_ram);
    }
    // the issue's sequence: a part read alone left its loan out, the next part beside decoding slots lends a tail
    // inside it.  The tail's routed experts move below the new loan from RAM; no trade reads the old loan's buffers.
    {
        Cache c;
        c.layers = 1;
        c.n = 8;
        c.slot = {0, 1, 2, 3, 4, 5, 6, 7};
        c.bytes = {0, 1, 2, 3, 4, 5, 6, 7};
        c.use = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f};
        c.in_ram = {1, 1, 1, 1, 1, 1, 1, 1};
        const Outcome fixed = relend(c, 3, 5, true);
        CHECK(!fixed.reads_lent_slot && fixed.table_ok && fixed.sides_ok && fixed.refill_ok);
        CHECK(fixed.trades == 3 && fixed.from_ram == 3);
        const Outcome old = relend(c, 3, 5, false);   // the trades blind to the loan: they copy its buffers
        CHECK(old.reads_lent_slot && !old.table_ok);
    }
    // random tables, both orders of the two loans' starts, across geometries
    {
        int64_t cases = 0, with_ram = 0, old_bad = 0;
        const int32_t geo[3][3] = {{1, 16, 12}, {3, 32, 70}, {4, 64, 200}};
        for (uint64_t seed = 1; seed <= 400; ++seed) {
            g_rng = seed;
            const int32_t* gm = geo[seed % 3];
            const Cache c = random_cache(gm[0], gm[1], gm[2]);
            const int64_t first1 = 1 + below(gm[2] - 1), first2 = 1 + below(gm[2] - 1);
            const Outcome o = relend(c, first1, first2, true);
            CHECK(!o.reads_lent_slot);
            CHECK(o.sides_ok);
            CHECK(o.table_ok);
            CHECK(o.refill_ok);
            ++cases;
            with_ram += o.from_ram > 0;
            const Outcome b = relend(c, first1, first2, false);
            old_bad += b.reads_lent_slot && !b.table_ok;
        }
        CHECK(with_ram > 50);   // the RAM path ran in many cases
        CHECK(old_bad > 50);    // and the old trades broke the table in many
        std::printf("loan_moves_test: %lld random re-lends, %lld moved experts from RAM, %lld broke without the mask\n",
                    (long long) cases, (long long) with_ram, (long long) old_bad);
    }
    if (g_fail == 0) std::printf("loan_moves_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
