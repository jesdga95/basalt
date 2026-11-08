// src/models/deepseek41/tier_policy_test.cpp - the tier admission and placement rules on synthetic id streams.
#include "models/deepseek41/tier_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::models::deepseek41::policy;

static int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

int main() {
    // admission window: only a repeat inside the window counts, the first miss never does
    CHECK(!repeat_miss(0, 100, kPromoteWindow));
    CHECK(repeat_miss(10 + 1, 10 + kPromoteWindow, kPromoteWindow));
    CHECK(!repeat_miss(10 + 1, 10 + kPromoteWindow + 1, kPromoteWindow));
    CHECK(repeat_miss(5 + 1, 5 + kAdmitWindow, kAdmitWindow));
    CHECK(!repeat_miss(5 + 1, 5 + kAdmitWindow + 1, kAdmitWindow));

    // ghost reset: a stream where one expert misses every 40 tokens is admitted on its second miss, one every 100 never
    for (int period : {40, 100}) {
        uint64_t last = 0;
        int admitted = 0;
        for (uint64_t t = 1; t < 1000; t += (uint64_t) period) {
            if (repeat_miss(last, t, kPromoteWindow)) ++admitted;
            last = t + 1;
        }
        CHECK(period == 40 ? admitted > 0 : admitted == 0);
    }

    // CPU-first split: the CPU keeps half (rounded up), the file-backed misses stay with it
    CHECK(bus_count(0, 0) == 0);
    CHECK(bus_count(1, 0) == 0);
    CHECK(bus_count(2, 0) == 1);
    CHECK(bus_count(3, 0) == 1);
    CHECK(bus_count(4, 0) == 2);
    CHECK(bus_count(2, 1) == 1);
    CHECK(bus_count(1, 2) == 1);
    CHECK(bus_count(3, 3) == 3);
    for (int p = 0; p <= 6; ++p)
        for (int f = 0; f <= 6 - p; ++f) {
            const int b = bus_count(p, f);
            CHECK(b >= 0 && b <= p);
            CHECK(p + f - b >= f);   // the CPU always takes every file-backed miss
        }

    // homes: a RAM shortfall puts homes on the main card, with or without a second card; RAM that takes the rest leaves
    // the main card all copies
    {
        Homes h = plan_homes(384, 34, 0, 305, -1);
        CHECK(h.main == 34 && h.peer == 0 && h.ram == 305 && h.file == 45);
        h = plan_homes(384, 34, 0, 384, -1);
        CHECK(h.main == 0 && h.peer == 0 && h.ram == 384 && h.file == 0);
        h = plan_homes(384, 34, 46, 320, -1);
        CHECK(h.main == 34 && h.peer == 46 && h.ram == 304 && h.file == 0);
        h = plan_homes(384, 34, 46, 420, -1);
        CHECK(h.main == 0 && h.peer == 46 && h.ram == 338 && h.file == 0);
        h = plan_homes(384, 34, 46, 220, -1);
        CHECK(h.main == 34 && h.peer == 46 && h.ram == 220 && h.file == 84);
        h = plan_homes(384, 34, 500, 10, -1);
        CHECK(h.main == 0 && h.peer == 384 && h.ram == 0 && h.file == 0);
        // a shortfall of n places puts n experts in the file, never more
        for (int n : {1, 2, 13}) {
            h = plan_homes(384, 34, 46, 304 - n, -1);
            CHECK(h.main == 34 && h.ram == 304 - n && h.file == n);
            h = plan_homes(384, 34, 0, 350 - n, -1);
            CHECK(h.main == 34 && h.file == n);
        }
        for (int mp : {0, 10, 80})
            for (int pp : {0, 7, 46})
                for (int rp : {0, 100, 384}) {
                    h = plan_homes(384, mp, pp, rp, -1);
                    CHECK(h.main + h.peer + h.ram + h.file == 384);
                    CHECK(h.main >= 0 && h.peer >= 0 && h.ram >= 0 && h.file >= 0);
                    // every rank has one place and the places hold what the plan says
                    int n[4] = {0, 0, 0, 0};
                    for (int r = 0; r < 384; ++r) ++n[(int) place_of_rank(h, 59, r)];
                    CHECK(n[(int) Place::Main] == h.main && n[(int) Place::Peer] == h.peer &&
                          n[(int) Place::Ram] == h.ram && n[(int) Place::File] == h.file);
                }
        // one card: the main card's homes head the order, then RAM and the file; RAM that takes all is the head when the
        // main card holds copies; two cards: the copies' head stays in RAM ahead of the second card
        h = plan_homes(384, 34, 0, 305, -1);
        CHECK(place_of_rank(h, 59, 33) == Place::Main && place_of_rank(h, 59, 34) == Place::Ram && place_of_rank(h, 59, 338) == Place::Ram &&
              place_of_rank(h, 59, 339) == Place::File);
        h = plan_homes(384, 34, 0, 384, -1);
        CHECK(place_of_rank(h, 59, 0) == Place::Ram && place_of_rank(h, 59, 383) == Place::Ram);
        h = plan_homes(384, 34, 46, 420, -1);
        CHECK(place_of_rank(h, 59, 58) == Place::Ram && place_of_rank(h, 59, 59) == Place::Peer && place_of_rank(h, 59, 105) == Place::Ram);
        h = plan_homes(384, 34, 46, 320, -1);
        CHECK(place_of_rank(h, 59, 33) == Place::Main && place_of_rank(h, 59, 34) == Place::Peer && place_of_rank(h, 59, 80) == Place::Ram);
    }

    // the machine's capacity decides homes against copies-only: the same live free RAM, different results
    {
        Homes h = plan_homes(384, 22, 46, 338, 345);   // fits, but not twice over: homes, the live room fills the rest
        CHECK(h.main == 22 && h.ram == 316 && h.file == 0);
        h = plan_homes(384, 22, 46, 330, 345);         // the live room moves RAM and the file, never the placement
        CHECK(h.main == 22 && h.ram == 316 && h.file == 0);
        h = plan_homes(384, 22, 46, 310, 345);
        CHECK(h.main == 22 && h.ram == 310 && h.file == 6);
        h = plan_homes(384, 22, 46, 338, 360);         // room for the second card's rest and the homes again: copies only
        CHECK(h.main == 0 && h.ram == 338 && h.file == 0);
        h = plan_homes(384, 22, 46, 330, 1000);        // other programs hold RAM now: a short tier, not homes
        CHECK(h.main == 0 && h.ram == 330 && h.file == 8);
        h = plan_homes(384, 34, 0, 384, 400);          // one card
        CHECK(h.main == 34 && h.ram == 350 && h.file == 0);
        h = plan_homes(384, 34, 0, 384, 418);
        CHECK(h.main == 0 && h.ram == 384);
    }

    // promotion budget: copies come in pieces of at most kPieceBytes
    CHECK(pieces(7'300'000) == 7);
    CHECK(pieces(kPieceBytes) == 1);
    CHECK(pieces(kPieceBytes + 1) == 2);

    // victim choice: least recently used pinned expert, ties to the lowest id, unpinned and busy ones skipped
    int32_t idx[6] = {0, -1, 2, 3, -1, 5};
    uint64_t st[6] = {50, 1, 20, 20, 0, 90};
    CHECK(pick_victim(idx, st, 6, nullptr, 0) == 2);
    int busy[1] = {2};
    CHECK(pick_victim(idx, st, 6, busy, 1) == 3);
    int32_t none[3] = {-1, -1, -1};
    CHECK(pick_victim(none, st, 3, nullptr, 0) == -1);

    // swap-slot accounting on a model of one layer: a spare slot, a victim and a candidate per swap; the slot set
    // stays a permutation and exactly one expert changes sides each time
    const int n = 8, cover = 5;
    std::vector<int32_t> ri(n, -1);
    for (int e = 0; e < cover; ++e) ri[(size_t) e] = e;
    int spare = cover;
    std::vector<uint64_t> stamp(n, 0);
    for (int step = 0; step < 50; ++step) {
        const int cand = cover + step % (n - cover);   // cycles through file-backed ids
        if (ri[(size_t) cand] >= 0) continue;
        const int v = pick_victim(ri.data(), stamp.data(), n, nullptr, 0);
        CHECK(v >= 0);
        const int32_t vs = ri[(size_t) v];
        ri[(size_t) cand] = spare;
        ri[(size_t) v] = -1;
        spare = vs;
        stamp[(size_t) cand] = (uint64_t) step + 1;
        std::vector<int> seen(cover + 1, 0);
        int pinned = 0;
        for (int e = 0; e < n; ++e)
            if (ri[(size_t) e] >= 0) {
                ++seen[(size_t) ri[(size_t) e]];
                ++pinned;
            }
        ++seen[(size_t) spare];
        CHECK(pinned == cover);
        for (int s : seen) CHECK(s == 1);
    }

    // the slot cap and the reserve hold when the tier grows and when it opens again
    {
        constexpr uint64_t one = 100, room = 1000;
        CHECK(slots_beside(500, 600, one, 8) == 0);
        CHECK(slots_beside(1000, 600, one, 2) == 2);
        CHECK(grow_slots(3000, room, 0, one, 4, 384, -1) == 20);
        CHECK(grow_slots(3000, room, 500, one, 4, 384, -1) == 15);
        CHECK(grow_slots(3000, room, 0, one, 4, 384, 10) == 6);
        CHECK(grow_slots(3000, room, 0, one, 10, 384, 10) == 0);
        CHECK(grow_slots(3000, room, 0, one, 12, 384, 10) == 0);
        CHECK(grow_slots(100000, room, 0, one, 4, 16, -1) == 12);
        CHECK(reclaim_slots(3000, room, 0, one, 8) == 8);
        CHECK(reclaim_slots(3000, room, 2300, one, 8) == 0);
        CHECK(reclaim_slots(3000, room, 2200, one, 8) == 8);
        CHECK(reclaim_slots(3000, room, 1500, one, 30) == 5);
        CHECK(reclaim_slots(3000, room, 2500, one, 30) == 0);
    }

    if (g_fail) return 1;
    std::printf("tier_policy_test: ok\n");
    return 0;
}
