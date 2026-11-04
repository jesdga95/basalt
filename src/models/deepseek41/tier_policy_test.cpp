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

    // the routing order holds every expert of a layer once
    static const uint16_t rank[40][384] =
#include "models/deepseek41/expert_rank.inc"
    ;
    for (const auto& row : rank) {
        std::vector<int> seen(384, 0);
        for (uint16_t e : row) {
            CHECK(e < 384);
            if (e < 384) ++seen[e];
        }
        for (int s : seen) CHECK(s == 1);
    }

    if (g_fail) return 1;
    std::printf("tier_policy_test: ok\n");
    return 0;
}
