// tests/core/resident_pack_room_test.cpp - how many experts of the complement a RAM copy may pack when the adaptive
// swaps' free places have to live beside them (CPU only: the sizing arithmetic, no GPU, no file, no model).
#include "basalt/core/ram_load.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>


using namespace basalt::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }
uint64_t mib(double m) { return (uint64_t) (m * 1048576.0); }

// A single Blackwell card's numbers: a RAM room of 41.8 GiB for the experts the expert cache does not hold, and
// 0.45 GiB of free places, no place bigger than 2.3 MiB.
const uint64_t kRoom = gib(41.8), kSpares = gib(0.45), kPlace = mib(2.3);

/// The room the pack leaves beside what it packs.
uint64_t beside(uint64_t packed) { return kRoom - packed; }
}

int main() {
    // A budget with the places' size beside it keeps every expert it was given: the places come out of the RAM room,
    // never out of the budget, so the copy does not lose an expert the user asked for.
    check(ram_tier_pack_bytes(gib(39), kRoom, kSpares, kPlace) == gib(39), "a budget below the places keeps its experts whole");
    check(ram_tier_pack_bytes(kRoom - kSpares, kRoom, kSpares, kPlace) == kRoom - kSpares,
          "a budget exactly at the places' edge keeps its experts whole");

    // A budget inside the places' own size of the room gives way to them: without a place an evicted expert is not
    // evicted at all, so the tier moves nothing, which costs far more than the coldest experts.
    check(ram_tier_pack_bytes(gib(41.6), kRoom, kSpares, kPlace) == kRoom - kSpares,
          "a budget inside the places' size funds the places instead");
    check(beside(ram_tier_pack_bytes(gib(41.6), kRoom, kSpares, kPlace)) == kSpares,
          "giving way leaves the places whole beside the pack");

    // The automatic sizing: the budget is the room itself, so it always lands in the places' band. This is the case
    // that silently froze the tier, because the pack reached the last byte of the room and the places were dropped.
    check(ram_tier_pack_bytes(kRoom, kRoom, kSpares, kPlace) == kRoom - kSpares, "a budget equal to the room funds the places");
    check(ram_tier_pack_bytes(std::numeric_limits<uint64_t>::max(), kRoom, kSpares, kPlace) == kRoom - kSpares,
          "a budget above the room is the room's, not the caller's");

    // What every arm above has in common: the places are left whole beside the pack, and the pack passes neither the
    // budget nor the room. A budget below the places' edge is the binding one, so it keeps its own figure exactly.
    for (const uint64_t budget : {gib(1), gib(20), gib(39), gib(41.6), kRoom, std::numeric_limits<uint64_t>::max()}) {
        const uint64_t packed = ram_tier_pack_bytes(budget, kRoom, kSpares, kPlace);
        check(packed <= budget && packed <= kRoom - kSpares, "the pack passes neither the budget nor the room less places");
        check(beside(packed) >= kSpares, "the places stay whole beside the pack");
    }
    check(beside(ram_tier_pack_bytes(gib(20), kRoom, kSpares, kPlace)) > kSpares, "a budget below the places keeps their room");
    // The room's own slack over the places is one place at most, or the pack would not have given way: the guard that
    // decides whether the places can exist at all is that slack, not a reserve taken out of the pack.
    check(kRoom - kSpares > kPlace, "the room holds the places and more than one place beside them");

    // No places, or a room that cannot hold them: nothing is given up for places that cannot exist, and the caller
    // clears them and says so. The room alone filling with them is the same case for a second reading.
    check(ram_tier_pack_bytes(gib(39), kRoom, 0, kPlace) == gib(39), "no places asked for packs to the budget");
    check(ram_tier_pack_bytes(gib(39), gib(0.4), kSpares, kPlace) == gib(39), "a room below the places gives up nothing");
    check(ram_tier_pack_bytes(gib(39), kSpares, kSpares, kPlace) == gib(39), "a room equal to the places gives up nothing");
    check(ram_tier_pack_bytes(gib(39), kSpares + kPlace, kSpares, kPlace) == gib(39),
          "a room of the places alone gives up nothing");
    check(ram_tier_pack_bytes(0, kRoom, kSpares, kPlace) == 0, "a zero budget packs nothing");
    check(ram_tier_pack_bytes(gib(39), kRoom, kSpares, 0) == gib(39), "an unnamed place size is still a pack with places");
    check(ram_tier_pack_bytes(kRoom, kRoom, kSpares, 0) == kRoom - kSpares, "an unnamed place size still funds the places");
    check(ram_tier_pack_bytes(kRoom, kRoom, kSpares, gib(41.5)) == kRoom,
          "places that fill the room leave the pack as it was");

    // The room the places may fill is the tier's room, driver tracking included: a pinned tier's room is below the
    // free RAM it was read from, and the pack follows it down rather than the reading.
    const uint64_t avail = gib(51), reserve = gib(9);
    const uint64_t room = ram_tier_room(avail, reserve, true);
    check(room < ram_tier_room(avail, reserve, false), "a pinned room is below a pageable one");
    check(ram_tier_pack_bytes(room, room, kSpares, kPlace) == room - kSpares, "the pack is bound by the pinned room");
    check(ram_tier_room(avail, reserve, true) - ram_tier_pack_bytes(room, room, kSpares, kPlace) == kSpares,
          "the pinned room leaves the places beside the pack");

    std::printf("resident_pack_room_test: %d checks\n", checks);
    return 0;
}
