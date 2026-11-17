// src/models/deepseek41/tier_policy.hpp - the admission and placement rules of the expert tiers, free of CUDA so a
// CPU test can drive them with synthetic id streams.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace basalt::models::deepseek41::policy {

constexpr int kPromoteWindow = 64;              // tokens within which a repeat miss earns a VRAM slot
constexpr uint64_t kPieceBytes = 1ull << 20;    // largest single promotion copy
constexpr int kPeerGapPieces = 3;               // second card exchange pieces per layer: about what one head leaves its bus
constexpr uint64_t kPeerMoveAge = 1280;         // lookups (~32 windows) the second card's victim must be older than the main card's
constexpr int kPromoteUses = 12;                // CPU uses that earn a VRAM slot when the RAM tier caches the file
constexpr int kUseHalfLife = 128;               // tokens after which those counts halve

/// One more CPU use of an expert: true, and the count starts over, when it reaches kPromoteUses.
inline bool count_use(uint16_t& uses) {
    if (++uses < kPromoteUses) return false;
    uses = 0;
    return true;
}

/// Whether the tokens from `token` to `token + n` cross a kUseHalfLife boundary.
inline bool use_half_life(uint64_t token, uint64_t n) {
    return token / kUseHalfLife != (token + n) / kUseHalfLife;
}

/// `last` is the token + 1 of the previous miss of an expert (0 = never); true when it falls within `window` tokens.
inline bool repeat_miss(uint64_t last, uint64_t token, int window) {
    return last != 0 && token + 1 - last <= (uint64_t) window;
}

/// How many experts of every layer each place holds.
struct Homes {
    int main = 0;   ///< on the main card and nowhere else; 0 when its slots are only copies of the RAM tier
    int peer = 0;   ///< on the second card and nowhere else
    int ram = 0;    ///< pinned in RAM
    int file = 0;   ///< read from the pack
};

/// `main_slots`: what the main card can hold through a prompt at the full context; `peer_slots`: the room on the second
/// card (0 without one); `ram_slots`: the room in RAM above the floor right now; `ram_cap`: the places per layer the
/// machine has for the tier whatever else is running (its total RAM less the engine's own buffers and the floor, not
/// limited to one layer's experts), or -1 when the tier's size was given and `ram_slots` is all there is.
/// The main card keeps its homes unless `ram_cap` takes what the second card does not and as many places again as the
/// main card would have homed: copies-only decodes slower than homes, so it is picked only where RAM leaves no doubt.
/// The choice reads `ram_cap` alone, so page-cache state and other readings of free memory never move it; `ram_slots`
/// only sizes the tier. What RAM cannot take stays in the file, so a shortfall of N places puts N experts there.
inline Homes plan_homes(int n_expert, int main_slots, int peer_slots, int ram_slots, int ram_cap) {
    Homes h;
    h.peer = peer_slots < n_expert ? peer_slots : n_expert;
    const int left = n_expert - h.peer;
    const int homed = main_slots < left ? main_slots : left;
    const bool roomy = ram_cap < 0 ? ram_slots >= left : ram_cap >= left + homed;
    if (!roomy) h.main = homed;
    const int rest = left - h.main;
    h.ram = ram_slots < rest ? ram_slots : rest;
    h.file = rest - h.ram;
    return h;
}

/// Where the expert at routing rank `r` lives: the main card's homes first, then the head of the RAM tier when the
/// main card's slots copy it (`copies` slots), the second card's homes, the rest of the RAM tier, the file.
enum class Place { Main, Peer, Ram, File };
inline Place place_of_rank(const Homes& h, int copies, int r) {
    if (r < h.main) return Place::Main;
    const int lead = h.main > 0 ? 0 : copies < h.ram ? copies : h.ram;
    if (r - h.main < lead) return Place::Ram;
    if (r - h.main - lead < h.peer) return Place::Peer;
    return r < h.main + h.peer + h.ram ? Place::Ram : Place::File;
}

/// Several second cards: card c holds the second-card homes of layers [lo[c], hi[c]) and of no other layer, so each layer
/// still has exactly one second card. Blocks are contiguous and differ by at most one layer, the larger ones first.
struct PeerBlocks {
    std::vector<int> lo, hi;
};
inline PeerBlocks split_layers(int n_layers, int n_cards) {
    PeerBlocks b;
    const int each = n_layers / n_cards, extra = n_layers % n_cards;
    for (int c = 0, at = 0; c < n_cards; ++c) {
        b.lo.push_back(at);
        at += each + (c < extra ? 1 : 0);
        b.hi.push_back(at);
    }
    return b;
}
inline int owner_of(const PeerBlocks& b, int layer) {
    for (size_t c = 0; c < b.lo.size(); ++c)
        if (layer >= b.lo[c] && layer < b.hi[c]) return (int) c;
    return -1;
}
/// The second-card homes every layer gets: the fewest any card holds for its own block. `block_bytes[c]`: one slot in
/// every layer of card c's block (256-rounded blobs, the per-layer term of the tier's `one_each`); `reserve`: what each
/// card leaves free. One card with every layer is the single second card's count.
inline int uniform_peer_slots(const std::vector<uint64_t>& room, const std::vector<uint64_t>& block_bytes, uint64_t reserve,
                              int n_expert) {
    uint64_t slots = (uint64_t) n_expert;
    for (size_t c = 0; c < room.size(); ++c)
        slots = std::min<uint64_t>(slots, room[c] > reserve && block_bytes[c] > 0 ? (room[c] - reserve) / block_bytes[c] : 0);
    return (int) slots;
}
/// Blocks sized to each card's room: the contiguous split, in card order and with every card holding at least one layer,
/// that gives every layer the most second-card homes `homes` such that each card's block fits:
/// homes * (one slot in every layer of the block) <= room - reserve. `layer_bytes[l]`: one slot of layer l (256-rounded,
/// as the tier's one_each). `homes` 0 (blocks by count, for the record) when even one home per layer does not fit or there
/// are more cards than layers. Feasibility only falls as homes rise, so a binary search finds the best count; each count
/// is checked exactly over every contiguous split (reach[c][i]: the first c cards can hold the first i layers). Filling
/// each card as far as it goes is NOT enough when the cards' rooms differ: a big early block can starve a small card.
inline PeerBlocks split_by_bytes(const std::vector<uint64_t>& layer_bytes, const std::vector<uint64_t>& room, uint64_t reserve,
                                 int n_expert, int& homes) {
    const int L = (int) layer_bytes.size(), n = (int) room.size();
    auto fill = [&](uint64_t h, PeerBlocks& b) {
        // from[c][i]: where card c-1's block starts when the first c cards hold exactly the first i layers, -1: cannot
        std::vector<std::vector<int>> from((size_t) n + 1, std::vector<int>((size_t) L + 1, -1));
        from[0][0] = 0;
        for (int c = 0; c < n; ++c) {
            const uint64_t cap = room[(size_t) c] > reserve ? room[(size_t) c] - reserve : 0;
            for (int i = 0; i < L; ++i) {
                if (from[(size_t) c][(size_t) i] < 0) continue;
                uint64_t sum = 0;
                for (int j = i; j < L; ++j) {   // card c takes layers [i, j]
                    sum += layer_bytes[(size_t) j];
                    if (h * sum > cap) break;
                    if (from[(size_t) c + 1][(size_t) j + 1] < 0) from[(size_t) c + 1][(size_t) j + 1] = i;
                }
            }
        }
        if (from[(size_t) n][(size_t) L] < 0) return false;
        b.lo.assign((size_t) n, 0);
        b.hi.assign((size_t) n, 0);
        for (int c = n, i = L; c > 0; --c) {
            b.hi[(size_t) c - 1] = i;
            i = from[(size_t) c][(size_t) i];
            b.lo[(size_t) c - 1] = i;
        }
        return true;
    };
    homes = 0;
    PeerBlocks best = split_layers(L, n > 0 ? n : 1);
    if (n < 1 || n > L) return best;
    int lo = 1, hi = n_expert;
    PeerBlocks b;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (fill((uint64_t) mid, b)) {
            homes = mid;
            best = b;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

/// A second-card slot in its own card's numbering: the card's first layer starts at 0. The tier's global slot of
/// (layer, n) is layer * homes_peer + n.
inline int32_t local_slot(const PeerBlocks& b, int homes_peer, int layer, int n) {
    return (int32_t) ((layer - b.lo[(size_t) owner_of(b, layer)]) * homes_peer + n);
}

/// Misses of a layer go half to the CPU (rounded up) and the rest to the bus; the file-backed ones can only go to the
/// CPU. Returns how many of the `pinned` misses the bus fetches, given `file` file-backed misses.
inline int bus_count(int pinned, int file) {
    const int m = pinned + file;
    const int cpu = (m + 1) / 2 > file ? (m + 1) / 2 : file;
    const int n = m - cpu;
    return n < 0 ? 0 : n > pinned ? pinned : n;
}

/// Scratch slots on a second card for the misses its own link fetches: the most a layer sends it.
constexpr int kPeerFetchSlots = 4;

/// Of the bus's misses (bus_count), how many a second card fetches over its own link and computes, the main card's link
/// taking the rest. A copy takes longer than a CPU expert, so one link carrying as many misses as the CPU ends well after
/// it: half of them, rounded down, go over the second card's link instead. When the CPU has one more, it ends last
/// anyway, and a second link would only slow it.
inline int peer_bus_count(int pinned, int file) {
    const int bus = bus_count(pinned, file);
    return 2 * bus >= pinned + file ? bus / 2 : 0;
}

/// The second card's share is queued before its plan, so its grouped kernels run this many groups side by side and stride
/// over the rest: about the experts a window of `n_tok` rows sends it.
inline int64_t peer_groups(int n_tok) { return 2 * (int64_t) n_tok + 1; }

/// Copies of at most kPieceBytes needed for `bytes`.
inline int pieces(uint64_t bytes) { return (int) ((bytes + kPieceBytes - 1) / kPieceBytes); }

/// Slots per layer that fit in `free_bytes` with `keep` left free; at most `most`.
inline int64_t slots_beside(uint64_t free_bytes, uint64_t keep, uint64_t one_each, int64_t most) {
    if (free_bytes <= keep || one_each == 0 || most <= 0) return 0;
    const uint64_t n = (free_bytes - keep) / one_each;
    return n < (uint64_t) most ? (int64_t) n : most;
}

/// Extra slots per layer beside `per_layer` open ones: what `free_bytes` holds above the tier's own room (`room`) and the
/// reserve (`floor`), within the expert count and the slot cap (`cap` < 0: none).
inline int64_t grow_slots(uint64_t free_bytes, uint64_t room, uint64_t floor, uint64_t one_each, int64_t per_layer, int n_expert, int64_t cap) {
    const int64_t limit = cap >= 0 && cap < n_expert ? cap : n_expert;
    return slots_beside(free_bytes, room + floor, one_each, limit - per_layer);
}

/// Whether a tier of `base` + `ext` slots per layer honours an exact count of `asked` per layer (`asked` <= 0: auto,
/// anything goes). The plan caps its slots at the count, so fewer than it means the card could not hold it.
inline bool holds_exact(int64_t asked, int64_t base, int64_t ext) { return asked <= 0 || base + ext >= asked; }

/// Slots per layer a lent arena opens again with: all `lent` when they fit above the reserve, else what fits above the
/// tier's own room and the reserve.
inline int64_t reclaim_slots(uint64_t free_bytes, uint64_t room, uint64_t floor, uint64_t one_each, int64_t lent) {
    const int64_t whole = slots_beside(free_bytes, floor, one_each, lent);
    return whole == lent ? lent : slots_beside(free_bytes, room + floor, one_each, lent);
}

/// The pinned expert of a layer to give up: the least recently used one not in `busy[0..n_busy)`; ties to the lowest id.
/// `idx[e] >= 0` marks pinned experts, `stamp[e]` the last use. Returns -1 when none qualifies.
inline int pick_victim(const int32_t* idx, const uint64_t* stamp, int n_expert, const int* busy, int n_busy) {
    int best = -1;
    for (int e = 0; e < n_expert; ++e) {
        if (idx[e] < 0) continue;
        bool skip = false;
        for (int i = 0; i < n_busy; ++i) skip = skip || busy[i] == e;
        if (skip) continue;
        if (best < 0 || stamp[e] < stamp[best]) best = e;
    }
    return best;
}

}  // namespace basalt::models::deepseek41::policy
