// src/models/deepseek41/tier_policy.hpp - the admission and placement rules of the expert tiers, free of CUDA so a
// CPU test can drive them with synthetic id streams.
#pragma once

#include <cstdint>

namespace basalt::models::deepseek41::policy {

constexpr int kPromoteWindow = 64;              // tokens within which a repeat miss earns a VRAM slot
constexpr int kAdmitWindow = 256;               // tokens within which a repeat file miss earns a pinned slot
constexpr int kSwapsPerToken = 2;               // pinned-slot swaps started per token
constexpr uint64_t kPieceBytes = 1ull << 20;    // largest single promotion copy

/// `last` is the token + 1 of the previous miss of an expert (0 = never); true when it falls within `window` tokens.
inline bool repeat_miss(uint64_t last, uint64_t token, int window) {
    return last != 0 && token + 1 - last <= (uint64_t) window;
}

/// Misses of a layer go half to the CPU (rounded up) and the rest to the bus; the file-backed ones can only go to the
/// CPU. Returns how many of the `pinned` misses the bus fetches, given `file` file-backed misses.
inline int bus_count(int pinned, int file) {
    const int m = pinned + file;
    const int cpu = (m + 1) / 2 > file ? (m + 1) / 2 : file;
    const int n = m - cpu;
    return n < 0 ? 0 : n > pinned ? pinned : n;
}

/// Copies of at most kPieceBytes needed for `bytes`.
inline int pieces(uint64_t bytes) { return (int) ((bytes + kPieceBytes - 1) / kPieceBytes); }

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
