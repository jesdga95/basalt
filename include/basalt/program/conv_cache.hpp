// include/basalt/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in src/program/serve/state.hpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it;
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// The most checkpoints the chains can hold together: `prompt_cache` in the solo chain and in a batch slot's request, two
/// in every other slot (`slots` 0: the solo chain alone).
/// The checkpoints every chain of a server holds together (the solo session's and every batch slot's) under a pool of
/// `pool_bytes`: as many as it fits past the one a save fills before the least recently used leaves, and no more than
/// the chains can hold (`prompt_cache` in the solo chain and in a batch slot's request, two in every other slot; `slots`
/// 0: the solo chain alone).  A checkpoint with no bytes takes nothing from the pool.  0: the pool fits fewer than two.
inline size_t pool_most(int64_t prompt_cache, int64_t slots) {
    if (prompt_cache <= 0) return 0;
    return (size_t) (slots > 0 ? 2 * (uint64_t) prompt_cache + 2 * (uint64_t) slots : (uint64_t) prompt_cache);
}
inline size_t pool_checkpoints(uint64_t pool_bytes, uint64_t checkpoint_bytes, int64_t prompt_cache, int64_t slots) {
    if (prompt_cache <= 0) return 0;
    const uint64_t most = pool_most(prompt_cache, slots);
    if (checkpoint_bytes == 0) return (size_t) most;
    const uint64_t fit = pool_bytes / checkpoint_bytes;
    return fit < 2 ? 0 : (size_t) std::min(fit - 1, most);
}

/// The checkpoint to drop once the chains hold more than the pool: `chains[c]` are chain c's stamps, its root first.
/// The least recently used one that is no chain's root; a root (a head other chats share) only when nothing else is
/// left.  {chains.size(), 0} when every chain is empty.
inline std::pair<size_t, size_t> pool_victim(const std::vector<std::vector<uint64_t>>& chains) {
    size_t vc = chains.size(), vi = 0;
    bool v_root = true;
    for (size_t c = 0; c < chains.size(); ++c)
        for (size_t i = 0; i < chains[c].size(); ++i) {
            const bool root = i == 0;
            if (vc == chains.size() || (v_root && !root) || (v_root == root && chains[c][i] < chains[vc][vi])) {
                vc = c;
                vi = i;
                v_root = root;
            }
        }
    return {vc, vi};
}

}  // namespace basalt::program::conv_cache
