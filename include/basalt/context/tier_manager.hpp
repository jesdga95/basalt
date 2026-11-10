// include/basalt/context/tier_manager.hpp - where each context's state lives, and what may move.
//
// The tier ledger sits between the engine (which moves bytes: device allocation, snapshot publish through
// `SnapshotStore`, payload release) and the registry FSM (which owns the logical HOT/WARM/COLD state and leases).
// It answers three questions and refuses everything else:
//
//   room    - does a placement fit the target tier's budget (`fits`), and which contexts could be shed to make
//             room (`evictable`: least-recently-used first, never a pinned one - a lease means someone is reading
//             or writing that state right now), and
//   truth   - the per-tier byte accounting, updated only by `commit_move`, which the engine calls once the move
//             actually happened - a failed publish leaves the ledger where it was, so the previous complete
//             placement stays the recorded one, and
//   fencing - every move carries the generation the engine fenced its work with; a move at or below the last
//             committed generation is refused, so a worker that died and woke up late cannot overwrite the state
//             a newer lease already moved.
//
// Budgets are hard caps: `commit_move` refuses a move that would cross the target tier's budget, so the caller
// evicts first instead of oversubscribing. The clock is injectable; tests run it on a virtual one.
//
// Thread-safe: every call takes the manager's lock.
#pragma once

#include "basalt/context/ids.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace basalt::context {

class TierManager {
public:
    struct Budget {
        uint64_t gpu_bytes = 0;    ///< device pages the engine may hold for contexts
        uint64_t host_bytes = 0;   ///< RAM the parked-state pool may hold
        uint64_t disk_bytes = 0;   ///< snapshot bytes the store may hold
    };
    /// Milliseconds; any monotone source. The default reads the system clock.
    using Clock = std::function<int64_t()>;

    struct Placement {
        Tier tier = Tier::missing;
        uint64_t bytes = 0;
        Generation generation = 0;   // last committed move's fencing token
        int64_t last_use_ms = 0;
    };

    struct EvictionCandidate {
        ContextId context_id;
        uint64_t bytes = 0;
        Generation generation = 0;   // fence the move with this
    };

    struct Stats {
        uint64_t used_gpu = 0, used_host = 0, used_disk = 0;
        uint64_t budget_gpu = 0, budget_host = 0, budget_disk = 0;
        size_t contexts_gpu = 0, contexts_host = 0, contexts_disk = 0;
        size_t pinned = 0;
    };

    explicit TierManager(Budget budget, Clock clock = nullptr);

    Budget budget() const;

    /// Records where a context's state stands after a move that already happened (or an engine-side allocation).
    /// Replaces any earlier placement; `generation` must exceed the last committed one for a tracked context.
    void track(const ContextId& ctx, Tier tier, uint64_t bytes, Generation generation);
    /// Forgets the context (tombstoned, or its state fully released). Pins go with it.
    void untrack(const ContextId& ctx);

    bool known(const ContextId& ctx) const;
    bool placement(const ContextId& ctx, Placement& out) const;
    Tier tier_of(const ContextId& ctx) const;   // missing for an unknown context
    /// The generation to fence the next move with: strictly above the last committed one.
    Generation next_generation(const ContextId& ctx) const;

    /// Whether `bytes` of new state fits the tier's remaining budget. A context's own bytes on that tier are
    /// excluded, so growing in place or shrinking is judged on the delta.
    bool fits(Tier tier, uint64_t bytes) const;
    bool fits(const ContextId& ctx, Tier to, uint64_t bytes) const;

    /// Commits a completed move to tier `to` with `bytes`. False (no change) for an unknown context, a generation
    /// at or below the last committed one (stale fencing), or a move that would cross the target budget.
    bool commit_move(const ContextId& ctx, Generation generation, Tier to, uint64_t bytes);

    /// A lease holds the state in place: `evictable` skips pinned contexts, and the engine refuses to cool or
    /// release a pinned context's bytes. Any number of leases may pin one context; it unpins when the last goes.
    void pin(const ContextId& ctx, const std::string& lease);
    bool unpin(const ContextId& ctx, const std::string& lease);   // false when that lease was not holding it
    bool pinned(const ContextId& ctx) const;

    /// Marks the context as just used; eviction order follows this.
    void touch(const ContextId& ctx);

    /// Unpinned contexts on `from`, least-recently-used first, until `bytes_needed` is covered (or all of them).
    /// The caller moves them in this order; each move must be fenced with the candidate's generation.
    std::vector<EvictionCandidate> evictable(Tier from, uint64_t bytes_needed) const;

    uint64_t used(Tier tier) const;
    Stats stats() const;

private:
    struct Entry {
        Placement at;
        std::unordered_set<std::string> leases;
    };

    static int64_t system_clock_ms();

    const Budget budget_;
    const Clock clock_;
    mutable std::mutex mu_;
    std::unordered_map<ContextId, Entry> entries_;
    uint64_t used_[4] = {0, 0, 0, 0};   // by Tier
    size_t count_[4] = {0, 0, 0, 0};
};

}  // namespace basalt::context
