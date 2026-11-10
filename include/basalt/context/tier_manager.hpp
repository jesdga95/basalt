// Residency of agent contexts across GPU, host RAM and disk. Host-only; no CUDA headers.
//
// The manager decides and accounts; a TierMover the caller supplies copies the bytes (device memory, its pinned host
// pool, snapshot files) and frees them, so the manager allocates no payload. Each tier has the byte budget the caller
// grants. Making room in a tier moves its least recently used idle contexts one tier down (a disk copy past the disk
// budget is dropped, leaving replay), and the space they leave goes to the request that moved them. A move copies
// first and commits after: until then the source stays complete and authoritative, so a cancel or a failure leaves
// the context where it was. A context with a lease or a move in flight is never moved. Mutating calls name the
// generation they expect and are refused under another. A copy must reproduce the digest recorded by the last move
// unless a writer has held the context since; when it does not, the binary state is dropped and the call says replay.
#pragma once

#include "basalt/context/ids.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace basalt::context {

enum class TierStatus : uint8_t {
    ok,
    invalid_argument,
    unknown_context,
    context_exists,
    stale_generation,
    in_use,           ///< a lease pins the context
    moving,           ///< a move of the context is in flight
    over_budget,      ///< the tier cannot make room: too few idle bytes to move down
    canceled,         ///< the move was canceled; the context stays where it was
    move_failed,      ///< the mover failed; the context stays where it was
    replay_required,  ///< no valid binary state: rebuild from the transcript
};
const char* tier_status_name(TierStatus s);

struct TierResult {
    TierStatus status = TierStatus::ok;
    std::string reason;  ///< why, with the figures that decided it; empty when ok
    explicit operator bool() const { return status == TierStatus::ok; }
};

struct TierMove {
    ContextId context;
    Tier from = Tier::missing, to = Tier::missing;
    Generation generation = 0;
    uint64_t bytes = 0;
    bool verify = false;  ///< the copy must reproduce `digest`
    Digest digest{};
};

/// The caller's data path. Calls for one context never overlap; calls for different contexts may.
class TierMover {
public:
    enum class Outcome : uint8_t { done, canceled, failed, corrupt };
    virtual ~TierMover() = default;
    /// Copies the context from `move.from` to `move.to`, leaving the source intact, and sets `digest` to the digest
    /// of what `to` now holds. Should return `canceled` soon after `cancel` turns true. `corrupt`: the source failed
    /// its own checks (a snapshot of another generation included).
    virtual Outcome copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest, std::string& error) = 0;
    /// Frees the context's copy in `tier`, complete or partial; nothing when there is none.
    virtual void release(const ContextId& context, Tier tier) = 0;
};

struct TierBudget { uint64_t gpu = 0, host = 0, disk = 0; };

struct TierInfo {
    Tier tier = Tier::missing;
    ContextState state = ContextState::recorded;
    Generation generation = 0;
    uint64_t bytes = 0;
    uint32_t readers = 0;
    bool writer = false, moving = false, verified = false;
};

struct TierLease {
    ContextId context;
    uint64_t id = 0;
    bool writer = false;
    Generation generation = 0;
};

class TierManager {
public:
    struct Stats {
        uint64_t used[3] = {}, budget[3] = {};  ///< by Tier: gpu, host, disk
        size_t contexts = 0;
        uint64_t moves = 0, evictions = 0, drops = 0, cancels = 0, failures = 0, replays = 0, rejections = 0;
    };

    TierManager(TierMover& mover, TierBudget budget);
    TierManager(const TierManager&) = delete;
    TierManager& operator=(const TierManager&) = delete;

    /// Makes room in `tier` and registers the context there under `generation`, returning the writer lease its
    /// materialization holds (the context cannot be moved before its bytes exist). A context whose binary state was
    /// dropped may be admitted again at the same or a later generation.
    TierResult admit(const ContextId& id, Tier tier, uint64_t bytes, Generation generation, TierLease& writer);
    /// Makes room in `to` and moves the context there.
    TierResult move(const ContextId& id, Tier to, Generation expected);
    /// Cancels the context's move in flight.
    TierResult cancel(const ContextId& id);
    /// Pins the context where it is. A writer lease is exclusive and starts the next generation.
    TierResult acquire(const ContextId& id, Generation expected, bool writer, TierLease& out);
    TierResult release(const TierLease& lease);
    /// Frees every copy of an idle context and forgets it.
    TierResult remove(const ContextId& id, Generation expected);

    bool info(const ContextId& id, TierInfo& out) const;
    Stats stats() const;

private:
    struct Entry {
        Tier tier = Tier::missing, target = Tier::missing;
        Generation generation = 0;
        uint64_t bytes = 0, used_at = 0;
        uint32_t readers = 0;
        bool writer = false, moving = false, verified = false, failed = false;
        Digest digest{};
        std::shared_ptr<std::atomic<bool>> cancel;
        std::vector<std::pair<uint64_t, bool>> leases;  // id, writer
    };

    TierResult reserve(Tier t, uint64_t need);
    TierResult evict(const ContextId& id, Tier from);
    TierResult transfer(const ContextId& id, Tier to);
    TierResult check_locked(const ContextId& id, Generation expected, Entry*& e);
    void settle_locked(Entry& e);

    TierMover& mover_;
    mutable std::mutex mu_;
    std::condition_variable settled_;  // a reservation, eviction or move finished
    uint64_t budget_[3] = {}, used_[3] = {}, leaving_[3] = {};
    uint64_t clock_ = 0, lease_seq_ = 0;
    std::unordered_map<ContextId, Entry> entries_;
    Stats counts_;
};

}  // namespace basalt::context
