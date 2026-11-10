#include "basalt/context/tier_manager.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace basalt::context {
namespace {

size_t at(Tier t) { return static_cast<size_t>(t); }
bool resident(Tier t) { return t == Tier::gpu || t == Tier::host || t == Tier::disk; }
std::string bytes_text(uint64_t n) { return std::to_string(n) + " B"; }
TierResult fail(TierStatus s, std::string reason) { return TierResult{s, std::move(reason)}; }

}  // namespace

const char* tier_status_name(TierStatus s) {
    switch (s) {
        case TierStatus::ok: return "ok";
        case TierStatus::invalid_argument: return "invalid_argument";
        case TierStatus::unknown_context: return "unknown_context";
        case TierStatus::context_exists: return "context_exists";
        case TierStatus::stale_generation: return "stale_generation";
        case TierStatus::in_use: return "in_use";
        case TierStatus::moving: return "moving";
        case TierStatus::over_budget: return "over_budget";
        case TierStatus::canceled: return "canceled";
        case TierStatus::move_failed: return "move_failed";
        case TierStatus::replay_required: return "replay_required";
    }
    return "unknown";
}

TierManager::TierManager(TierMover& mover, TierBudget budget) : mover_(mover) {
    budget_[0] = budget.gpu;
    budget_[1] = budget.host;
    budget_[2] = budget.disk;
}

TierResult TierManager::check_locked(const ContextId& id, Generation expected, Entry*& e) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return fail(TierStatus::unknown_context, "no context " + id);
    e = &it->second;
    if (e->generation != expected)
        return fail(TierStatus::stale_generation, id + ": generation " + std::to_string(expected) +
                                                      " is not the current " + std::to_string(e->generation));
    if (e->moving) return fail(TierStatus::moving, id + ": a move is in flight");
    return {};
}

void TierManager::settle_locked(Entry& e) {
    e.moving = false;
    e.target = Tier::missing;
    e.cancel.reset();
    settled_.notify_all();
}

// Claims `need` bytes of `t`. When the tier is short, its least recently used idle contexts move down first and the
// bytes they leave are claimed at once, so no concurrent request takes them. Only downward moves count as leaving a
// tier: they never wait on the tier they leave, so waiting for them cannot close a cycle.
TierResult TierManager::reserve(Tier t, uint64_t need) {
    const size_t ti = at(t);
    std::set<ContextId> stuck;  // victims whose move failed during this call
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        if (need > budget_[ti]) {
            ++counts_.rejections;
            return fail(TierStatus::over_budget, std::string(tier_name(t)) + ": need " + bytes_text(need) +
                                                     ", budget " + bytes_text(budget_[ti]));
        }
        if (used_[ti] + need <= budget_[ti]) {
            used_[ti] += need;
            return {};
        }
        if (used_[ti] - leaving_[ti] + need <= budget_[ti]) {  // unclaimed bytes on their way out suffice
            settled_.wait(lk);
            continue;
        }
        const uint64_t deficit = used_[ti] - leaving_[ti] + need - budget_[ti];
        std::vector<std::pair<uint64_t, const ContextId*>> idle;
        uint64_t idle_bytes = 0;
        for (const auto& [id, e] : entries_)
            if (e.tier == t && !e.moving && !e.writer && !e.readers && !stuck.count(id)) {
                idle.emplace_back(e.used_at, &id);
                idle_bytes += e.bytes;
            }
        if (idle_bytes < deficit) {
            ++counts_.rejections;
            return fail(TierStatus::over_budget,
                        std::string(tier_name(t)) + ": need " + bytes_text(need) + ", " + bytes_text(used_[ti]) +
                            " in use (" + bytes_text(leaving_[ti]) + " leaving) of " + bytes_text(budget_[ti]) +
                            ", " + bytes_text(idle_bytes) + " idle to move down");
        }
        std::sort(idle.begin(), idle.end());
        std::vector<ContextId> victims;
        uint64_t freed = 0;
        for (const auto& [when, id] : idle) {
            if (freed >= deficit) break;
            Entry& e = entries_.at(*id);
            e.moving = true;
            e.target = t == Tier::disk ? Tier::missing : static_cast<Tier>(ti + 1);
            e.cancel = std::make_shared<std::atomic<bool>>(false);
            freed += e.bytes;
            victims.push_back(*id);
        }
        used_[ti] += need;
        leaving_[ti] += freed;
        lk.unlock();
        bool all = true;
        for (const auto& id : victims)
            if (!evict(id, t)) {
                all = false;
                stuck.insert(id);
            }
        lk.lock();
        if (all) return {};
        used_[ti] -= need;  // a victim stayed: give the claim back and look again
        settled_.notify_all();
    }
}

// The victim is marked moving out of `from` and its bytes count as leaving it.
TierResult TierManager::evict(const ContextId& id, Tier from) {
    if (from == Tier::disk) {  // the last tier: the binary state goes, the transcript replays it
        mover_.release(id, Tier::disk);
        std::lock_guard<std::mutex> lk(mu_);
        Entry& e = entries_.at(id);
        used_[at(from)] -= e.bytes;
        leaving_[at(from)] -= e.bytes;
        e.tier = Tier::missing;
        e.verified = false;
        ++counts_.drops;
        ++counts_.evictions;
        settle_locked(e);
        return {};
    }
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        bytes = entries_.at(id).bytes;
    }
    const Tier to = static_cast<Tier>(at(from) + 1);
    if (TierResult r = reserve(to, bytes); !r) {
        std::lock_guard<std::mutex> lk(mu_);
        leaving_[at(from)] -= bytes;
        settle_locked(entries_.at(id));
        return r;
    }
    TierResult r = transfer(id, to);
    if (r) {
        std::lock_guard<std::mutex> lk(mu_);
        ++counts_.evictions;
    }
    return r;
}

// The context is marked moving and `to` holds a reservation of its bytes. Copies, checks and commits; on any other
// outcome the destination is freed and the context stays where it was, unless its source proved corrupt.
TierResult TierManager::transfer(const ContextId& id, Tier to) {
    TierMove m;
    std::shared_ptr<std::atomic<bool>> cancel;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const Entry& e = entries_.at(id);
        m = TierMove{id, e.tier, to, e.generation, e.bytes, e.verified, e.digest};
        cancel = e.cancel;
    }
    Digest got{};
    std::string error;
    auto o = mover_.copy(m, *cancel, got, error);
    if (o == TierMover::Outcome::done && m.verify && got != m.digest) {
        o = TierMover::Outcome::corrupt;
        error = "the copy does not reproduce the recorded digest";
    }
    if (o == TierMover::Outcome::done && cancel->load()) o = TierMover::Outcome::canceled;
    const bool down = at(to) > at(m.from);
    if (o == TierMover::Outcome::done) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry& e = entries_.at(id);
            e.tier = to;
            e.digest = got;
            e.verified = true;
            e.used_at = ++clock_;
            ++counts_.moves;
        }
        mover_.release(id, m.from);  // the old copy is gone before its bytes count as free
        std::lock_guard<std::mutex> lk(mu_);
        used_[at(m.from)] -= m.bytes;
        if (down) leaving_[at(m.from)] -= m.bytes;
        settle_locked(entries_.at(id));
        return {};
    }
    mover_.release(id, to);
    const bool corrupt = o == TierMover::Outcome::corrupt;
    if (corrupt) mover_.release(id, m.from);
    std::lock_guard<std::mutex> lk(mu_);
    Entry& e = entries_.at(id);
    used_[at(to)] -= m.bytes;
    if (down) leaving_[at(m.from)] -= m.bytes;
    const std::string where = id + ": " + tier_name(m.from) + " to " + tier_name(to);
    if (corrupt) {
        used_[at(m.from)] -= m.bytes;
        e.tier = Tier::missing;
        e.verified = false;
        e.failed = true;
        ++counts_.replays;
        settle_locked(e);
        return fail(TierStatus::replay_required, where + ": source failed verification (" + error + ")");
    }
    const bool canceled = o == TierMover::Outcome::canceled;
    ++(canceled ? counts_.cancels : counts_.failures);
    settle_locked(e);
    return canceled ? fail(TierStatus::canceled, where + " canceled; the context stays")
                    : fail(TierStatus::move_failed, where + " failed (" + error + "); the context stays");
}

TierResult TierManager::admit(const ContextId& id, Tier tier, uint64_t bytes, Generation generation,
                              TierLease& writer) {
    if (!valid_context_id(id)) return fail(TierStatus::invalid_argument, "invalid context id");
    if (!resident(tier) || bytes == 0) return fail(TierStatus::invalid_argument, id + ": admit needs a tier and bytes");
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = entries_.find(id);
        fresh = it == entries_.end();
        if (!fresh && (it->second.tier != Tier::missing || it->second.moving))
            return fail(TierStatus::context_exists, id + " is already admitted");
        if (!fresh && generation < it->second.generation)
            return fail(TierStatus::stale_generation, id + ": generation " + std::to_string(generation) +
                                                          " is older than " + std::to_string(it->second.generation));
        Entry& e = entries_[id];
        e.moving = true;  // materializing: nobody moves, leases or removes it meanwhile
        e.target = tier;
    }
    if (TierResult r = reserve(tier, bytes); !r) {
        std::lock_guard<std::mutex> lk(mu_);
        if (fresh) {
            entries_.erase(id);
            settled_.notify_all();
        } else {
            settle_locked(entries_.at(id));
        }
        return r;
    }
    std::lock_guard<std::mutex> lk(mu_);
    Entry& e = entries_.at(id);
    e.tier = tier;
    e.generation = generation;
    e.bytes = bytes;
    e.verified = false;
    e.failed = false;
    e.writer = true;
    e.readers = 0;
    e.leases = {{++lease_seq_, true}};
    e.used_at = ++clock_;
    writer = TierLease{id, lease_seq_, true, generation};
    settle_locked(e);
    return {};
}

TierResult TierManager::move(const ContextId& id, Tier to, Generation expected) {
    if (!resident(to)) return fail(TierStatus::invalid_argument, id + ": a move goes to GPU, HOST or DISK");
    uint64_t bytes = 0;
    bool down = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        Entry* e = nullptr;
        if (TierResult r = check_locked(id, expected, e); !r) return r;
        if (e->tier == to) return {};
        if (e->tier == Tier::missing) return fail(TierStatus::replay_required, id + " has no binary state");
        if (e->writer || e->readers)
            return fail(TierStatus::in_use, id + ": " + std::to_string(e->readers) + " reader(s)" +
                                                (e->writer ? " and a writer" : "") + " pin it");
        e->moving = true;
        e->target = to;
        e->cancel = std::make_shared<std::atomic<bool>>(false);
        bytes = e->bytes;
        down = at(to) > at(e->tier);
        if (down) leaving_[at(e->tier)] += bytes;
    }
    if (TierResult r = reserve(to, bytes); !r) {
        std::lock_guard<std::mutex> lk(mu_);
        Entry& e = entries_.at(id);
        if (down) leaving_[at(e.tier)] -= bytes;
        settle_locked(e);
        return r;
    }
    return transfer(id, to);
}

TierResult TierManager::cancel(const ContextId& id) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = entries_.find(id);
    if (it == entries_.end()) return fail(TierStatus::unknown_context, "no context " + id);
    if (!it->second.cancel) return fail(TierStatus::invalid_argument, id + " has no move in flight");
    it->second.cancel->store(true);
    return {};
}

TierResult TierManager::acquire(const ContextId& id, Generation expected, bool writer, TierLease& out) {
    std::lock_guard<std::mutex> lk(mu_);
    Entry* e = nullptr;
    if (TierResult r = check_locked(id, expected, e); !r) return r;
    if (e->tier == Tier::missing) return fail(TierStatus::replay_required, id + " has no binary state");
    if (writer && e->writer) return fail(TierStatus::in_use, id + " already has a writer");
    if (writer) {
        e->writer = true;
        ++e->generation;
        e->verified = false;  // the state is about to change
    } else {
        ++e->readers;
    }
    e->leases.emplace_back(++lease_seq_, writer);
    e->used_at = ++clock_;
    out = TierLease{id, lease_seq_, writer, e->generation};
    return {};
}

TierResult TierManager::release(const TierLease& lease) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = entries_.find(lease.context);
    if (it == entries_.end()) return fail(TierStatus::unknown_context, "no context " + lease.context);
    Entry& e = it->second;
    const auto held = std::find_if(e.leases.begin(), e.leases.end(), [&](const auto& l) { return l.first == lease.id; });
    if (held == e.leases.end())
        return fail(TierStatus::invalid_argument, lease.context + ": lease " + std::to_string(lease.id) + " is not held");
    if (held->second) e.writer = false;
    else --e.readers;
    e.leases.erase(held);
    e.used_at = ++clock_;
    return {};
}

TierResult TierManager::remove(const ContextId& id, Generation expected) {
    Tier tier = Tier::missing;
    {
        std::lock_guard<std::mutex> lk(mu_);
        Entry* e = nullptr;
        if (TierResult r = check_locked(id, expected, e); !r) return r;
        if (e->writer || e->readers) return fail(TierStatus::in_use, id + ": leases pin it");
        e->moving = true;
        tier = e->tier;
    }
    if (resident(tier)) mover_.release(id, tier);
    std::lock_guard<std::mutex> lk(mu_);
    if (resident(tier)) used_[at(tier)] -= entries_.at(id).bytes;
    entries_.erase(id);
    settled_.notify_all();
    return {};
}

bool TierManager::info(const ContextId& id, TierInfo& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    const Entry& e = it->second;
    out = TierInfo{e.tier, ContextState::cold, e.generation, e.bytes, e.readers, e.writer, e.moving, e.verified};
    if (e.moving && e.tier == Tier::missing) out.state = ContextState::materializing;
    else if (e.moving && resident(e.target) && at(e.target) < at(e.tier)) out.state = ContextState::restoring;
    else if (e.tier == Tier::gpu) out.state = e.moving ? ContextState::cooling : ContextState::hot;
    else if (e.tier == Tier::host) out.state = ContextState::warm;
    else if (e.tier == Tier::missing && e.failed) out.state = ContextState::failed;
    return true;  // DISK, or a dropped copy: COLD (a replay-only marker when missing)
}

TierManager::Stats TierManager::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = counts_;
    for (size_t i = 0; i < 3; ++i) {
        s.used[i] = used_[i];
        s.budget[i] = budget_[i];
    }
    s.contexts = entries_.size();
    return s;
}

}  // namespace basalt::context
