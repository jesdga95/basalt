// src/context/tier_manager.cpp - see include/basalt/context/tier_manager.hpp.
#include "basalt/context/tier_manager.hpp"

#include <algorithm>
#include <chrono>

namespace basalt::context {
namespace {
constexpr size_t slot(Tier t) { return static_cast<size_t>(t); }
}  // namespace

int64_t TierManager::system_clock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

TierManager::TierManager(Budget budget, Clock clock)
    : budget_(budget), clock_(clock ? std::move(clock) : &system_clock_ms) {}

TierManager::Budget TierManager::budget() const { return budget_; }

namespace {
uint64_t budget_of(const TierManager::Budget& b, Tier t) {
    switch (t) {
        case Tier::gpu: return b.gpu_bytes;
        case Tier::host: return b.host_bytes;
        case Tier::disk: return b.disk_bytes;
        case Tier::missing: return 0;
    }
    return 0;
}
}  // namespace

void TierManager::track(const ContextId& ctx, Tier tier, uint64_t bytes, Generation generation) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) {
        entries_[ctx] = Entry{Placement{tier, bytes, generation, clock_()}, {}};
        used_[slot(tier)] += bytes;
        ++count_[slot(tier)];
        return;
    }
    // Fenced like a move: a stale report cannot rewrite the ledger.
    if (generation <= it->second.at.generation) return;
    used_[slot(it->second.at.tier)] -= it->second.at.bytes;
    --count_[slot(it->second.at.tier)];
    it->second.at = Placement{tier, bytes, generation, clock_()};
    used_[slot(tier)] += bytes;
    ++count_[slot(tier)];
}

void TierManager::untrack(const ContextId& ctx) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) return;
    used_[slot(it->second.at.tier)] -= it->second.at.bytes;
    --count_[slot(it->second.at.tier)];
    entries_.erase(it);
}

bool TierManager::known(const ContextId& ctx) const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.count(ctx) > 0;
}

bool TierManager::placement(const ContextId& ctx, Placement& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) return false;
    out = it->second.at;
    return true;
}

Tier TierManager::tier_of(const ContextId& ctx) const {
    Placement p;
    return placement(ctx, p) ? p.tier : Tier::missing;
}

Generation TierManager::next_generation(const ContextId& ctx) const {
    Placement p;
    return placement(ctx, p) ? p.generation + 1 : 1;
}

bool TierManager::fits(Tier tier, uint64_t bytes) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (tier == Tier::missing) return true;
    return used_[slot(tier)] + bytes <= budget_of(budget_, tier);
}

bool TierManager::fits(const ContextId& ctx, Tier to, uint64_t bytes) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (to == Tier::missing) return true;
    const auto it = entries_.find(ctx);
    const uint64_t own = (it != entries_.end() && it->second.at.tier == to) ? it->second.at.bytes : 0;
    return used_[slot(to)] - own + bytes <= budget_of(budget_, to);
}

bool TierManager::commit_move(const ContextId& ctx, Generation generation, Tier to, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) return false;
    if (generation <= it->second.at.generation) return false;   // stale fence
    if (to == Tier::missing && bytes != 0) return false;        // missing holds no bytes
    const Tier from = it->second.at.tier;
    const uint64_t own = from == to ? it->second.at.bytes : 0;
    if (to != Tier::missing && used_[slot(to)] - own + bytes > budget_of(budget_, to)) return false;
    used_[slot(from)] -= it->second.at.bytes;
    --count_[slot(from)];
    it->second.at = Placement{to, bytes, generation, clock_()};
    used_[slot(to)] += bytes;
    ++count_[slot(to)];
    return true;
}

void TierManager::pin(const ContextId& ctx, const std::string& lease) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) return;
    it->second.leases.insert(lease);
}

bool TierManager::unpin(const ContextId& ctx, const std::string& lease) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it == entries_.end()) return false;
    return it->second.leases.erase(lease) > 0;
}

bool TierManager::pinned(const ContextId& ctx) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    return it != entries_.end() && !it->second.leases.empty();
}

void TierManager::touch(const ContextId& ctx) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = entries_.find(ctx);
    if (it != entries_.end()) it->second.at.last_use_ms = clock_();
}

std::vector<TierManager::EvictionCandidate> TierManager::evictable(Tier from, uint64_t bytes_needed) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<EvictionCandidate> out;
    if (from == Tier::missing || bytes_needed == 0) return out;
    for (const auto& [ctx, entry] : entries_) {
        if (entry.at.tier != from || !entry.leases.empty()) continue;
        out.push_back({ctx, entry.at.bytes, entry.at.generation});
    }
    std::sort(out.begin(), out.end(), [this](const EvictionCandidate& a, const EvictionCandidate& b) {
        const int64_t la = entries_.at(a.context_id).at.last_use_ms;
        const int64_t lb = entries_.at(b.context_id).at.last_use_ms;
        if (la != lb) return la < lb;
        return a.context_id < b.context_id;   // a stable order for equal ages
    });
    uint64_t covered = 0;
    size_t keep = 0;
    for (; keep < out.size() && covered < bytes_needed; ++keep) covered += out[keep].bytes;
    out.resize(keep);
    return out;
}

uint64_t TierManager::used(Tier tier) const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_[slot(tier)];
}

TierManager::Stats TierManager::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats s;
    s.used_gpu = used_[slot(Tier::gpu)];
    s.used_host = used_[slot(Tier::host)];
    s.used_disk = used_[slot(Tier::disk)];
    s.budget_gpu = budget_.gpu_bytes;
    s.budget_host = budget_.host_bytes;
    s.budget_disk = budget_.disk_bytes;
    s.contexts_gpu = count_[slot(Tier::gpu)];
    s.contexts_host = count_[slot(Tier::host)];
    s.contexts_disk = count_[slot(Tier::disk)];
    for (const auto& [ctx, entry] : entries_)
        if (!entry.leases.empty()) ++s.pinned;
    return s;
}

}  // namespace basalt::context
