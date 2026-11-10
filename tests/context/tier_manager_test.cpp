#include "basalt/context/tier_manager.hpp"
#include "basalt/core/host_arena.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace basalt::context;

namespace {
void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
void ok(const TierResult& r, const char* message) {
    if (!r) { std::fprintf(stderr, "FAIL: %s: %s %s\n", message, tier_status_name(r.status), r.reason.c_str()); std::abort(); }
}
bool has(const TierResult& r, TierStatus s, const char* text = "") {
    return r.status == s && r.reason.find(text) != std::string::npos;
}

constexpr uint64_t U = 4096;  // one context's bytes in every test

// Copies live in one fixed range per tier, sized to that tier's budget and placed by core::HostArena: a manager that
// let a tier's copies outgrow its budget makes an allocation here fail.
class Mover : public TierMover {
public:
    using Hook = std::function<Outcome(const TierMove&, const std::atomic<bool>&)>;
    explicit Mover(TierBudget b) {
        const uint64_t sizes[3] = {b.gpu, b.host, b.disk};
        for (size_t i = 0; i < 3; ++i) {
            base_[i] = static_cast<uint8_t*>(::operator new(sizes[i] ? sizes[i] : 1, std::align_val_t(4096)));
            if (sizes[i]) check(arena_[i].reset(base_[i], sizes[i]), "tier range");
        }
    }
    ~Mover() override {
        for (auto* p : base_) ::operator delete(p, std::align_val_t(4096));
    }
    static uint8_t byte_of(const ContextId& c, size_t i) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char ch : c) h = (h ^ ch) * 1099511628211ull;
        return static_cast<uint8_t>((h >> 24) + i * 131);
    }
    bool materialize(const ContextId& c, Tier t, uint64_t n) {
        auto* p = static_cast<uint8_t*>(arena_[size_t(t)].allocate(n));
        if (!p) return false;
        for (size_t i = 0; i < n; ++i) p[i] = byte_of(c, i);
        std::lock_guard<std::mutex> lk(mu_);
        copies_[{c, int(t)}] = p;
        return true;
    }
    uint8_t* where(const ContextId& c, Tier t) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = copies_.find({c, int(t)});
        return it == copies_.end() ? nullptr : it->second;
    }
    size_t copies_of(const ContextId& c) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (int t = 0; t < 3; ++t) n += copies_.count({c, t});
        return n;
    }
    bool intact(const uint8_t* p, const ContextId& c, uint64_t n) {
        for (size_t i = 0; i < n; ++i)
            if (p[i] != byte_of(c, i)) return false;
        return true;
    }
    uint64_t arena_used(Tier t) const { return arena_[size_t(t)].stats().used; }
    void pin(const ContextId& c, int d) { std::lock_guard<std::mutex> lk(mu_); pinned_[c] += d; }

    Outcome copy(const TierMove& m, const std::atomic<bool>& cancel, Digest& digest, std::string& error) override {
        uint8_t* src = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const auto it = copies_.find({m.context, int(m.from)});
            src = it == copies_.end() ? nullptr : it->second;
            if (pinned_[m.context]) ++leased_moves;
            if (copies_.count({m.context, int(m.to)})) ++stale_targets;
        }
        if (!src) { ++bad_source; error = "no source copy"; return Outcome::failed; }
        if (!intact(src, m.context, m.bytes)) ++bad_source;
        auto* dst = static_cast<uint8_t*>(arena_[size_t(m.to)].allocate(m.bytes));
        if (!dst) { ++full; error = "tier range full"; return Outcome::failed; }
        {
            std::lock_guard<std::mutex> lk(mu_);
            copies_[{m.context, int(m.to)}] = dst;
        }
        std::memcpy(dst, src, m.bytes);
        if (hook) {
            const Outcome o = hook(m, cancel);
            if (o != Outcome::done) { error = "injected"; return o; }
        }
        digest = Digest{};
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < m.bytes; ++i) h = (h ^ dst[i]) * 1099511628211ull;
        std::memcpy(digest.data(), &h, sizeof(h));
        return Outcome::done;
    }
    void release(const ContextId& c, Tier t) override {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = copies_.find({c, int(t)});
        if (it == copies_.end()) return;
        check(arena_[size_t(t)].release(it->second), "release a copy once");
        copies_.erase(it);
    }

    Hook hook;
    std::atomic<uint64_t> full{0}, bad_source{0}, leased_moves{0}, stale_targets{0};

private:
    uint8_t* base_[3] = {};
    basalt::core::HostArena arena_[3];
    std::mutex mu_;
    std::map<std::pair<ContextId, int>, uint8_t*> copies_;
    std::map<ContextId, int> pinned_;
};

// Admits `id` in `t`, materializes it there and lets the writer go.
void put(TierManager& tm, Mover& mv, const ContextId& id, Tier t, Generation g = 1) {
    TierLease w;
    ok(tm.admit(id, t, U, g, w), "admit");
    check(mv.materialize(id, t, U), "an admitted context has its space");
    ok(tm.release(w), "release the materializing writer");
}
TierInfo info_of(const TierManager& tm, const ContextId& id) {
    TierInfo i;
    check(tm.info(id, i), "known context");
    return i;
}

void basics() {
    const TierBudget b{2 * U, 2 * U, 2 * U};
    Mover mv(b);
    TierManager tm(mv, b);
    TierLease w;
    check(has(tm.admit("bad id", Tier::gpu, U, 1, w), TierStatus::invalid_argument), "context id checked");
    check(has(tm.admit("big", Tier::gpu, 3 * U, 1, w), TierStatus::over_budget, "budget"), "larger than the tier");
    ok(tm.admit("a", Tier::gpu, U, 1, w), "admit");
    check(w.writer && w.generation == 1, "admission holds the writer");
    check(has(tm.admit("a", Tier::gpu, U, 1, w), TierStatus::context_exists), "admitted once");
    check(mv.materialize("a", Tier::gpu, U), "materialize");
    check(has(tm.move("a", Tier::host, 1), TierStatus::in_use, "writer"), "materializing context cannot move");
    ok(tm.release(w), "release");
    check(has(tm.release(w), TierStatus::invalid_argument), "a lease is released once");
    check(has(tm.move("a", Tier::host, 2), TierStatus::stale_generation, "generation 2"), "fenced by generation");
    ok(tm.move("a", Tier::host, 1), "demote");
    TierInfo i = info_of(tm, "a");
    check(i.tier == Tier::host && i.state == ContextState::warm && i.verified, "warm with a digest");
    check(!mv.where("a", Tier::gpu) && mv.intact(mv.where("a", Tier::host), "a", U), "one complete copy");
    ok(tm.move("a", Tier::disk, 1), "to disk");
    check(info_of(tm, "a").state == ContextState::cold, "cold");
    ok(tm.move("a", Tier::gpu, 1), "restore verifies and lands");
    check(info_of(tm, "a").state == ContextState::hot && mv.copies_of("a") == 1, "hot again");
    TierLease wr, rd, other;
    ok(tm.acquire("a", 1, true, wr), "writer");
    check(wr.generation == 2 && !info_of(tm, "a").verified, "a writer starts a generation and voids the digest");
    check(has(tm.acquire("a", 2, true, other), TierStatus::in_use), "one writer");
    ok(tm.acquire("a", 2, false, rd), "a reader alongside");
    check(has(tm.move("a", Tier::host, 2), TierStatus::in_use, "reader"), "a lease pins the context");
    check(has(tm.remove("a", 2), TierStatus::in_use), "no removal under a lease");
    ok(tm.release(rd), "release reader");
    ok(tm.release(wr), "release writer");
    check(has(tm.move("a", Tier::host, 1), TierStatus::stale_generation), "the old generation stays fenced");
    ok(tm.move("a", Tier::host, 2), "move under the new generation");
    ok(tm.remove("a", 2), "remove");
    check(!tm.info("a", i) && mv.copies_of("a") == 0, "every copy freed");
    const auto s = tm.stats();
    check(s.used[0] == 0 && s.used[1] == 0 && s.used[2] == 0 && s.contexts == 0 && s.moves == 4, "accounting");
    check(mv.full == 0 && mv.bad_source == 0 && mv.stale_targets == 0, "mover never refused");
}

void verification_and_replay() {
    const TierBudget b{2 * U, 2 * U, 2 * U};
    Mover mv(b);
    TierManager tm(mv, b);
    put(tm, mv, "a", Tier::gpu);
    ok(tm.move("a", Tier::host, 1), "demote records the digest");
    mv.where("a", Tier::host)[7] ^= 1;  // the host copy rots
    const auto r = tm.move("a", Tier::gpu, 1);
    check(has(r, TierStatus::replay_required, "verification"), "a copy off its digest says replay");
    const TierInfo i = info_of(tm, "a");
    check(i.tier == Tier::missing && i.state == ContextState::failed && !i.verified, "binary state dropped");
    check(mv.copies_of("a") == 0 && tm.stats().used[0] == 0 && tm.stats().used[1] == 0, "both copies freed");
    TierLease l;
    check(has(tm.move("a", Tier::host, 1), TierStatus::replay_required), "no move without state");
    check(has(tm.acquire("a", 1, false, l), TierStatus::replay_required), "no lease without state");
    check(has(tm.admit("a", Tier::gpu, U, 0, l), TierStatus::stale_generation), "replay cannot go back a generation");
    put(tm, mv, "a", Tier::gpu, 1);  // rebuilt from the transcript
    ok(tm.move("a", Tier::disk, 1), "to disk");
    mv.hook = [](const TierMove& m, const std::atomic<bool>&) {
        return m.from == Tier::disk ? TierMover::Outcome::corrupt : TierMover::Outcome::done;
    };
    check(has(tm.move("a", Tier::gpu, 1), TierStatus::replay_required), "the mover's own check failing says replay");
    mv.hook = nullptr;
    check(tm.stats().replays == 2 && mv.copies_of("a") == 0, "replays counted, nothing left behind");
}

void cancel_and_failure() {
    const TierBudget b{2 * U, 2 * U, 2 * U};
    Mover mv(b);
    TierManager tm(mv, b);
    put(tm, mv, "a", Tier::gpu);
    mv.hook = [](const TierMove&, const std::atomic<bool>&) { return TierMover::Outcome::failed; };
    check(has(tm.move("a", Tier::host, 1), TierStatus::move_failed, "stays"), "a failed copy");
    TierInfo i = info_of(tm, "a");
    check(i.tier == Tier::gpu && i.state == ContextState::hot && !i.moving, "previous state kept");
    check(mv.intact(mv.where("a", Tier::gpu), "a", U) && !mv.where("a", Tier::host), "source whole, partial freed");
    check(tm.stats().used[1] == 0 && mv.arena_used(Tier::host) == 0 && tm.stats().failures == 1, "reservation back");
    std::atomic<bool> entered{false};
    mv.hook = [&](const TierMove&, const std::atomic<bool>& cancel) {
        entered = true;
        while (!cancel.load()) std::this_thread::yield();
        return TierMover::Outcome::canceled;
    };
    TierResult moved;
    std::thread t([&] { moved = tm.move("a", Tier::host, 1); });
    while (!entered.load()) std::this_thread::yield();
    i = info_of(tm, "a");
    check(i.moving && i.state == ContextState::cooling, "move in flight is visible");
    TierLease l;
    check(has(tm.acquire("a", 1, false, l), TierStatus::moving), "no lease during a move");
    check(has(tm.move("a", Tier::disk, 1), TierStatus::moving), "one move at a time");
    check(has(tm.remove("a", 1), TierStatus::moving), "no removal during a move");
    ok(tm.cancel("a"), "cancel");
    t.join();
    check(moved.status == TierStatus::canceled, "the move reports the cancel");
    i = info_of(tm, "a");
    check(i.tier == Tier::gpu && !i.moving && mv.intact(mv.where("a", Tier::gpu), "a", U), "canceled move left it whole");
    check(!mv.where("a", Tier::host) && tm.stats().used[1] == 0 && tm.stats().cancels == 1, "partial copy freed");
    check(has(tm.cancel("a"), TierStatus::invalid_argument), "nothing to cancel");
    mv.hook = nullptr;
    ok(tm.move("a", Tier::host, 1), "the next move goes through");
}

void evacuate_and_reuse() {
    const TierBudget b{U, 2 * U, 2 * U};
    Mover mv(b);
    TierManager tm(mv, b);
    for (const char* id : {"a", "b"}) {
        put(tm, mv, id, Tier::gpu);
        ok(tm.move(id, Tier::host, 1), "park in host");
    }
    uint8_t* const a_host = mv.where("a", Tier::host);
    put(tm, mv, "c", Tier::gpu);
    ok(tm.move("c", Tier::host, 1), "host full: its oldest idle context goes to disk");
    check(mv.where("c", Tier::host) == a_host, "the incoming context takes the space the evacuated one left");
    TierInfo i = info_of(tm, "a");
    check(i.tier == Tier::disk && i.verified && mv.intact(mv.where("a", Tier::disk), "a", U), "evacuated whole");
    check(tm.stats().evictions == 1 && tm.stats().used[1] == 2 * U && mv.arena_used(Tier::host) == 2 * U, "at budget");
    TierLease rb, rd;
    ok(tm.acquire("b", 1, false, rb), "lease b");
    mv.pin("b", 1);
    put(tm, mv, "d", Tier::gpu);
    uint8_t* const c_host = mv.where("c", Tier::host);
    ok(tm.move("d", Tier::host, 1), "a leased context is passed over");
    check(info_of(tm, "b").tier == Tier::host && info_of(tm, "c").tier == Tier::disk, "the idle one went instead");
    check(mv.where("d", Tier::host) == c_host, "and its space is reused");
    ok(tm.acquire("d", 1, false, rd), "lease d");
    mv.pin("d", 1);
    put(tm, mv, "e", Tier::gpu);
    check(has(tm.move("e", Tier::host, 1), TierStatus::over_budget, "idle"), "nothing idle to evacuate");
    check(info_of(tm, "e").tier == Tier::gpu && !info_of(tm, "e").moving, "the refused move left it in place");
    mv.pin("b", -1);
    ok(tm.release(rb), "unpin b");
    ok(tm.move("e", Tier::host, 1), "b goes to disk, the disk drops its oldest copy");
    i = info_of(tm, "a");
    check(i.tier == Tier::missing && i.state == ContextState::cold && mv.copies_of("a") == 0, "replay-only marker");
    check(has(tm.move("a", Tier::gpu, 1), TierStatus::replay_required), "a dropped context replays");
    check(tm.stats().drops == 1 && info_of(tm, "b").tier == Tier::disk, "drop counted");
    ok(tm.move("c", Tier::gpu, 1), "restore from disk into the GPU space e left");
    check(info_of(tm, "c").state == ContextState::hot && mv.intact(mv.where("c", Tier::gpu), "c", U), "restored whole");
    mv.pin("d", -1);
    ok(tm.release(rd), "unpin d");
    const auto s = tm.stats();
    for (int t = 0; t < 3; ++t)
        check(s.used[t] == mv.arena_used(Tier(t)) && s.used[t] <= s.budget[t], "accounting matches the ranges");
    check(mv.full == 0 && mv.bad_source == 0 && mv.leased_moves == 0 && mv.stale_targets == 0, "valid moves only");
}

// Threads admit, move, lease and remove their own contexts in tiers too small for all of them, so every request
// evicts other threads' idle contexts; another thread cancels moves at random.
void contended() {
    constexpr int threads = 8, per = 6, rounds = 400;
    const TierBudget b{4 * U, 8 * U, uint64_t(threads * per) * U};
    Mover mv(b);
    TierManager tm(mv, b);
    auto name = [](int t, int k) { return "w" + std::to_string(t) + "-" + std::to_string(k); };
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> moved{0}, refused{0};
    std::thread canceler([&] {
        std::mt19937 r(5);
        while (!stop.load()) {
            (void) tm.cancel(name(int(r() % threads), int(r() % per)));
            std::this_thread::yield();
        }
    });
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t)
        workers.emplace_back([&, t] {
            std::mt19937 r(uint32_t(1000 + t));
            Generation gen[per];
            bool live[per] = {};
            for (auto& g : gen) g = 1;
            for (int i = 0; i < rounds; ++i) {
                const int k = int(r() % per);
                const std::string id = name(t, k);
                if (!live[k]) {
                    const Tier where = r() % 2 ? Tier::gpu : Tier::host;
                    TierLease w;
                    const TierResult a = tm.admit(id, where, U, gen[k], w);
                    if (!a) {
                        check(a.status == TierStatus::over_budget, "admission refuses only for room");
                        ++refused;
                        continue;
                    }
                    check(mv.materialize(id, where, U), "admitted bytes have their space");
                    ok(tm.release(w), "materialized");
                    live[k] = true;
                    continue;
                }
                const unsigned op = r() % 4;
                if (op == 0) {
                    const TierResult m = tm.move(id, static_cast<Tier>(r() % 3), gen[k]);
                    if (m) ++moved;
                    else check(m.status == TierStatus::over_budget || m.status == TierStatus::moving ||
                                   m.status == TierStatus::canceled, "a move fails only for room, a move or a cancel");
                } else if (op == 1) {
                    TierLease l;
                    const bool writer = r() % 3 == 0;
                    const TierResult a = tm.acquire(id, gen[k], writer, l);
                    if (!a) {
                        check(a.status == TierStatus::moving, "a lease is refused only during a move");
                        continue;
                    }
                    mv.pin(id, 1);
                    const TierInfo inf = info_of(tm, id);
                    const uint8_t* p = mv.where(id, inf.tier);
                    check(!inf.moving && p && mv.intact(p, id, U), "a leased context is whole where it is");
                    if (writer) gen[k] = l.generation;
                    std::this_thread::yield();
                    mv.pin(id, -1);
                    ok(tm.release(l), "release");
                } else if (op == 2) {
                    const TierResult d = tm.remove(id, gen[k]);
                    if (d) live[k] = false;
                    else check(d.status == TierStatus::moving, "removal is refused only during a move");
                }
            }
        });
    for (auto& w : workers) w.join();
    stop = true;
    canceler.join();
    check(mv.full == 0 && mv.bad_source == 0 && mv.leased_moves == 0 && mv.stale_targets == 0,
          "every copy had its space and a whole, unleased source");
    const auto s = tm.stats();
    check(s.replays == 0 && s.drops == 0 && s.failures == 0, "no state lost under contention");
    const uint64_t ranges[3] = {mv.arena_used(Tier::gpu), mv.arena_used(Tier::host), mv.arena_used(Tier::disk)};
    uint64_t in[3] = {};
    for (int t = 0; t < threads; ++t)
        for (int k = 0; k < per; ++k) {
            const std::string id = name(t, k);
            TierInfo inf;
            if (!tm.info(id, inf)) {
                check(mv.copies_of(id) == 0, "a removed context left no copy");
                continue;
            }
            const uint8_t* p = mv.where(id, inf.tier);
            check(!inf.moving && !inf.readers && !inf.writer, "quiescent");
            check(mv.copies_of(id) == 1 && p && mv.intact(p, id, U), "one whole copy, where the manager says");
            in[size_t(inf.tier)] += inf.bytes;
            ok(tm.remove(id, inf.generation), "remove");
        }
    for (int t = 0; t < 3; ++t) check(s.used[t] == in[t] && in[t] == ranges[t], "accounting equals the copies");
    const auto end = tm.stats();
    check(end.used[0] == 0 && end.used[1] == 0 && end.used[2] == 0 && end.contexts == 0, "all freed");
    check(mv.arena_used(Tier::gpu) == 0 && mv.arena_used(Tier::host) == 0 && mv.arena_used(Tier::disk) == 0,
          "every range empty");
    std::printf("tier_manager contended: %llu moves (%llu asked), %llu evictions, %llu canceled, %llu refused\n",
                (unsigned long long) s.moves, (unsigned long long) moved.load(), (unsigned long long) s.evictions,
                (unsigned long long) s.cancels, (unsigned long long) refused.load());
}

}  // namespace

int main() {
    basics();
    verification_and_replay();
    cancel_and_failure();
    evacuate_and_reuse();
    contended();
    std::printf("tier_manager_test: ok\n");
    return 0;
}
