// tests/context/tier_manager_test.cpp - budgets, fencing, leases and eviction order on a virtual clock.
#include "basalt/context/tier_manager.hpp"

#include <cstdio>
#include <string>

namespace {
using namespace basalt::context;
int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) { ++failures; std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

int64_t virtual_now = 1000;
int64_t virtual_clock() { return virtual_now; }

TierManager make() {
    return TierManager{TierManager::Budget{1000, 4000, 10000}, &virtual_clock};
}

void accounting_and_budget() {
    auto t = make();
    t.track("a", Tier::host, 3000, 1);
    t.track("b", Tier::host, 500, 1);
    CHECK(t.used(Tier::host) == 3500);
    CHECK(t.tier_of("a") == Tier::host);
    CHECK(t.tier_of("nobody") == Tier::missing);
    CHECK(!t.fits(Tier::host, 600));            // 3500 + 600 > 4000
    CHECK(t.fits("a", Tier::host, 3500));       // growing in place is judged on the delta (500 + 3500 == 4000)
    CHECK(!t.fits("a", Tier::host, 3501));
    CHECK(t.commit_move("a", 2, Tier::disk, 3000));    // disk budget 10000: fits
    CHECK(t.tier_of("a") == Tier::disk);
    CHECK(t.used(Tier::host) == 500 && t.used(Tier::disk) == 3000);
    CHECK(!t.commit_move("b", 2, Tier::disk, 8000));   // 3000 + 8000 > 10000
    CHECK(t.tier_of("b") == Tier::host);               // refused: nothing moved
    CHECK(t.commit_move("b", 2, Tier::disk, 6000));
    CHECK(t.commit_move("b", 3, Tier::missing, 0));    // releasing state is always a legal target
    CHECK(t.used(Tier::disk) == 3000 && t.tier_of("b") == Tier::missing);
    CHECK(!t.commit_move("b", 4, Tier::missing, 7));   // missing holds no bytes
    t.untrack("b");
    CHECK(!t.known("b"));
    auto st = t.stats();
    CHECK(st.used_host == 0 && st.used_disk == 3000 && st.contexts_disk == 1);
}

void fencing_refuses_stale_moves() {
    auto t = make();
    t.track("a", Tier::host, 100, 5);
    CHECK(t.next_generation("a") == 6);
    CHECK(!t.commit_move("a", 5, Tier::disk, 100));    // at the committed generation: stale
    CHECK(!t.commit_move("a", 4, Tier::disk, 100));    // below it: stale
    CHECK(t.tier_of("a") == Tier::host);
    CHECK(t.commit_move("a", 6, Tier::disk, 100));
    t.track("a", Tier::host, 100, 3);                  // a stale report cannot rewrite the ledger either
    CHECK(t.tier_of("a") == Tier::disk);
    CHECK(!t.known("ghost") && t.next_generation("ghost") == 1);
    CHECK(!t.commit_move("ghost", 1, Tier::host, 1));  // unknown contexts do not materialize
}

void eviction_order() {
    auto t = make();
    virtual_now = 10;
    t.track("old", Tier::host, 100, 1);
    virtual_now = 20;
    t.track("mid", Tier::host, 100, 1);
    virtual_now = 30;
    t.track("new", Tier::host, 100, 1);
    t.pin("mid", "lease-1");
    auto gone = t.evictable(Tier::host, 150);          // "mid" is pinned: old, then new
    CHECK(gone.size() == 2);
    CHECK(gone.size() == 2 && gone[0].context_id == "old" && gone[1].context_id == "new");
    CHECK(gone.size() == 2 && gone[0].generation == 1);
    CHECK(t.evictable(Tier::host, 0).empty());
    t.unpin("mid", "lease-1");
    CHECK(t.evictable(Tier::host, 350).size() == 3);   // everything, oldest first
    auto ev = t.evictable(Tier::host, 350);
    CHECK(ev[0].context_id == "old" && ev[1].context_id == "mid" && ev[2].context_id == "new");
    virtual_now = 25;
    t.touch("old");                                    // old overtakes mid: mid becomes the eldest
    ev = t.evictable(Tier::host, 100);
    CHECK(ev.size() == 1 && ev[0].context_id == "mid");
    CHECK(t.evictable(Tier::disk, 100).empty());
    CHECK(!t.unpin("mid", "lease-1"));                 // already gone
    CHECK(!t.pinned("mid"));
}

void leases_hold_in_place() {
    auto t = make();
    t.track("a", Tier::gpu, 500, 1);
    t.pin("a", "w1");
    t.pin("a", "r1");
    CHECK(t.pinned("a"));
    CHECK(t.unpin("a", "w1"));
    CHECK(t.pinned("a"));                              // the reader still holds it
    CHECK(t.unpin("a", "r1"));
    CHECK(!t.pinned("a"));
    CHECK(t.stats().pinned == 0);
    t.pin("a", "w2");
    CHECK(t.stats().pinned == 1);
    t.untrack("a");                                    // pins go with the context
    CHECK(!t.pinned("a"));
}

}  // namespace

int main() {
    accounting_and_budget();
    fencing_refuses_stale_moves();
    eviction_order();
    leases_hold_in_place();
    if (failures) { std::printf("%d tier_manager test failures\n", failures); return 1; }
    std::printf("tier_manager tests ok\n");
    return 0;
}
