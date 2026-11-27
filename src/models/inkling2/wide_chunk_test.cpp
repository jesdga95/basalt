// src/models/inkling2/wide_chunk_test.cpp - a long read's wider chunk (wide_chunk.hpp): taken, backed and given its tap in
// order, and given back once at the end; no room leaves everything untouched; a mapping or a tap that fails gives the room
// back; a rollback whose unmap fails, a mapping or a tap that failed other than out of memory, and a tap release that fails
// keep the room taken and never open again; a take or a return that does not put the copies back (or whose last wait for the
// card failed) fails the read; a read whose drain was not confirmed touches nothing; a throw during the end's cleanup does not
// leave it. And the copies a take sends back are found again in a placement that moved before it (a cache grow, a
// request-start swap), with a return that stops part-way told apart from a full one.
#include "models/inkling2/wide_chunk.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

using namespace basalt::models::inkling2;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

struct FakeOps {
    Take take_as = Take::Taken;
    WideStep map_as = WideStep::Done, tap_as = WideStep::Done;
    bool unmappable = true, tap_frees = true, restores = true, give_throws = false;
    uint64_t taken = 0, given = 0;
    int takes = 0, gives = 0, mapped = 0, taps = 0, calls = 0;
    Take take(uint64_t bytes) {
        ++calls;
        if (take_as == Take::Taken) {
            ++takes;
            taken += bytes;
        }
        return take_as;
    }
    bool give(uint64_t bytes) {
        ++calls;
        if (give_throws) throw std::bad_alloc();
        ++gives;
        given += bytes;
        return restores;
    }
    WideStep map() {
        ++calls;
        if (map_as == WideStep::Done) ++mapped;
        return map_as;
    }
    bool unmap() {
        ++calls;
        if (!unmappable) return false;
        --mapped;
        return true;
    }
    WideStep tap_on() {
        ++calls;
        if (tap_as == WideStep::Done) ++taps;
        return tap_as;
    }
    bool tap_off() {
        ++calls;
        if (!tap_frees) return false;
        --taps;
        return true;
    }
    bool even() const { return taken == given && takes == gives && mapped == 0 && taps == 0; }
};

// A card's copies, the highest ranked first, as the model keeps them: a demote drops the lowest, a promote appends while
// it has room.
struct FakePlacement {
    std::vector<size_t> held;
    std::vector<uint8_t> has = std::vector<uint8_t>(64, 0);
    void set(std::vector<size_t> v) {
        std::fill(has.begin(), has.end(), 0);
        held = std::move(v);
        for (size_t p : held) has[p] = 1;
    }
    void demote(size_t n) {
        for (size_t i = 0; i < n && !held.empty(); ++i) {
            has[held.back()] = 0;
            held.pop_back();
        }
    }
    void promote(const std::vector<size_t>& pairs, size_t room) {
        for (size_t p : pairs) {
            if (room == 0) break;
            held.push_back(p);
            has[p] = 1;
            --room;
        }
    }
};
}  // namespace

int main() {
    {
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide && w.wide(), "room, mapping and tap: wide");
        expect(o.takes == 1 && o.mapped == 1 && o.taps == 1 && o.gives == 0, "wide: taken, mapped, tapped, nothing given yet");
        expect(w.open(1000, true) == WideOpen::Narrow && o.takes == 1, "a second open while wide takes nothing");
        expect(w.close(true) == WideClose::Done && !w.wide() && o.even() && o.given == 1000, "the end gives back what was taken");
        expect(w.close(true) == WideClose::Done && o.gives == 1, "a second close gives nothing more");
    }
    {
        FakeOps o;
        o.take_as = Take::Refused;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Narrow && !w.wide(), "no room: narrow");
        expect(o.calls == 1 && o.gives == 0 && o.mapped == 0, "no room: nothing else called");
    }
    {
        FakeOps o;
        o.take_as = Take::Failed;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Failed && !w.wide() && !w.stuck(),
               "a take that could not put the copies back fails the read");
    }
    {
        FakeOps o;
        o.map_as = WideStep::Short;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Narrow && o.even() && o.given == 1000,
               "the mapping fails: the room comes back, narrow");
    }
    {
        FakeOps o;
        o.map_as = WideStep::Short;
        o.restores = false;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Failed, "the mapping fails and the copies do not all return: the read fails");
    }
    {
        FakeOps o;
        o.tap_as = WideStep::Short;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Narrow && o.even() && o.given == 1000,
               "the tap fails: unmapped, the room comes back");
    }
    {   // the tap fails and so does the unmap of the rollback
        FakeOps o;
        o.tap_as = WideStep::Short;
        o.unmappable = false;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Failed && w.stuck(), "tap and unmap fail: the read fails, stuck");
        expect(o.gives == 0 && o.mapped == 1, "tap and unmap fail: the mapping and its room stay taken");
        const int calls = o.calls;
        expect(w.open(1000, true) == WideOpen::Failed && o.calls == calls, "stuck: never opens again");
        expect(w.close(true) == WideClose::Stuck && o.calls == calls, "stuck: the end touches nothing");
    }
    {
        FakeOps o;
        o.tap_as = WideStep::Short;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, false) == WideOpen::Wide && o.taps == 0, "no drafter: wide without a tap");
        expect(w.close(true) == WideClose::Done && o.even(), "no drafter: the end gives back");
    }
    {
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide, "wide");
        o.unmappable = false;
        expect(w.close(true) == WideClose::Stuck && o.gives == 0 && w.stuck(),
               "an unmap that fails at the end keeps the room taken");
    }
    {
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide, "wide");
        o.restores = false;
        expect(w.close(true) == WideClose::RestoreFailed && o.mapped == 0 && o.gives == 1,
               "a return short of the copies is told");
    }
    {   // a read that failed part-way and whose drain was not confirmed
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide, "wide");
        const int calls = o.calls;
        expect(w.close(false) == WideClose::Stuck && o.calls == calls && w.stuck(),
               "drain not confirmed: nothing released or touched");
    }
    {   // a read that threw part-way, drained: the end's cleanup throws
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide, "wide");
        o.give_throws = true;
        bool thrown = false;
        WideClose c = WideClose::Done;
        try {
            c = w.close(true);
        } catch (...) {
            thrown = true;
        }
        expect(!thrown && c == WideClose::Stuck, "a throw in the end's cleanup stays inside it");
    }
    {   // a mapping that failed some other way than out of memory may have left a mapping behind
        FakeOps o;
        o.map_as = WideStep::Broken;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Failed && w.stuck() && o.gives == 0, "a broken mapping: stuck, no refund");
    }
    {   // a tap that failed some other way than out of memory
        FakeOps o;
        o.tap_as = WideStep::Broken;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Failed && w.stuck(), "a broken tap: stuck");
        expect(o.gives == 0 && o.mapped == 1, "a broken tap: the mapping and the room stay taken");
    }
    {   // the tap's release fails at the end
        FakeOps o;
        WideChunk<FakeOps> w(o);
        expect(w.open(1000, true) == WideOpen::Wide, "wide");
        o.tap_frees = false;
        expect(w.close(true) == WideClose::Stuck && w.stuck(), "a tap release that fails: stuck");
        expect(o.mapped == 1 && o.gives == 0, "a tap release that fails: nothing unmapped or refunded");
    }
    {   // the copies sent back, after the placement moved before the read
        FakePlacement p;
        p.set({7, 3, 9, 1, 4, 12, 30, 5});
        p.demote(2);                                            // a cache grow before the read
        std::swap(p.held[1], p.held.back());                     // a request-start swap
        p.has[p.held.back()] = 0;
        p.held.back() = 20;
        p.has[20] = 1;
        const std::vector<size_t> before = p.held;              // in rank order
        std::vector<uint8_t> kept(64, 0);
        std::vector<size_t> out;
        out.reserve(before.size());
        const size_t cap = out.capacity();
        p.demote(3);                                            // take
        left_out(before, p.held, kept, out);
        expect(out == std::vector<size_t>(before.end() - 3, before.end()),
               "the copies sent back: the three lowest, in rank order");
        expect(out.capacity() == cap && std::all_of(kept.begin(), kept.end(), [](uint8_t k) { return k == 0; }),
               "left_out allocates nothing and leaves its scratch clean");
        FakePlacement part = p;
        part.promote(out, 1);                                   // a return that stops after one
        expect(!restored(out, [&](size_t q) { return part.has[q] != 0; }, true), "a return that stops part-way is not restored");
        p.promote(out, out.size());
        expect(restored(out, [&](size_t q) { return p.has[q] != 0; }, true) && p.held == before,
               "a full return: the set as before");
        expect(!restored(out, [&](size_t q) { return p.has[q] != 0; }, false),
               "every copy published but the last wait for the card failed: not restored");
    }
    if (failures) return 1;
    std::printf("inkling2_wide_chunk_test: ok\n");
    return 0;
}
