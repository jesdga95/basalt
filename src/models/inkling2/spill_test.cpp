// src/models/inkling2/spill_test.cpp - the spill plan: nothing leaves when nothing is needed, the coldest experts leave
// first, slots are a permutation with the kept experts first, the bytes it frees cover what was asked net of the shared
// staging and the rounding, copies go back until a keep is free, and a resume trims the caches only when it frees half.
#include "models/inkling2/spill.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace basalt::models::inkling2;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    const int L = 4, E = 16;
    const uint64_t gran = 2u << 20;
    const std::vector<uint64_t> stride = {4479120, 4479120, 4479120, 8957952};
    // a profile: layer 0 hottest, expert e of layer l ranked by (l, e)
    std::vector<std::pair<int32_t, int32_t>> rank;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) rank.emplace_back(l, e);
    std::string err;
    SpillPlan p;
    expect(plan_spill(L, E, stride, gran, rank, 0, -1, p, err) && !p.any() && p.device == p.full && p.staging == 0, "no need, no spill");
    for (auto& s : p.slot) expect(s.empty(), "no slot maps without a spill");

    const uint64_t need = 100ull << 20;
    expect(plan_spill(L, E, stride, gran, rank, need, -1, p, err), "a need the experts cover");
    expect(p.any() && p.full >= p.device + need, "the bytes freed cover the need");
    expect(p.n_spill[3] > 0 && p.n_spill[0] == 0, "the coldest (last ranked) layer gives up first");
    for (int l = 0; l < L; ++l) {
        if (p.slot[(size_t) l].empty()) {
            expect(p.n_spill[(size_t) l] == 0, "a layer with no map keeps all");
            continue;
        }
        std::vector<int> seen(E, 0);
        for (int e = 0; e < E; ++e) {
            const int s = p.slot[(size_t) l][(size_t) e];
            expect(s >= 0 && s < E, "slot in range");
            if (s >= 0 && s < E) ++seen[(size_t) s];
        }
        for (int s = 0; s < E; ++s) expect(seen[(size_t) s] == 1, "slots are a permutation");
        // the spilled ones are the highest experts (the coldest by this profile) and take the last slots
        const int keep = E - p.n_spill[(size_t) l];
        for (int e = 0; e < E; ++e) expect((p.slot[(size_t) l][(size_t) e] >= keep) == (e >= keep), "kept experts first, RAM ones last");
        // the layer's own backing and the staging cover its slots and the zeroed guard after them
        const uint64_t own = ((uint64_t) keep * stride[(size_t) l] + gran - 1) / gran * gran;
        expect(own + p.staging >= (uint64_t) E * stride[(size_t) l] + kSlotGuard, "the staging covers the RAM slots and the guard");
    }
    // the diagnostic cap: at most `keep` experts stay
    expect(plan_spill(L, E, stride, gran, rank, 0, 40, p, err) && L * E - p.spilled <= 40, "the keep cap");
    // no profile: the highest numbered experts of every layer in turn
    expect(plan_spill(L, E, stride, gran, {}, 0, L * E - 8, p, err) && p.n_spill[0] == 2 && p.n_spill[3] == 2, "no profile: turns");
    // more than the experts can give
    SpillPlan q;
    expect(!plan_spill(L, E, stride, gran, rank, p.full + 1, -1, q, err) && !err.empty(), "an impossible need fails");
    // the copies' layout: mixed strides, copies sharing a granule, copies over several, reclaim by whole granules
    {
        const uint64_t G = 2u << 20;
        const std::vector<uint64_t> st = {4479120, 4479120, 8957952, 4479120, 8957952, 4479120};
        const CopyLayout c = copy_layout(st, G);
        bool ok = c.at.size() == st.size() && c.at[0] == 0;
        for (size_t k = 0; k < st.size(); ++k) {
            ok = ok && c.at[k] % 256 == 0 && c.end[k] == c.at[k] + st[k];
            if (k > 0) ok = ok && c.at[k] >= c.end[k - 1] && c.at[k] - c.end[k - 1] < 256;
        }
        expect(ok, "copy layout: back to back, 256-byte aligned, no overlap");
        expect(c.pages(0) == 0 && c.pages(1) == 3 && c.bytes() == c.pages((int64_t) st.size()) * G, "copy layout: pages");
        const int64_t n = (int64_t) st.size();
        const uint64_t mapped = c.pages(n);
        expect(c.keep_for(n, mapped, 0) == n, "reclaim nothing: every copy stays");
        expect(c.keep_for(n, mapped, ~0ull) == 0, "reclaim more than all: every copy goes");
        for (uint64_t want = 1; want <= mapped * G; want += G / 3) {
            const int64_t m = c.keep_for(n, mapped, want);
            const uint64_t freed = (mapped - c.pages(m)) * G;
            // enough is freed, and keeping one more copy would not have freed enough
            expect(freed >= want && (m == n || (mapped - c.pages(m + 1)) * G < want), "reclaim: the fewest copies dropped");
        }
        // copies 0 and 1 share granule 2: dropping copy 1 alone frees granules 3..4, not granule 2
        expect(c.pages(1) == 3 && c.pages(2) == 5, "a shared granule stays while a copy uses it");
    }
    {   // give_back: a fake room and copies ranked hottest first; a demote drops the coldest until enough granules are free
        const uint64_t G = 2u << 20, copy = 3u << 20;   // copies straddle granules, so a drop may free one or two
        struct Card {
            CopyLayout lay;
            int64_t n = 0;
            uint64_t mapped = 0, room = 0;
            int calls = 0;
        };
        const auto fill = [&](Card& c, int64_t n, uint64_t room) {
            c.lay = copy_layout(std::vector<uint64_t>((size_t) n, copy), G);
            c.n = n;
            c.mapped = c.lay.pages(n);
            c.room = room;
            c.calls = 0;
        };
        const auto run = [](Card& c, uint64_t free, bool fail = false) {
            return give_back(free, [&] { return c.room; }, [&] { return c.n; }, [&](uint64_t bytes, uint64_t& freed) {
                ++c.calls;
                if (fail) return false;
                c.n = c.lay.keep_for(c.n, c.mapped, bytes);
                freed = (c.mapped - c.lay.pages(c.n)) * G;
                c.mapped -= freed / G;
                c.room += freed;
                return true;
            });
        };
        const uint64_t keep = 512ull << 20;
        Card c;
        fill(c, 300, 7ull << 20);   // a refused map: the card read again holds 7 MiB, not the counted room
        expect(run(c, keep) && c.room >= keep, "shortage: copies go back until the keep is free");
        expect(c.n < 300 && c.n > 0, "shortage: only as many copies as the keep needs");
        expect(c.room - G * 2 < keep, "shortage: the fewest copies go (one more drop frees at most two granules)");
        const int64_t after = c.n;
        expect(run(c, keep) && c.n == after, "a second pass with the keep free changes nothing");
        fill(c, 300, keep + G);
        expect(run(c, keep) && c.n == 300 && c.calls == 0, "no shortage: no demote, the set stays");
        fill(c, 4, 0);
        expect(run(c, keep) && c.n == 0 && c.room < keep, "every copy back and still short: stops with none left");
        fill(c, 4, 0);
        expect(!run(c, keep, true) && c.calls == 1, "a failed demote stops and says so");
    }
    {   // the caches a prompt keeps, and when a resume trims them
        const int64_t step = 8192, base = 32768, ctx = 262144;
        expect(kept_positions(1, step, base, ctx) == base, "keep: never below the base");
        expect(kept_positions(36986, step, base, ctx) == 40960 && kept_positions(40960, step, base, ctx) == 40960,
               "keep: whole steps");
        expect(kept_positions(ctx + 5, step, base, ctx) == ctx, "keep: never past the context");
        // a long conversation backed to 229376, then a compaction resumes at its 18553-token prefix with 18433 to read
        int64_t backed = 229376;
        const int64_t short_keep = kept_positions(18553 + 18433, step, base, ctx);
        expect(resume_trims(short_keep, backed), "a compaction to a sixth trims");
        expect(!resume_trims(kept_positions(224008 + 600, step, base, ctx), backed), "the next turn: no trim");
        expect(!resume_trims(kept_positions(200000, step, base, ctx), backed), "an edit a few turns back: no trim");
        expect(resume_trims(backed / 2, backed) && !resume_trims(backed / 2 + step, backed), "the rule: half or less");
        // two conversations on one prefix, alternating: one trim per switch to the short one, none while it goes on
        int trims = 0;
        for (int sw = 0; sw < 4; ++sw) {
            const bool to_short = sw % 2 == 0;
            const int64_t keep = to_short ? short_keep : kept_positions(229000, step, base, ctx);
            if (resume_trims(keep, backed)) {
                ++trims;
                backed = keep;
            } else {
                backed = std::max(backed, keep);   // the deeper one grows back
            }
            if (to_short)
                for (int64_t turn = 1; turn <= 8; ++turn) {   // the short one's turns go deeper by a step each
                    const int64_t k = kept_positions(short_keep + turn * step, step, base, ctx);
                    expect(!resume_trims(k, backed), "a conversation's own turns never trim");
                    backed = std::max(backed, k);
                }
        }
        expect(trims == 2, "alternating: one trim per switch to the shallow conversation");
    }
    if (failures) return 1;
    std::printf("spill plan: ok\n");
    return 0;
}
