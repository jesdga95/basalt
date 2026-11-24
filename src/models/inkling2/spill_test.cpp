// src/models/inkling2/spill_test.cpp - the spill plan: nothing leaves when nothing is needed, the coldest experts leave
// first, slots are a permutation with the kept experts first, the bytes it frees cover what was asked net of the shared
// staging and the rounding, copies go back until a keep is free, a resume trims the caches only when it frees half,
// and a prompt chunk's staging plan puts every expert it reads in its slot.
#include "models/inkling2/spill.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
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
    // the staging plan against a slot-by-slot model, over random copies and routing: every slot a chunk reads holds its
    // expert, every slot the products can read into (after a read or kept one) is staged or guarded, adjacent RAM slots go
    // in one copy, and a slot neither read nor reachable is left alone
    // fixed layouts: card, a RAM run, a guard after a read slot, an untouched one, a RAM run, card; a kept layer whose first
    // RAM slot is skipped; a layer all in RAM with nothing kept
    const auto same = [](const std::vector<StageOp>& a, const std::vector<StageOp>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].kind != b[i].kind || a[i].slot != b[i].slot || a[i].n != b[i].n) return false;
        return true;
    };
    {
        const uint8_t card[8] = {1, 0, 0, 1, 0, 0, 0, 1}, want[8] = {1, 1, 1, 0, 0, 1, 1, 1};
        expect(same(stage_plan(8, card, want, true), {{StageOp::Card, 0, 1}, {StageOp::Ram, 1, 2}, {StageOp::Guard, 3, 1},
                                                      {StageOp::Ram, 5, 2}, {StageOp::Card, 7, 1}}),
               "stage plan: mixed copies with sparse routing");
        const uint8_t none[3] = {0, 0, 0}, mid[3] = {0, 1, 0};
        expect(same(stage_plan(3, none, mid, true), {{StageOp::Guard, 0, 1}, {StageOp::Ram, 1, 1}, {StageOp::Guard, 2, 1}}),
               "stage plan: a skipped slot after the kept ones is guarded");
        const uint8_t ram[5] = {0, 0, 0, 0, 0};
        expect(same(stage_plan(5, ram, nullptr, false), {{StageOp::Ram, 0, 5}}), "stage plan: a layer all in RAM is one copy");
    }
    std::mt19937 gen(12345);
    const auto next = [&] { return (uint32_t) gen(); };
    for (int trial = 0; trial < 20000; ++trial) {
        const int n = 1 + (int) (next() % 40);
        const bool kept = next() % 4 != 0;
        const int pc = (int) (next() % 3), pw = (int) (next() % 4);   // copy and routing densities
        std::vector<uint8_t> card((size_t) n), want((size_t) n);
        for (int j = 0; j < n; ++j) {
            card[(size_t) j] = pc == 0 ? 0 : pc == 1 ? (uint8_t) (next() % 2) : 1;
            want[(size_t) j] = pw == 3 ? 1 : (uint8_t) (next() % 4 < (uint32_t) pw + 1 ? 1 : 0);
        }
        const bool all = next() % 5 == 0;
        enum : uint8_t { Stale, Expert, Zero };
        std::vector<uint8_t> state((size_t) n, Stale);
        std::vector<int> touched((size_t) n, 0);
        bool ok = true, adjacent_ram = false;
        int last_ram_end = -1;
        for (const StageOp& op : stage_plan(n, card.data(), all ? nullptr : want.data(), kept)) {
            if (op.slot < 0 || op.n < 1 || op.slot + op.n > n) { ok = false; break; }
            if (op.kind != StageOp::Ram && op.n != 1) ok = false;
            for (int j = op.slot; j < op.slot + op.n; ++j) {
                ++touched[(size_t) j];
                state[(size_t) j] = op.kind == StageOp::Guard ? Zero : Expert;
                if (op.kind == StageOp::Ram && card[(size_t) j]) ok = false;    // a slot with a copy comes from the card
                if (op.kind == StageOp::Card && !card[(size_t) j]) ok = false;
            }
            if (op.kind == StageOp::Ram) {
                if (op.slot == last_ram_end) adjacent_ram = true;
                last_ram_end = op.slot + op.n;
            }
        }
        for (int j = 0; j < n; ++j) {
            const bool read = all || want[(size_t) j];
            const bool reachable = j > 0 ? (all || want[(size_t) j - 1]) : kept;
            if (touched[(size_t) j] > 1) ok = false;
            if (read && state[(size_t) j] != Expert) ok = false;
            if (!read && reachable && state[(size_t) j] != Zero) ok = false;
            if (!read && !reachable && state[(size_t) j] != Stale) ok = false;
        }
        expect(ok, "stage plan: read slots staged, reachable ones guarded, the rest untouched");
        expect(!adjacent_ram, "stage plan: adjacent RAM slots in one copy");
        if (!ok || adjacent_ram) break;
    }
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
