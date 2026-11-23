// src/models/inkling2/spill_test.cpp - the spill plan: nothing leaves when nothing is needed, the coldest experts leave
// first, slots are a permutation with the kept experts first, and the bytes it frees cover what was asked net of the shared
// staging and the rounding; a prompt chunk's staging plan puts every expert it reads in its slot.
#include "models/inkling2/spill.hpp"

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
    if (failures) return 1;
    std::printf("spill plan: ok\n");
    return 0;
}
