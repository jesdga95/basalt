// src/models/inkling2/spill_test.cpp - the spill plan: nothing leaves when nothing is needed, the coldest experts leave
// first, slots are a permutation with the kept experts first, and the bytes it frees cover what was asked net of the shared
// staging and the rounding.
#include "models/inkling2/spill.hpp"

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
    if (failures) return 1;
    std::printf("spill plan: ok\n");
    return 0;
}
