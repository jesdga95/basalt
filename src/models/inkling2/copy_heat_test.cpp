// src/models/inkling2/copy_heat_test.cpp - the copies' ranking: the spill order before any lookup, the session's own counts after,
// decay when the traffic changes, and swaps only where a RAM expert clearly beats a copy of its size.
#include "models/inkling2/copy_heat.hpp"

#include <algorithm>
#include <cstdio>

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
    const int n_expert = 8, layers = 4;
    const size_t pairs = (size_t) layers * n_expert;
    std::vector<std::pair<int32_t, int32_t>> order;
    for (int e = 0; e < n_expert; ++e)
        for (int l = layers - 1; l >= 0; --l) order.emplace_back(l, e);
    CopyHeat h;
    h.start(pairs, n_expert, order);
    std::vector<size_t> ram;
    for (const auto& [l, e] : order) ram.push_back((size_t) l * n_expert + (size_t) e);
    expect(by_score_desc(h, std::vector<size_t>(ram.rbegin(), ram.rend())) == ram, "unseen: the spill order");
    expect(by_score_desc(h, ram) == ram, "unseen: the order already sorted stays");

    std::vector<uint32_t> look(pairs, 0u);
    look[ram[20]] = 50;
    look[ram[21]] = 20;
    h.learn(look, 64);
    const std::vector<size_t> r = by_score_desc(h, ram);
    expect(r[0] == ram[20] && r[1] == ram[21] && r[2] == ram[0], "counted pairs lead, the rest keep the spill order");

    // decay: after many rows without lookups the old counts are gone, the order returns to the prior
    h.learn(std::vector<uint32_t>(pairs, 0u), (int64_t) (CopyHeat::kHalfLifeRows * 40));
    expect(by_score_desc(h, ram) == ram, "stale counts decay away");

    // swaps: copies are ram[0..3]; ram[10] (same class) is hot, ram[11] (other class) is hot, ram[12] is barely warmer
    CopyHeat s;
    s.start(pairs, n_expert, order);
    std::vector<uint32_t> w(pairs, 0u);
    for (int i = 0; i < 4; ++i) w[ram[(size_t) i]] = 100;
    w[ram[3]] = 5;
    w[ram[10]] = 400;
    w[ram[11]] = 300;
    w[ram[12]] = 110;
    s.learn(w, 8);
    const std::vector<size_t> copies(ram.begin(), ram.begin() + 4), rest(ram.begin() + 4, ram.end());
    auto one_class = [](size_t) { return 0; };
    std::vector<CopySwap> sw = plan_swaps(s, copies, rest, one_class, 8);
    expect(sw.size() == 2, "two clear gains, the third (110 vs 100) is under the margin");
    expect(sw.size() >= 1 && sw[0].in == ram[10] && sw[0].out == ram[3], "the hottest takes the coldest copy's place");
    expect(sw.size() >= 2 && sw[1].in == ram[11] && sw[1].out != sw[0].out, "each copy gives up once");
    expect(plan_swaps(s, copies, rest, one_class, 1).size() == 1, "max_swaps bounds the plan");
    auto split = [&](size_t p) { return p == ram[11] ? 1 : 0; };
    sw = plan_swaps(s, copies, rest, split, 8);
    expect(sw.size() == 1 && sw[0].in == ram[10], "a RAM expert never takes a copy of another size");
    expect(plan_swaps(h, copies, rest, one_class, 8).empty(), "no counts: no swap");
    // every swap names a distinct copy and a distinct RAM expert, so applying them in order keeps copies and tables one-to-one
    CopyHeat m;
    m.start(pairs, n_expert, order);
    std::vector<uint32_t> many(pairs, 0u);
    for (size_t i = 0; i < ram.size(); ++i) many[ram[i]] = (uint32_t) (5 + 37 * ((i * 7) % 11));
    m.learn(many, 8);
    const std::vector<size_t> held(ram.begin(), ram.begin() + 12), free_(ram.begin() + 12, ram.end());
    const std::vector<CopySwap> all = plan_swaps(m, held, free_, one_class, 100);
    bool distinct = !all.empty();
    for (size_t i = 0; i < all.size(); ++i)
        for (size_t j = i + 1; j < all.size(); ++j) distinct = distinct && all[i].out != all[j].out && all[i].in != all[j].in;
    for (const CopySwap& x : all) distinct = distinct && std::find(held.begin(), held.end(), x.out) != held.end() &&
                                              std::find(free_.begin(), free_.end(), x.in) != free_.end();
    expect(distinct, "swaps are distinct, each from a copy to a RAM expert");
    if (failures == 0) std::printf("inkling2_copy_heat_test: ok\n");
    return failures == 0 ? 0 : 1;
}
