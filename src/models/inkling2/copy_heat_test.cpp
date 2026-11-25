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
    const std::vector<int> once(layers, 1);
    CopyHeat h;
    h.start(pairs, n_expert, order, once);
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
    s.start(pairs, n_expert, order, once);
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
    m.start(pairs, n_expert, order, once);
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
    // a looped layer: layers 1 and 2 run three times a token, so their counts weigh 3^-kVisitPower (about 0.44)
    const std::vector<int> looped = {1, 3, 3, 1};
    CopyHeat v;
    v.start(pairs, n_expert, order, looped);
    const size_t pre = 0 * n_expert + 5, core = 1 * n_expert + 5, coda = 3 * n_expert + 5;
    std::vector<uint32_t> per(pairs, 0u);
    per[pre] = 100;    // one visit a token: 100 lookups per visit
    per[core] = 200;   // three visits: about 88
    per[coda] = 90;
    v.learn(per, 8);
    const std::vector<size_t> ranked = by_score_desc(v, {core, coda, pre});
    expect(ranked == std::vector<size_t>({pre, coda, core}), "per visit: 100 and 90 outrank 200 over three visits");
    per[core] = 240;   // about 105: ahead of the prelude expert's 100
    CopyHeat v1;
    v1.start(pairs, n_expert, order, looped);
    v1.learn(per, 8);
    expect(by_score_desc(v1, {core, coda, pre})[0] == core, "the discount is below one per visit: 240 over three visits leads");
    per[core] = 200;
    expect(by_score_desc(h, {core, coda, pre}) == by_score_desc(h, {pre, core, coda}), "one visit each: lookups alone");
    per[core] = 330;   // about 145
    CopyHeat v2;
    v2.start(pairs, n_expert, order, looped);
    v2.learn(per, 8);
    expect(by_score_desc(v2, {pre, coda, core})[0] == core, "a looped expert hotter per visit still leads");
    // the same counts give the same order, whatever order the candidates come in
    CopyHeat v3;
    v3.start(pairs, n_expert, order, looped);
    v3.learn(many, 8);
    CopyHeat v4;
    v4.start(pairs, n_expert, order, looped);
    v4.learn(many, 8);
    expect(by_score_desc(v3, ram) == by_score_desc(v4, std::vector<size_t>(ram.rbegin(), ram.rend())),
           "deterministic: equal counts, equal rank, whatever the candidates' order");
    // unseen pairs keep the profile's order below one lookup per visit
    CopyHeat u;
    u.start(pairs, n_expert, order, looped);
    expect(by_score_desc(u, std::vector<size_t>(ram.rbegin(), ram.rend())) == ram, "unseen, looped: the spill order");
    std::vector<uint32_t> one(pairs, 0u);
    const size_t last_looped = ram[ram.size() - 2];   // layer 1, near the end of the profile's order
    one[last_looped] = 1;                              // one lookup, at the smallest weight
    u.learn(one, 0);
    expect(last_looped / n_expert == 1 && by_score_desc(u, ram)[0] == last_looped, "one lookup outranks every unseen pair");
    {   // the largest order the loader admits (24 layers x 512 experts, the looped layers at weight 3^-0.75): one lookup at
        // the profile's last place still outranks its unseen first place
        const int L = 24, E = 512;
        std::vector<int> vis(L, 1);
        for (int l = 6; l < 18; ++l) vis[(size_t) l] = 3;
        std::vector<std::pair<int32_t, int32_t>> big;
        for (int e = 0; e < E; ++e)
            for (int l = 0; l < L; ++l) big.emplace_back(l, e);
        CopyHeat b;
        b.start((size_t) L * E, E, big, vis);
        const size_t first = (size_t) big.front().first * E + big.front().second;
        const auto [ll, le] = big[big.size() - 7];    // a looped layer's pair near the end
        const size_t late = (size_t) ll * E + le;
        std::vector<uint32_t> once((size_t) L * E, 0u);
        once[late] = 1;
        b.learn(once, 0);
        expect(ll >= 6 && ll < 18 && by_score_desc(b, {first, late})[0] == late, "12,288 pairs: one lookup outranks the unseen first");
        expect(by_score_desc(b, {first, (size_t) big[1].first * E + big[1].second})[0] == first, "12,288 pairs: unseen keep the order");
    }
    {   // swaps across weights: a copy at a single-visit layer against a RAM expert of a looped one, and back
        const size_t edge = 0 * n_expert + 2, loop = 1 * n_expert + 2;
        auto heat = [&](uint32_t e, uint32_t l) {
            CopyHeat x;
            x.start(pairs, n_expert, order, looped);
            std::vector<uint32_t> c(pairs, 0u);
            c[edge] = e;
            c[loop] = l;
            x.learn(c, 0);
            return x;
        };
        auto swaps = [&](const CopyHeat& x, size_t copy, size_t ram_pair) {
            return plan_swaps(x, {copy}, {ram_pair}, [](size_t) { return 0; }, 8).size();
        };
        expect(swaps(heat(40, 160), edge, loop) == 1, "160 over three visits (about 70) takes a copy of 40: past the margin and 16");
        expect(swaps(heat(40, 120), edge, loop) == 0, "120 over three visits (about 53) is under 16 more than 40");
        expect(swaps(heat(60, 100), loop, edge) == 1, "60 takes a looped copy of 100 (about 44): 16 more");
        expect(swaps(heat(58, 100), loop, edge) == 0, "58 is under 16 more than about 44");
    }
    {   // equal scores: the profile's order, then the pair, whatever order the candidates come in
        CopyHeat t;
        t.start(pairs, n_expert, {}, looped);           // no profile: every prior and rank equal
        expect(by_score_desc(t, {5, 3, 9}) == std::vector<size_t>({3, 5, 9}), "full ties: by pair");
    }
    if (failures == 0) std::printf("inkling2_copy_heat_test: ok\n");
    return failures == 0 ? 0 : 1;
}
