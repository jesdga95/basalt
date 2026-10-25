// tests/program/multi_gpu_plan_test.cpp - what the cards do together, over fake card tables: the prices, the pick, the
// single-card cases (no choice, no other card touched).
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/multi_gpu_plan.hpp"

#include "basalt/platform/process.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static constexpr uint64_t GiB = 1ull << 30;

static GpuCard card(int smi, const char* name, double gbps, uint64_t gib) {
    GpuCard c;
    c.token = std::to_string(smi);
    c.smi = smi;
    c.name = name;
    c.cc_major = 12;
    c.mem_gbps = gbps;
    c.vram = gib * GiB;
    return c;
}

static MgModel model() {
    MgModel m;
    m.main_overhead = 4300ull * (1 << 20) + 7ull * GiB / 2;
    m.peer_overhead = 1280ull * (1 << 20);
    return m;
}

static const MgPrice& price(const MgPlan& p, Mechanism m) {
    for (const MgPrice& x : p.prices)
        if (x.mechanism == m) return x;
    static MgPrice none;
    return none;
}

static double tok_s(double ms) { return 2.15 * 1000.0 / ms; }

int main() {
    // unlike cards (a 5090 and a 5060 Ti): the expert tier, the split loses to it by far
    {
        const MgPlan p = plan_multi_gpu({card(0, "RTX 5090", 1792, 32), card(1, "RTX 5060 Ti", 448, 16)}, model());
        CHECK(!p.equal);
        CHECK(p.pick == Mechanism::ExpertTier);
        const double tier = price(p, Mechanism::ExpertTier).ms, split = price(p, Mechanism::LayerSplit).ms;
        CHECK(tier > 7.0 && tier < 9.5);          // measured 8.0 ms
        CHECK(split > 12.0 && split < 16.0);      // measured 13.5 ms (the cut at layer 40)
        CHECK(p.line.find("expert tier") != std::string::npos && p.line.find("picked expert tier") != std::string::npos);
    }
    // two equal 16 GB cards (2 x RTX 5080): the expert tier within 10% of the reported pair rate, the split far behind
    {
        const MgPlan p = plan_multi_gpu({card(0, "RTX 5080", 960, 16), card(1, "RTX 5080", 960, 16)}, model());
        CHECK(p.equal);
        CHECK(p.pick == Mechanism::ExpertTier);
        const double rate = tok_s(price(p, Mechanism::ExpertTier).ms);
        CHECK(rate > 0.9 * 147.5 && rate < 1.1 * 147.5);
        CHECK(price(p, Mechanism::LayerSplit).ms > 1.1 * price(p, Mechanism::ExpertTier).ms);
        CHECK(!price(p, Mechanism::TensorParallel).runs);   // priced, not built
    }
    // overlapped windows: only two alike cards that can run them (a drafter, a server, one conversation) price the split
    // that way, and the split is picked where it prices at least 5% below the tier
    {
        MgModel m = model();
        m.overlapped = true;
        const MgPlan unlike = plan_multi_gpu({card(0, "RTX 5090", 1792, 32), card(1, "RTX 5060 Ti", 448, 16)}, m);
        CHECK(!unlike.equal && unlike.pick == Mechanism::ExpertTier);   // never picked, whatever the hold
        for (double hold : {0.0, 0.5, 1.0}) {
            m.hold = hold;
            CHECK(plan_multi_gpu({card(0, "RTX 5090", 1792, 32), card(1, "RTX 5060 Ti", 448, 16)}, m).pick == Mechanism::ExpertTier);
        }
        m.hold = MgModel{}.hold;
        const std::vector<GpuCard> two = {card(0, "RTX 5080", 960, 16), card(1, "RTX 5080", 960, 16)};
        const MgPlan on = plan_multi_gpu(two, m);
        const double tier = price(on, Mechanism::ExpertTier).ms, split = price(on, Mechanism::LayerSplit).ms;
        CHECK(on.equal && price(on, Mechanism::LayerSplit).overlapped);
        CHECK(split < 0.95 * tier && on.pick == Mechanism::LayerSplit);
        CHECK(on.line.find("windows overlapped") != std::string::npos && on.line.find("picked layer split") != std::string::npos);
        MgModel off = m;
        off.overlapped = false;   // no drafter, no server or batch slots: the plain split, far behind the tier
        const MgPlan plain = plan_multi_gpu(two, off);
        CHECK(!price(plain, Mechanism::LayerSplit).overlapped && plain.pick == Mechanism::ExpertTier);
        CHECK(price(plain, Mechanism::LayerSplit).ms > price(on, Mechanism::LayerSplit).ms);
        MgModel low = m;
        low.hold = 0.1;           // windows that seldom find their successor running do not pay
        CHECK(plan_multi_gpu(two, low).pick == Mechanism::ExpertTier);
        // three and four cards: the engine overlaps two stages only, so the split is priced plain
        for (int n = 3; n <= 4; ++n) {
            std::vector<GpuCard> v;
            for (int i = 0; i < n; ++i) v.push_back(card(i, "RTX 5080", 960, 16));
            const MgPlan p = plan_multi_gpu(v, m);
            CHECK(!price(p, Mechanism::LayerSplit).overlapped && p.line.find("overlapped") == std::string::npos);
        }
    }
    // for two to four alike cards of any size the pick follows the prices: the split exactly where it prices 5% below
    // the tier, the tier otherwise; unlike cards never leave the tier
    for (int n = 2; n <= 4; ++n)
        for (const double gbps : {448.0, 672.0, 960.0, 1792.0})
            for (const uint64_t gib : {8, 12, 16, 24, 32}) {
                std::vector<GpuCard> v;
                for (int i = 0; i < n; ++i) v.push_back(card(i, "RTX", gbps, gib));
                MgModel m = model();
                m.overlapped = true;
                const MgPlan p = plan_multi_gpu(v, m);
                const double tier = price(p, Mechanism::ExpertTier).ms, split = price(p, Mechanism::LayerSplit).ms;
                CHECK(p.pick == (split < 0.95 * tier ? Mechanism::LayerSplit : Mechanism::ExpertTier));
                v.back() = card(n - 1, "RTX", gbps / 2, gib);
                CHECK(plan_multi_gpu(v, m).pick == Mechanism::ExpertTier);
            }
    // three and four equal cards, and equal cards big enough to hold every expert: priced, one line, a valid pick
    for (int n = 2; n <= 4; ++n) {
        std::vector<GpuCard> v;
        for (int i = 0; i < n; ++i) v.push_back(card(i, "RTX 5090", 1792, 32));
        const MgPlan p = plan_multi_gpu(v, model());
        CHECK(p.equal && p.prices.size() == 3 && !p.line.empty());
        CHECK(p.line.find(std::to_string(n) + " cards") != std::string::npos);
        CHECK(price(p, p.pick).runs);
    }
    // a card of another size or speed breaks "alike"; 10% apart does not
    CHECK(cards_alike(card(0, "a", 960, 16), card(1, "b", 1000, 16)));
    CHECK(!cards_alike(card(0, "a", 960, 16), card(1, "b", 448, 16)));
    CHECK(!cards_alike(card(0, "a", 960, 16), card(1, "b", 960, 24)));
    CHECK(cards_alike(card(0, "a", 0, 0), card(1, "b", 960, 16)));   // unknown figures match
    // one card: no choice at all
    {
        const MgPlan p = plan_multi_gpu({card(0, "RTX 5090", 1792, 32)}, model());
        CHECK(p.prices.empty() && p.line.empty());
    }
    // the gpu setting: a single card, whichever is named, makes no multi-card choice and narrows the CUDA view to it
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 1792, 32), card(1, "RTX 5060 Ti", 448, 16), card(2, "RTX 5080", 960, 16)};
        for (const char* text : {"", "0", "1", "2"}) {
            GpuSpec s;
            std::string err;
            CHECK(parse_gpu_spec(text, s, err));
            const GpuChoice c = choose_gpus(s, v, true);
            CHECK(c.error.empty() && c.cards.size() == 1);
            RoleInputs in;
            for (int i : c.cards) in.cards.push_back(v[(size_t) i]);
            const std::vector<GpuRole> r = choose_roles(in);
            CHECK(r.size() == 1 && r[0] == GpuRole::Main);   // no tier, no idle card
            apply_gpus(v, c.cards);
            const char* env = std::getenv("CUDA_VISIBLE_DEVICES");
            CHECK(env != nullptr && std::string(env) == v[(size_t) c.cards[0]].token);   // the other cards are hidden
        }
        // two cards named: both are used, the first is main
        GpuSpec s;
        std::string err;
        CHECK(parse_gpu_spec("2,1", s, err));
        const GpuChoice c = choose_gpus(s, v, true);
        CHECK(c.cards.size() == 2 && c.cards[0] == 2 && c.cards[1] == 1);
    }
    if (g_fail) { std::printf("multi_gpu_plan_test: %d failures\n", g_fail); return 1; }
    std::printf("multi_gpu_plan_test: ok\n");
    return 0;
}
