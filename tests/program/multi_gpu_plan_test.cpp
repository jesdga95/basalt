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
    // the prompt-time split: a main card and a tier card, the experts each streams per chunk spread evenly over layers
    {
        auto side = [](double gbps, double link, int64_t resident) {
            PsCard c;
            c.mem_gbps = gbps;
            c.link_gbps = link;
            c.stream_bytes.assign(48, (double) (48 * 512 - resident) * 2.76e6 / 48.0);
            return c;
        };
        auto chunks = [](std::initializer_list<int64_t> v) { return std::vector<int64_t>(v); };
        PsPart part;
        part.row_bytes = 4 * 2560 * 4;
        part.setup_ms = 40;
        // a 64K prompt on a 32 GB main card with a 16 GB tier card of a quarter of its bandwidth: read across both, the
        // tier card taking the last ten or so layers, both prices near what such a pair reads it in
        part.tier_chunks = chunks({16384, 24576, 24575});
        part.split_chunks = chunks({16384, 24576, 24575});
        part.draft_rows = 8257;
        const PsCard main_tier = side(1792, 28.9, 9460 - 3241), main_split = side(1792, 28.9, 9460 - 2883);
        const PsCard slow = side(448, 28.8, 5650 - 2900);
        PsPrice pr = price_prompt_split(main_tier, main_split, slow, part, 30, 48);
        CHECK(pr.split && pr.lb >= 36 && pr.lb <= 40);
        CHECK(pr.tier_ms > 0.95 * 7540 && pr.tier_ms < 1.05 * 7540);
        CHECK(pr.split_ms > 0.95 * 6740 && pr.split_ms < 1.05 * 6740);
        // the same prompt in one chunk and a short one: nothing to overlap
        part.split_chunks = chunks({65535});
        pr = price_prompt_split(main_tier, main_split, slow, part, 30, 48);
        CHECK(!pr.split && pr.lb == 0);
        // two alike cards: about half the layers each, and far cheaper than one card
        part.split_chunks = chunks({16384, 24576, 24575});
        const PsCard alike_tier = side(960, 25, 4000 - 2900), alike_split = side(960, 25, 4000 - 2883);
        const PsCard alike_peer = side(960, 25, 4000 - 2900);
        pr = price_prompt_split(alike_tier, alike_split, alike_peer, part, 8, 48);
        CHECK(pr.split && pr.lb >= 18 && pr.lb <= 30 && pr.split_ms < 0.8 * pr.tier_ms);
        // a tier card of a sixteenth of the bandwidth: not worth a split
        pr = price_prompt_split(main_tier, main_split, side(112, 28.8, 2750), part, 30, 48);
        CHECK(!pr.split);
    }
    if (g_fail) { std::printf("multi_gpu_plan_test: %d failures\n", g_fail); return 1; }
    std::printf("multi_gpu_plan_test: ok\n");
    return 0;
}
