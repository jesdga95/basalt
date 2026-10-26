// tests/program/multi_gpu_plan_test.cpp - what the cards do together, over fake card tables: the prices, the pick, the
// single-card cases (no choice, no other card touched).
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/multi_gpu_plan.hpp"

#include "basalt/platform/process.hpp"

#include <algorithm>
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
    // the prompt-time split: a main card and tier cards, the experts each streams per chunk spread evenly over layers
    {
        auto side = [](double gbps, double link, int64_t resident, double setup_ms = 0) {
            PsCard c;
            c.mem_gbps = gbps;
            c.link_gbps = link;
            c.setup_ms = setup_ms;
            c.stream_bytes.assign(48, (double) (48 * 512 - resident) * 2.76e6 / 48.0);
            return c;
        };
        auto chunks = [](std::initializer_list<int64_t> v) { return std::vector<int64_t>(v); };
        // equal chunks of at most `cap`
        auto even = [](int64_t n, int64_t cap) {
            std::vector<int64_t> v;
            const int64_t parts = (n + cap - 1) / cap, c = (n + parts - 1) / parts;
            for (int64_t at = 0; at < n; at += c) v.push_back(std::min(c, n - at));
            return v;
        };
        PsPart part;
        part.row_bytes = 4 * 2560 * 4;
        // a 64K prompt on a 32 GB main card with a 16 GB tier card of a quarter of its bandwidth: read across both, the
        // tier card taking the last ten or so layers, both prices near what such a pair reads it in
        part.tier_chunks = chunks({16384, 24576, 24575});
        part.split_chunks = chunks({16384, 24576, 24575});
        part.draft_rows = 8257;
        const PsCard main_tier = side(1792, 28.9, 9460 - 3241), main_split = side(1792, 28.9, 9460 - 2883);
        PsCard slow = side(448, 28.8, 5650 - 2900, 40);
        slow.layer_lo = 30;
        PsPrice pr = price_prompt_split(main_tier, main_split, {slow}, part, 2, 48);
        CHECK(pr.split && pr.stages == 2 && pr.cuts.size() == 1 && pr.lb() >= 36 && pr.lb() <= 40);
        CHECK(pr.tier_ms > 0.95 * 7540 && pr.tier_ms < 1.05 * 7540);
        CHECK(pr.split_ms > 0.95 * 6740 && pr.split_ms < 1.05 * 6740);
        CHECK(pr.stage_ms.size() == 3 && pr.stage_ms[2] == pr.split_ms && pr.stage_cuts[2] == pr.cuts);
        // its weights from layer 39 on: the cut follows them
        slow.layer_lo = 39;
        CHECK(price_prompt_split(main_tier, main_split, {slow}, part, 2, 48).lb() >= 39);
        slow.layer_lo = 30;
        // one stage allowed, or one chunk: no split
        CHECK(!price_prompt_split(main_tier, main_split, {slow}, part, 1, 48).split);
        part.split_chunks = chunks({65535});
        pr = price_prompt_split(main_tier, main_split, {slow}, part, 2, 48);
        CHECK(!pr.split && pr.lb() == 0 && pr.stages == 0);
        // two alike cards: about half the layers each, and far cheaper than one card
        part.split_chunks = chunks({16384, 24576, 24575});
        const PsCard alike_tier = side(960, 25, 4000 - 2900), alike_split = side(960, 25, 4000 - 2883);
        const PsCard alike_peer = side(960, 25, 4000 - 2900);
        pr = price_prompt_split(alike_tier, alike_split, {alike_peer}, part, 2, 48);
        CHECK(pr.split && pr.lb() >= 18 && pr.lb() <= 30 && pr.split_ms < 0.8 * pr.tier_ms);
        // a tier card of a sixteenth of the bandwidth: not worth a split
        pr = price_prompt_split(main_tier, main_split, {side(112, 28.8, 2750)}, part, 2, 48);
        CHECK(!pr.split);

        // three and four alike 16 GB cards: the split uses them all on long prompts, cut about evenly, and every stage
        // added makes it cheaper; a quarter or a third of the layers on each card
        const PsCard p16 = side(960, 25, 4000 - 2900, 60);
        for (int64_t n : {65535LL, 131071LL, 997000LL}) {
            PsPart lp;
            lp.row_bytes = part.row_bytes;
            lp.tier_chunks = lp.split_chunks = even(n, 16384);
            lp.draft_rows = 8257;
            const PsPrice two = price_prompt_split(alike_tier, alike_split, {p16}, lp, 2, 48);
            const PsPrice three = price_prompt_split(alike_tier, alike_split, {p16, p16}, lp, 3, 48);
            const PsPrice four = price_prompt_split(alike_tier, alike_split, {p16, p16, p16}, lp, 4, 48);
            CHECK(two.split && three.split && four.split);
            CHECK(three.stages == 3 && four.stages == 4 && three.cuts.size() == 2 && four.cuts.size() == 3);
            CHECK(four.split_ms < three.split_ms && three.split_ms < two.split_ms && two.split_ms < 0.8 * two.tier_ms);
            // a price over fewer stages is the one the smaller set of cards gets
            CHECK(four.stage_ms[2] == two.split_ms && four.stage_ms[3] == three.split_ms);
            CHECK(three.cuts[0] >= 12 && three.cuts[0] <= 20 && three.cuts[1] >= 28 && three.cuts[1] <= 36);
            CHECK(four.cuts[0] >= 8 && four.cuts[0] <= 14 && four.cuts[1] >= 19 && four.cuts[1] <= 26 &&
                  four.cuts[2] >= 31 && four.cuts[2] <= 38);
            // a stage count above the cards' is not priced; a cap below them is kept
            CHECK(price_prompt_split(alike_tier, alike_split, {p16, p16}, lp, 8, 48).stages == 3);
            CHECK(price_prompt_split(alike_tier, alike_split, {p16, p16, p16}, lp, 2, 48).stages == 2);
            if (n == 997000) {   // a deep prompt: four cards read it in well under a third of one card's time
                CHECK(four.split_ms < 0.33 * four.tier_ms);
            }
        }
        // the windows: the start's plans for two, three and four stages give each card the layers it may read, widened
        // by three, the card that reads the last layers to the end
        {
            PsPart lp;
            lp.row_bytes = part.row_bytes;
            lp.tier_chunks = lp.split_chunks = even(131071, 16384);
            lp.draft_rows = 8257;
            const PsPrice ref = price_prompt_split(alike_tier, alike_split, {p16, p16, p16}, lp, 4, 48);
            const std::vector<std::pair<int, int>> w = prompt_split_windows(ref, 3, 3, 48);
            CHECK(w.size() == 3 && w[0].second == 48);
            CHECK(w[0].first == std::max(2, std::min({ref.stage_cuts[2][0], ref.stage_cuts[3][1], ref.stage_cuts[4][2]}) - 3));
            CHECK(w[1].first == std::min(ref.stage_cuts[3][0], ref.stage_cuts[4][1]) - 3 &&
                  w[1].second == std::max(ref.stage_cuts[3][1], ref.stage_cuts[4][2]) + 3);
            CHECK(w[2].first == ref.stage_cuts[4][0] - 3 && w[2].second == ref.stage_cuts[4][1] + 3);
            // every plan of the start fits the windows: priced inside them, the same plans come back
            std::vector<PsCard> win = {p16, p16, p16};
            for (size_t i = 0; i < 3; ++i) {
                win[i].layer_lo = w[i].first;
                win[i].layer_hi = w[i].second;
            }
            const PsPrice again = price_prompt_split(alike_tier, alike_split, win, lp, 4, 48);
            CHECK(again.stage_cuts == ref.stage_cuts && again.split_ms == ref.split_ms);
            // a card with no window cannot take a stage: one window short, the split stops a stage earlier
            win[2] = p16;
            win[2].layer_lo = 47;
            win[2].layer_hi = 47;
            CHECK(price_prompt_split(alike_tier, alike_split, win, lp, 4, 48).stage_ms[4] == 0);
            // a window that does not reach the last layer cannot be the last stage
            win[0].layer_hi = 40;
            CHECK(price_prompt_split(alike_tier, alike_split, win, lp, 4, 48).stages == 0);
            // no stage count worth it: no windows
            PsPrice none = ref;
            for (double& ms : none.stage_ms) ms = ms > 0 ? none.tier_ms : 0;
            for (const auto& x : prompt_split_windows(none, 3, 3, 48)) CHECK(x.first == 0 && x.second == 0);
        }
        // mixed cards: the faster card joins first and reads the last layers; with a slow third card a long prompt
        // still gains from it, a shorter one keeps to two stages
        {
            const std::vector<size_t> order = prompt_split_order({448, 960, 960, 1792});
            CHECK(order.size() == 4 && order[0] == 3 && order[1] == 1 && order[2] == 2 && order[3] == 0);
            const PsCard p80 = side(960, 25, 5000 - 2900, 40), p60 = side(448, 28.8, 5650 - 2900, 40);
            PsPart lp;
            lp.row_bytes = part.row_bytes;
            lp.draft_rows = 8257;
            lp.tier_chunks = lp.split_chunks = even(65535, 24576);
            pr = price_prompt_split(main_tier, main_split, {p80, p60}, lp, 3, 48);
            CHECK(pr.split && pr.stages == 2 && pr.stage_ms[3] > pr.split_ms);
            lp.tier_chunks = lp.split_chunks = even(997000, 24576);
            pr = price_prompt_split(main_tier, main_split, {p80, p60}, lp, 3, 48);
            CHECK(pr.split && pr.stages == 3 && pr.split_ms < pr.stage_ms[2]);
            // the slower card's stage is the shorter one
            CHECK(pr.cuts[1] - pr.cuts[0] < 48 - pr.cuts[1]);
        }
    }
    if (g_fail) { std::printf("multi_gpu_plan_test: %d failures\n", g_fail); return 1; }
    std::printf("multi_gpu_plan_test: ok\n");
    return 0;
}
