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
        // four alike 16 GB cards behind one host link, from reads of 133,205 and 266,381 tokens: the tier cards stream
        // nine tenths of their layers' experts per chunk, the main card all of its own.  Each card's link alone gives the
        // prices those reads were priced at; the cards streaming at once over the host's 54 GB/s give the reads
        {
            const PsCard m4 = side(960, 26.8, 0), t4 = side(960, 26.8, 2458, 60);
            auto read = [&](int64_t n, int64_t cap, double host) {
                PsPart lp;
                lp.row_bytes = part.row_bytes;
                lp.draft_rows = 8257;
                lp.host_gbps = host;
                lp.tier_chunks = lp.split_chunks = even(n, cap);
                return price_prompt_split(m4, m4, {t4, t4, t4}, lp, 4, 48);
            };
            auto near = [](double a, double b, double tol) { return a > (1 - tol) * b && a < (1 + tol) * b; };
            const PsPrice a1 = read(133205, 22528, 0), a2 = read(266381, 16384, 0);
            CHECK(near(a1.stage_ms[2], 16141, 0.03) && near(a1.stage_ms[3], 14359, 0.03) && near(a1.stage_ms[4], 13562, 0.03));
            CHECK(near(a2.stage_ms[2], 32158, 0.03) && near(a2.stage_ms[3], 24809, 0.03) && near(a2.stage_ms[4], 21118, 0.03));
            const PsPrice b1 = read(133205, 22528, 54), b2 = read(266381, 16384, 54);
            CHECK(near(b2.stage_ms[2], 32865, 0.06) && near(b2.stage_ms[3], 27270, 0.06) && near(b2.stage_ms[4], 25669, 0.06));
            CHECK(b2.split && b2.stages == 4 && b2.stage_ms[4] < b2.stage_ms[3] && b2.stage_ms[3] < b2.stage_ms[2]);
            CHECK(b1.stage_ms[4] >= 12891 && b1.stage_ms[4] < 1.1 * 12891 && b1.stages == 4);
            // a host that feeds every card's link whole changes no price
            const PsPrice c2 = read(266381, 16384, 4 * 26.8);
            CHECK(c2.stage_ms == a2.stage_ms && c2.stage_cuts == a2.stage_cuts);
            // nor does it on two cards whose links it can feed: the pair's price is the one each link alone gives
            PsPart lp = part;
            lp.host_gbps = 28.9 + 28.8;
            const PsPrice d = price_prompt_split(main_tier, main_split, {slow}, lp, 2, 48);
            CHECK(d.split_ms == price_prompt_split(main_tier, main_split, {slow}, part, 2, 48).split_ms);
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
    // the draft layer's rows a split read holds back: two chunks in place, more past the chunk in both buffers
    {
        auto room = [](size_t n, int64_t chunk, int64_t hand) {
            return prompt_split_draft_room(std::vector<int64_t>(n, chunk), chunk, hand);
        };
        CHECK(room(2, 16384, 16384) == 16384);
        CHECK(room(3, 16384, 16384) == 32768);
        CHECK(room(4, 16384, 16384) == 32768);
        // a deep part: smaller chunks than the buffers were made for leave the rest of both buffers
        std::vector<int64_t> deep(17, 14848);
        deep.front() = 16384;
        deep.back() = 9000;
        CHECK(prompt_split_draft_room(deep, 14848, 27392) == 2 * 14848 + 2 * (27392 - 14848));
        CHECK(prompt_split_draft_room(deep, 14848, 27392) >= 32832);
        CHECK(2 * 14848 < 32832);   // the window the two chunks before the kept one alone could not hold
        CHECK(room(4, 4096, 16128) == 2 * 4096 + 2 * (16128 - 4096));
        CHECK(room(2, 4096, 2048) == 4096);
    }
    // a split read's chunks are the prompt path's: the main card keeps its last chunk, and the stages' state comes back
    // once the last stage has read the chunk before it, so the planned last chunk must start where the prompt path's
    // does, for every part length and cap (deep parts on smaller cards take caps just above the head chunk)
    {
        constexpr int64_t kHead = 16384;
        auto path_chunks = [](int64_t n, int64_t T, int64_t head_set) {   // Prefill::run_impl's chunk_len
            std::vector<int64_t> v;
            const int64_t head = n > T ? std::min(head_set, T) : 0;
            for (int64_t at = 0; T > 0 && at < n;) {
                const int64_t t = std::min(at == 0 && head > 0 ? head : T, n - at);
                v.push_back(t);
                at += t;
            }
            return v;
        };
        int bad = 0, old_off = 0, cases = 0;
        std::vector<int64_t> caps;
        for (int64_t c = 12800; c <= 17920; c += 256) caps.push_back(c);
        caps.push_back(27392);
        caps.push_back(32768);
        for (int64_t cap : caps)
            for (int64_t n = 4096; n <= 560000; n += n < 200000 ? 997 : 61) {
                const auto [chunk, head] = prompt_one_card_chunks(n, cap, true, kHead);
                const std::vector<int64_t> cl = prompt_chunk_list(n, chunk, head);
                const std::vector<int64_t> ref = path_chunks(n, chunk, head);
                int64_t sum = 0;
                for (int64_t t : cl) sum += t;
                ++cases;
                if (cl != ref || sum != n || chunk > cap || (head != 0 && head >= chunk)) ++bad;
                // the old plan: the last chunk's start from a head the prompt path clamps to the chunk
                if (head == 0 && n > chunk) {
                    const int64_t h2 = n > cap && n >= 2 * kHead && cap > kHead ? kHead : 0;
                    if (h2 > 0) {
                        const int64_t old_last = h2 + ((n - h2 - 1) / chunk) * chunk;
                        if (old_last != n - ref.back()) ++old_off;
                    }
                }
            }
        CHECK(cases > 100000);
        CHECK(bad == 0);
        CHECK(old_off > 0);   // the band the old arithmetic missed is inside this range
        // one of the failing reads: a 16640-token cap, a 235161-token part
        const auto [c1, h1] = prompt_one_card_chunks(235161, 16640, true, kHead);
        CHECK(c1 < kHead && h1 == 0);
    }
    if (g_fail) { std::printf("multi_gpu_plan_test: %d failures\n", g_fail); return 1; }
    std::printf("multi_gpu_plan_test: ok\n");
    return 0;
}
