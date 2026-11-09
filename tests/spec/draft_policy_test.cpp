// tests/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's? How many of a drafter's
// stated-chance drafts does a window take?
//
// Simulated rounds with measured window costs (about 10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap;
//   6. a size measured too dear long ago is measured again.
// Drafter windows (stated chances, DeepSeek's measured shape: about 3.6 ms more per row on 13 ms):
//   7. drafts that are always accepted fill the window, which grows past a size measured too dear at the start;
//   8. middling chances stop the window where another row costs more than it returns;
//   9. a head that overstates its chances is calibrated down, and the window shrinks with it;
//  10. rows that cost little take more drafts at the same chances;
//  11. never more rows than the drafts, nor beyond the cap.
#include "basalt/spec/draft_policy.hpp"

#include <cstdio>
#include <random>

using basalt::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
double ds_cost(int t) { return 13.0 + 3.6 * (t - 1); }  // DeepSeek's round: verify + commit + draft
// rounds of a drafter whose drafts are each accepted with chance `q` (a fixed-seed draw), stating `stated`; returns the
// window the policy picks most often over the last half
int run(DraftPolicy& p, float stated, float q, int rounds, double (*c)(int)) {
    const float ch[5] = {stated, stated, stated, stated, stated};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    int seen[DraftPolicy::kMaxT + 1] = {};
    for (int i = 0; i < rounds; ++i) {
        const int t = p.window(ch, 5);
        int a = 0;
        while (a < t - 1 && u(rng) < q) ++a;
        p.observe_window(ch, t, a, c(t));
        if (2 * i >= rounds) ++seen[t];
    }
    int best = 1;
    for (int t = 1; t <= DraftPolicy::kMaxT; ++t)
        if (seen[t] > seen[best]) best = t;
    return best;
}
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
        for (int i = 0; i < 300; ++i) p.observe(false, 4, 3, 0, cost(4));
        check(p.choose(4, 5, 40).lookup, "a size measured long ago is probed again");
    }
    {
        DraftPolicy p(6);
        const float ch[5] = {0.999f, 0.999f, 0.999f, 0.999f, 0.999f};
        for (int i = 0; i < 20; ++i) {   // a cold start: size 6 measured dear beside size 1
            p.observe_window(nullptr, 1, 0, ds_cost(1));
            p.observe_window(ch, 6, 5, 3.0 * ds_cost(6));
        }
        check(p.window(ch, 5) < 6, "drafter: a size measured dear is not taken at first");
        const int t = run(p, 0.999f, 1.0f, 800, ds_cost);
        check(t == 6, "drafter: accepted drafts fill the window once its cost is known");
        check(p.cost_ms(6) < 1.2 * ds_cost(6), "drafter: the dear size's cost is measured again");
    }
    {
        DraftPolicy p(6);
        const float ch[5] = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f};
        const int t = run(p, 0.6f, 0.6f, 300, ds_cost);
        // E/cost at q 0.6: t1 1/13 = .0769, t2 1.6/16.6 = .0964, t3 1.96/20.2 = .0970, t4 2.18/23.8 = .0914
        check(t == 2 || t == 3, "drafter: middling chances stop at 2 or 3 rows");
        check(p.window(ch, 5) <= 3, "drafter: no deeper than the rows pay");
    }
    {
        DraftPolicy p(6);
        const float ch[5] = {0.95f, 0.95f, 0.95f, 0.95f, 0.95f};
        const int t0 = run(p, 0.95f, 0.95f, 200, ds_cost);
        DraftPolicy o(6);
        const int t1 = run(o, 0.95f, 0.45f, 300, ds_cost);   // stated 0.95, accepted 0.45
        check(o.chance_rate(0.95f) < 0.7, "drafter: an overstated chance is calibrated down");
        check(t1 < t0, "drafter: the window shrinks with the calibrated chance");
        (void) ch;
    }
    {
        DraftPolicy p(6), q(6);
        const int t_dear = run(p, 0.75f, 0.75f, 300, ds_cost);
        const int t_cheap = run(q, 0.75f, 0.75f, 300, [](int t) { return 13.0 + 0.8 * (t - 1); });
        check(t_cheap > t_dear, "drafter: cheap rows take more drafts at the same chances");
    }
    {
        DraftPolicy p(4);
        const float ch[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        check(p.window(ch, 2) <= 3, "drafter: never more rows than drafts + 1");
        check(p.window(ch, 5) <= 4, "drafter: never beyond the cap");
        check(p.window(ch, 0) == 1, "drafter: no drafts, one row");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
