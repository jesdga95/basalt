// tests/program/probe_derive_test.cpp - the worker pick, the balance share and the suggestion lines.
#include "basalt/program/probe_derive.hpp"

#include <cmath>
#include <cstdio>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    ProbeRates r;
    r.pool_workers = r.active_workers = 15;
    for (auto w : {std::pair<int, double>{1, 80}, {2, 62}, {4, 54}, {8, 50.5}, {15, 50}}) {
        r.cpu_expert.push_back({w.first, 8, w.second});
        r.cpu_expert.push_back({w.first, 1, w.second + 3});   // another entry count is ignored
    }
    WorkerPick p;
    // 50.5 is within 2% of 50: the smallest count that is
    CHECK(pick_workers(r, 8, 0.02, p) && p.workers == 8 && p.us_best == 50 && p.us_pick == 50.5);
    CHECK(!suggest_workers(r, p).empty() && suggest_workers(r, p)== "--pool-workers 8 (50 us/expert, best 50 us)");
    // 54 is not
    r.cpu_expert[6].per_entry_us = 56;
    CHECK(pick_workers(r, 8, 0.02, p) && p.workers == 15);
    CHECK(suggest_workers(r, p).empty());   // the pick is the current count: nothing to say
    CHECK(pick_workers(r, 8, 0.10, p) && p.workers == 4);
    CHECK(!pick_workers(r, 5, 0.02, p));
    ProbeRates none;
    CHECK(!pick_workers(none, 8, 0.02, p));

    CHECK(std::fabs(balance_share(1000, 1000) - 0.5) < 1e-9);
    CHECK(std::fabs(balance_share(2127, 2084) - 0.5) < 1e-9);
    CHECK(std::fabs(balance_share(1000, 3000) - 0.25) < 1e-9);
    CHECK(balance_share(100, 100000) == 0.05 && balance_share(100000, 100) == 0.95);
    CHECK(balance_share(0, 5) == 0.0);

    std::vector<ShareTrial> t = {{0.0, 2100}, {1.0, 2100}, {0.6, 2000}, {0.5, 1900}, {0.5, 1860}, {0.6, 1900}};
    CHECK(std::fabs(mean_ms_at(t, 0.6) - 1950) < 1e-9 && std::fabs(mean_ms_at(t, 0.5) - 1880) < 1e-9);
    CHECK(mean_ms_at(t, 0.3) == 0.0);
    CHECK(std::fabs(share_gain(t, 0.6, 0.5) - (1950.0 / 1880.0 - 1.0)) < 1e-9);
    CHECK(share_gain(t, 0.6, 0.3) == 0.0);
    CHECK(suggest_share(0.5, 0.6, 0.037) == "--prefill-cpu-share 0.50 (prefill +4%)");
    CHECK(suggest_share(0.6, 0.6, 0.1).empty());

    // the adaptive tier: the band from two baseline passes, at least the floor
    CHECK(std::fabs(adapt_band(250, 255, 0.01) - 5.0 / 252.5) < 1e-9);
    CHECK(adapt_band(250, 250, 0.01) == 0.01);
    const AdaptTrial base{1, 96, 0.85, false, 0};
    std::vector<AdaptTrial> at = {{1, 96, 0.85, false, 250}, {64, 256, 0.70, false, 190}, {1, 160, 0.85, false, 253},
                                  {1, 96, 0.85, false, 252}};
    CHECK(suggest_adapt(base, 251, 0.01, at).empty());   // 253 is inside the band
    at[2].tps = 262;
    CHECK(suggest_adapt(base, 251, 0.01, at) == "--adapt-swaps 160 (adapt test +4%)");
    at[1].tps = 280;
    CHECK(suggest_adapt(base, 251, 0.01, at) == "--adapt-every 64 --adapt-swaps 256 --adapt-decay 0.70 (adapt test +12%)");
    at.push_back({1, 96, 0.85, true, 300});   // the saved profile is never the adapt pick
    CHECK(suggest_adapt(base, 251, 0.01, at) == "--adapt-every 64 --adapt-swaps 256 --adapt-decay 0.70 (adapt test +12%)");
    CHECK(suggest_saved_profile("p.bin", 251, 262, 0.01) == "--expert-profile-save p.bin (adapt test +4%)");
    CHECK(suggest_saved_profile("p.bin", 251, 240, 0.01) == "no --expert-profile-save (adapt test +5% on the model's profile)");
    CHECK(suggest_saved_profile("p.bin", 251, 252, 0.01).empty());
    CHECK(format_adapt_line({{1, 96, 0.85, false, 250}, {64, 256, 0.7, false, 190}, {1, 96, 0.85, false, 252},
                             {1, 96, 0.85, true, 255}}, 0.01) ==
          "1/96/0.85 250 252 | 64/256/0.70 190 | saved profile 255 tok/s (band 1.0%)");

    std::printf(g_fail ? "probe_derive_test: FAILED\n" : "probe_derive_test: ok\n");
    return g_fail ? 1 : 0;
}
