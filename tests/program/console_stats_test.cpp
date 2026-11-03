// tests/program/console_stats_test.cpp - the console's per-slot counts (console_stats.hpp) and the expert map's sample
// (core/expert_map.hpp): windows and drafts by position per slot, a round's lookups to every slot in it, a request's
// start from zero, the lines' text; the sample's spread, determinism and counting.
#include "console_stats.hpp"
#include "basalt/core/expert_map.hpp"

#include <cstdio>
#include <set>
#include <string>

using namespace basalt::program::serve;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {

void slot_counts() {
    ConsoleStats s;
    s.on = true;
    // a solo window of slot 0: 3 drafts, 2 kept, 3 tokens sent; its lookups are its own
    s.lookups_begin({0, 0, 0, 0});
    s.window(0, 3, 2, 3);
    s.lookups_end({8, 10, 2, 1});
    // a round of slots 0 and 2: both get its lookups
    s.lookups_begin({8, 10, 2, 1});
    s.window(0, 2, 2, 3);
    s.window(2, 1, 0, 1);
    s.lookups_end({14, 20, 2, 1});
    const SlotTally* a = s.slot(0);
    const SlotTally* c = s.slot(2);
    CHECK(a && c && s.slot(1) && s.slot(1)->windows == 0);
    CHECK(a->windows == 2 && a->tokens == 6 && a->positions == 3);
    CHECK(a->offered[0] == 2 && a->offered[1] == 2 && a->offered[2] == 1);
    CHECK(a->accepted[0] == 2 && a->accepted[1] == 2 && a->accepted[2] == 0);
    CHECK(a->look.hits == 14 && a->look.lookups == 20 && a->look.offloaded == 2 && a->look.files == 1);
    CHECK(c->look.hits == 6 && c->look.lookups == 10 && c->look.offloaded == 0 && c->windows == 1);
    CHECK(s.slot_line(0) == "XSLOT 0 2 6 14 20 2 1 2/2,2/2,0/1\n");
    CHECK(s.slot_line(2) == "XSLOT 2 1 1 6 10 0 0 0/1\n");
    // the lines of the slots that moved, once
    const std::string lines = s.slot_lines();
    CHECK(lines == s.slot_line(0) + s.slot_line(1) + s.slot_line(2) || lines == s.slot_line(0) + s.slot_line(2));
    CHECK(s.slot_lines().empty());
    // a request starts in slot 0: from zero, and its line goes out
    s.start(0);
    CHECK(s.slot(0)->windows == 0 && s.slot(0)->positions == 0);
    CHECK(s.slot_lines() == "XSLOT 0 0 0 0 0 0 0 -\n");
    // accepted never above offered, positions past the table clipped
    s.window(0, 40, 50, 41);
    CHECK(s.slot(0)->positions == SlotTally::kPositions && s.slot(0)->accepted[SlotTally::kPositions - 1] == 1);
    // a window outside any round adds no lookups
    s.window(1, 0, 0, 1);
    CHECK(s.slot(1)->look.lookups == 0);
}

void timing() {
    ConsoleStats s;
    const auto t0 = ConsoleStats::Clock::now();
    CHECK(!s.due(t0, true));   // off until STATS
    s.on = true;
    CHECK(s.due(t0, false));
    CHECK(!s.due(t0 + std::chrono::milliseconds(100), false));
    CHECK(s.due(t0 + std::chrono::milliseconds(100), true));
    CHECK(s.due(t0 + std::chrono::milliseconds(700), false));
}

void map_line() {
    CHECK(ConsoleStats::map_line(20480, {0, 1, 2, 3}, {5, 0, 1, 2}) == "XMAP 20480 0123 5,0,1,2\n");
}

void sample() {
    const std::vector<int32_t> p = basalt::core::expert_map_pairs(20480, 640);
    CHECK(p.size() == 640);
    CHECK(p == basalt::core::expert_map_pairs(20480, 640));   // the same on every start
    CHECK(p.front() == 16 && p.back() == 20480 - 16);         // the middle of each equal stretch
    std::set<int32_t> layers;
    for (const int32_t x : p) layers.insert(x / 512);
    CHECK(layers.size() == 40);                               // every layer of 40 x 512 is in it
    for (size_t i = 1; i < p.size(); ++i) CHECK(p[i] > p[i - 1]);
    CHECK(basalt::core::expert_map_pairs(100, 640).size() == 100);   // fewer pairs than cells: all of them
    CHECK(basalt::core::expert_map_pairs(0, 640).empty());

    basalt::core::ExpertMapTally t;
    t.note(0, 16);   // before open: nothing
    CHECK(!t.is_open());
    t.open(40, 512);
    CHECK(t.total() == 20480 && t.pairs() == p);
    t.note(0, 16);
    t.note(0, 16);
    t.note(0, 17);   // not sampled
    t.note(39, 496);
    std::vector<uint32_t> r;
    t.take(r);
    CHECK(r.size() == 640 && r[0] == 2 && r[639] == 1 && r[1] == 0);
    t.take(r);
    CHECK(r[0] == 0 && r[639] == 0);   // counts start again
}

}  // namespace

int main() {
    slot_counts();
    timing();
    map_line();
    sample();
    if (g_fail) return 1;
    std::printf("console_stats_test: ok\n");
    return 0;
}
