// tests/program/image_loan_test.cpp - the room an image encode may borrow, the elastic KV give-back order and the
// elastic KV's ceiling.
#include "basalt/program/image_loan.hpp"

#include <algorithm>
#include <cstdio>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    // every expert in RAM: the whole cache is lendable
    {
        const std::vector<int32_t> slot_of = {0, 5, -1, 9};
        CHECK(image_lend_floor(slot_of, {}, 10) == 0);
        CHECK(image_lend_floor(slot_of, {1, 1, 1, 1}, 10) == 0);
    }
    // an expert only in VRAM pins every slot up to its own
    {
        const std::vector<int32_t> slot_of = {0, 5, 7, 9};
        CHECK(image_lend_floor(slot_of, {0, 1, 1, 1}, 10) == 1);
        CHECK(image_lend_floor(slot_of, {1, 0, 1, 1}, 10) == 6);
        CHECK(image_lend_floor(slot_of, {1, 0, 0, 1}, 10) == 8);
        // a slot past the live cache (lent to the KV) does not count
        CHECK(image_lend_floor(slot_of, {1, 1, 1, 0}, 9) == 0);
        // an unplaced expert without a RAM copy does not count either
        CHECK(image_lend_floor({-1, 3}, {0, 1}, 10) == 0);
    }
    // idle sessions first, largest spare first; never below keep or the floor; nothing spare is left out
    {
        std::vector<KvHold> h(4);
        h[0] = {20000, 19000, true};    // decoding, 1000 spare
        h[1] = {40000, 8000, false};    // idle, 32000 spare
        h[2] = {4096, 100, false};      // idle at the floor: nothing
        h[3] = {12000, 2000, false};    // idle, 10000 spare
        const auto o = kv_giveback_order(h, 4096);
        CHECK(o.size() == 3);
        if (o.size() == 3) {
            CHECK(o[0].first == 1 && o[0].second == 8000);
            CHECK(o[1].first == 3 && o[1].second == 4096);
            CHECK(o[2].first == 0 && o[2].second == 19000);
        }
    }
    // a session that holds all it backs gives nothing
    {
        std::vector<KvHold> h = {{200000, 200000, false}, {28672, 30000, true}};
        CHECK(kv_giveback_order(h, 4096).empty());
    }
    // the cache's tail in granules, down to the floor: 3 MiB slots over 2 MiB granules, a shrink keeping whole
    // granules (what is left of a slot cut in two is not a slot)
    {
        const int64_t mib = 1 << 20, gran = 2 * mib, mapped = 15 * gran;
        const auto slots_after = [&](int64_t keep) { return (keep + gran - 1) / gran * gran / (3 * mib); };
        CHECK(kv_cache_spare_granules(mapped, gran, 4, slots_after) == 9);    // 4 slots need 12 MiB: 6 granules stay
        CHECK(kv_cache_spare_granules(mapped, gran, 5, slots_after) == 7);    // 15 MiB: 8 granules (16 MiB) stay
        CHECK(kv_cache_spare_granules(mapped, gran, 0, slots_after) == 15);
        CHECK(kv_cache_spare_granules(mapped, gran, 11, slots_after) == 0);   // already below the floor: nothing
        CHECK(kv_cache_spare_granules(0, gran, 0, slots_after) == 0);
    }
    // the ceiling: each session in turn backs what the spare bytes pay for, on the 256-cell grid, up to its context
    {
        const auto linear = [](const std::vector<int64_t>& backed) {
            return [backed](size_t i, int64_t cells) { return std::max<int64_t>(cells - backed[i], 0) * 1000; };
        };
        // one session at its floor: 100000 more cells' bytes, 99840 on the grid
        std::vector<int64_t> b = {32768}, m = {262144};
        CHECK(kv_ceiling_cells(b, m, 100000LL * 1000, linear(b)) == 32768 + 99840);
        CHECK(kv_ceiling_cells(b, m, 0, linear(b)) == 32768);
        CHECK(kv_ceiling_cells(b, m, 1LL << 40, linear(b)) == 262144);   // the whole context fits
        // a context off the grid is reached whole
        m = {40000};
        CHECK(kv_ceiling_cells(b, m, 1LL << 40, linear(b)) == 40000);
        // two slots: the first fills to its context, the second takes the rest
        b = {4096, 4096};
        m = {32768, 32768};
        CHECK(kv_ceiling_cells(b, m, 40000LL * 1000, linear(b)) == 32768 + 4096 + 11264);
        // a session that is not elastic holds its whole context
        b = {-1, 4096};
        CHECK(kv_ceiling_cells(b, m, 0, linear(b)) == 32768 + 4096);
    }
    if (g_fail == 0) std::printf("image_loan_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
