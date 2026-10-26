// tests/program/image_loan_test.cpp - the room an image encode may borrow and the elastic KV give-back order.
#include "basalt/program/image_loan.hpp"

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
    if (g_fail == 0) std::printf("image_loan_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
