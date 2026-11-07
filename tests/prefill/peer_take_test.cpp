// tests/prefill/peer_take_test.cpp - the split of a layer's RAM experts between the main card and a second card (host only).
#include "basalt/prefill/peer_share.hpp"

#include <cstdio>

using basalt::prefill::peer_take;
using basalt::prefill::ShareRates;

static int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

int main() {
    const double blob = 8e6, in_row = 4608, out_row = 16384;
    // no rates: everything stays on the main card
    CHECK(peer_take(ShareRates{}, std::vector<int64_t>(100, 40), 0, 0, 0, blob, in_row, out_row) == 0);
    // equal links, products far faster than the bus, few rows: the bytes split about evenly
    {
        const ShareRates r{30e9, 30e9, 1e12, 1e12};
        const int k = peer_take(r, std::vector<int64_t>(300, 4), 0, 0, 0, blob, in_row, out_row);
        CHECK(k >= 140 && k <= 150);
    }
    // a link half as fast takes about a third
    {
        const ShareRates r{30e9, 15e9, 1e12, 1e12};
        const int k = peer_take(r, std::vector<int64_t>(300, 4), 0, 0, 0, blob, in_row, out_row);
        CHECK(k >= 95 && k <= 105);
    }
    // the second card's products bind: it takes what its rows allow, less than the bytes alone would give it
    {
        const ShareRates bus{30e9, 30e9, 1e12, 1e12}, slow{30e9, 30e9, 1e7, 2e5};
        const std::vector<int64_t> rows(300, 250);
        const int kb = peer_take(bus, rows, 0, 0, 0, blob, in_row, out_row);
        const int ks = peer_take(slow, rows, 0, 0, 0, blob, in_row, out_row);
        CHECK(ks < kb);
        CHECK(ks > 0);
    }
    // its own experts' rows already fill the second card: it takes no RAM expert
    {
        const ShareRates r{30e9, 30e9, 1e7, 1e6};
        CHECK(peer_take(r, std::vector<int64_t>(50, 10), 100, 0, 100000, blob, in_row, out_row) == 0);
    }
    // nothing to take
    CHECK(peer_take(ShareRates{30e9, 30e9, 1e12, 1e12}, {}, 10, 0, 10, blob, in_row, out_row) == 0);
    if (g_fail == 0) std::printf("peer_take_test: ok\n");
    return g_fail ? 1 : 0;
}
