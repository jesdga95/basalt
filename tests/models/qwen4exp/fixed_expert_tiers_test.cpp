// Fixed expert tiers gate the peer and mixed-size staging extension without changing the original single-card batch path.
#include "basalt/models/qwen4exp/options.hpp"

#include <cstdio>

using basalt::models::qwen4exp::Options;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    Options o;
    CHECK(!o.fixed_expert_tiers());   // the default adaptive tiers

    o.adapt_every = 0;
    CHECK(o.fixed_expert_tiers());

    o.adapt_every = 4;
    o.adapt_swaps = 0;
    CHECK(o.fixed_expert_tiers());   // the peer inherits the main card's zero

    o.peer_device = 1;
    o.peer_adapt_swaps = 96;
    CHECK(!o.fixed_expert_tiers());   // an adaptive peer still moves experts and uses the link

    o.adapt_swaps = 96;
    o.peer_adapt_swaps = 0;
    CHECK(!o.fixed_expert_tiers());

    o.adapt_swaps = 0;
    o.peer_adapt_swaps = 96;
    o.peer_device = -1;
    CHECK(o.fixed_expert_tiers());   // a peer-only value is irrelevant without a peer

    o.peer_devices = {1};
    CHECK(!o.fixed_expert_tiers());

    if (g_fail) return 1;
    std::printf("fixed_expert_tiers_test: ok\n");
    return 0;
}
