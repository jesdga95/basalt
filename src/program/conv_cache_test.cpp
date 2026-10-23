// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain - stamps[0] is
// the root, the deepest point every request so far has shared (the end of the system prompt, in practice).
// A stamp advances on creation, when a new checkpoint lands on an existing length, and when a request
// mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim.
#include "basalt/program/conv_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using basalt::program::conv_cache::eviction_victim;
using basalt::program::conv_cache::context_admits;
using basalt::program::conv_cache::kAdmitSlack;
using basalt::program::conv_cache::reply_keep;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("conv_cache_test\n");
    {
        const std::vector<uint64_t> stamps = {1, 2, 3, 4, 5, 6, 7};   // chat A: root + six turns, one over
        check(eviction_victim(stamps.data(), stamps.size(), 6) == 1,
              "over budget with fresh stamps: the oldest leaf leaves (the FIFO it was)");
    }
    {
        // chat B mounts through the root long after it was created; its stamp stays ancient
        const std::vector<uint64_t> stamps = {1, 100, 101, 102, 103, 104, 105};
        const size_t v = eviction_victim(stamps.data(), stamps.size(), 6);
        check(v == 1, "the root's stale stamp never makes it the victim");
        check(v != 0, "the root - the shared prefix - is never the victim");
    }
    {
        // the same chain after mounting checkpoint 1: its stamp jumps, the rotation moves to the next leaf
        const std::vector<uint64_t> stamped = {1, 8, 3, 4, 5, 6, 7};
        check(eviction_victim(stamped.data(), stamped.size(), 6) == 2,
              "after a mount, the rotation takes the next least recently used leaf");
    }
    {
        const std::vector<uint64_t> stamps = {1, 90, 91, 92};
        check(eviction_victim(stamps.data(), stamps.size(), 2) == 1,
              "a two-slot budget keeps root + newest and drops the leaf in between");
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0,
              "a one-slot budget has no pin: the oldest leaves, the newest point stays");
    }
    {
        const std::vector<uint64_t> stamps = {7, 3, 3, 5};
        const size_t a = eviction_victim(stamps.data(), stamps.size(), 4);
        const size_t b = eviction_victim(stamps.data(), stamps.size(), 4);
        check(a == 1 && b == 1, "ties among equally stale leaves: the earlier index, deterministically");
    }
    {
        // The state keeps one row past the row of the reply's last produced token, so the last pick itself is never
        // kept: a stop token ends the reply (it is the next row, never committed), the accepted tokens after it were
        // never emitted, and a reply that reaches its cap has the same shape.  The first row is always kept.
        const std::vector<int64_t> stops = {248044, 248046};
        const auto keep = [&](std::initializer_list<int32_t> picks, int64_t produced, int64_t max_new,
                              size_t n_stops = 2) {
            const std::vector<int32_t> p(picks);
            return reply_keep(p.data(), (int) p.size(), stops.data(), n_stops, produced, max_new);
        };
        check(keep({10, 11, 12}, 0, 100) == 3, "a window with no stop token commits every accepted row");
        check(keep({248046, 11, 12}, 0, 100) == 1, "the first pick stops: the row before it is kept, not the pick");
        check(keep({10, 248044, 12, 13}, 0, 100) == 2, "the stop token and the accepted tokens after it are not kept");
        check(keep({10, 11, 248046}, 0, 100) == 3, "a stop token as the last accepted pick is not kept either");
        check(keep({10, 11, 12}, 3, 4) == 1, "the row of the token that reaches the cap is kept, the rest were not emitted");
        check(keep({10, 11, 12}, 2, 4) == 2, "a cap reached inside the window keeps up to its row");
        check(keep({248046, 11, 12}, 0, 100, 0) == 3, "the scoring path ignores stop tokens and commits every row");
        check(keep({10, 11}, 0, 1) == 1, "a cap of one token keeps the first row only");
        // The batch path calls this function now, so it must reproduce what that loop computed on its own: walk every
        // accepted width, every stop position and every cap, and compare against the loop's arithmetic written out.
        const auto batch_loop = [&](const std::vector<int32_t>& picks, int64_t produced, int64_t max_new) {
            int k = (int) picks.size();
            for (int e = 0; e < k; ++e) {
                const bool eos = picks[(size_t) e] == 248044 || picks[(size_t) e] == 248046;
                const bool fin = eos || produced + e + 1 >= max_new;
                if (fin) { k = e + 1; break; }
            }
            return k;
        };
        int compared = 0;
        for (int accepted = 1; accepted <= 8; ++accepted) {
            for (int stop_at = -1; stop_at < accepted; ++stop_at) {   // -1: no stop token in the window
                for (int64_t produced = 0; produced <= 3; ++produced) {
                    for (int64_t max_new = 1; max_new <= 12; ++max_new) {
                        std::vector<int32_t> picks((size_t) accepted, 10);
                        if (stop_at >= 0) picks[(size_t) stop_at] = 248046;
                        const int shared = reply_keep(picks.data(), accepted, stops.data(), stops.size(), produced, max_new);
                        const int own = batch_loop(picks, produced, max_new);
                        if (shared != own) {
                            std::printf("  reply_keep(%d, stop=%d, produced=%lld, max_new=%lld) = %d, batch loop = %d\n",
                                        accepted, stop_at, (long long) produced, (long long) max_new, shared, own);
                        }
                        check(shared == own, "the shared rule reproduces the batch loop's committed count");
                        ++compared;
                    }
                }
            }
        }
        check(compared == 44 * 4 * 12, "the table covered every accepted width, stop position and cap");
    }
    {
        // Acceptance boundaries: exactly at ctx is admitted, the first count past it is not. A test that only checked
        // the middle would not catch a `>=` regression, so the exact points are the ones asserted.
        check(context_admits(8, 0, 16), "prompt + slack exactly at the context is admitted");
        check(!context_admits(9, 0, 16), "one token past the context is refused");
        check(context_admits(0, 8, 16), "an empty prompt with the budget exactly at the context is admitted");
        check(!context_admits(0, 9, 16), "a budget one past the context is refused");
        check(context_admits(4, 4, 16), "prompt and budget sharing the slack are admitted at the boundary");
        check(!context_admits(4, 5, 16), "one over that shared boundary is refused");
        check(kAdmitSlack == 8, "the reserved slack is the eight tokens the call sites used inline");
        check(context_admits(0, 0, kAdmitSlack), "an empty prompt with no budget fits the slack");
        check(!context_admits(0, 0, kAdmitSlack - 1), "a context smaller than the slack admits nothing");
        check(!context_admits(-1, 100, 16), "a negative prompt cannot slip past the boundary");
        for (int64_t ctx = kAdmitSlack; ctx < 40; ++ctx) {   // the exact point, for every context in a small range
            check(context_admits(ctx - kAdmitSlack, 0, ctx) && !context_admits(ctx - kAdmitSlack + 1, 0, ctx),
                  "the boundary is the first count past the context, for every context in the range");
        }
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
