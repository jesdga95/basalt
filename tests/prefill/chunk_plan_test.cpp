// tests/prefill/chunk_plan_test.cpp - the chunks a prompt part is read in on one card, and the draft layer's rows a split read
// across the cards hands back.
#include "basalt/prefill/chunk_plan.hpp"

#include <cstdio>
#include <vector>

using namespace basalt::prefill;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    // the draft layer's rows a split read holds back: two chunks in place, more past the chunk in both buffers
    {
        auto room = [](size_t n, int64_t chunk, int64_t hand) {
            return split_draft_room(std::vector<int64_t>(n, chunk), chunk, hand);
        };
        CHECK(room(2, 16384, 16384) == 16384);
        CHECK(room(3, 16384, 16384) == 32768);
        CHECK(room(4, 16384, 16384) == 32768);
        // a deep part: smaller chunks than the buffers were made for leave the rest of both buffers
        std::vector<int64_t> deep(17, 14848);
        deep.front() = 16384;
        deep.back() = 9000;
        CHECK(split_draft_room(deep, 14848, 27392) == 2 * 14848 + 2 * (27392 - 14848));
        CHECK(split_draft_room(deep, 14848, 27392) >= 32832);
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
                const auto [chunk, head] = one_card_chunks(n, cap, true, kHead);
                const std::vector<int64_t> cl = chunk_list(n, chunk, head);
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
        const auto [c1, h1] = one_card_chunks(235161, 16640, true, kHead);
        CHECK(c1 < kHead && h1 == 0);
    }
    if (g_fail) return 1;
    std::printf("chunk_plan_test: ok\n");
    return 0;
}
