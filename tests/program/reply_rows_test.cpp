// tests/program/reply_rows_test.cpp - the rows a verify window keeps (reply_rows.hpp).
// Fixed cases for each way a window ends, then random requests run window by window as request_decode.cpp runs them:
// after every window the kept rows are the prompt and the picks sent but the last, the next token is the last pick
// sent at the next position, and a window that ends the reply keeps no row past its last pick.
#include "reply_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::program::serve;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {

const std::vector<int64_t> kEos = {9, 8};

uint64_t g_rng = 0;
uint64_t next_u64() {   // splitmix64: the same sequence on every compiler
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
int below(int n) { return (int) (next_u64() % (uint64_t) n); }

void fixed_cases() {
    const int32_t picks[8] = {1, 2, 3, 4, 5, 6, 7, 1};
    ReplyRows r = reply_rows(picks, 0, 5, 0, 100, kEos, false);   // mid reply: every accepted row, the last pick next
    CHECK(r.emit == 6 && !r.ends && r.keep == 6);
    r = reply_rows(picks, 0, 2, 0, 100, kEos, false);   // a rejection after row 2
    CHECK(r.emit == 3 && !r.ends && r.keep == 3);
    r = reply_rows(picks, 0, 5, 98, 100, kEos, false);   // max_new two picks in: rows 0-1, pick 1 is the next token
    CHECK(r.emit == 2 && r.ends && r.keep == 2);
    const int32_t eos_mid[6] = {1, 2, 9, 4, 5, 6};
    r = reply_rows(eos_mid, 0, 5, 0, 100, kEos, false);   // an end token at pick 2: rows 0-2
    CHECK(r.emit == 3 && r.ends && r.keep == 3);
    const int32_t eos_last[4] = {1, 2, 3, 8};
    r = reply_rows(eos_last, 0, 3, 0, 100, kEos, false);   // an end token at the last accepted row: every row
    CHECK(r.emit == 4 && r.ends && r.keep == 4);
    r = reply_rows(picks, 4, 4, 0, 1, kEos, false);   // a first window (four prompt rows), one pick: all five rows
    CHECK(r.emit == 1 && r.ends && r.keep == 5);
    r = reply_rows(eos_mid, 0, 5, 99, 100, kEos, true);   // scored: every row, nothing ends
    CHECK(!r.ends && r.keep == 6);
}

// One request: a prompt, then windows whose drafts the target confirms at random, until max_new or an end token.
// The target's token at each position is fixed, so a confirmed draft is that token.
void random_request() {
    const int S = 1 + below(8);
    const int n = 1 + below(40);
    const int64_t max_new = 1 + below(30);
    std::vector<int32_t> seq;   // the conversation: prompt, then the target's tokens
    for (int i = 0; i < n + 64 + 8; ++i) seq.push_back(below(16) == 0 ? (int32_t) kEos[(size_t) below(2)] : 10 + below(50));
    for (int i = 0; i < n; ++i)
        if (seq[(size_t) i] < 10) seq[(size_t) i] = 10;
    const int fold = std::max(0, n - S);   // the first window holds the prompt's last rows
    std::vector<int32_t> consumed(seq.begin(), seq.begin() + fold), sent;
    int64_t p = fold, produced = 0;
    int32_t x = seq[(size_t) fold];
    bool first = true;
    std::vector<int32_t> window((size_t) S), picks((size_t) S);
    while (produced < max_new) {
        const int T = first ? n - fold : 1 + below(S);
        const int e0 = first ? T - 1 : 0;
        window[0] = x;
        for (int i = 1; i < T; ++i)   // prompt rows, or drafts that are right until a random miss
            window[(size_t) i] = i <= e0 || below(4) != 0 ? seq[(size_t) (p + i)] : 10 + below(50);
        int a = 0;
        for (int i = 0; i < T; ++i) picks[(size_t) i] = seq[(size_t) (p + i + 1)];
        if (e0 > 0) a = T - 1;
        else
            while (a < T - 1 && window[(size_t) a + 1] == picks[(size_t) a]) ++a;
        const ReplyRows r = reply_rows(picks.data(), e0, a, produced, max_new, kEos, false);
        CHECK(r.keep >= 1 && r.keep <= a + 1);
        for (int i = 0; i < r.keep; ++i) consumed.push_back(window[(size_t) i]);
        bool eos = false;
        for (int i = e0; i < e0 + r.emit; ++i) {
            sent.push_back(picks[(size_t) i]);
            eos = picks[(size_t) i] == kEos[0] || picks[(size_t) i] == kEos[1];
        }
        produced += r.emit;
        x = picks[(size_t) r.keep - 1];
        p += r.keep;
        first = false;
        // the kept rows are the prompt and every pick sent but the last; the last is the next token, at p
        CHECK((int64_t) consumed.size() == p);
        CHECK(x == sent.back());
        CHECK(p == n + (int64_t) sent.size() - 1);
        for (int64_t i = 0; i < p; ++i) CHECK(consumed[(size_t) i] == seq[(size_t) i]);
        CHECK(r.ends == (eos || produced >= max_new));
        if (r.ends) break;
    }
}

}  // namespace

int main() {
    fixed_cases();
    for (int i = 0; i < 20000 && g_fail == 0; ++i) random_request();
    if (g_fail != 0) {
        std::printf("reply_rows_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("reply_rows_test: ok\n");
    return 0;
}
