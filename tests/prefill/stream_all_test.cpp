// tests/prefill/stream_all_test.cpp - stream_all_min_for: the stream-all chunk size priced from the link.
//
// Checks that the size
//   1. is on the 256-token grid and grows with the copy time (more bytes, or a slower link);
//   2. is clamped to the floor (a small copy) and to the default (a long copy);
//   3. is the default when the link rate or the bytes are unknown (zero or negative).
#include "basalt/prefill/stream_all.hpp"

#include <cstdio>

using basalt::prefill::kStreamAllDefault;
using basalt::prefill::kStreamAllFloor;
using basalt::prefill::stream_all_min_for;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr uint64_t kGiB = 1ull << 30;
}  // namespace

int main() {
    std::printf("stream_all_test\n");
    bool grid = true, rising = true;
    int64_t prev = 0;
    for (uint64_t gib = 1; gib <= 80; ++gib) {
        const int64_t t = stream_all_min_for(gib * kGiB, 57.5);
        grid = grid && t % 256 == 0;
        rising = rising && t >= prev;
        prev = t;
    }
    check(grid, "every size is on the 256-token grid");
    check(rising, "more non-resident bytes never lower the size");
    check(stream_all_min_for(37 * kGiB, 28.0) >= stream_all_min_for(37 * kGiB, 57.5), "a slower link never lowers the size");
    const int64_t mid = stream_all_min_for(37 * kGiB, 57.5);
    check(mid > kStreamAllFloor && mid < kStreamAllDefault, "a mid-size copy prices between the floor and the default");
    check(stream_all_min_for(1 * kGiB, 57.5) == kStreamAllFloor, "a small copy is clamped to the floor");
    check(stream_all_min_for(200 * kGiB, 8.0) == kStreamAllDefault, "a long copy is clamped to the default");
    check(stream_all_min_for(~0ull, 1e-9) == kStreamAllDefault, "a huge copy time is clamped before the integer cast");
    check(stream_all_min_for(37 * kGiB, 0.0) == kStreamAllDefault, "an unknown link keeps the default");
    check(stream_all_min_for(37 * kGiB, -1.0) == kStreamAllDefault, "a negative link rate keeps the default");
    check(stream_all_min_for(0, 57.5) == kStreamAllDefault, "no non-resident bytes keeps the default");
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
