// src/models/inkling2/kernels/slices_test.cpp - the slice count a card's decode attention uses (slices.hpp): whole waves of
// the eight key/value heads' blocks at the plans' residencies, the larger count on a tie, within its bounds.
#include "models/inkling2/kernels/slices.hpp"

#include <cstdio>

using namespace basalt::models::inkling2;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    const std::vector<int> two_three = {2, 3};
    // 84 multiprocessors: 8 x 63 = 504 blocks, 3 waves at 2 a multiprocessor and 2 at 3
    expect(slice_count(84, two_three, 8, 32, 64) == 63, "84 SMs: 63");
    expect(slice_count(84, {2}, 8, 32, 64) == 63, "84 SMs, one plan: 63");
    // 170: 8 x 63 = 504 (2 waves of 340, 1 of 510) against 42's 336 (1 and 1) ties per slice: the larger
    expect(slice_count(170, two_three, 8, 32, 64) == 63, "170 SMs: 63");
    // 188: 8 x 47 = 376, one whole wave at 2 a multiprocessor
    expect(slice_count(188, two_three, 8, 32, 64) == 47, "188 SMs: 47");
    // 70: 8 x 52 = 416 blocks, 3 waves of 140 and 2 of 210: the least waves per slice
    const int c70 = slice_count(70, two_three, 8, 32, 64);
    expect(c70 == 52, "70 SMs: 52");
    // one plan alone may want another count: 170 SMs at 2 a multiprocessor take 42 (336 blocks, one wave)
    expect(slice_count(170, {2}, 8, 32, 64) == 42, "170 SMs, one plan: 42");
    std::printf("slice counts: 70 SMs %d, 84 SMs %d, 170 SMs %d, 188 SMs %d\n", c70, slice_count(84, two_three, 8, 32, 64),
                slice_count(170, two_three, 8, 32, 64), slice_count(188, two_three, 8, 32, 64));
    expect(slice_count(84, {}, 8, 32, 64) == 64, "no plan: the cap");
    if (failures) return 1;
    std::printf("inkling2_slices_test: ok\n");
    return 0;
}
