// src/models/inkling2/kernels/window_plan_test.cpp - a deep INT8 window's plan (window_plan.hpp): the register form with more
// blocks a multiprocessor, the second V tile on a tie; the rings when no register form beats them, when none fits, or when
// the register forms were not probed.
#include "models/inkling2/kernels/window_plan.hpp"

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
    expect(window_plan(true, 3, 3, 2) == WindowPlan::RegistersPingPong, "a tie takes the second V tile");
    expect(window_plan(true, 3, 2, 2) == WindowPlan::Registers, "the plain form where the second V tile fits fewer");
    expect(window_plan(true, 2, 3, 2) == WindowPlan::RegistersPingPong, "the second V tile where it fits more");
    expect(window_plan(true, 2, 2, 2) == WindowPlan::Rings, "no more blocks than the rings: the rings");
    expect(window_plan(true, 0, 0, 2) == WindowPlan::Rings, "no register form fits: the rings");
    expect(window_plan(false, 3, 3, 2) == WindowPlan::Rings, "not probed: the rings");
    if (failures) return 1;
    std::printf("inkling2_window_plan_test: ok\n");
    return 0;
}
