// src/program/prefix_reuse_test.cpp - which part of a held sequence the next prompt keeps.
#include "basalt/program/prefix_reuse.hpp"

#include <cstdio>
#include <vector>

using basalt::program::prefix_reuse::reusable;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    const std::vector<int32_t> held = {1, 2, 3, 4};
    check(reusable(held, {1, 2, 3, 4, 5, 6}) == 4, "a prompt that extends the held tokens keeps all of them");
    check(reusable(held, {1, 2, 3, 4, 5}) == 4, "one new token is enough");
    check(reusable(held, {1, 2, 3, 4}) == 0, "an identical prompt has nothing left to decode: fresh");
    check(reusable(held, {1, 2, 3}) == 0, "a shorter prompt: fresh");
    check(reusable(held, {1, 2, 9, 4, 5}) == 0, "a divergence inside the held tokens: fresh");
    check(reusable({}, {1, 2}) == 0, "nothing held: fresh");
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
