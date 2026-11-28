// core::KvPool: takes stop at the budget with nothing counted, gives never go below zero, and a smaller budget leaves no
// free bytes until enough are given back (CPU only).
#include "basalt/core/kv_pool.hpp"

#include <cstdio>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    basalt::core::KvPool p;
    check(p.free() == 0 && !p.take(1) && p.take(0), "an empty pool takes nothing but zero");
    p.set_budget(100);
    check(p.take(60) && p.taken() == 60 && p.free() == 40, "a take within the budget");
    check(!p.take(41) && p.taken() == 60, "a take past the budget counts nothing");
    check(p.take(40) && p.free() == 0, "the last byte");
    p.give(30);
    check(p.taken() == 70 && p.free() == 30, "a give");
    p.set_budget(50);
    check(p.free() == 0 && !p.take(1), "a budget below what is taken leaves nothing free");
    p.give(1000);
    check(p.taken() == 0 && p.free() == 50, "a give never goes below zero");
    if (fails == 0) std::printf("kv_pool_test: ok\n");
    return fails == 0 ? 0 : 1;
}
