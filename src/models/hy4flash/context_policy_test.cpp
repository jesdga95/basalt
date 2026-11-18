// src/models/hy4flash/context_policy_test.cpp - the context sizes and the refusal rule.
#include "models/hy4flash/context_policy.hpp"

#include <cstdio>

using namespace basalt::models::hy4flash::ctxp;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    // 12 sparse layers: 13,824 B of rows per position (design: 14.5 GB at 1M, BF16)
    check(cache_bytes(1 << 20, 12) >= 12ull * (1 << 20) * 1152 && cache_bytes(1 << 20, 12) < 17ull << 30, "1M positions: BF16 rows plus pooled keys");
    check(ring_bytes(512, 30) == 30ull * 512 * 1152, "window rings");
    const uint64_t gib = 1ull << 30;
    check(plan_context(8192, 1 << 20, 20 * gib, 12).context == 8192, "8K fits");
    check(!plan_context(1 << 20, 1 << 20, 20 * gib, 12).error.empty(), "1M refused with 20 GiB free (half rule)");
    check(plan_context(1 << 20, 1 << 20, 40 * gib, 12).context == (1 << 20), "1M accepted with 40 GiB free");
    check(!plan_context(512, 1 << 20, 40 * gib, 12).error.empty(), "below the least refused");
    check(!plan_context(2 << 20, 1 << 20, 40 * gib, 12).error.empty(), "past the native maximum refused");
    return failures ? 1 : 0;
}
