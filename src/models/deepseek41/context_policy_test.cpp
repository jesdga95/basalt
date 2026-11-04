// src/models/deepseek41/context_policy_test.cpp - the context a DeepSeek run gets from --max-context, the model's
// maximum and the free memory (host only).
#include "models/deepseek41/context_policy.hpp"

#include <cstdio>

using namespace basalt::models::deepseek41::ctxp;

static int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

int main() {
    const std::vector<int> ratios = {2, 2, 2, 1};   // the kv-source layers 2, 8, 14 and 20
    constexpr int64_t kNative = 1048576;
    constexpr uint64_t GiB = 1ull << 30;

    // cache sizes: 640 B per position at ratio 2, 1280 at ratio 1
    CHECK(cache_bytes(1000000, ratios) > 3ull * 640 * 1000000 + 1280ull * 1000000 - 16384);
    CHECK(cache_bytes(1000000, ratios) < 3ull * 640 * 1000000 + 1280ull * 1000000 + 16384);
    CHECK(cache_bytes(0, {}) == 0);
    CHECK(owns_cache(2, 2) && owns_cache(20, 1) && !owns_cache(3, 2) && !owns_cache(2, 0));

    // no flag: the model's maximum when the caches take at most a quarter of the free memory
    Plan p = plan_context(0, false, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == kNative && p.note.empty());
    // less memory lowers the default to a multiple of the step, never past the quarter
    p = plan_context(0, false, kNative, 8 * GiB, ratios);
    CHECK(p.error.empty() && p.context > 0 && p.context < kNative && p.context % kStep == 0);
    CHECK(context_bytes(p.context, ratios) <= 2 * GiB && !p.note.empty());
    p = plan_context(0, false, kNative, 1, ratios);
    CHECK(p.context == kMinContext);
    // a pack that names no length keeps the legacy cap
    p = plan_context(0, false, 0, 20 * GiB, ratios);
    CHECK(p.context == kLegacyContext);

    // an explicit value is honoured up to the model's maximum
    p = plan_context(32768, true, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == 32768 && p.note.empty());
    p = plan_context(kNative, true, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == kNative);
    p = plan_context(kNative + 1, true, kNative, 20 * GiB, ratios);
    CHECK(!p.error.empty() && p.context == 0);
    p = plan_context(100, true, kNative, 20 * GiB, ratios);
    CHECK(!p.error.empty());
    // more than half of the free memory is refused, a quarter or more warns
    p = plan_context(kNative, true, kNative, 6 * GiB, ratios);
    CHECK(!p.error.empty());
    p = plan_context(kNative, true, kNative, 12 * GiB, ratios);
    CHECK(p.error.empty() && !p.note.empty());

    if (g_fail == 0) std::printf("context_policy_test: ok\n");
    return g_fail ? 1 : 0;
}
