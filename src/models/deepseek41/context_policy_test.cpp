// src/models/deepseek41/context_policy_test.cpp - the context a DeepSeek run gets from --max-context, the model's
// maximum and the free memory (host only).
#include "models/deepseek41/context_policy.hpp"

#include "basalt/program/options.hpp"

#include <cstdio>

using namespace basalt::models::deepseek41::ctxp;
using basalt::models::deepseek41::CacheFormat;

static int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

namespace {
Plan plan_context_f(int64_t requested, int64_t native, uint64_t free_bytes, const std::vector<int>& ratios) {
    return plan_context(requested, native, free_bytes, ratios, basalt::models::deepseek41::CacheFormat::Fp8);
}
}  // namespace

int main() {
    const std::vector<int> ratios = {2, 2, 2, 1};   // the kv-source layers 2, 8, 14 and 20
    constexpr int64_t kNative = 1048576;
    constexpr uint64_t GiB = 1ull << 30;

    // cache sizes: a packed row of 288 + 80 B, 184 B per position at ratio 2, 368 at ratio 1; fp16 rows 1024 + 256 B
    const CacheFormat F8 = CacheFormat::Fp8, F16 = CacheFormat::Fp16;
    CHECK(latent_row_bytes(F8) == 288 && index_row_bytes(F8) == 80 && latent_row_bytes(F16) == 1024 && index_row_bytes(F16) == 256);
    CHECK(cache_bytes(1000000, ratios, F8) > 3ull * 184 * 1000000 + 368ull * 1000000 - 16384);
    CHECK(cache_bytes(1000000, ratios, F8) < 3ull * 184 * 1000000 + 368ull * 1000000 + 16384);
    CHECK(cache_bytes(1000000, ratios, F16) > 3ull * 640 * 1000000 + 1280ull * 1000000 - 16384);
    CHECK(cache_bytes(1000000, ratios, F16) < 3ull * 640 * 1000000 + 1280ull * 1000000 + 16384);
    CHECK(cache_bytes(0, {}, F8) == 0);
    // --kv-resident: later latents in RAM, index keys on the card
    CHECK(resident_positions(1000000, 0) == 1000000 && resident_positions(1000000, 1000000) == 1000000 &&
          resident_positions(1000000, 65536) == 65536);
    CHECK(cache_bytes(1000000, ratios, F8, 1000000) == cache_bytes(1000000, ratios, F8) &&
          host_cache_bytes(1000000, ratios, F8, 0) == 0);
    CHECK(cache_bytes(1000000, ratios, F8, 65536) + host_cache_bytes(1000000, ratios, F8, 65536) == cache_bytes(1000000, ratios, F8));
    CHECK(host_cache_bytes(1000000, ratios, F8, 65536) == (3 * (500001 - 32769) + (1000001 - 65537)) * 288ull);
    CHECK(prompt_stage_bytes(1000000, ratios, F8, 0) == 0 && prompt_stage_bytes(1000000, ratios, F8, 65536) == 1000001ull * 288);
    CHECK(context_bytes(1000000, ratios, F8, 65536) == cache_bytes(1000000, ratios, F8, 65536) + workspace_bytes(1000000));
    // the array rounds the rows on the card up to its granule: 65,536 positions are 9,437,472 and 18,874,656 B of rows
    constexpr uint64_t kGran = 2ull << 20;
    CHECK(device_latent_bytes(1000000, 2, F8, 65536, kGran) == 5 * kGran && device_latent_bytes(1000000, 1, F8, 65536, kGran) == 10 * kGran);
    CHECK(host_latent_bytes(1000000, 2, F8, 65536, kGran) == 69 * kGran - 5 * kGran);   // 144,000,288 B round up to 69 granules
    CHECK(cache_bytes(1000000, ratios, F8, 65536, kGran) == 25 * kGran + (3 * 500001 + 1000001) * 80ull);
    CHECK(host_cache_bytes(1000000, ratios, F8, 65536, kGran) == (3 * 64 + 128) * kGran);
    // a tail the rounding leaves nothing of: the whole cache on the card, nothing pinned, no prompt copy
    CHECK(device_latent_bytes(8192, 2, F8, 8000, kGran) == 4097 * 288ull && host_latent_bytes(8192, 2, F8, 8000, kGran) == 0);
    CHECK(cache_bytes(8192, ratios, F8, 8000, kGran) == cache_bytes(8192, ratios, F8) && host_cache_bytes(8192, ratios, F8, 8000, kGran) == 0 &&
          prompt_stage_bytes(8192, ratios, F8, 8000, kGran) == 0);
    // only the caches with rows in RAM size the prompt copy: at 8,192 positions the ratio-1 cache alone passes a granule
    CHECK(host_latent_bytes(8192, 2, F8, 1024, kGran) == 0 && host_latent_bytes(8192, 1, F8, 1024, kGran) == kGran &&
          prompt_stage_bytes(8192, ratios, F8, 1024, kGran) == 8193 * 288ull);
    CHECK(cell_format(F8) == 1 && cell_format(F16) == 0);
    CacheFormat f = F16;
    CHECK(cache_format("fp8", f) && f == F8 && cache_format("fp16", f) && f == F16 && !cache_format("int8", f) &&
          !cache_format("", f));
    // fp16 caches of a full context need 3.2 GiB: refused beside 6 GiB free, a warning beside 12 GiB
    CHECK(!plan_context(kNative, kNative, 6 * GiB, ratios, F16).error.empty());
    CHECK(!plan_context(kNative, kNative, 12 * GiB, ratios, F16).note.empty());
    // with --kv-resident only the part on the card counts: the same context fits beside 6 GiB
    CHECK(plan_context(kNative, kNative, 6 * GiB, ratios, F16, 65536, kGran).error.empty());
    CHECK(!plan_context(kNative, kNative, GiB, ratios, F16, 65536, kGran).error.empty());
    CHECK(owns_cache(2, 2) && owns_cache(20, 1) && !owns_cache(3, 2) && !owns_cache(2, 0));

    // the shared default and an explicit value are honoured from kMinContext up to the model's maximum
    Plan p = plan_context_f(basalt::program::kDefaultContext, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == basalt::program::kDefaultContext && p.note.empty());
    p = plan_context_f(32768, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == 32768 && p.note.empty());
    p = plan_context_f(kNative, kNative, 20 * GiB, ratios);
    CHECK(p.error.empty() && p.context == kNative);
    p = plan_context_f(kNative + 1, kNative, 20 * GiB, ratios);
    CHECK(!p.error.empty() && p.context == 0);
    p = plan_context_f(100, kNative, 20 * GiB, ratios);
    CHECK(!p.error.empty());
    // a pack that names no length is capped at the legacy maximum
    p = plan_context_f(kLegacyContext + 1, 0, 20 * GiB, ratios);
    CHECK(!p.error.empty());
    // more than half of the free memory is refused, a quarter or more warns
    p = plan_context_f(kNative, kNative, 3 * GiB / 2, ratios);
    CHECK(!p.error.empty());
    p = plan_context_f(kNative, kNative, 3 * GiB, ratios);
    CHECK(p.error.empty() && !p.note.empty());
    p = plan_context_f(kNative, kNative, 4 * GiB, ratios);
    CHECK(p.error.empty() && p.note.empty());

    if (g_fail == 0) std::printf("context_policy_test: ok\n");
    return g_fail ? 1 : 0;
}
