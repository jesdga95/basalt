#include "models/deepseek4/context_policy.hpp"

#include "basalt/log.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool expect(bool ok, const char* message) {
    if (!ok) basalt::log::error("basalt deepseek4 context test", "%s", message);
    return ok;
}

}  // namespace

int main() {
    using namespace basalt::models::deepseek4::ctxp;
    const std::vector<int> ratios{0, 4, 128};
    const Footprint at_zero = cache_footprint(0, ratios);
    const uint64_t fixed_window = 3ull * 128 * 512 * sizeof(float);
    const uint64_t fixed_partial = (8ull * 1024 * 2 + 8ull * 128 * 2 + 128ull * 512 * 2) * sizeof(float);
    if (!expect(at_zero.error.empty() && at_zero.window == fixed_window && at_zero.compressed == 0 &&
                at_zero.index_keys == 0 && at_zero.partial_state == fixed_partial &&
                at_zero.total == fixed_window + fixed_partial,
                "empty sequence cache rows or fixed compressor state differ")) return 1;

    const Footprint at_four = cache_footprint(4, ratios);
    // 3 x 128 x 512 float sliding rows; one CSA output, one CSA index row,
    // and fixed CSA/HCA compressor state.
    const uint64_t expected_window = 3ull * 128 * 512 * sizeof(float);
    const uint64_t expected_compressed = 512ull * sizeof(float);
    const uint64_t expected_index = 128ull * sizeof(float);
    const uint64_t expected_partial = (8ull * 1024 * 2 + 8ull * 128 * 2 + 128ull * 512 * 2) * sizeof(float);
    if (!expect(at_four.error.empty() && at_four.window == expected_window &&
                at_four.compressed == expected_compressed && at_four.index_keys == expected_index &&
                at_four.partial_state == expected_partial &&
                at_four.total == expected_window + expected_compressed + expected_index + expected_partial,
                "mixed CSA/HCA footprint differs from its cache geometry")) return 1;

    const Plan default_plan = plan_context(0, 1 << 20, 16ull << 30, ratios);
    if (!expect(default_plan.error.empty() && default_plan.context == (1 << 20), "valid native context was refused")) return 1;
    const Plan too_small = plan_context(1023, 1 << 20, 16ull << 30, ratios);
    if (!expect(too_small.context == 0 && too_small.error.find("below") != std::string::npos,
                "context below the minimum was not refused")) return 1;
    const Plan too_large = plan_context((1 << 20) + 1ll, 1 << 20, 16ull << 30, ratios);
    if (!expect(too_large.context == 0 && too_large.error.find("exceeds") != std::string::npos,
                "context beyond indexer capacity was not refused")) return 1;
    const Plan no_room = plan_context(1 << 20, 1 << 20, 1, ratios);
    if (!expect(no_room.context == 0 && no_room.error.find("half") != std::string::npos,
                "cache memory refusal did not trigger")) return 1;
    const Footprint invalid_ratio = cache_footprint(4096, {16});
    if (!expect(invalid_ratio.error.find("unsupported") != std::string::npos,
                "unsupported compression ratio was accepted")) return 1;
    const Footprint overflow = cache_footprint(INT64_MAX, {4});
    if (!expect(overflow.error.find("overflow") != std::string::npos,
                "cache size overflow was not detected")) return 1;

    std::puts("deepseek4 context policy: mixed caches, bounds, memory refusal, and overflow passed");
    return 0;
}
