// src/models/inkling2/expert_map_test.cpp - the expert map from the card's running counts: the sample, all in VRAM, and
// each sampled pair's lookups since the last take (the first take starts the counting; a wrapped count still differs right).
#include "models/inkling2/expert_map.hpp"

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
    const int n_layers = 24, n_expert = 128;
    std::vector<uint32_t> counts((size_t) n_layers * n_expert, 0u);
    const std::vector<int32_t> pairs = basalt::core::expert_map_pairs((int64_t) counts.size(), basalt::core::ExpertMapTally::kCells);
    ResidentExpertMap map;
    int64_t total = 0;
    std::vector<uint8_t> homes;
    std::vector<uint32_t> routed;
    for (size_t i = 0; i < counts.size(); ++i) counts[i] = (uint32_t) (7 * i);
    map.take(counts, total, homes, routed);
    expect(total == 3072, "the total is every (layer, expert) pair");
    expect(homes.size() == pairs.size() && routed.size() == pairs.size() && pairs.size() == 640, "the sample's size");
    bool vram = true, zero = true;
    for (size_t i = 0; i < homes.size(); ++i) {
        vram = vram && homes[i] == (uint8_t) basalt::core::ExpertHome::Vram;
        zero = zero && routed[i] == 0;
    }
    expect(vram, "every sampled expert lives in VRAM");
    expect(zero, "the first take starts the counting");
    // some lookups: pair p looked up p % 5 times, the first sampled pair's count wrapping past 2^32
    counts[(size_t) pairs[0]] = 0xfffffffeu;
    map.take(counts, total, homes, routed);
    for (size_t i = 0; i < counts.size(); ++i) counts[i] += (uint32_t) (i % 5);
    counts[(size_t) pairs[0]] = 3u;   // 0xfffffffe + 5
    map.take(counts, total, homes, routed);
    bool right = routed[0] == 5;
    for (size_t i = 1; i < pairs.size(); ++i) right = right && routed[i] == (uint32_t) (pairs[i] % 5);
    expect(right, "each sampled pair's lookups since the last take");
    map.take(counts, total, homes, routed);
    zero = true;
    for (uint32_t r : routed) zero = zero && r == 0;
    expect(zero, "no lookups since the last take");
    if (failures) return 1;
    std::printf("inkling2 expert map: ok\n");
    return 0;
}
