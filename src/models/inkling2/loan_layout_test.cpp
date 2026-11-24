// src/models/inkling2/loan_layout_test.cpp - the loan layout: experts keep their order, lie 256-byte aligned inside their
// region without overlap, go to the first region with room, and stop at the first one that fits nowhere.
#include "models/inkling2/loan_layout.hpp"

#include <cstdio>
#include <vector>

using namespace basalt::models::inkling2;

namespace {
/// Packs a prefix of the ranked experts in order.
std::vector<LoanSlot> plan_loan(const std::vector<uint64_t>& strides, const std::vector<uint64_t>& regions) {
    std::vector<LoanSlot> out;
    std::vector<uint64_t> at(regions.size(), 0);
    for (const uint64_t stride : strides) {
        const auto slot = next_loan_slot(stride, regions, at);
        if (!slot) break;
        out.push_back(*slot);
    }
    return out;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    const uint64_t a = 4479120, b = 8957952;
    expect(plan_loan({}, {1 << 20}).empty(), "no experts, no slots");
    expect(plan_loan({a, a}, {}).empty(), "no regions, no slots");
    expect(plan_loan({a}, {a - 1}).empty(), "an expert that fits nowhere stops the plan");

    // a mixed list into two regions: the first fills, the rest go to the second
    std::vector<uint64_t> s;
    for (int i = 0; i < 40; ++i) s.push_back(i % 5 == 4 ? b : a);
    const std::vector<uint64_t> regions = {100ull << 20, 60ull << 20};
    const std::vector<LoanSlot> p = plan_loan(s, regions);
    expect(!p.empty() && p.size() <= s.size(), "a prefix of the experts is placed");
    uint64_t last_end[2] = {0, 0};
    for (size_t k = 0; k < p.size(); ++k) {
        const size_t r = (size_t) p[k].region;
        expect(p[k].offset % 256 == 0, "256-byte aligned");
        expect(p[k].offset + s[k] <= regions[r], "inside its region");
        expect(p[k].offset >= last_end[r], "no overlap, in order within a region");
        last_end[r] = p[k].offset + s[k];
    }
    expect(p[0].region == 0 && p[0].offset == 0, "the first expert opens the first region");
    uint64_t used = 0;
    for (size_t k = 0; k < p.size(); ++k) used += s[k];
    expect(used <= regions[0] + regions[1], "no more bytes than the regions hold");
    // the stop: the next expert really fits in neither region's remainder
    if (p.size() < s.size()) {
        uint64_t end[2] = {0, 0};
        for (size_t k = 0; k < p.size(); ++k) end[p[k].region] = p[k].offset + s[k];
        const uint64_t next = s[p.size()];
        expect(((end[0] + 255) / 256 * 256 + next > regions[0]) && ((end[1] + 255) / 256 * 256 + next > regions[1]), "stopped only when it fits nowhere");
    }
    // exact fit
    expect(plan_loan({256, 256, 256}, {768}).size() == 3, "exact fit places all");
    expect(plan_loan({256, 256, 256}, {767}).size() == 2, "one byte short places two");
    if (failures) return 1;
    std::printf("loan layout: ok\n");
    return 0;
}
