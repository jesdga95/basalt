// src/program/image_loan.cpp - see include/basalt/program/image_loan.hpp.
#include "basalt/program/image_loan.hpp"

#include <algorithm>

namespace basalt::program {

int64_t image_lend_floor(const std::vector<int32_t>& slot_of, const std::vector<uint8_t>& in_ram, int64_t live) {
    int64_t floor = 0;
    if (in_ram.empty()) return floor;
    for (size_t i = 0; i < slot_of.size(); ++i) {
        const int64_t s = slot_of[i];
        if (s < 0 || s >= live) continue;
        if (i >= in_ram.size() || in_ram[i] == 0) floor = std::max<int64_t>(floor, s + 1);
    }
    return floor;
}

std::vector<std::pair<int, int64_t>> kv_giveback_order(const std::vector<KvHold>& holds, int64_t floor) {
    struct Spare { int i; int64_t kept, spare; bool active; };
    std::vector<Spare> v;
    for (size_t i = 0; i < holds.size(); ++i) {
        const KvHold& h = holds[i];
        const int64_t kept = std::max(h.keep, floor);
        if (h.backed > kept) v.push_back({(int) i, kept, h.backed - kept, h.active});
    }
    std::stable_sort(v.begin(), v.end(), [](const Spare& a, const Spare& b) {
        if (a.active != b.active) return !a.active;
        return a.spare > b.spare;
    });
    std::vector<std::pair<int, int64_t>> out;
    out.reserve(v.size());
    for (const Spare& s : v) out.emplace_back(s.i, s.kept);
    return out;
}

}  // namespace basalt::program
