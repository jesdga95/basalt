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

int64_t kv_cache_spare_granules(int64_t mapped, int64_t gran, int64_t floor,
                                const std::function<int64_t(int64_t)>& slots_after) {
    if (gran <= 0 || mapped <= 0) return 0;
    // what a shrink keeps falls with every granule given, so the slots it leaves do too
    int64_t lo = 0, hi = mapped / gran;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (slots_after(mapped - mid * gran) >= floor) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

int64_t kv_ceiling_cells(const std::vector<int64_t>& backed, const std::vector<int64_t>& max_cells, int64_t spare,
                         const std::function<int64_t(size_t, int64_t)>& grow) {
    int64_t total = 0;
    for (size_t i = 0; i < backed.size() && i < max_cells.size(); ++i) {
        if (backed[i] < 0) { total += max_cells[i]; continue; }
        int64_t best = std::min(backed[i], max_cells[i]);
        for (int64_t lo = best / 256 + 1, hi = (max_cells[i] + 255) / 256; lo <= hi;) {
            const int64_t mid = lo + (hi - lo) / 2;
            const int64_t c = std::min<int64_t>(mid * 256, max_cells[i]);
            if (grow(i, c) <= spare) { best = c; lo = mid + 1; }
            else hi = mid - 1;
        }
        spare -= grow(i, best);
        total += best;
    }
    return total;
}

}  // namespace basalt::program
