// ---- Jev: bounded sequence scheduling and target-only draft ownership ----
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace basalt::core {

struct DecisionWindow {
    std::array<int, 8> slots{}, lengths{};
    int size = 0, rows = 0;
};

inline DecisionWindow decision_window(std::span<const int> remaining, int capacity) {
    DecisionWindow out;
    if (remaining.size() > out.slots.size() || capacity < (int) remaining.size() || capacity > 8) return out;
    for (int i = 0; i < (int) remaining.size(); ++i)
        if (remaining[i] > 0) out.slots[out.size++] = i;
    for (int j = 0; j < out.size; ++j) {
        out.lengths[j] = std::min(remaining[out.slots[j]], (capacity - out.rows) / (out.size - j));
        out.rows += out.lengths[j];
    }
    for (int j = 0; j < out.size && out.rows < capacity; ++j) {
        const int extra = std::min(remaining[out.slots[j]] - out.lengths[j], capacity - out.rows);
        out.lengths[j] += extra; out.rows += extra;
    }
    return out;
}

inline int64_t decision_draft_prefix(int64_t previous, int64_t resumed, bool replaced) {
    return replaced || previous < 0 ? resumed : std::min(previous, resumed);
}

} // namespace basalt::core
// ---- end Jev ----
