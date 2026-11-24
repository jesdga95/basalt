// src/models/inkling2/loan_layout.hpp - where the loaned expert copies lie: the next hottest RAM experts, packed into the
// device regions the prompt path leaves idle while decode runs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "models/inkling2/spill.hpp"

namespace basalt::models::inkling2 {

/// One loaned copy's place: `region` index and byte `offset` in it.
struct LoanSlot {
    int region = 0;
    uint64_t offset = 0;
};

/// Claims the next aligned slot in the first region with room, without crossing a region boundary.
inline std::optional<LoanSlot> next_loan_slot(uint64_t stride, std::span<const uint64_t> regions, std::span<uint64_t> at) {
    for (size_t r = 0; r < regions.size(); ++r) {
        if (at[r] > regions[r] || stride > regions[r] - at[r]) continue;
        const LoanSlot slot{(int) r, at[r]};
        at[r] = CopyLayout::aligned(at[r] + stride);
        return slot;
    }
    return std::nullopt;
}

}  // namespace basalt::models::inkling2
