// src/models/inkling2/spill.cpp - the spill plan. Read the header first.
#include "models/inkling2/spill.hpp"

#include <algorithm>

namespace basalt::models::inkling2 {
namespace {
uint64_t round_up(uint64_t v, uint64_t g) { return g ? (v + g - 1) / g * g : v; }
}  // namespace

bool plan_spill(int n_layers, int n_expert, const std::vector<uint64_t>& stride, uint64_t gran,
                const std::vector<std::pair<int32_t, int32_t>>& rank, uint64_t need, int64_t keep, SpillPlan& out,
                std::string& err) {
    out = SpillPlan{};
    out.slot.assign((size_t) n_layers, {});
    out.n_spill.assign((size_t) n_layers, 0);
    if ((int) stride.size() != n_layers) {
        err = "spill plan: a stride per layer is needed";
        return false;
    }
    for (int l = 0; l < n_layers; ++l) out.full += (uint64_t) n_expert * stride[(size_t) l];
    out.device = out.full;
    if (need == 0 && keep < 0) return true;

    // the coldest pair first
    std::vector<std::pair<int32_t, int32_t>> order;
    if (!rank.empty()) {
        if ((int64_t) rank.size() != (int64_t) n_layers * n_expert) {
            err = "spill plan: the profile does not rank every expert";
            return false;
        }
        order.assign(rank.rbegin(), rank.rend());
    } else {
        for (int e = n_expert - 1; e >= 0; --e)
            for (int l = 0; l < n_layers; ++l) order.emplace_back(l, e);
    }
    std::vector<std::vector<uint8_t>> away((size_t) n_layers, std::vector<uint8_t>((size_t) n_expert, 0));
    // a spilled layer's device bytes: its kept slots, rounded up; the rest of its range is the shared staging
    const auto own = [&](int l) { return round_up((uint64_t) (n_expert - out.n_spill[(size_t) l]) * stride[(size_t) l], gran); };
    const auto tail = [&](int l) { return round_up((uint64_t) n_expert * stride[(size_t) l] + kSlotGuard, gran) - own(l); };
    const auto device = [&](uint64_t& staging) {
        uint64_t d = 0;
        staging = 0;
        for (int l = 0; l < n_layers; ++l) {
            if (!out.n_spill[(size_t) l]) {
                d += (uint64_t) n_expert * stride[(size_t) l];
                continue;
            }
            d += own(l);
            staging = std::max(staging, tail(l));
        }
        return d + staging;
    };
    const int64_t total = (int64_t) n_layers * n_expert;
    bool met = false;
    for (const auto& [l, e] : order) {
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert || away[(size_t) l][(size_t) e]) {
            err = "spill plan: the profile ranks a pair twice or out of range";
            return false;
        }
        away[(size_t) l][(size_t) e] = 1;
        ++out.n_spill[(size_t) l];
        ++out.spilled;
        out.device = device(out.staging);
        if (out.full >= out.device + need && (keep < 0 || total - out.spilled <= keep)) {
            met = true;
            break;
        }
    }
    if (!met) {
        err = "spill plan: even with every expert in RAM the card is " +
              std::to_string((out.device + need - std::min(out.full, out.device + need)) >> 20) + " MiB short";
        return false;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (!out.n_spill[(size_t) l]) continue;
        std::vector<int32_t>& s = out.slot[(size_t) l];
        s.assign((size_t) n_expert, -1);
        int32_t next = 0;
        for (int pass = 0; pass < 2; ++pass)
            for (int e = 0; e < n_expert; ++e)
                if (away[(size_t) l][(size_t) e] == pass) s[(size_t) e] = next++;
    }
    return true;
}

std::vector<StageOp> stage_plan(int n_spill, const uint8_t* card, const uint8_t* want, bool kept) {
    std::vector<StageOp> out;
    const auto staged = [&](int j) { return !want || want[j]; };
    for (int j = 0; j < n_spill;) {
        if (!staged(j)) {
            if (j > 0 ? staged(j - 1) : kept) out.push_back({StageOp::Guard, j, 1});
            ++j;
        } else if (card[j]) {
            out.push_back({StageOp::Card, j, 1});
            ++j;
        } else {
            int n = 1;
            while (j + n < n_spill && staged(j + n) && !card[j + n]) ++n;
            out.push_back({StageOp::Ram, j, n});
            j += n;
        }
    }
    return out;
}

CopyLayout copy_layout(const std::vector<uint64_t>& strides, uint64_t gran) {
    CopyLayout c;
    c.gran = gran;
    for (uint64_t stride : strides) c.append(stride);
    return c;
}

int64_t CopyLayout::keep_for(int64_t n, uint64_t mapped, uint64_t bytes) const {
    int64_t m = n;
    while (m > 0 && (mapped - std::min(mapped, pages(m))) * gran < bytes) --m;
    return m;
}

}  // namespace basalt::models::inkling2
