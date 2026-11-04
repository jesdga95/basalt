// basalt/program/serve/loan_moves.cpp - see loan_moves.hpp.
#include "loan_moves.hpp"

#include <algorithm>

namespace basalt::program::serve {

LoanSplit split_loan(const std::vector<LentEntry>& lent, int64_t first) {
    LoanSplit s;
    for (const LentEntry& x : lent) (x.second >= first ? s.again : s.back).push_back(x);
    return s;
}

int64_t plan_routed_trades(const float* use, const int32_t* slot, const uint8_t* in_ram, const uint8_t* lent_out,
                           int32_t n, int64_t first, std::vector<RoutedTrade>& out) {
    int64_t routed = 0;
    std::vector<int32_t> hot, cold;
    for (int32_t ex = 0; ex < n; ++ex) {
        if (slot[ex] >= first) {
            if (use[ex] <= 0.0f) continue;
            ++routed;
            if (lent_out == nullptr || lent_out[ex] == 0 || in_ram[ex] != 0) hot.push_back(ex);
        } else if (slot[ex] >= 0 && in_ram[ex] != 0) {
            cold.push_back(ex);
        }
    }
    if (hot.empty() || cold.empty()) return routed;
    std::sort(hot.begin(), hot.end(), [&](int32_t a, int32_t b) { return use[a] > use[b]; });
    std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) {
        return use[a] != use[b] ? use[a] < use[b] : slot[a] > slot[b];   // then the profile's coldest
    });
    for (size_t k = 0; k < hot.size() && k < cold.size() && use[hot[k]] > use[cold[k]]; ++k)
        out.push_back({hot[k], cold[k], slot[hot[k]], slot[cold[k]], lent_out != nullptr && lent_out[hot[k]] != 0});
    return routed;
}

}  // namespace basalt::program::serve
