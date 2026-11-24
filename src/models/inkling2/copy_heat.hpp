// src/models/inkling2/copy_heat.hpp - which RAM experts earn a card copy: each (layer, expert) pair's recent lookups by the rows
// a conversation keeps, decayed with those rows, ranked in place of the static spill order. With no lookups seen the rank
// is the spill order itself.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::models::inkling2 {

class CopyHeat {
public:
    /// Kept rows after which a lookup counts half: a stale genre's counts give way within a few hundred tokens, while a
    /// genre's own hot set stays stable over many more.
    static constexpr double kHalfLifeRows = 320.0;
    /// The profile's order as a tie-break below one lookup: pair order[i] outranks order[i + 1] while both are unseen.
    static constexpr float kPriorStep = 1e-4f;

    /// `order`: the RAM experts, the most routed first, as (layer, expert); `pairs` = layers * n_expert.
    void start(size_t pairs, int n_expert, const std::vector<std::pair<int32_t, int32_t>>& order) {
        heat_.assign(pairs, 0.0f);
        prior_.assign(pairs, 0.0f);
        for (size_t i = 0; i < order.size(); ++i)
            prior_[(size_t) order[i].first * (size_t) n_expert + (size_t) order[i].second] = kPriorStep * (float) (order.size() - i);
    }
    bool started() const { return !heat_.empty(); }

    /// `lookups`: each pair's lookups since the last call; `rows`: the kept rows they cover (older counts decay by them).
    void learn(const std::vector<uint32_t>& lookups, int64_t rows) {
        if (heat_.empty() || lookups.size() != heat_.size()) return;
        const float keep = (float) std::exp2(-(double) std::max<int64_t>(rows, 0) / kHalfLifeRows);
        for (size_t i = 0; i < heat_.size(); ++i) heat_[i] = heat_[i] * keep + (float) lookups[i];
    }
    float score(size_t pair) const { return heat_[pair] + prior_[pair]; }

private:
    std::vector<float> heat_, prior_;
};

/// `pairs` ordered by score, highest first (equal scores keep their order).
inline std::vector<size_t> by_score_desc(const CopyHeat& h, std::vector<size_t> pairs) {
    std::stable_sort(pairs.begin(), pairs.end(), [&](size_t a, size_t b) { return h.score(a) > h.score(b); });
    return pairs;
}

struct CopySwap {
    size_t out = 0, in = 0;   ///< the copy to give up, the RAM expert that takes its allocation
};

/// Swaps that move a card copy from a colder pair to a hotter RAM one of the same size, at most `max_swaps`, the largest
/// gains first. A pair must beat the copy by a quarter and by `min_gain` lookups, so noise does not churn the card.
/// `copies`: pairs with a copy; `ram`: RAM pairs without; `size_class(pair)`: equal for pairs whose copies can swap storage.
template <class SizeClass>
std::vector<CopySwap> plan_swaps(const CopyHeat& h, const std::vector<size_t>& copies, const std::vector<size_t>& ram, SizeClass size_class,
                                 int max_swaps, float min_gain = 16.0f) {
    std::vector<size_t> cold = by_score_desc(h, copies), hot = by_score_desc(h, ram);
    std::reverse(cold.begin(), cold.end());
    std::vector<CopySwap> all;
    std::vector<size_t> classes;
    for (size_t p : cold) classes.push_back((size_t) size_class(p));
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    for (size_t c : classes) {
        size_t i = 0;
        for (size_t in : hot) {
            if ((size_t) size_class(in) != c) continue;
            while (i < cold.size() && (size_t) size_class(cold[i]) != c) ++i;
            if (i == cold.size()) break;
            const float have = h.score(cold[i]), want = h.score(in);
            if (want < have * 1.25f || want - have < min_gain) break;
            all.push_back({cold[i], in});
            ++i;
        }
    }
    std::stable_sort(all.begin(), all.end(), [&](const CopySwap& a, const CopySwap& b) {
        return h.score(a.in) - h.score(a.out) > h.score(b.in) - h.score(b.out);
    });
    if ((int) all.size() > max_swaps) all.resize((size_t) std::max(max_swaps, 0));
    return all;
}

}  // namespace basalt::models::inkling2
