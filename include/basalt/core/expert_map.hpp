// include/basalt/core/expert_map.hpp - a fixed sample of a model's routed experts for the server's expert map: where
// each one lives and how often decode looks it up.
//
// The sample is the same on every start of a pack (pair i sits at the middle of the i-th equal stretch of all
// layer * n_expert pairs), so the map's cells keep their experts across restarts. Counting is a table read and an
// increment per looked-up expert, on the thread that looks them up, and only once the server has asked for the map.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace basalt::core {

/// Where an expert lives, as the map draws it.
enum class ExpertHome : uint8_t { Vram = 0, Peer = 1, Ram = 2, File = 3 };

/// The sample's pairs (layer * n_expert + expert), `cells` of them spread evenly over `total`, or all when fewer.
inline std::vector<int32_t> expert_map_pairs(int64_t total, int cells) {
    std::vector<int32_t> pairs;
    if (total <= 0 || cells <= 0) return pairs;
    const int64_t n = total < cells ? total : cells;
    pairs.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) pairs.push_back((int32_t) ((2 * i + 1) * total / (2 * n)));
    return pairs;
}

/// The sampled pairs' lookups since the last take().
class ExpertMapTally {
public:
    static constexpr int kCells = 640;

    /// Starts counting the sample of n_layers * n_expert pairs (again from zero).
    void open(int n_layers, int n_expert, int cells = kCells) {
        n_expert_ = n_expert;
        const int64_t total = (int64_t) n_layers * n_expert;
        pairs_ = expert_map_pairs(total, cells);
        idx_.assign((size_t) (total > 0 ? total : 0), -1);
        for (size_t i = 0; i < pairs_.size(); ++i) idx_[(size_t) pairs_[i]] = (int16_t) i;
        n_.assign(pairs_.size(), 0);
    }
    bool is_open() const { return !idx_.empty(); }
    int64_t total() const { return (int64_t) idx_.size(); }
    const std::vector<int32_t>& pairs() const { return pairs_; }
    /// One lookup of (layer, e); nothing before open().
    void note(int layer, int e) {
        if (idx_.empty()) return;
        const int16_t i = idx_[(size_t) layer * (size_t) n_expert_ + (size_t) e];
        if (i >= 0) ++n_[(size_t) i];
    }
    /// The counts since the last take, in sample order; they start again from zero.
    void take(std::vector<uint32_t>& routed) {
        routed.assign(n_.begin(), n_.end());
        std::fill(n_.begin(), n_.end(), 0u);
    }

private:
    int n_expert_ = 0;
    std::vector<int32_t> pairs_;
    std::vector<int16_t> idx_;   ///< [pair] its place in the sample, or -1
    std::vector<uint32_t> n_;
};

}  // namespace basalt::core
