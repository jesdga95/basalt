#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::program {

/// The K cut points of a layer split as `--layer-split` spells them: the FIRST LAYER of each later stage
/// ("8,20,34" for four stages of 8/12/14/14).  False when the string is not a comma-separated list of numbers.
inline bool parse_split_cuts(const std::string& str, std::vector<int64_t>& out) {
    if (str.empty()) return false;
    size_t a = 0;
    while (true) {
        const size_t b = str.find(',', a);
        const std::string t = str.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
        out.push_back((int64_t) std::atoll(t.c_str()));
        if (b == std::string::npos) break;
        a = b + 1;                                  // a trailing comma leaves an empty token, and is refused
    }
    return true;
}

/// Whether `cuts` is a usable set of cut points for a split over `n_stage_cards` later stages: at least two
/// layers before the first cut (stage 0 must compute something) and strictly rising, so every stage owns at
/// least one layer.  The count must match the devices, which the caller checks.
inline bool split_cuts_ordered(const std::vector<int64_t>& cuts) {
    for (size_t i = 0; i < cuts.size(); ++i)
        if (cuts[i] < 2 || (i > 0 && cuts[i] <= cuts[i - 1])) return false;
    return true;
}

}  // namespace basalt::program
