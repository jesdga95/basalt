// basalt/vision/mrope_positions.hpp - a prompt's 3-D rotary positions with pictures in it (Qwen3-VL's get_rope_index).
//
// Text takes the running position on all three axes (t, h, w). Each run of pad tokens takes the next grid of merged
// tokens: row y, column x at t = p, h = p + y, w = p + x, and the text after it continues at p + max(nx, ny). A video
// is split into its two-frame groups, each a grid of its own between its timestamp text, so a picture's pad and a
// video group's pad follow one rule.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::vision {

struct MediaGrid {
    int64_t rows = 0, nx = 0, ny = 0;   ///< merged tokens, nx * ny of them
};

/// Fills `pos` (`cells` x 3: t, h, w) for the first `n` tokens of `ids` and the `cells - n` positions after them, and
/// `runs` with each grid's first cell. Every run of the ids in `pads` takes the next of `grids`, in order. Returns ""
/// or what does not match (a run shorter than its grid, more runs than grids or fewer, a prompt ending in a run).
std::string mrope_positions(const int64_t* ids, int64_t n, const std::vector<int64_t>& pads, const std::vector<MediaGrid>& grids,
                            int64_t cells, int32_t* pos, std::vector<int64_t>& runs);

}  // namespace basalt::vision
