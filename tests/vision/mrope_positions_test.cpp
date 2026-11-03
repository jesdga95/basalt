// tests/vision/mrope_positions_test.cpp - prompt positions with pictures and video groups against the reference
// (transformers 5.22.2 Qwen4ExpModel.get_rope_index on the same token runs; 900 a picture's pad, 901 a video group's,
// 5 text). CPU only.
#include "basalt/vision/mrope_positions.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::vision;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("mrope %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}

struct Case {
    const char* name;
    std::vector<int64_t> ids;
    std::vector<int32_t> want;   // t, h, w per token
    std::vector<MediaGrid> grids;
};
}  // namespace

int main() {
    const std::vector<Case> cases = {
        // a 2x3 picture (grid 1x4x6)
        {"picture", {5,5,5,900,900,900,900,900,900,5,5,5,5},
         {0,0,0,1,1,1,2,2,2,3,3,3,3,3,4,3,3,5,3,4,3,3,4,4,3,4,5,6,6,6,7,7,7,8,8,8,9,9,9},
         {{6, 3, 2}}},
        // a video of three 2x3 groups (grid 3x4x6), each after three timestamp tokens and its start, before its end
        {"video", {5,5,5,5,5,5,901,901,901,901,901,901,5,5,5,5,5,901,901,901,901,901,901,5,5,5,5,5,901,901,901,901,901,901,5,5,5},
         {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,6,7,6,6,8,6,7,6,6,7,7,6,7,8,9,9,9,10,10,10,11,11,11,12,12,12,13,13,13,
          14,14,14,14,14,15,14,14,16,14,15,14,14,15,15,14,15,16,17,17,17,18,18,18,19,19,19,20,20,20,21,21,21,22,22,22,22,22,
          23,22,22,24,22,23,22,22,23,23,22,23,24,25,25,25,26,26,26,27,27,27},
         {{6, 3, 2}, {6, 3, 2}, {6, 3, 2}}},
        // a 2x3 picture of 3 rows (grid 1x6x4), then a video of two 4x1 groups (grid 2x2x8)
        {"picture and video", {5,5,900,900,900,900,900,900,5,5,5,5,5,901,901,901,901,5,5,5,901,901,901,901,5,5},
         {0,0,0,1,1,1,2,2,2,2,2,3,2,3,2,2,3,3,2,4,2,2,4,3,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10,10,11,10,10,12,10,10,13,
          14,14,14,15,15,15,16,16,16,17,17,17,17,17,18,17,17,19,17,17,20,21,21,21,22,22,22},
         {{6, 2, 3}, {4, 4, 1}, {4, 4, 1}}},
    };
    const std::vector<int64_t> pads = {900, 901};
    for (const Case& c : cases) {
        const int64_t n = (int64_t) c.ids.size();
        std::vector<int32_t> pos((size_t) (n + 4) * 3, -1);
        std::vector<int64_t> runs;
        const std::string err = mrope_positions(c.ids.data(), n, pads, c.grids, n + 4, pos.data(), runs);
        check(err.empty(), std::string(c.name) + ": accepted" + (err.empty() ? "" : " (" + err + ")"));
        check(std::vector<int32_t>(pos.begin(), pos.begin() + n * 3) == c.want, std::string(c.name) + ": the reference's positions");
        const int32_t last = c.want[(size_t) (n - 1) * 3];
        check(pos[(size_t) n * 3] == last + 1 && pos[(size_t) (n + 3) * 3 + 2] == last + 4, std::string(c.name) + ": the cells after it continue");
        check(runs.size() == c.grids.size(), std::string(c.name) + ": one run per grid");
    }
    // mismatches are refused
    std::vector<int32_t> pos(64 * 3);
    std::vector<int64_t> runs;
    const std::vector<int64_t> two = {5, 900, 900, 5, 901, 901, 5};
    check(mrope_positions(two.data(), 7, pads, {{2, 2, 1}}, 64, pos.data(), runs).find("more images than the embeddings") != std::string::npos,
          "a run with no grid left");
    check(mrope_positions(two.data(), 7, pads, {{2, 2, 1}, {2, 2, 1}, {2, 2, 1}}, 64, pos.data(), runs).find("more images than the prompt") != std::string::npos,
          "a grid with no run left");
    check(mrope_positions(two.data(), 7, pads, {{3, 3, 1}, {2, 2, 1}}, 64, pos.data(), runs).find("fewer pad tokens") != std::string::npos,
          "a run shorter than its grid");
    const std::vector<int64_t> tail = {5, 901, 901};
    check(mrope_positions(tail.data(), 3, pads, {{2, 2, 1}}, 64, pos.data(), runs).find("cannot end") != std::string::npos,
          "a prompt ending in a run");
    return failures ? 1 : 0;
}
