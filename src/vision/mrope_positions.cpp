// src/vision/mrope_positions.cpp - see basalt/vision/mrope_positions.hpp.
#include "basalt/vision/mrope_positions.hpp"

#include <algorithm>

namespace basalt::vision {

std::string mrope_positions(const int64_t* ids, int64_t n, const std::vector<int64_t>& pads, const std::vector<MediaGrid>& grids,
                            int64_t cells, int32_t* pos, std::vector<int64_t>& runs) {
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        pos[c * 3] = (int32_t) t;
        pos[c * 3 + 1] = (int32_t) h;
        pos[c * 3 + 2] = (int32_t) w;
    };
    auto is_pad = [&](int64_t id) { return std::find(pads.begin(), pads.end(), id) != pads.end(); };
    runs.clear();
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (!is_pad(ids[i])) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= grids.size()) return "the prompt has more images than the embeddings file";
        const MediaGrid& m = grids[k++];
        for (int64_t j = 0; j < m.rows; ++j)
            if (i + j >= n || !is_pad(ids[i + j]))
                return "image " + std::to_string(k) + " has " + std::to_string((long long) m.rows) + " rows but fewer pad tokens";
        runs.push_back(i);
        for (int64_t j = 0; j < m.rows; ++j) put(i + j, p, p + j / m.nx, p + j % m.nx);
        i += m.rows;
        p += std::max(m.nx, m.ny);
    }
    if (k != grids.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && is_pad(ids[n - 1])) return "the prompt cannot end in an image";
    for (int64_t c = n; c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
    return "";
}

}  // namespace basalt::vision
