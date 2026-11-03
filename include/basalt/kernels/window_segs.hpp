// include/basalt/kernels/window_segs.hpp - a batch window's segments as kernels see them: which rows of one call belong to
// which conversation (docs/design/batching.md). Header-only; a model's kernels take it by value in their arguments, beside
// each segment's cache addresses.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define BASALT_SEGS_HD __host__ __device__
#else
#define BASALT_SEGS_HD
#endif

namespace basalt::kernels {

constexpr int kWindowSegsMax = 16;   ///< segments one call holds at most

/// Segment i is rows row0[i] .. row0[i + 1] - 1 of the call, at consecutive positions of one conversation. n = 0: the call
/// is one conversation's (row0 unused), as before batching.
struct WindowSegs {
    int32_t n = 0;
    int32_t row0[kWindowSegsMax + 1] = {};
};

/// The segment row r is in (0 when n is 0).
BASALT_SEGS_HD inline int seg_of(const WindowSegs& s, int r) {
    int k = 0;
    while (k + 1 < s.n && r >= s.row0[k + 1]) ++k;
    return k;
}

/// The first row of row r's segment: the rows before it in the call that its convolutions may read start there (0 when n
/// is 0).
BASALT_SEGS_HD inline int seg_first(const WindowSegs& s, int r) { return s.n > 0 ? s.row0[seg_of(s, r)] : 0; }

}  // namespace basalt::kernels

#undef BASALT_SEGS_HD
