// include/basalt/core/exact_slots.hpp - the sizing rules for an exact `--expert-cache N`, free of CUDA so a CPU test
// can drive them. An exact count is honoured or the start is refused; it is never trimmed to what fits.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::core {

/// How many leading blobs fit in `cap` bytes when each takes its size rounded up to 256 (the slot stride).
inline int64_t blobs_within(const std::vector<int64_t>& blob_bytes, uint64_t cap) {
    uint64_t used = 0;
    int64_t n = 0;
    for (const int64_t b : blob_bytes) {
        const uint64_t padded = ((uint64_t) b + 255) / 256 * 256;
        if (used + padded > cap) break;
        used += padded;
        ++n;
    }
    return n;
}

/// The one-line refusal of an exact count that does not fit: the flag's value, what fits (`per` is " per layer" where
/// the count is per layer, else empty), the free VRAM and the reserve.
inline std::string exact_slots_refusal(long long value, long long fit, const char* per, uint64_t free_bytes,
                                       int reserve_mib) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "--expert-cache %lld: only %lld%s fit (%.2f GiB free, %d MiB reserve); use a smaller value or auto",
                  value, std::max<long long>(fit, 0), per, (double) free_bytes / 1073741824.0, reserve_mib);
    return buf;
}

}  // namespace basalt::core
