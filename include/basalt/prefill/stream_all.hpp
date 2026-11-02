// include/basalt/prefill/stream_all.hpp - the chunk size from which a prompt chunk streams every non-resident expert.
//
// Below it a chunk stages only the experts it routes to; from it on every non-resident expert streams through the
// ring.  Streaming copies all of those experts over the link whatever the chunk, while the routed-only path spends
// more per token, so streaming pays from the chunk whose extra routed-only time outlasts that copy.  The default
// holds when the link or the bytes are unknown and caps the priced value, since the prompt buffers are sized by it;
// the floor keeps the smallest chunks, where few experts are routed, on the routed path.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace basalt::prefill {

constexpr int64_t kStreamAllDefault = 6144, kStreamAllFloor = 2048;
constexpr double kRoutedExtraMs = 0.24;   // the routed-only path's time per token beyond the streamed path's

/// The stream-all chunk size for `bytes` of non-resident experts over a host->device link of `gbps` GB/s: the smallest
/// chunk on the 256-token grid whose extra routed-only time exceeds copying them all, between the floor and the
/// default.  The default when either is unknown (zero).
inline int64_t stream_all_min_for(uint64_t bytes, double gbps) {
    if (gbps <= 0.0 || bytes == 0) return kStreamAllDefault;
    const double copy_ms = (double) bytes / (gbps * 1e6);
    const double t = std::ceil(copy_ms / kRoutedExtraMs / 256.0) * 256.0;
    return (int64_t) std::min(std::max(t, (double) kStreamAllFloor), (double) kStreamAllDefault);
}

}  // namespace basalt::prefill
