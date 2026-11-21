// include/basalt/core/checkpoint_regions.hpp - how a model describes its running state to the checkpoint transfer.
//
// A model lists the device memory its running state lives in, piece by piece; checkpoint_transfer.hpp moves the pieces
// to and from host bytes as one packed transfer.  No GPU runtime is needed to include this.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::core {

/// One piece of running state in device memory.
struct DeviceRegion {
    void* dev = nullptr;
    size_t bytes = 0;
};

/// Host bytes that hold `regions` packed back to back (a checkpoint's `state_a`, ...).  `P` is `uint8_t` to write them
/// and `const uint8_t` to read them.
template <class P>
struct Segment {
    P* host = nullptr;
    std::vector<DeviceRegion> regions;
    size_t bytes() const {
        size_t n = 0;
        for (const DeviceRegion& r : regions) n += r.bytes;
        return n;
    }
};
using SaveSegment = Segment<uint8_t>;
using RestoreSegment = Segment<const uint8_t>;

/// What a model session gives the shared engine for its running state: the pieces, in a fixed order whatever the
/// position, and the stream its work runs on (null: the default stream).  `head` is the state the session keeps on the
/// host (a position offset, a count): the engine stores it first in the checkpoint's bytes, the regions after it, and
/// gives it back on a restore (a save fills it; a restore's call needs only its size).  A model split across two cards lists
/// the second card's pieces in `peer_regions` with that card's stream and device: they follow the others in the bytes.
struct RunningState {
    std::vector<uint8_t> head;
    std::vector<DeviceRegion> regions;
    void* stream = nullptr;
    std::vector<DeviceRegion> peer_regions;
    void* peer_stream = nullptr;
    int peer_device = -1;
    static size_t sum(const std::vector<DeviceRegion>& rs) {
        size_t n = 0;
        for (const DeviceRegion& r : rs) n += r.bytes;
        return n;
    }
    size_t regions_bytes() const { return sum(regions); }
    size_t bytes() const { return head.size() + regions_bytes() + sum(peer_regions); }
};

}  // namespace basalt::core
