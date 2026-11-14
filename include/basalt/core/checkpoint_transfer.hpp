// include/basalt/core/checkpoint_transfer.hpp - a conversation checkpoint's running state between the device and the host.
//
// Every model's checkpoint is the same job: a set of device regions copied into host bytes when it is taken and back
// when a request resumes from it.  The host bytes come from the pooled, page-locked blocks of CheckpointBytes
// (conversation_cache.hpp), so the copies run at link speed and need no staging by the driver; the many small pieces
// of a model's state (a ring and a compressor per layer) go through one device-side gather and one host copy.
//
//   copy_out / copy_in   the state lands (or is back) when the call returns;
//   CheckpointDeferral   the state is snapshotted on the device at once, in the order of the stream's own work, and
//                        reaches the host while the caller goes on; its bytes are valid after finish().
#pragma once

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace basalt::core {

/// Page-locks pooled checkpoint blocks: pass it to `CheckpointBytes::allocator_type::reserve`.  A lock is refused (the
/// block stays ordinary memory) when the driver's registration allowance cannot take it.
extern const HostPin kCheckpointPin;

/// Whether the host range is page-locked.
bool host_range_pinned(const void* p);

/// The regions of every segment into its host bytes; returns when they have landed.  The regions are read as they are
/// now: the caller has the stream idle of writes to them.
bool checkpoint_copy_out(const std::vector<SaveSegment>& segments, cudaStream_t stream, std::string& error);
/// The host bytes of every segment into its regions, queued on `stream` (the work after it sees them); returns once
/// the host bytes have been read, so they may be freed.
bool checkpoint_copy_in(const std::vector<RestoreSegment>& segments, cudaStream_t stream, std::string& error);

/// Saves a checkpoint off the caller's critical path.  save() copies the regions into a spare device buffer at device
/// speed, ordered after the stream's earlier work and before its later work, so decode may change the live state at
/// once; the spare then moves to the host on a side stream while the caller carries on.  Nothing may read, move-from or
/// free the segments' host bytes before finish() has returned true.  One save is in flight at a time: save() finishes the
/// previous one.  Host bytes that are not page-locked (a few KiB of small parts) go through a bounce buffer and are copied
/// on by finish(); a save with more of them than the bounce buffer holds is made at once.
class CheckpointDeferral {
public:
    static constexpr size_t kBounceBytes = size_t(1) << 20;
    CheckpointDeferral() = default;
    ~CheckpointDeferral();
    CheckpointDeferral(const CheckpointDeferral&) = delete;
    CheckpointDeferral& operator=(const CheckpointDeferral&) = delete;
    /// Allocates the spare buffer on the current device (a larger one replaces it); without one save() copies out at
    /// once.  Regions saved afterwards must lie on this device.
    bool reserve(size_t bytes, std::string& error);
    size_t capacity() const { return cap_; }
    /// `key` names the save in flight (pending()): the address of the checkpoint's bytes.
    bool save(const std::vector<SaveSegment>& segments, cudaStream_t stream, const void* key, std::string& error);
    /// The key of the save still in flight (null: none).
    const void* pending() const { return dst_; }
    /// Waits for the host copy; false if it failed (the saved bytes are then unusable).
    bool finish(std::string& error);

private:
    struct Bounced {
        uint8_t* host;
        const uint8_t* bounce;
        size_t bytes;
    };
    void release();
    uint8_t* spare_ = nullptr;
    uint8_t* bounce_ = nullptr;
    size_t cap_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t snapshot_ = nullptr, landed_ = nullptr;
    std::vector<Bounced> tail_;
    const void* dst_ = nullptr;
};

}  // namespace basalt::core
