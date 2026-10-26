// include/basalt/core/staged_copy.hpp - device memory from one card to another through pinned host buffers.
//
// Without GPU-to-GPU access a peer copy is staged by the driver piece by piece, each piece waiting for the last.  Here
// the source card's device-to-host copies run on its own stream ahead of the destination card's host-to-device ones,
// through a few pinned buffers taken in turn, so both links work at once.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::core {

class StagedCopy {
public:
    StagedCopy() = default;
    ~StagedCopy();
    StagedCopy(const StagedCopy&) = delete;
    StagedCopy& operator=(const StagedCopy&) = delete;

    /// Copies from card `src_dev` to card `dst_dev` through `kBufs` pinned buffers of `piece_bytes`.
    bool init(int src_dev, int dst_dev, size_t piece_bytes, std::string& err);
    bool ready() const { return dst_stream_ != nullptr; }
    /// Queues `bytes` from `src` (on the source card) to `dst` (on the destination card).
    bool copy(void* dst, const void* src, uint64_t bytes, std::string& err);
    /// Waits until every queued copy has landed.
    bool wait(std::string& err);

private:
    static constexpr int kBufs = 4;
    void release();
    int src_dev_ = -1, dst_dev_ = -1;
    size_t piece_ = 0;
    uint8_t* buf_[kBufs] = {};
    cudaEvent_t d2h_[kBufs] = {}, h2d_[kBufs] = {};   // d2h_ on the source card, h2d_ on the destination card
    bool live_[kBufs] = {};
    cudaStream_t src_stream_ = nullptr, dst_stream_ = nullptr;
    int next_ = 0;
};

}  // namespace basalt::core
