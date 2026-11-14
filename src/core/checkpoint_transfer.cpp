// src/core/checkpoint_transfer.cpp - a conversation checkpoint's running state between the device and the host.
#include "basalt/core/checkpoint_transfer.hpp"

#include "basalt/core/ram_load.hpp"

#include <cstring>

namespace basalt::core {
namespace {

bool lock_block(void* p, size_t bytes) {
    uint64_t allowance = 0;
    std::string why;
    if (host_pin_budget(allowance, why) && allowance < bytes) return false;
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) == cudaSuccess) return true;
    cudaGetLastError();
    return false;
}

void unlock_block(void* p) {
    if (cudaHostUnregister(p) != cudaSuccess) cudaGetLastError();
}

bool fail(std::string& error, const char* what, cudaError_t status) {
    error = std::string("checkpoint transfer ") + what + ": " + cudaGetErrorString(status);
    return false;
}

}  // namespace

const HostPin kCheckpointPin = {lock_block, unlock_block};

bool host_range_pinned(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeHost;
}

bool checkpoint_copy_out(const std::vector<SaveSegment>& segments, cudaStream_t stream, std::string& error) {
    for (const SaveSegment& s : segments) {
        size_t at = 0;
        for (const DeviceRegion& r : s.regions) {
            if (r.bytes == 0) continue;
            if (!r.dev || !s.host) { error = "checkpoint transfer: a running-state region has no buffer"; return false; }
            if (const cudaError_t e = cudaMemcpyAsync(s.host + at, r.dev, r.bytes, cudaMemcpyDeviceToHost, stream); e != cudaSuccess)
                return fail(error, "copy out", e);
            at += r.bytes;
        }
    }
    if (const cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return fail(error, "synchronize", e);
    return true;
}

bool checkpoint_copy_in(const std::vector<RestoreSegment>& segments, cudaStream_t stream, std::string& error) {
    for (const RestoreSegment& s : segments) {
        size_t at = 0;
        for (const DeviceRegion& r : s.regions) {
            if (r.bytes == 0) continue;
            if (!r.dev || !s.host) { error = "checkpoint transfer: a running-state region has no buffer"; return false; }
            if (const cudaError_t e = cudaMemcpyAsync(r.dev, s.host + at, r.bytes, cudaMemcpyHostToDevice, stream); e != cudaSuccess)
                return fail(error, "copy in", e);
            at += r.bytes;
        }
    }
    // the host bytes are free to move once the last copy has left them
    if (const cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return fail(error, "synchronize", e);
    return true;
}

CheckpointDeferral::~CheckpointDeferral() { release(); }

void CheckpointDeferral::release() {
    if (landed_) cudaEventSynchronize(landed_);
    if (landed_) cudaEventDestroy(landed_);
    if (snapshot_) cudaEventDestroy(snapshot_);
    if (stream_) cudaStreamDestroy(stream_);
    if (spare_) cudaFree(spare_);
    if (bounce_) cudaFreeHost(bounce_);
    spare_ = nullptr;
    bounce_ = nullptr;
    stream_ = nullptr;
    snapshot_ = landed_ = nullptr;
    cap_ = 0;
    dst_ = nullptr;
    tail_.clear();
}

bool CheckpointDeferral::reserve(size_t bytes, std::string& error) {
    if (bytes <= cap_ || !bytes) return true;
    if (dst_ && !finish(error)) return false;
    release();
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&snapshot_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&landed_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        release();
        error = "cannot create the deferred checkpoint stream";
        return false;
    }
    if (cudaMalloc((void**) &spare_, bytes) != cudaSuccess || cudaHostAlloc((void**) &bounce_, kBounceBytes, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        release();
        error = "cannot allocate the deferred checkpoint buffer";
        return false;
    }
    cap_ = bytes;
    return true;
}

bool CheckpointDeferral::save(const std::vector<SaveSegment>& segments, cudaStream_t stream, const void* key, std::string& error) {
    if (dst_ && !finish(error)) return false;
    size_t total = 0, loose = 0;
    for (const SaveSegment& s : segments) {
        const size_t n = s.bytes();
        total += n;
        if (n && !host_range_pinned(s.host)) loose += n;
    }
    // host bytes the driver would stage itself make the copy wait for the host: more than the bounce buffer holds, and
    // the save is made at once instead
    if (!spare_ || total > cap_ || !total || loose > kBounceBytes) return checkpoint_copy_out(segments, stream, error);
    // snapshot: every region into the spare, in the stream's order, so work queued after this call cannot change it
    size_t at = 0;
    for (const SaveSegment& s : segments)
        for (const DeviceRegion& r : s.regions) {
            if (r.bytes == 0) continue;
            if (!r.dev || !s.host) { error = "checkpoint transfer: a running-state region has no buffer"; return false; }
            if (const cudaError_t e = cudaMemcpyAsync(spare_ + at, r.dev, r.bytes, cudaMemcpyDeviceToDevice, stream); e != cudaSuccess)
                return fail(error, "snapshot", e);
            at += r.bytes;
        }
    cudaError_t e = cudaEventRecord(snapshot_, stream);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(stream_, snapshot_, 0);
    if (e != cudaSuccess) return fail(error, "snapshot order", e);
    // one host copy per segment; a segment on ordinary memory goes through the bounce buffer and is copied on at finish()
    size_t from = 0, bounced = 0;
    tail_.clear();
    for (const SaveSegment& s : segments) {
        const size_t n = s.bytes();
        if (n == 0) continue;
        const bool direct = host_range_pinned(s.host);
        uint8_t* to = direct ? s.host : bounce_ + bounced;
        if (!direct) {
            tail_.push_back({s.host, bounce_ + bounced, n});
            bounced += n;
        }
        if (e = cudaMemcpyAsync(to, spare_ + from, n, cudaMemcpyDeviceToHost, stream_); e != cudaSuccess) {
            tail_.clear();
            return fail(error, "copy out", e);
        }
        from += n;
    }
    if (e = cudaEventRecord(landed_, stream_); e != cudaSuccess) {
        tail_.clear();
        return fail(error, "copy out", e);
    }
    dst_ = key;
    return true;
}

bool CheckpointDeferral::finish(std::string& error) {
    if (!dst_) return true;
    const cudaError_t c = cudaEventSynchronize(landed_);
    if (c == cudaSuccess)
        for (const Bounced& b : tail_) std::memcpy(b.host, b.bounce, b.bytes);
    tail_.clear();
    dst_ = nullptr;
    if (c == cudaSuccess) return true;
    error = std::string("conversation checkpoint deferred copy: ") + cudaGetErrorString(c);
    return false;
}

}  // namespace basalt::core
