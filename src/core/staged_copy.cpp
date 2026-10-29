// src/core/staged_copy.cpp - see include/basalt/core/staged_copy.hpp.
#include "basalt/core/staged_copy.hpp"

#include "basalt/core/on_device.hpp"

#include <algorithm>

namespace basalt::core {

StagedCopy::~StagedCopy() { release(); }

void StagedCopy::release() {
    for (int b = 0; b < kBufs; ++b) {
        if (buf_[b] != nullptr) cudaFreeHost(buf_[b]);
        buf_[b] = nullptr;
        live_[b] = false;
    }
    if (src_dev_ >= 0) {
        const OnDevice on(src_dev_);
        if (src_stream_ != nullptr) { cudaStreamSynchronize(src_stream_); cudaStreamDestroy(src_stream_); }
        for (cudaEvent_t& e : d2h_) { if (e != nullptr) cudaEventDestroy(e); e = nullptr; }
    }
    if (dst_dev_ >= 0) {
        const OnDevice on(dst_dev_);
        if (dst_stream_ != nullptr) { cudaStreamSynchronize(dst_stream_); cudaStreamDestroy(dst_stream_); }
        for (cudaEvent_t& e : h2d_) { if (e != nullptr) cudaEventDestroy(e); e = nullptr; }
    }
    src_stream_ = dst_stream_ = nullptr;
    src_dev_ = dst_dev_ = -1;
}

bool StagedCopy::init(int src_dev, int dst_dev, size_t piece_bytes, std::string& err) {
    release();
    src_dev_ = src_dev;
    dst_dev_ = dst_dev;
    piece_ = piece_bytes;
    bool ok = piece_ > 0;
    for (int b = 0; ok && b < kBufs; ++b) ok = cudaHostAlloc((void**) &buf_[b], piece_, cudaHostAllocPortable) == cudaSuccess;
    {
        const OnDevice on(src_dev_);
        ok = ok && cudaStreamCreateWithFlags(&src_stream_, cudaStreamNonBlocking) == cudaSuccess;
        for (int b = 0; ok && b < kBufs; ++b) ok = cudaEventCreateWithFlags(&d2h_[b], cudaEventDisableTiming) == cudaSuccess;
    }
    {
        const OnDevice on(dst_dev_);
        ok = ok && cudaStreamCreateWithFlags(&dst_stream_, cudaStreamNonBlocking) == cudaSuccess;
        for (int b = 0; ok && b < kBufs; ++b) ok = cudaEventCreateWithFlags(&h2d_[b], cudaEventDisableTiming) == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        release();
        err = "a staged copy between two cards: its pinned buffers or streams";
        return false;
    }
    return true;
}

namespace {
bool is_host(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeHost;
}
}  // namespace

bool StagedCopy::copy(void* dst, const void* src, uint64_t bytes, std::string& err) {
    // pinned host memory on one side: one copy over that card's link, ordered on the destination stream
    if (bytes > 0 && (is_host(src) || is_host(dst))) {
        bool ok = true;
        if (is_host(src)) {
            const OnDevice on(dst_dev_);
            ok = cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyHostToDevice, dst_stream_) == cudaSuccess;
        } else {
            const int b = next_;
            next_ = (next_ + 1) % kBufs;
            {
                const OnDevice on(src_dev_);
                ok = cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToHost, src_stream_) == cudaSuccess &&
                     cudaEventRecord(d2h_[b], src_stream_) == cudaSuccess;
            }
            const OnDevice on(dst_dev_);
            ok = ok && cudaStreamWaitEvent(dst_stream_, d2h_[b], 0) == cudaSuccess &&
                 cudaEventRecord(h2d_[b], dst_stream_) == cudaSuccess;
            live_[b] = ok;
        }
        if (!ok) err = std::string("a staged copy between two cards: ") + cudaGetErrorString(cudaGetLastError());
        return ok;
    }
    for (uint64_t at = 0; at < bytes;) {
        const size_t n = (size_t) std::min<uint64_t>(piece_, bytes - at);
        const int b = next_;
        next_ = (next_ + 1) % kBufs;
        bool ok = true;
        {
            const OnDevice on(src_dev_);
            if (live_[b]) ok = cudaStreamWaitEvent(src_stream_, h2d_[b], 0) == cudaSuccess;   // the buffer's last upload
            ok = ok && cudaMemcpyAsync(buf_[b], (const uint8_t*) src + at, n, cudaMemcpyDeviceToHost, src_stream_) == cudaSuccess &&
                 cudaEventRecord(d2h_[b], src_stream_) == cudaSuccess;
        }
        {
            const OnDevice on(dst_dev_);
            ok = ok && cudaStreamWaitEvent(dst_stream_, d2h_[b], 0) == cudaSuccess &&
                 cudaMemcpyAsync((uint8_t*) dst + at, buf_[b], n, cudaMemcpyHostToDevice, dst_stream_) == cudaSuccess &&
                 cudaEventRecord(h2d_[b], dst_stream_) == cudaSuccess;
        }
        if (!ok) {
            err = std::string("a staged copy between two cards: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        live_[b] = true;
        at += n;
    }
    return true;
}

bool StagedCopy::wait(std::string& err) {
    const OnDevice on(dst_dev_);
    if (cudaStreamSynchronize(dst_stream_) != cudaSuccess) {
        err = std::string("a staged copy between two cards: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

}  // namespace basalt::core
