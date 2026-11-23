#pragma once

#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::inkling2 {

// The streams whose windows wait on the CPU experts' completion, watched by the drain after an abort. An owner notes
// its stream before its first window, and forgets it after synchronizing it and before destroying it. The drain holds
// the list across its queries, so a forgotten stream is never queried and a watched one is not destroyed while it is.
// A forgotten stream whose synchronize failed may still hold work: every later drain reports completion unconfirmed.
class MergeStreams {
public:
    void note(cudaStream_t s) {
        std::lock_guard<std::mutex> lk(mu_);
        if (std::find(streams_.begin(), streams_.end(), s) == streams_.end()) streams_.push_back(s);
    }
    // `synced`: the stream's synchronize status. Waits for a drain in progress.
    void forget(cudaStream_t s, cudaError_t synced) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto gone = std::remove(streams_.begin(), streams_.end(), s);
        if (gone != streams_.end() && retired_ == cudaSuccess) retired_ = synced;
        streams_.erase(gone, streams_.end());
    }
    // Report whether queued GPU visits drained after the bell was aborted: `own` (stream 0), then the noted streams.
    template <class Query>
    void drain(cudaError_t selected, cudaStream_t own, Query query) {
        constexpr int kDrainMs = 5000;
        const auto start = std::chrono::steady_clock::now();
        const auto elapsed_ms = [&] {
            return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        };
        cudaError_t last_bad = selected;
        char where[32] = "device";
        if (selected == cudaSuccess) {
            std::lock_guard<std::mutex> lk(mu_);
            if (retired_ != cudaSuccess) {
                last_bad = retired_;
                std::snprintf(where, sizeof where, "a retired stream");
            }
            // A stream in a capture is not queried, which would break its owner's capture: it is polled until the
            // capture ends, then what was queued on it before the capture is waited on as any other work.
            const auto poll = [&](cudaStream_t s) {
                cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
                if (const cudaError_t e = cudaStreamIsCapturing(s, &capture); e != cudaSuccess) return e;
                return capture == cudaStreamCaptureStatusNone ? query(s) : cudaErrorStreamCaptureUnsupported;
            };
            const auto waiting = [](cudaError_t e) {
                return e == cudaErrorNotReady || e == cudaErrorStreamCaptureUnsupported;
            };
            for (std::size_t i = 0; i <= streams_.size(); ++i) {
                const cudaStream_t s = i ? streams_[i - 1] : own;
                cudaError_t status = poll(s);
                while (waiting(status) && elapsed_ms() < kDrainMs) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    status = poll(s);
                }
                if (status != cudaSuccess) {
                    last_bad = status;
                    std::snprintf(where, sizeof where, "stream %d", (int) i);
                }
            }
        }
        (void) cudaGetLastError();
        if (last_bad != cudaSuccess) {
            basalt::core::gpu_left_running().store(true);
            basalt::log::error("basalt inkling2", "the CPU experts' doorbell was aborted: GPU completion unconfirmed after %lld ms (%s, %s)",
                               elapsed_ms(), cudaGetErrorString(last_bad), where);
        } else {
            basalt::log::info("basalt inkling2", "the CPU experts' doorbell was aborted: the GPU finished in %lld ms", elapsed_ms());
        }
    }

private:
    std::mutex mu_;
    std::vector<cudaStream_t> streams_;
    cudaError_t retired_ = cudaSuccess;   // the first failed synchronize of a forgotten stream
};

}  // namespace basalt::models::inkling2
