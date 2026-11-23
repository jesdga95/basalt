#pragma once

#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::inkling2 {

// The streams whose windows wait on the CPU experts' completion, watched by the drain after an abort. An owner forgets
// its stream once its work has finished and before destroying it. The drain holds the list across its queries, so a
// forgotten stream is never queried and a watched one is not destroyed while it is.
class MergeStreams {
public:
    void note(cudaStream_t s) {
        std::lock_guard<std::mutex> lk(mu_);
        if (std::find(streams_.begin(), streams_.end(), s) == streams_.end()) streams_.push_back(s);
    }
    // Waits for a drain in progress.
    void forget(cudaStream_t s) {
        std::lock_guard<std::mutex> lk(mu_);
        streams_.erase(std::remove(streams_.begin(), streams_.end(), s), streams_.end());
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
        int bad_stream = -1;  // device selection failed before any stream query
        if (selected == cudaSuccess) {
            std::lock_guard<std::mutex> lk(mu_);
            for (std::size_t i = 0; i <= streams_.size(); ++i) {
                const cudaStream_t s = i ? streams_[i - 1] : own;
                cudaError_t status = query(s);
                while (status == cudaErrorNotReady && elapsed_ms() < kDrainMs) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    status = query(s);
                }
                if (status != cudaSuccess) {
                    last_bad = status;
                    bad_stream = (int) i;
                }
            }
        }
        (void) cudaGetLastError();
        if (last_bad != cudaSuccess) {
            basalt::core::gpu_left_running().store(true);
            basalt::log::error("basalt inkling2", "the CPU experts' doorbell was aborted: GPU completion unconfirmed after %lld ms (%s, stream %d)",
                               elapsed_ms(), cudaGetErrorString(last_bad), bad_stream);
        } else {
            basalt::log::info("basalt inkling2", "the CPU experts' doorbell was aborted: the GPU finished in %lld ms", elapsed_ms());
        }
    }

private:
    std::mutex mu_;
    std::vector<cudaStream_t> streams_;
};

}  // namespace basalt::models::inkling2
