#pragma once

#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <span>
#include <thread>

namespace basalt::models::inkling2 {

// Report whether queued GPU visits drained after the bell was aborted.
template <class Query>
void drain_cpu_expert_streams(cudaError_t selected, std::span<const cudaStream_t> watch, Query query) {
    constexpr int kDrainMs = 5000;
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed_ms = [&] {
        return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    };
    cudaError_t last_bad = selected;
    int bad_stream = -1;  // device selection failed before any stream query
    if (selected == cudaSuccess) {
        for (std::size_t i = 0; i < watch.size(); ++i) {
            cudaError_t status = query(watch[i]);
            while (status == cudaErrorNotReady && elapsed_ms() < kDrainMs) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                status = query(watch[i]);
            }
            if (status == cudaErrorInvalidResourceHandle) {
                // A retired merge stream can still be in the watch list.
                (void) cudaGetLastError();
            } else if (status != cudaSuccess) {
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

}  // namespace basalt::models::inkling2
