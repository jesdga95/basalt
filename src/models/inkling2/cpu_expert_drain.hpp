#pragma once

#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace basalt::models::inkling2 {

// Admission to the windows that wait on the CPU experts. A call holds it from before its first such window until the
// sync that ends them. The abort closes it before releasing the bell: no call enters after that, a call already in
// holds it until its sync, however many windows it still launches, and discards its results. The abort then waits for
// the calls in flight: once none holds it, every window that could wait on the bell has finished.
class WindowGate {
public:
    // False once closed.
    bool enter() {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return false;
        ++held_;
        return true;
    }
    // `synced`: the call's window streams' synchronize status. False once closed: the call's results are discarded.
    bool leave(cudaError_t synced) {
        std::lock_guard<std::mutex> lk(mu_);
        --held_;
        if (synced != cudaSuccess && failed_ == cudaSuccess) failed_ = synced;
        cv_.notify_all();
        return !closed_;
    }
    // A failure that leaves a window's fork without its join: completion can no longer be confirmed.
    void fail(cudaError_t e) {
        std::lock_guard<std::mutex> lk(mu_);
        if (failed_ == cudaSuccess) failed_ = e;
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(mu_);
        return closed_;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    // After close and the bell's release: report whether the calls in flight finished with their streams synchronized
    // within `bound_ms`.
    void drain(int bound_ms) {
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mu_);
        const bool idle = cv_.wait_for(lk, std::chrono::milliseconds(bound_ms), [&] { return held_ == 0; });
        const long long ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (idle && failed_ == cudaSuccess) {
            basalt::log::info("basalt inkling2", "the CPU experts' doorbell was aborted: the GPU finished in %lld ms", ms);
            return;
        }
        char why[96];
        if (!idle) std::snprintf(why, sizeof why, "%d calls still in flight", held_);
        else std::snprintf(why, sizeof why, "%s in a call's windows", cudaGetErrorString(failed_));
        basalt::core::gpu_left_running().store(true);
        basalt::log::error("basalt inkling2", "the CPU experts' doorbell was aborted: GPU completion unconfirmed after %lld ms (%s)", ms, why);
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool closed_ = false;
    int held_ = 0;
    cudaError_t failed_ = cudaSuccess;   // the first failed sync of a call that left, or a fork left unjoined
};

}  // namespace basalt::models::inkling2
