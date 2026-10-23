// Helpers for the per-device dynamic shared-memory opt-in of a launch family.  A layer split launches the same
// kernels from one thread per card, so the lazy per-device state is set under a lock and published with a release
// store; a failed opt-in is fatal because the launch would otherwise fail later as invalid.
#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "basalt/log.hpp"

namespace basalt::kernels {

inline int opt_in_device(const char* who) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        basalt::log::error("basalt", "%s: device %d out of range", who, dev);
        std::abort();
    }
    return dev;
}

inline void opt_in_check(cudaError_t e, const char* who) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: shared-memory opt-in failed: %s", who, cudaGetErrorString(e));
        std::abort();
    }
}

inline std::mutex& opt_in_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace basalt::kernels
