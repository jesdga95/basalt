#pragma once

#include "basalt/kernels/dp4a.hpp"

#include <cuda/atomic>

namespace basalt::kernels {

using DoorbellWord = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

__device__ __forceinline__ uint32_t doorbell_load_relaxed(const uint32_t* word) {
    return DoorbellWord(*const_cast<uint32_t*>(word)).load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint32_t doorbell_load_acquire(const uint32_t* word) {
    return DoorbellWord(*const_cast<uint32_t*>(word)).load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void doorbell_store_relaxed(uint32_t* word, uint32_t value) {
    DoorbellWord(*word).store(value, cuda::memory_order_relaxed);
}

__device__ __forceinline__ void doorbell_store_release(uint32_t* word, uint32_t value) {
    DoorbellWord(*word).store(value, cuda::memory_order_release);
}

// A non-null abort word takes priority on each designated poll; abort_every is positive.
__device__ __forceinline__ bool doorbell_spin(const uint32_t* flag, uint32_t want, const uint32_t* abort, int abort_every) {
    int until_abort = abort_every;
    for (;;) {
        if (abort && --until_abort == 0) {
            if (doorbell_load_relaxed(abort)) return false;
            until_abort = abort_every;
        }
        if (doorbell_load_relaxed(flag) == want) {
            cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
            return true;
        }
        spin_pause();
    }
}

}  // namespace basalt::kernels
