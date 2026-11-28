// src/core/elastic_array.cpp - see elastic_array.hpp.
#include "basalt/core/elastic_array.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

namespace basalt::core {

namespace {
// The new bytes are zero before any other stream can read them.
bool clear_extent(void* data, uint64_t bytes, bool& idle) {
    const cudaError_t clear = cudaMemsetAsync(data, 0, bytes, nullptr);
    idle = cudaStreamSynchronize(nullptr) == cudaSuccess;
    return clear == cudaSuccess && idle;
}
}  // namespace

ElasticArray::~ElasticArray() {
    if (!base_) return;
    const OnDevice on_device(device_);
    if (plain_) {
        cudaFree(base_);
        return;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return;
    for (const Extent& e : extents_) vmm::unmap(base_ + e.off, e.bytes, e.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err, uint32_t step_granules) {
    if (bytes == 0) bytes = 1;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void) cudaGetLastError();
        dev = -1;
    }
    if (vmm::supported()) {
        constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
        const uint64_t g = vmm::granularity();
        if (g > 0 && bytes <= kMax - (g - 1)) {
            const uint64_t mult = step_granules > 0 ? (uint64_t) step_granules : 1;
            const uint64_t step = g <= kMax / mult ? g * mult : g;
            const uint64_t res = (bytes + g - 1) / g * g;
            if (uint8_t* va = vmm::reserve(res); va != nullptr) {
                device_ = dev;
                gran_ = g;
                step_ = step;
                reserved_ = res;
                base_ = va;
                return true;
            }
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t) bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "out of device memory for the array";
        return false;
    }
    bool idle = false;
    if (!clear_extent(p, bytes, idle)) {
        if (idle) cudaFree(p);
        (void) cudaGetLastError();
        err = "cannot initialize the array";
        return false;
    }
    device_ = dev;
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    plain_ = true;
    return true;
}

uint64_t ElasticArray::growth(uint64_t bytes) const {
    if (!base_ || plain_ || gran_ == 0 || step_ == 0 || bytes <= backed_ || bytes > reserved_) return 0;
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const uint64_t rounded = bytes <= kMax - (step_ - 1) ? (bytes + step_ - 1) / step_ * step_ : reserved_;
    const uint64_t target = std::min(reserved_, rounded);
    return target > backed_ ? target - backed_ : 0;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    out_of_memory_ = false;
    if (unusable_) {
        err = "the array's backing is uncertain";
        return false;
    }
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    const uint64_t n = growth(bytes);
    if (n == 0) return true;
    const OnDevice on_device(device_);
    const uint64_t want = backed_ + n;
    if (keep_free > 0) {
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!device_memory(-1, free_b, total_b, why)) {
            err = "cannot read the free device memory: " + why;
            return false;
        }
        if (keep_free > free_b || n > free_b - keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            return false;
        }
    }
    const unsigned long long h = vmm::map(base_ + backed_, n, &out_of_memory_);
    if (!h) {   // any failure but a clean out-of-memory may leave the range mapped untracked
        unusable_ = !out_of_memory_;
        err = out_of_memory_ ? "out of device memory for the array" : "the array's new backing could not be mapped";
        return false;
    }
    bool idle = false;
    if (!clear_extent(base_ + backed_, n, idle)) {
        if (!idle) extents_.push_back({backed_, n, h});
        unusable_ = !idle || !vmm::unmap(base_ + backed_, n, h);
        err = "cannot initialize the array's new backing";
        return false;
    }
    extents_.push_back({backed_, n, h});
    backed_ = want;
    return true;
}

void ElasticArray::trim(uint64_t bytes) {
    if (unusable_ || plain_ || !base_ || extents_.empty() || extents_.back().off < bytes) return;
    const OnDevice on_device(device_);
    if (cudaDeviceSynchronize() != cudaSuccess) {   // a reader may still use what would go
        unusable_ = true;
        return;
    }
    while (!extents_.empty() && extents_.back().off >= bytes) {
        const Extent& e = extents_.back();
        if (!vmm::unmap(base_ + e.off, e.bytes, e.handle)) {   // its record stays: the range is not reused
            unusable_ = true;
            break;
        }
        extents_.pop_back();
    }
    backed_ = std::min(backed_, extents_.empty() ? 0 : extents_.back().off + extents_.back().bytes);
}

}  // namespace basalt::core
