// src/core/elastic_array.cpp - see elastic_array.hpp.
#include "basalt/core/elastic_array.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::core {

ElasticArray::~ElasticArray() {
    if (!base_) return;
    if (plain_) {
        cudaFree(base_);
        return;
    }
    cudaDeviceSynchronize();
    for (const Extent& e : extents_) vmm::unmap(base_ + e.off, e.bytes, e.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err) {
    if (bytes == 0) bytes = 1;
    if (vmm::supported()) {
        gran_ = vmm::granularity();
        if (gran_) {
            reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
            base_ = vmm::reserve(reserved_);
            if (base_) return true;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    plain_ = true;
    return true;
}

// grow in steps of a few granules so a long generation maps rarely
uint64_t ElasticArray::growth(uint64_t bytes) const {
    if (bytes <= backed_ || bytes > reserved_) return 0;
    const uint64_t step = gran_ * 4;
    return std::min(reserved_, (bytes + step - 1) / step * step) - backed_;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    const uint64_t n = growth(bytes);
    const uint64_t want = backed_ + n;
    if (keep_free > 0) {
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!device_memory(-1, free_b, total_b, why)) {
            err = "cannot read the free device memory: " + why;
            return false;
        }
        if (free_b < n + keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            return false;
        }
    }
    const unsigned long long h = vmm::map(base_ + backed_, n);
    if (!h) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(base_ + backed_, 0, n);
    extents_.push_back({backed_, n, h});
    backed_ = want;
    return true;
}

void ElasticArray::trim(uint64_t bytes) {
    if (plain_ || !base_ || extents_.empty() || extents_.back().off < bytes) return;
    cudaDeviceSynchronize();
    while (!extents_.empty() && extents_.back().off >= bytes) {
        const Extent& e = extents_.back();
        vmm::unmap(base_ + e.off, e.bytes, e.handle);
        extents_.pop_back();
    }
    backed_ = extents_.empty() ? 0 : extents_.back().off + extents_.back().bytes;
}

}  // namespace basalt::core
