// src/models/deepseek41/elastic.cpp - see elastic.hpp.
#include "models/deepseek41/elastic.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::deepseek41 {

ElasticBuf::~ElasticBuf() {
    if (!base_) return;
    if (plain_) {
        cudaFree(base_);
        return;
    }
    cudaDeviceSynchronize();
    for (const Extent& e : extents_) core::vmm::unmap(base_ + e.off, e.bytes, e.handle);
    core::vmm::release_range(base_, reserved_);
}

bool ElasticBuf::open(uint64_t bytes, std::string& err) {
    if (bytes == 0) bytes = 1;
    if (core::vmm::supported()) {
        gran_ = core::vmm::granularity();
        if (gran_) {
            reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
            base_ = core::vmm::reserve(reserved_);
            if (base_) return true;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the compressed caches";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    plain_ = true;
    return true;
}

bool ElasticBuf::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the compressed cache is full";
        return false;
    }
    // grow in steps of a few granules so a long generation maps rarely
    const uint64_t step = gran_ * 4;
    const uint64_t want = std::min(reserved_, (bytes + step - 1) / step * step);
    const uint64_t n = want - backed_;
    if (keep_free > 0) {
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!core::device_memory(-1, free_b, total_b, why)) {
            err = "cannot read the free device memory: " + why;
            return false;
        }
        if (free_b < n + keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            return false;
        }
    }
    const unsigned long long h = core::vmm::map(base_ + backed_, n);
    if (!h) {
        err = "out of device memory for the compressed caches";
        return false;
    }
    cudaMemset(base_ + backed_, 0, n);
    extents_.push_back({backed_, n, h});
    backed_ = want;
    return true;
}

void ElasticBuf::trim(uint64_t bytes) {
    if (plain_ || !base_ || extents_.empty() || extents_.back().off < bytes) return;
    cudaDeviceSynchronize();
    while (!extents_.empty() && extents_.back().off >= bytes) {
        const Extent& e = extents_.back();
        core::vmm::unmap(base_ + e.off, e.bytes, e.handle);
        extents_.pop_back();
    }
    backed_ = extents_.empty() ? 0 : extents_.back().off + extents_.back().bytes;
}

}  // namespace basalt::models::deepseek41
