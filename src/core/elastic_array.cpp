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
    for (const Extent& e : host_extents_) vmm::unmap(base_ + e.off, e.bytes, e.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err, uint64_t host_from) {
    if (bytes == 0) bytes = 1;
    const bool host = host_from < bytes;
    if (vmm::supported()) {
        gran_ = vmm::granularity();
        const uint64_t host_gran = host ? vmm::host_granularity() : 1;
        if (gran_ && host_gran) {
            gran_ = std::max(gran_, host_gran);   // both are powers of two
            reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
            host_from_ = host ? std::min(reserved_, (host_from + gran_ - 1) / gran_ * gran_) : reserved_;
            base_ = vmm::reserve(reserved_);
            if (base_) return map_host_tail(err);
        }
    }
    if (host) {
        err = "no host virtual memory management to map pinned memory into the array";
        return false;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = host_from_ = bytes;
    plain_ = true;
    return true;
}

bool ElasticArray::map_host_tail(std::string& err) {
    // pieces below the driver's single-allocation limit
    constexpr uint64_t kPiece = 1ull << 30;
    for (uint64_t off = host_from_; off < reserved_;) {
        const uint64_t n = std::min(kPiece, reserved_ - off);
        const unsigned long long h = vmm::map_host(base_ + off, n);
        if (!h) {
            err = "out of page-locked host memory for the array";
            return false;   // the destructor unmaps the pieces
        }
        host_extents_.push_back({off, n, h});
        off += n;
    }
    if (host_from_ < reserved_) cudaMemset(base_ + host_from_, 0, reserved_ - host_from_);
    return true;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    if (bytes <= backed_ || backed_ == host_from_) return true;   // the rest is host memory
    // grow in steps of a few granules so a long generation maps rarely
    const uint64_t step = gran_ * 4;
    const uint64_t want = std::min(host_from_, (bytes + step - 1) / step * step);
    const uint64_t n = want - backed_;
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
