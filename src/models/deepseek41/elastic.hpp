// src/models/deepseek41/elastic.hpp - a device array that reserves its whole address range up front and backs it with
// physical memory as it is used. The address never moves, so kernel arguments and captured graphs stay valid.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class ElasticBuf {
public:
    ElasticBuf() = default;
    ~ElasticBuf();
    ElasticBuf(const ElasticBuf&) = delete;
    ElasticBuf& operator=(const ElasticBuf&) = delete;

    /// Reserves `bytes` of address space. Without virtual memory management the whole array is allocated at once.
    bool open(uint64_t bytes, std::string& err);
    /// Backs at least the first `bytes` bytes (rounded up to the allocation granularity; the new bytes are zeroed).
    /// False with `err` when the device has no memory for them with `keep_free` left; what was backed stays.
    bool ensure(uint64_t bytes, uint64_t keep_free, std::string& err);
    /// Gives back the whole allocations that lie past `bytes` (the device is synchronised first); the address range stays.
    void trim(uint64_t bytes);
    uint8_t* data() const { return base_; }
    uint64_t backed() const { return backed_; }
    uint64_t capacity() const { return reserved_; }

private:
    struct Extent {
        uint64_t off, bytes;
        unsigned long long handle;
    };
    uint8_t* base_ = nullptr;
    uint64_t reserved_ = 0, backed_ = 0, gran_ = 0;
    bool plain_ = false;   // one cudaMalloc, no virtual memory
    std::vector<Extent> extents_;
};

}  // namespace basalt::models::deepseek41
