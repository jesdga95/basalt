// include/basalt/core/elastic_array.hpp - a device array that reserves its whole address range up front and backs it with
// physical memory as it is used. The address never moves, so kernel arguments and captured graphs stay valid.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class ElasticArray {
public:
    ElasticArray() = default;
    ~ElasticArray();
    ElasticArray(const ElasticArray&) = delete;
    ElasticArray& operator=(const ElasticArray&) = delete;

    /// Reserves `bytes` of address space. Without virtual memory management the whole array is allocated at once.
    bool open(uint64_t bytes, std::string& err);
    /// Backs at least the first `bytes` bytes (rounded up to the allocation granularity; the new bytes are zeroed).
    /// False with `err` when the device has no memory for them with `keep_free` left; what was backed stays.
    bool ensure(uint64_t bytes, uint64_t keep_free, std::string& err);
    /// The device bytes ensure(bytes) would back beyond what is backed now (its rounding included); 0 when it backs nothing.
    uint64_t growth(uint64_t bytes) const;
    /// Gives back the whole allocations that lie past `bytes` (the device is synchronised first); the address range stays.
    /// Nothing is released when the device cannot be confirmed idle or the array's backing is uncertain.
    void trim(uint64_t bytes);
    /// False once a new extent's map, clear or release could not be confirmed: the array is not grown or trimmed again.
    bool usable() const { return !unusable_; }
    /// The last ensure failed only because the device lacked the memory (nothing was mapped).
    bool out_of_memory() const { return out_of_memory_; }
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
    bool unusable_ = false;   // a map, clear or release could not be confirmed: the range's backing is not known
    bool out_of_memory_ = false;
    bool plain_ = false;   // one cudaMalloc, no virtual memory
    std::vector<Extent> extents_;
};

}  // namespace basalt::core
