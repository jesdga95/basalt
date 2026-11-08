#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace basalt::core {

// A fixed backing store. Exhaustion never allocates a second backing region.
class HostArena {
public:
    struct Stats { uint64_t capacity = 0, used = 0, peak = 0, failures = 0; size_t allocations = 0; };
    bool reset(void* base, size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!live_.empty()) return false;
        base_ = static_cast<uint8_t*>(base);
        free_.clear(); live_.clear(); stats_ = {};
        if (base && bytes) { free_.emplace(0, bytes); stats_.capacity = bytes; }
        return true;
    }
    void* allocate(size_t bytes, size_t alignment = 256) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!bytes || !alignment || (alignment & (alignment - 1))) { ++stats_.failures; return nullptr; }
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(base_) + it->first;
            if (address > std::numeric_limits<uintptr_t>::max() - (alignment - 1)) continue;
            const uintptr_t aligned = (address + alignment - 1) & ~(uintptr_t(alignment) - 1);
            const size_t padding = aligned - address;
            if (padding > it->second || bytes > it->second - padding) continue;
            const size_t offset = it->first, at = offset + padding, remaining = it->second - padding - bytes;
            void* result = base_ + at;
            // Bookkeeping succeeds before the free list changes.
            live_.emplace(at, bytes);
            try {
                if (remaining) free_.emplace(at + bytes, remaining);
            } catch (...) { live_.erase(at); throw; }
            if (padding) it->second = padding;
            else free_.erase(it);
            stats_.used += bytes; stats_.peak = std::max(stats_.peak, stats_.used); ++stats_.allocations;
            return result;
        }
        ++stats_.failures;
        return nullptr;
    }
    bool owns(const void* p) const {
        return owns_range(p, 1);
    }
    bool owns_range(const void* p, size_t bytes) const {
        std::lock_guard<std::mutex> lock(mu_);
        const uintptr_t a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(base_);
        return base_ && bytes > 0 && a >= b && a - b < stats_.capacity && bytes <= stats_.capacity - (a - b);
    }
    bool release(void* p) {
        std::lock_guard<std::mutex> lock(mu_);
        const uintptr_t address = reinterpret_cast<uintptr_t>(p), base = reinterpret_cast<uintptr_t>(base_);
        if (!base_ || address < base || address - base >= stats_.capacity) return false;
        const auto found = live_.find(address - base);
        if (found == live_.end()) return false;
        size_t offset = found->first, bytes = found->second;
        const size_t charged = bytes;
        auto next = free_.lower_bound(offset);
        if (next != free_.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == offset) {
                offset = prev->first; bytes += prev->second;
                // Reuse its node so freeing cannot need heap memory.
                prev->second = bytes;
                if (next != free_.end() && offset + bytes == next->first) {
                    prev->second += next->second; free_.erase(next);
                }
                live_.erase(found); stats_.used -= charged; --stats_.allocations;
                return true;
            }
        }
        auto node = live_.extract(found);
        if (next != free_.end() && offset + bytes == next->first) {
            node.mapped() += next->second; free_.erase(next);
        }
        free_.insert(std::move(node));
        stats_.used -= charged; --stats_.allocations;
        return true;
    }
    Stats stats() const { std::lock_guard<std::mutex> lock(mu_); return stats_; }
    size_t max_live_end() const {
        std::lock_guard<std::mutex> lock(mu_);
        return live_.empty() ? 0 : live_.rbegin()->first + live_.rbegin()->second;
    }
    size_t allocation_bytes(const void* p) const {
        std::lock_guard<std::mutex> lock(mu_);
        const uintptr_t address = reinterpret_cast<uintptr_t>(p), base = reinterpret_cast<uintptr_t>(base_);
        if (!base_ || address < base) return 0;
        const auto it = live_.find(address - base);
        return it == live_.end() ? 0 : it->second;
    }
    template<class Visitor> void visit_free(Visitor visitor) {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& [off, bytes] : free_) visitor(base_ + off, bytes);
    }
private:
    uint8_t* base_ = nullptr;
    mutable std::mutex mu_;
    std::map<size_t, size_t> free_;
    std::map<size_t, size_t> live_;
    Stats stats_;
};

}  // namespace basalt::core
