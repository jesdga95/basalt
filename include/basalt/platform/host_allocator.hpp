#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <limits>
#include <vector>

namespace basalt::platform {

struct HostAllocatorHooks {
    void* (*allocate)(size_t bytes, size_t alignment);
    bool (*release)(void* p);
    uint64_t (*available)();
};
inline std::atomic<const HostAllocatorHooks*> host_allocator_hooks{nullptr};

inline void* allocate_host_buffer(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    if (const auto* hooks = host_allocator_hooks.load(std::memory_order_acquire)) {
        if (void* p = hooks->allocate(bytes, std::max(alignment, alignof(std::max_align_t)))) return p;
        throw std::bad_alloc();
    }
    return alignment > alignof(std::max_align_t) ? ::operator new(bytes, std::align_val_t(alignment)) : ::operator new(bytes);
}
inline void release_host_buffer(void* p, size_t alignment = alignof(std::max_align_t)) noexcept {
    if (!p) return;
    if (const auto* hooks = host_allocator_hooks.load(std::memory_order_acquire))
        if (hooks->release(p)) return;
    if (alignment > alignof(std::max_align_t)) ::operator delete(p, std::align_val_t(alignment));
    else ::operator delete(p);
}

inline uint64_t reserved_host_available() {
    const auto* hooks = host_allocator_hooks.load(std::memory_order_acquire);
    return hooks ? hooks->available() : UINT64_MAX;
}

template<class T> struct HostAllocator {
    using value_type = T;
    HostAllocator() = default;
    template<class U> HostAllocator(const HostAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
        return static_cast<T*>(allocate_host_buffer(n * sizeof(T), alignof(T)));
    }
    void deallocate(T* p, size_t) noexcept { release_host_buffer(p, alignof(T)); }
    template<class U> bool operator==(const HostAllocator<U>&) const noexcept { return true; }
};
template<class T> using HostVector = std::vector<T, HostAllocator<T>>;

}  // namespace basalt::platform
