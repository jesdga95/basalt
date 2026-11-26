// src/core/ram_load.cpp - the RAM tier's memory: allocation, pinning and registration.
#include "basalt/core/ram_load.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace basalt::core {

bool host_pin_budget(uint64_t& limit, std::string& why) {
    if constexpr (!platform::kBoundHostRegistration) return platform::pin_budget(limit, why);
    limit = 0;
    why.clear();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    std::vector<cudaDeviceProp> props((size_t) count);
    std::vector<bool> valid((size_t) count, false);
    for (int d = 0; d < count; ++d) {
        if (cudaGetDeviceProperties(&props[(size_t) d], d) != cudaSuccess) { (void) cudaGetLastError(); continue; }
        uint64_t unused = 0;
        std::string note;
        const auto& prop = props[(size_t) d];
        if (!platform::pin_budget(unused, note, prop.luid, prop.luidDeviceNodeMask)) continue;
        valid[(size_t) d] = true;
        // Create mapping consumers before registering portable pages, without changing the caller's device.
        const cudaError_t initialized = cudaInitDevice(d, 0, 0);
        if (initialized != cudaSuccess) {
            why = "CUDA" + std::to_string(d) + " context initialization refused: " + cudaGetErrorString(initialized);
            (void) cudaGetLastError();
            return true;
        }
    }
    bool known = false;
    for (int d = 0; d < count; ++d) {
        if (!valid[(size_t) d]) continue;
        const auto& prop = props[(size_t) d];
        uint64_t allowance = 0;
        std::string note;
        if (!platform::pin_budget(allowance, note, prop.luid, prop.luidDeviceNodeMask, count > 1)) continue;
        if (!known || allowance < limit) {
            limit = allowance;
            why = "CUDA" + std::to_string(d) + " " + note;
        }
        known = true;
    }
    return known;
}

std::vector<uint64_t> ram_pin_slices(std::vector<uint64_t> starts, uint64_t end, uint64_t page, uint64_t target) {
    std::vector<uint64_t> cuts{0};
    if (!end) return cuts;
    page = std::max<uint64_t>(page, 1);
    target = std::max<uint64_t>(target, 1);
    std::sort(starts.begin(), starts.end());
    for (uint64_t at : starts)
        if (at < end && at % page == 0 && at - cuts.back() >= target) cuts.push_back(at);
    cuts.push_back(end);
    return cuts;
}

const uint8_t* ram_pin_alias(const std::vector<uint64_t>& starts,
                             const std::vector<const uint8_t*>& devices, uint64_t off) {
    if (starts.size() != devices.size()) return nullptr;
    const auto next = std::upper_bound(starts.begin(), starts.end(), off);
    if (next == starts.begin()) return nullptr;
    const size_t i = (size_t) (next - starts.begin() - 1);
    return devices[i] ? devices[i] + (size_t) (off - starts[i]) : nullptr;
}

namespace {
RamPin pin_ram_bounded(void* base, uint64_t bytes, uint64_t lock_bytes, const std::vector<uint64_t>& bounds,
                        bool mapped, const uint8_t** device, uint64_t remaining = ~uint64_t(0)) {
    RamPin p;
    if (device) *device = nullptr;
    uint64_t allowance = 0;
    const bool known = host_pin_budget(allowance, p.cap_why);
    if (known) remaining = std::min(remaining, allowance);
    if (remaining < bytes) {
        p.capped = true;
        p.cap = remaining;
        p.refused = "over the pin budget";
    }
    const unsigned flags = cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0u);
    const auto register_range = [&](uint64_t off, uint64_t n, const uint8_t*& alias) {
        void* at = (uint8_t*) base + off;
        const char* operation = "cudaHostRegister";
        cudaError_t e = cudaHostRegister(at, (size_t) n, flags);
        if (e == cudaSuccess && device) {
            operation = "cudaHostGetDevicePointer";
            void* d = nullptr;
            e = cudaHostGetDevicePointer(&d, at, 0);
            if (e == cudaSuccess && d == nullptr) e = cudaErrorInvalidValue;
            if (e == cudaSuccess) alias = (const uint8_t*) d;
            else (void) cudaHostUnregister(at);
        }
        if (e == cudaSuccess) return true;
        p.refused = cudaGetErrorString(e);
        if (off != 0 || n != bytes)
            basalt::log::warn("RAM registration", "%s at %llu MiB for %llu MiB: %d (%s)", operation,
                (unsigned long long) (off >> 20), (unsigned long long) (n >> 20), (int) e, p.refused.c_str());
        (void) cudaGetLastError();
        return false;
    };
    const uint8_t* alias = nullptr;
    if (bytes && !p.capped && register_range(0, bytes, alias)) {
        p.registered_bytes = bytes;
        if (device) *device = alias;
    } else if (bytes && bounds.size() >= 2) {
        p.sliced = true;
        const uint64_t page = platform::system_page_size();
        for (size_t i = 0; i + 1 < bounds.size(); ++i) {
            const uint64_t off = bounds[i], end = bounds[i + 1];
            if (off != p.registered_bytes || end <= off || end > bytes || !page || off % page || end % page ||
                (p.capped && end > p.cap)) break;
            alias = nullptr;
            if (!register_range(off, end - off, alias)) break;
            p.slice_starts.push_back(off);
            if (device) p.slice_devices.push_back(alias);
            p.registered_bytes = end;
            ++p.slices;
        }
        if (device && !p.slice_devices.empty()) *device = p.slice_devices.front();
        if (p.registered_bytes == bytes) p.refused.clear();
    }
    if (!bytes) p.refused = "over the pin budget";
    if (p.registered_bytes < lock_bytes) {
        p.lock_off = p.registered_bytes;
        const auto lr = platform::lock_resident((uint8_t*) base + p.lock_off, lock_bytes - p.lock_off);
        p.locked_bytes = lr.locked_bytes;
        p.lock_note = lr.note;
    }
    return p;
}
}  // namespace

RamPin pin_ram(void* base, uint64_t register_bytes, uint64_t lock_bytes, const std::vector<uint64_t>& bounds, bool mapped,
               const uint8_t** device) {
    if (platform::kBoundHostRegistration || bounds.size() >= 2)
        return pin_ram_bounded(base, register_bytes, lock_bytes, bounds, mapped, device);
    RamPin p;
    const unsigned flags = cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0u);
    const auto lock_rest = [&](uint64_t from) {
        const platform::LockResult lr = platform::lock_resident((uint8_t*) base + from, lock_bytes - from);
        p.locked_bytes = lr.locked_bytes;
        p.lock_off = from;
        p.lock_note = lr.note;
    };
    if (register_bytes == 0) {
        p.refused = "over the pin budget";
        lock_rest(0);
        return p;
    }
    const cudaError_t e = cudaHostRegister(base, (size_t) register_bytes, flags);
    if (e == cudaSuccess) {
        if (device == nullptr) {
            p.registered_bytes = register_bytes;
            return p;
        }
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, base, 0) == cudaSuccess && alias != nullptr) {
            *device = (const uint8_t*) alias;
            p.registered_bytes = register_bytes;
            return p;
        }
        (void) cudaHostUnregister(base);
    }
    p.refused = cudaGetErrorString(e);
    // read here: an error left set would be reported by the next unrelated call that checks
    (void) cudaGetLastError();
    lock_rest(0);
    return p;
}

void unpin_ram(void* base, const RamPin& p) {
    if (p.allocation_backed) return;
    if (p.locked_bytes > 0) platform::unlock_resident((uint8_t*) base + p.lock_off, p.locked_bytes);
    if (p.sliced) {
        for (uint64_t off : p.slice_starts) (void) cudaHostUnregister((uint8_t*) base + off);
    } else if (p.registered_bytes > 0) {
        (void) cudaHostUnregister(base);
    }
}

uint64_t ram_padded(uint64_t bytes) {
    constexpr uint64_t kHuge = 2ull << 20;
    return (bytes + kHuge - 1) / kHuge * kHuge;
}

uint8_t* alloc_ram(uint64_t bytes) {
    return (uint8_t*) platform::alloc_aligned((size_t) ram_padded(bytes), 2ull << 20);
}

RamAllocation::~RamAllocation() { reset(); }

RamAllocation::RamAllocation(RamAllocation&& other) noexcept { *this = std::move(other); }

RamAllocation& RamAllocation::operator=(RamAllocation&& other) noexcept {
    if (this == &other) return *this;
    reset();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    capacity_ = std::exchange(other.capacity_, 0);
    allocation_pinned_ = std::exchange(other.allocation_pinned_, false);
    return *this;
}

bool RamAllocation::allocate(uint64_t bytes, bool pin) {
    reset();
    if (!bytes) return true;
    if (bytes > SIZE_MAX - ((2ull << 20) - 1)) return false;
    if constexpr (platform::kPreferHostVmm) {
        if (pin) data_ = vmm::alloc_host(bytes, capacity_);
        allocation_pinned_ = data_ != nullptr;
    }
    if (!data_) {
        data_ = alloc_ram(bytes);
        if (data_) capacity_ = ram_padded(bytes);
    }
    if (!data_) return false;
    bytes_ = bytes;
    return true;
}

void RamAllocation::reset() {
    if (allocation_pinned_) vmm::free_host(data_, capacity_);
    else platform::free_aligned(data_);
    data_ = nullptr;
    bytes_ = capacity_ = 0;
    allocation_pinned_ = false;
}

RamPin pin_ram(const RamAllocation& allocation, const std::vector<uint64_t>& bounds, bool mapped,
               const uint8_t** device) {
    if (device) *device = nullptr;
    if (!allocation.data()) return {};
    if (allocation.allocation_pinned()) {
        RamPin p;
        p.registered_bytes = allocation.capacity();
        p.allocation_backed = true;
        if (device) *device = allocation.data();
        return p;
    }
    return pin_ram(allocation.data(), allocation.capacity(), allocation.bytes(), bounds, mapped, device);
}

std::vector<RamPin> pin_ram_spans(const std::vector<RamAllocation>& allocations, bool mapped) {
    std::vector<uint8_t*> ordinary;
    std::vector<uint64_t> bytes;
    bool has_ordinary = false;
    for (const auto& a : allocations) {
        ordinary.push_back(a.allocation_pinned() ? nullptr : a.data());
        bytes.push_back(a.bytes());
        has_ordinary = has_ordinary || ordinary.back() != nullptr;
    }
    auto pins = has_ordinary ? pin_ram_spans(ordinary, bytes, mapped) : std::vector<RamPin>(allocations.size());
    for (size_t i = 0; i < allocations.size(); ++i)
        if (allocations[i].allocation_pinned()) pins[i] = pin_ram(allocations[i], {}, mapped);
    return pins;
}

std::vector<RamPin> pin_ram_spans(const std::vector<uint8_t*>& spans, const std::vector<uint64_t>& bytes, bool mapped) {
    uint64_t budget = 0, total = 0;
    std::string why;
    for (uint64_t b : bytes) total += ram_padded(b);
    const bool limited = (platform::kBoundHostRegistration ? host_pin_budget(budget, why) : platform::pin_budget(budget, why))
                         && budget < total;
    std::vector<RamPin> pins(spans.size());
    uint64_t pinned = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i] == nullptr) continue;
        const uint64_t padded = ram_padded(bytes[i]);
        const bool room = !limited || (pinned <= budget && padded <= budget - pinned);
        if constexpr (platform::kBoundHostRegistration) {
            // DXGI usage can lag registration. Also charge each span against the initial allowance.
            const uint64_t remaining = limited ? (budget > pinned ? budget - pinned : 0) : ~uint64_t(0);
            pins[i] = pin_ram_bounded(spans[i], padded, bytes[i], {}, mapped, nullptr, remaining);
        } else {
            pins[i] = pin_ram(spans[i], room ? padded : 0, bytes[i], {}, mapped);
        }
        pinned += pins[i].registered_bytes;
    }
    return pins;
}

}  // namespace basalt::core
