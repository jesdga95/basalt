#include "basalt/core/host_pool.hpp"
#include "basalt/platform/host_allocator.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <thread>
#include <vector>

namespace basalt::core {
namespace {
// The resident backing outlives CUDA and pooled statics. Only leased ranges have CUDA mappings.
HostArena& arena() { static HostArena* value = new HostArena; return *value; }
std::atomic<bool> enabled{false};
std::mutex pool_mu;
uint8_t* pool_base = nullptr;
std::map<size_t, size_t> discarded_ranges;

void mark_discarded(size_t s, size_t e) {
    if (s >= e) return;
    auto it = discarded_ranges.upper_bound(s);
    if (it != discarded_ranges.begin()) {
        auto prev = std::prev(it);
        if (prev->second >= s) {
            s = std::min(s, prev->first);
            e = std::max(e, prev->second);
            it = discarded_ranges.erase(prev);
        }
    }
    while (it != discarded_ranges.end() && it->first <= e) {
        e = std::max(e, it->second);
        it = discarded_ranges.erase(it);
    }
    discarded_ranges.emplace(s, e);
}

bool ensure_committed(size_t s, size_t e) {
    if (s >= e || discarded_ranges.empty()) return true;
    auto it = discarded_ranges.upper_bound(s);
    if (it != discarded_ranges.begin()) {
        auto prev = std::prev(it);
        if (prev->second > s) it = prev;
    }
    while (it != discarded_ranges.end() && it->first < e) {
        const size_t ds = it->first, de = it->second;
        const size_t cs = std::max(s, ds), ce = std::min(e, de);
        const auto locked = basalt::platform::commit_resident_pool(pool_base + cs, ce - cs);
        if (locked.locked_bytes != ce - cs) {
            (void) basalt::platform::discard_resident_pool(pool_base + cs, ce - cs);
            return false;
        }
        it = discarded_ranges.erase(it);
        if (ds < cs) discarded_ranges.emplace(ds, cs);
        if (ce < de) {
            discarded_ranges.emplace(ce, de);
            break;
        }
    }
    return true;
}

void* allocate(size_t bytes, size_t alignment) {
    if (bytes > std::numeric_limits<size_t>::max() - 4095) return nullptr;
    bytes = (bytes + 4095) & ~size_t{4095};
    std::lock_guard<std::mutex> lk(pool_mu);
    void* p = arena().allocate(bytes, std::max<size_t>(4096, alignment));
    if (!p) return nullptr;
    const size_t off = static_cast<size_t>(static_cast<uint8_t*>(p) - pool_base);
    const size_t end = off + bytes;
    if (!ensure_committed(off, end)) {
        arena().release(p); return nullptr;
    }
    if (cudaHostRegister(p, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        arena().release(p); return nullptr;
    }
    return p;
}
bool release(void* p) {
    std::lock_guard<std::mutex> lk(pool_mu);
    if (!arena().owns(p)) return false;
    const size_t bytes = arena().allocation_bytes(p);
    if (!bytes) std::abort();
    if (cudaHostUnregister(p) != cudaSuccess) std::abort();
    if (!arena().release(p)) std::abort();
    return true;
}
const basalt::platform::HostAllocatorHooks hooks{allocate, release, host_pool_available};
}

bool open_host_pool(uint64_t bytes, uint64_t permitted, std::string& error) {
    if (enabled || bytes == 0 || bytes > std::numeric_limits<size_t>::max()) {
        error = "invalid host pool size or a pool is already open"; return false;
    }
    // MemAvailable estimates reclaimability; it is not a successful physical reservation.
    const uint64_t tables = bytes / 256;
    if (permitted == 0 || bytes > permitted || tables > permitted - bytes) {
        error = "host pool needs " + std::to_string((bytes + tables) >> 20) + " MiB of allocation budget; " +
                std::to_string(permitted >> 20) + " MiB permitted";
        return false;
    }
    basalt::log::info("host pool", "reserving %.2f GiB of resident backing before model loading",
                      double(bytes) / 1073741824.0);
    void* p = basalt::platform::alloc_resident_pool(bytes);
    void* mapping = p; const uint64_t mapping_bytes = bytes;
    if (!p) { error = "host pool backing allocation failed"; return false; }
    const auto locked = basalt::platform::lock_resident(p, bytes);
    if (locked.locked_bytes != bytes) {
        error = "host pool needs a complete resident lock: " + locked.note;
        basalt::platform::free_region(mapping, mapping_bytes); return false;
    }
    basalt::log::info("host pool", "resident lock %llu MiB", (unsigned long long) (locked.locked_bytes >> 20));
    // The backing stays resident even when an external controller changes the swap policy.
    const size_t workers = std::min<size_t>(8, std::max<size_t>(1, bytes >> 30));
    auto touch = [&](size_t worker) {
        const size_t first = (bytes / workers / 4096) * 4096 * worker;
        const size_t last = worker + 1 == workers ? bytes : (bytes / workers / 4096) * 4096 * (worker + 1);
        std::memset(static_cast<uint8_t*>(p) + first, 0, last - first);
    };
    std::vector<std::thread> threads;
    try { for (size_t worker = 1; worker < workers; ++worker) threads.emplace_back(touch, worker); }
    catch (...) {
        for (auto& thread : threads) thread.join();
        basalt::platform::free_region(mapping, mapping_bytes);
        error = "host pool prefault workers could not start"; return false;
    }
    touch(0);
    for (auto& thread : threads) thread.join();
    try { arena().reset(p, static_cast<size_t>(bytes)); }
    catch (...) {
        basalt::platform::free_region(mapping, mapping_bytes);
        error = "host pool bookkeeping allocation failed"; return false;
    }
    {
        std::lock_guard<std::mutex> lk(pool_mu);
        pool_base = static_cast<uint8_t*>(p);
        discarded_ranges.clear();
    }
    enabled = true;
    basalt::platform::host_allocator_hooks.store(&hooks, std::memory_order_release);
    basalt::log::info("host pool", "%.2f GiB prefaulted and resident; fixed capacity; unused physical pages are returned",
                      double(bytes) / 1073741824.0);
    return true;
}
bool host_pool_enabled() { return enabled; }
HostArena::Stats host_pool_stats() { return arena().stats(); }
uint64_t host_pool_available() { const auto s = arena().stats(); return s.capacity - s.used; }
bool discard_host_pool_free(size_t keep_free_bytes) {
    if (!enabled) return true;
    std::lock_guard<std::mutex> lk(pool_mu);
    const size_t keep = (keep_free_bytes + 4095) & ~size_t{4095};
    size_t kept_committed = 0;
    std::vector<std::pair<size_t, size_t>> to_discard;
    arena().visit_free([&](void* p, size_t bytes) {
        const size_t raw_s = static_cast<size_t>(static_cast<uint8_t*>(p) - pool_base);
        const size_t raw_e = raw_s + bytes;
        const size_t fs = (raw_s + 4095) & ~size_t{4095};
        const size_t fe = raw_e & ~size_t{4095};
        if (fs >= fe) return;
        size_t cur = fs;
        auto it = discarded_ranges.upper_bound(fs);
        if (it != discarded_ranges.begin()) {
            auto prev = std::prev(it);
            if (prev->second > fs) it = prev;
        }
        auto consume_committed = [&](size_t cs, size_t ce) {
            if (cs >= ce) return;
            if (kept_committed < keep) {
                const size_t take = std::min(ce - cs, keep - kept_committed);
                kept_committed += take;
                cs += take;
            }
            if (cs < ce) to_discard.emplace_back(cs, ce);
        };
        while (it != discarded_ranges.end() && it->first < fe) {
            if (cur < it->first) consume_committed(cur, std::min(fe, it->first));
            cur = std::max(cur, it->second);
            if (cur >= fe) break;
            ++it;
        }
        if (cur < fe) consume_committed(cur, fe);
    });
    bool ok = true;
    for (const auto& [ds, de] : to_discard) {
        if (basalt::platform::discard_resident_pool(pool_base + ds, de - ds)) {
            mark_discarded(ds, de);
        } else {
            ok = false;
        }
    }
    return ok;
}

cudaError_t allocate_pinned_host(void** p, size_t bytes, unsigned int flags) {
    if (!enabled) return cudaHostAlloc(p, bytes, flags);
    if (!p) return cudaErrorInvalidValue;
    if (!bytes) { *p = nullptr; return cudaSuccess; }
    try { *p = allocate(bytes, 4096); }
    catch (const std::bad_alloc&) { *p = nullptr; }
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
cudaError_t allocate_pinned_host(void** p, size_t bytes) { return allocate_pinned_host(p, bytes, 0); }
cudaError_t release_pinned_host(void* p) {
    if (!p) return cudaSuccess;
    if (enabled && arena().owns(p)) return release(p) ? cudaSuccess : cudaErrorInvalidValue;
    return cudaFreeHost(p);
}
cudaError_t register_host_memory(void* p, size_t bytes, unsigned int flags) {
    if (enabled && arena().owns(p)) {
        if (!arena().owns_range(p, bytes)) return cudaErrorInvalidValue;
        return cudaSuccess;
    }
    return cudaHostRegister(p, bytes, flags);
}
cudaError_t unregister_host_memory(void* p) {
    return enabled && arena().owns(p) ? cudaSuccess : cudaHostUnregister(p);
}
void* allocate_plain_host(size_t bytes) {
    if (!enabled) return std::malloc(bytes);
    try { return allocate(bytes, alignof(std::max_align_t)); }
    catch (const std::bad_alloc&) { return nullptr; }
}
void release_plain_host(void* p) { if (!p || (enabled && release(p))) return; std::free(p); }

}  // namespace basalt::core
