// src/core/pinned.cu - the pinned host arena.
#include "basalt/core/pinned.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include "basalt/log.hpp"

namespace basalt::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         const std::string& shared_file, uint64_t shared_hash) : capacity(bytes) {
    if (bytes == 0) return;
    basalt::platform::RegionBacking region_backing = basalt::platform::RegionBacking::NormalPages;
    base = basalt::platform::alloc_region(bytes, region_backing, note, shared_file, shared_hash, mapping_base,
                                          mapping_bytes);
    backing = region_backing == basalt::platform::RegionBacking::LargePages ? PageBacking::LargePages
                                                                             : PageBacking::NormalPages;

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        const RamPin p = pin_ram(base, bytes, bytes, bounds, /*mapped=*/true);
        registered_bytes = p.registered_bytes;
        locked_bytes = p.locked_bytes;
        if (p.refused.empty()) {
            note = "cudaHostRegister PORTABLE ok; " + note;
        } else if (p.sliced) {
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            slice_starts = p.slice_starts;
            registered_slices = p.slices;
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) p.cap / (double) (1ull << 30));
            note = "cudaHostRegister of the whole arena FAILED (" + p.refused + "); " +
                   (p.capped ? "slices capped at " + std::string(gib) + " GiB (" + p.cap_why + "); " : std::string()) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) note = p.lock_note + "; " + note;
        } else {
            note = "cudaHostRegister FAILED (" + p.refused + ") - the arena is NOT pinned, so copies will be slow; " + note;
            note = p.lock_note + "; " + note;
        }
    }
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) basalt::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else if (!platform::kBoundHostRegistration || registered_bytes > 0) {
            cudaHostUnregister(base);
        }
        basalt::platform::free_region(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        basalt::log::error("basalt", "stream_bandwidth: cudaMalloc failed for %llu B", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace basalt::core
