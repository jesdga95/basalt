// include/basalt/core/gpu_mem_info.hpp - the current device's free and total memory, as a plan may spend it.
//
// Header-only, like device_copy.hpp, so every library and test that sizes memory uses it without linking `core`.
#pragma once

#include "basalt/core/device_copy.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace basalt::core {

/// Whether the current device is an integrated GPU (cudaDevAttrIntegrated); false when the runtime cannot say.
inline bool current_device_integrated() {
    if (!any_integrated_device()) return false;
    int d = 0, integrated = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, d) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return integrated != 0;
}

/// The host's MemAvailable in bytes (/proc/meminfo); 0 when it cannot be read.
inline size_t host_mem_available() {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    std::fclose(f);
    return (size_t) kib << 10;
#else
    return 0;
#endif
}

/// cudaMemGetInfo, with an integrated GPU's free memory as the host's MemAvailable (at most the total).
///
/// An integrated GPU's memory is the machine's. Its driver (a GB10) reports MemFree as free, so the page cache counts as
/// used: right after a large download or file read it reported 8 GiB free of 122 GiB while 120 GiB were available, and
/// every plan refused. Its allocations take page cache back on demand (40 GiB allocated and written with 8 GiB "free"
/// succeeded, the cache shrinking under it), so MemAvailable is what a plan can spend. A discrete card is reported as the
/// runtime reports it.
inline cudaError_t gpu_mem_info(size_t* free_bytes, size_t* total_bytes) {
    const cudaError_t e = cudaMemGetInfo(free_bytes, total_bytes);
    if (e != cudaSuccess || !current_device_integrated()) return e;
    *free_bytes = std::min(*total_bytes, std::max(*free_bytes, host_mem_available()));
    return e;
}

}  // namespace basalt::core
