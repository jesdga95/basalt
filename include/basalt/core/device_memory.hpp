// include/basalt/core/device_memory.hpp - how much memory the current device has left, as the sizing decisions read it.
//
// On a discrete card device_mem_info is cudaMemGetInfo, and nothing else is asked.
//
// An integrated GPU (the N1X, GB10) has no memory of its own. Measured on the RTX Spark N1X under Windows:
//
//   * A carve-out of the machine's RAM is dedicated to it (96 GiB of 128): the OS's dedicated segment, and what the
//     management library reports as the device's total. Past it comes a shared segment out of the RAM the OS uses
//     (half of it, 15 GiB). cudaMemGetInfo's total is both (110.5 GiB), so a cache sized from it would run 14 GiB
//     into the RAM the host needs.
//   * cudaMemGetInfo's free follows cudaMalloc and page-locked host memory, but not memory mapped through the virtual
//     memory API (core/vmm.cpp), which is how the expert cache and the KV arena are made.
//   * The management library's "used" (and the dedicated segment's committed bytes) count a device page only once it
//     has been written, for every process.
//   * The OS's usage figure for this process (DXGI, the local segment group) follows all three at once - cudaMalloc,
//     mapped virtual memory and page-locked host memory - as soon as they are made.
//
// So on an integrated GPU with a carve-out:
//
//   total = the carve-out
//   used  = the larger of  what every process has written (the management library), and
//                          what the others had written when this process first asked + what this process holds now
//   free  = total - used
//
// "What this process holds" is the OS's figure where there is one (Windows), else cudaMemGetInfo's. On Windows it
// includes this process's page-locked host memory, which lives in the RAM and not in the carve-out: free reads low by
// that much, the safe side. Without a carve-out the management library reports (a GPU sharing all of its memory),
// the reading stays cudaMemGetInfo's.
//
// Windows also charges an integrated GPU's memory to the commit limit (RAM plus page file): cudaMalloc and cuMemCreate
// each raise the commit charge by their full size when they are made, before a byte is written. The engine's first
// load on the N1X took the charge to 97.3 of 98.5 GiB with the carve-out a quarter free. So there, free is also at
// most the commit headroom less what a RAM tier leaves of the RAM (plan_reserve: the RAM guard's reserve,
// --ram-reserve-mib, plus what the engine takes after its plan), and the first reading where that binds says so in
// the log.
#pragma once

#include "basalt/core/device_copy.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace basalt::core {

/// cudaMemGetInfo for the current device as the sizing decisions should read it (see above): exactly cudaMemGetInfo
/// on a discrete card. Free is the smaller of the two bounds DeviceMemoryReading holds.
cudaError_t device_mem_info(size_t* free_b, size_t* total_b);

/// The two bounds behind device_mem_info's free. `device_free` and `total` are the device's own memory (the carve-out
/// reading on an integrated GPU, cudaMemGetInfo's elsewhere); `commit_room` is the commit headroom less the plan
/// reserve where the OS charges the device's memory to its commit limit, and UINT64_MAX where it does not.
/// The commit room lags a release: the driver keeps freed device memory charged for a while, small blocks longer
/// (measured on the N1X: one freed 64 MiB segment gave back none of its charge at once, four 512 MiB ones most of
/// theirs). So whether memory was given back is read from `device_free`.
struct DeviceMemoryReading {
    uint64_t device_free = 0, total = 0;
    uint64_t commit_room = UINT64_MAX;
};
cudaError_t device_mem_reading(DeviceMemoryReading& r);

/// What device_mem_info reads from, for the start line and the tests.
struct DeviceMemoryModel {
    bool integrated = false;      ///< cudaDevAttrIntegrated
    uint64_t dedicated = 0;       ///< the carve-out in bytes; 0: none known (a discrete card, or no management library)
    bool os_usage = false;        ///< this process's usage comes from the OS (DXGI), not from cudaMemGetInfo
    bool commit_bound = false;    ///< the OS charges the device's memory to its commit limit (an integrated GPU, Windows)
};
DeviceMemoryModel device_memory_model();

}  // namespace basalt::core
