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
#pragma once

#include "basalt/core/device_copy.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace basalt::core {

/// cudaMemGetInfo for the current device as the sizing decisions should read it (see above): exactly cudaMemGetInfo
/// on a discrete card.
cudaError_t device_mem_info(size_t* free_b, size_t* total_b);

/// What device_mem_info reads from, for the start line and the tests.
struct DeviceMemoryModel {
    bool integrated = false;      ///< cudaDevAttrIntegrated
    uint64_t dedicated = 0;       ///< the carve-out in bytes; 0: none known (a discrete card, or no management library)
    bool os_usage = false;        ///< this process's usage comes from the OS (DXGI), not from cudaMemGetInfo
};
DeviceMemoryModel device_memory_model();

}  // namespace basalt::core
