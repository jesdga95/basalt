// include/basalt/core/device_memory.hpp - how much memory the current device has left, as the sizing decisions read it.
//
// An integrated GPU's memory is the machine's RAM: a dedicated carve-out the OS does not see, and past it a shared
// segment out of the RAM the OS uses.  cudaMemGetInfo reports both, and can leave out what the virtual memory API maps,
// so a cache sized from it would spill into the shared segment (smaller pages) and take RAM the host needs.  Every
// sizing decision reads device_mem_info instead: the carve-out, as the management library reports it, like a discrete
// card's memory; without a carve-out, the RAM.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace basalt::core {

/// Whether the current device is an integrated GPU (cudaDevAttrIntegrated).
bool device_integrated();

/// Whether the current device's allocations come out of the RAM the OS uses: an integrated GPU with no dedicated
/// carve-out the management library reports.
bool device_memory_is_host_ram();

/// cudaMemGetInfo for the current device, and exactly that on a discrete card.  On an integrated GPU: the dedicated
/// carve-out's figures where the management library reports them; else `total` at most the machine's RAM and `free`
/// at most the RAM the OS can hand out without paging, less platform::ram_reserve().
cudaError_t device_mem_info(size_t* free_b, size_t* total_b);

}  // namespace basalt::core
