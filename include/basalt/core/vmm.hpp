// include/basalt/core/vmm.hpp - the driver's virtual memory management, looked up through the runtime.
//
// One reserved address range, backed piece by piece by physical allocations that can be unmapped and mapped again
// while every address in the range stays the same.  The expert cache's segments and the elastic KV pools both use
// it: captured graphs and kernel arguments hold addresses into these ranges, so only the backing moves.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::core::vmm {

/// Whether the current device supports virtual memory management (and the driver entry points were found).
bool supported();
/// The allocation granularity of the current device (every size and offset below is a multiple of it); 0 on failure.
uint64_t granularity();
/// The minimum host allocation granularity on the current device, or 0 without host VMM.
uint64_t host_granularity();
/// CUDA-pinned system memory accessible to the CPU and current GPU; other GPUs may copy through host endpoints.
/// Returns the padded mapping size for free_host; no portable registration is involved.
uint8_t* alloc_host(uint64_t bytes, uint64_t& mapped);
void free_host(uint8_t* ptr, uint64_t mapped);
/// Reserves `bytes` of address space (a multiple of the granularity); null on failure.
uint8_t* reserve(uint64_t bytes);
void release_range(uint8_t* va, uint64_t bytes);
/// Backs `[va, va + bytes)` with a new physical allocation on the current device, readable and writable by it.
/// Returns the handle (0 on failure). `out_of_memory`, when given, is set when the failure was the device lacking the
/// memory and nothing was allocated or mapped; any other failure may leave the range in an unknown state.
unsigned long long map(uint8_t* va, uint64_t bytes, bool* out_of_memory = nullptr);
/// Unmaps what `map` mapped there and frees its physical memory.  The caller has made sure no work reads it.
bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// A physical allocation of `bytes` on the current device, mapped nowhere yet (0 on failure): one backing that
/// `map_handle` can put under several ranges at once, so they share its bytes.  `release` drops the caller's reference;
/// the memory is freed once every mapping of it is gone too.
unsigned long long create(uint64_t bytes);
/// Maps the first `bytes` of `handle` at `va`, readable and writable by the current device.
bool map_handle(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// Unmaps `[va, va + bytes)` without releasing any handle (the mapping's reference goes with it).
void unmap_only(uint8_t* va, uint64_t bytes);
void release(unsigned long long handle);

}  // namespace basalt::core::vmm
