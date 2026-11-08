// include/basalt/core/host_plan.hpp - RAM held from before a RAM tier is sized for page-locked buffers made later.
//
// A model holds the RAM its later page-locked buffers take at their peak before its RAM tier or copy reads the free
// memory, so the tier sizes beside them.  A part allocates its buffer with host_alloc where it would call
// cudaHostAlloc: the held RAM goes back to the OS first, so the buffer replaces it and the free memory never drops
// below what the tier left.  The buffer itself is allocated when the part makes it, so every card sees the same
// page-locked memory at each step as without the hold.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace basalt::core {

/// Holds `bytes` of RAM, its pages written, for a later host_alloc.  False when the RAM is not there.
bool host_plan(size_t bytes);
/// cudaHostAlloc, after giving back the smallest hold of at least `bytes` (else the largest one).
cudaError_t host_alloc(void** p, size_t bytes, unsigned flags);
/// The RAM still held.
uint64_t host_plan_bytes();
/// Gives back every hold; returns their bytes.
uint64_t host_plan_release();

}  // namespace basalt::core
