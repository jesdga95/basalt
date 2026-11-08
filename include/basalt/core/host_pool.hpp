#pragma once

#include "basalt/core/host_arena.hpp"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <string>

namespace basalt::core {

// One process-wide pool, opened before loading the model's host buffers.
bool open_host_pool(uint64_t bytes, uint64_t permitted, std::string& error);
bool host_pool_enabled();
HostArena::Stats host_pool_stats();
uint64_t host_pool_available();
/// Returns unused physical pages while retaining the pool's fixed address capacity.
bool discard_host_pool_free(size_t keep_free_bytes = 0);
cudaError_t allocate_pinned_host(void** p, size_t bytes, unsigned int flags);
cudaError_t allocate_pinned_host(void** p, size_t bytes);
cudaError_t release_pinned_host(void* p);
cudaError_t register_host_memory(void* p, size_t bytes, unsigned int flags);
cudaError_t unregister_host_memory(void* p);
template<class T> cudaError_t allocate_pinned_host(T** p, size_t bytes, unsigned int flags) {
    return allocate_pinned_host(reinterpret_cast<void**>(p), bytes, flags);
}
template<class T> cudaError_t allocate_pinned_host(T** p, size_t bytes) {
    return allocate_pinned_host(reinterpret_cast<void**>(p), bytes);
}
void* allocate_plain_host(size_t bytes);
void release_plain_host(void* p);

}  // namespace basalt::core
