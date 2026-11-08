#pragma once

#include "basalt/platform/host_allocator.hpp"

#include <cstdint>
#include <istream>
#include <optional>

namespace basalt::core {

// Free bytes in the resident data pool, or host physical memory when no pool is open.
// Unknown physical telemetry is distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    // Pool space is already resident: consuming it does not need a physical-RAM margin again.
    if (basalt::platform::reserved_host_available() != UINT64_MAX) floor = 0;
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace basalt::core
