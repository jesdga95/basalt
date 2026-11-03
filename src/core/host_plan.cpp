// src/core/host_plan.cpp - see include/basalt/core/host_plan.hpp.
#include "basalt/core/host_plan.hpp"

#include "basalt/platform/memory.hpp"

#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace basalt::core {

namespace {
std::mutex g_mu;
std::vector<std::pair<void*, size_t>> g_held;
}  // namespace

bool host_plan(size_t bytes) {
    if (bytes == 0) return true;
    void* p = platform::alloc_aligned(bytes, 4096);
    if (p == nullptr) return false;
    std::memset(p, 0, bytes);
    const std::lock_guard<std::mutex> lk(g_mu);
    g_held.emplace_back(p, bytes);
    return true;
}

cudaError_t host_alloc(void** p, size_t bytes, unsigned flags) {
    {
        const std::lock_guard<std::mutex> lk(g_mu);
        size_t pick = g_held.size();
        for (size_t i = 0; i < g_held.size(); ++i) {
            const size_t b = g_held[i].second;
            if (pick == g_held.size()) { pick = i; continue; }
            const size_t c = g_held[pick].second;
            if ((b >= bytes && (c < bytes || b < c)) || (c < bytes && b > c)) pick = i;
        }
        if (pick < g_held.size()) {
            platform::free_aligned(g_held[pick].first);
            g_held.erase(g_held.begin() + (std::ptrdiff_t) pick);
        }
    }
    return cudaHostAlloc(p, bytes, flags);
}

uint64_t host_plan_bytes() {
    const std::lock_guard<std::mutex> lk(g_mu);
    uint64_t n = 0;
    for (const auto& h : g_held) n += h.second;
    return n;
}

uint64_t host_plan_release() {
    const std::lock_guard<std::mutex> lk(g_mu);
    uint64_t n = 0;
    for (const auto& h : g_held) {
        platform::free_aligned(h.first);
        n += h.second;
    }
    g_held.clear();
    return n;
}

}  // namespace basalt::core
