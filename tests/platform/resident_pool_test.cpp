#include "basalt/platform/memory.hpp"
#include <cstdio>
#include <cstring>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#endif

int main() {
    using namespace basalt::platform;
    constexpr uint64_t bytes = 4u << 20, half = bytes / 2;
    auto* p = static_cast<unsigned char*>(alloc_resident_pool(bytes));
    if (!p) return 1;
    const auto initial = commit_resident_pool(p, bytes);
    if (!initial.ok || initial.locked_bytes != bytes) {
        free_region(p, bytes);
        std::printf("resident_pool_test: cannot lock test backing: %s\n", initial.note.c_str());
        return 1;
    }
    std::memset(p, 0x5a, bytes);
    if (!discard_resident_pool(p + half, half)) return 2;
#if !defined(_WIN32)
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> resident(bytes / page);
    if (mincore(p, bytes, resident.data()) != 0) return 3;
    size_t kept = 0, returned = 0;
    for (size_t i = 0; i < resident.size(); ++i) {
        if (i < half / page) kept += resident[i] & 1;
        else returned += resident[i] & 1;
    }
    if (kept != half / page || returned != 0) return 4;
    std::printf("resident_pool_test: mincore kept %zu live pages, inactive pages resident %zu\n", kept, returned);
#endif
    for (size_t i = 0; i < half; ++i) if (p[i] != 0x5a) return 5;
    const auto reused = commit_resident_pool(p + half, half);
    if (!reused.ok || reused.locked_bytes != half) return 6;
    for (size_t i = half; i < bytes; ++i) if (p[i] != 0) return 7;
    std::memset(p + half, 0xa5, half);
    for (size_t i = 0; i < bytes; ++i) if (p[i] != (i < half ? 0x5a : 0xa5)) return 8;
    if (!discard_resident_pool(p, bytes)) return 9;
    free_region(p, bytes);
    std::printf("resident_pool_test: physical release, live-range preservation and recommit passed\n");
}
