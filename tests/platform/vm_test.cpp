// Virtual reservations and disjoint committed prefixes, with a decommit/recommit rollback.
#include "basalt/platform/memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    namespace p = basalt::platform;
    const uint64_t page = p::vm_page_size();
    if (page == 0 || p::vm_reserve(0) != nullptr || p::vm_reserve(page + 1) != nullptr) return 1;
    auto* const base = (uint8_t*) p::vm_reserve(16 * page);
    if (base == nullptr || (uintptr_t) base % page != 0) return 1;
    int bad = 0;
    if (!p::vm_commit(base, 2 * page) || !p::vm_commit(base + 8 * page, page)) {
        p::vm_release(base, 16 * page);
        return 1;
    }
    std::memset(base, 0x35, (size_t) (2 * page));
    std::memset(base + 8 * page, 0x72, (size_t) page);
    bad += p::vm_commit(base + 1, page) || p::vm_commit(base, page - 1) || p::vm_commit(nullptr, page);
    if (!p::vm_commit(base + 2 * page, page)) ++bad;
    else {
        std::memset(base + 2 * page, 0x49, (size_t) page);
        if (!p::vm_decommit(base + 2 * page, page) || !p::vm_commit(base + 2 * page, page)) ++bad;
        else for (uint64_t i = 0; i < page; ++i) if (base[2 * page + i] != 0) { ++bad; break; }
    }
    for (uint64_t i = 0; i < 2 * page; ++i) if (base[i] != 0x35) { ++bad; break; }
    for (uint64_t i = 0; i < page; ++i) if (base[8 * page + i] != 0x72) { ++bad; break; }
    bad += p::vm_decommit(base + 1, page) || p::vm_decommit(base, 0);
    p::vm_release(base, 16 * page);
    p::vm_release(nullptr, 0);
    std::printf("platform_vm_test: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
