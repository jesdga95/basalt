// tests/platform/region_test.cpp - platform::alloc_region and free_region. CPU only. Large pages are not assumed: the
// region has to come back usable whichever backing it got, and the note has to say which.
#include "basalt/platform/memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("region %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    const uint64_t bytes = 6ull << 20;   // not a multiple of the 2 MiB page the OS may back it with
    RegionBacking backing = RegionBacking::NormalPages;
    std::string note;
    void* base = nullptr;
    uint64_t base_bytes = 0;
    uint8_t* p = (uint8_t*) alloc_region(bytes, backing, note, "", 0, base, base_bytes);
    check(p != nullptr, "an anonymous region is allocated");
    check(!note.empty(), "the note says what backs it");
    std::printf("region note: %s\n", note.c_str());
    if (p != nullptr) {
        check(base != nullptr && base_bytes >= bytes, "the mapping to release covers the region");
        bool zero = true;
        for (uint64_t i = 0; zero && i < bytes; i += 4096) zero = p[i] == 0;
        check(zero, "the region starts zero-filled");
        std::memset(p, 0xa5, bytes);
        check(p[0] == 0xa5 && p[bytes - 1] == 0xa5, "every byte of it is writable");
        free_region(base, base_bytes);
    }
    free_region(nullptr, 0);

    // Non-2 MiB-multiple sizes must also allocate and release cleanly (Linux hugetlb requires 2 MiB-aligned munmap).
    {
        const uint64_t odd_bytes = (5ull << 20) + 1234;
        std::string odd_note;
        void* ob = nullptr;
        uint64_t on = 0;
        uint8_t* op = (uint8_t*) alloc_region(odd_bytes, backing, odd_note, "", 0, ob, on);
        check(op != nullptr && ob != nullptr && on >= odd_bytes, "non-2 MiB-multiple region allocates and tracks mapping bytes");
        if (op != nullptr) {
            op[0] = 0x5a;
            op[odd_bytes - 1] = 0xa5;
            check(op[0] == 0x5a && op[odd_bytes - 1] == 0xa5, "non-2 MiB-multiple region is writable end to end");
            free_region(ob, on);
        }
        std::string znote;
        void* zb = (void*) 1;
        uint64_t zn = 1;
        check(alloc_region(0, backing, znote, "", 0, zb, zn) == nullptr && zb == nullptr && zn == 0, "empty region is refused");
    }

    // a shared arena file is refused with a note where it cannot be made (no model hash here), and nothing is mapped
    std::string note2;
    void* b2 = (void*) 1;
    uint64_t n2 = 7;
    void* r = alloc_region(bytes, backing, note2, "region_test_no_such_dir/arena", 0, b2, n2);
    check(r == nullptr && b2 == nullptr && n2 == 0 && !note2.empty(), "a shared arena without a hash is refused");

    uint64_t limit = 0;
    std::string why;
    const bool budgeted = pin_budget(limit, why);
    check(!budgeted || !why.empty(), "a pin budget names where it comes from");
    return failures ? 1 : 0;
}
