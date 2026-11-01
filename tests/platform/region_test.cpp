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

static void test_budgeted_regions() {
    using namespace basalt::platform;
    RegionBacking backing = RegionBacking::NormalPages;
    std::string note;
    void* base = (void*) 1;
    uint64_t mapped = 1;
    auto refused = [&](uint64_t bytes, uint64_t reserve, const char* why) {
        base = (void*) 1;
        mapped = 1;
        note.clear();
        void* p = alloc_region_budgeted(bytes, reserve, backing, note, base, mapped);
        check(p == nullptr && base == nullptr && mapped == 0 && !note.empty(), why);
        if (p != nullptr) free_region(base, mapped);
    };
    refused(0, 0, "zero-sized budgeted region is refused");
    refused(UINT64_MAX, 0, "allocator alignment overflow is refused");
    refused(4096, UINT64_MAX - 4095, "region plus reserve overflow is refused");
    refused(1, UINT64_MAX - 1, "rounded size cannot overflow its reserve");
    // An impossible ordinary reserve must fail even on a machine with a configured hugetlb pool, before it
    // borrows any of those pages. No assumption about configured large pages or current free RAM is needed.
    refused(4096, UINT64_MAX / 2, "insufficient auxiliary reserve is refused");
    refused(1ull << 62, 0, "impossible output budget is refused without ordinary allocation");

    // Non-page-sized regions exercise rounding and the ownership span used for both backing types.
    constexpr uint64_t bytes = (2ull << 20) + 19;
    for (int cycle = 0; cycle < 2; ++cycle) {
        uint8_t* p = (uint8_t*) alloc_region_budgeted(bytes, 4096, backing, note, base, mapped);
        check(p != nullptr, "small budgeted region is allocated");
        if (p == nullptr) continue;
        check(base == p && mapped >= bytes, "budgeted ownership covers the rounded region");
        bool zero = true;
        for (uint64_t i = 0; i < bytes; ++i) zero &= p[i] == 0;
        check(zero, "budgeted region is zero-filled, including after a release");
        std::memset(p, 0xa5, bytes);
        check(p[0] == 0xa5 && p[bytes - 1] == 0xa5, "budgeted region is writable through its final byte");
        free_region(base, mapped);
    }
}

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

    // a shared arena file is refused with a note where it cannot be made (no model hash here), and nothing is mapped
    std::string note2;
    void* b2 = (void*) 1;
    uint64_t n2 = 7;
    void* r = alloc_region(bytes, backing, note2, "region_test_no_such_dir/arena", 0, b2, n2);
    check(r == nullptr && b2 == nullptr && n2 == 0 && !note2.empty(), "a shared arena without a hash is refused");

    test_budgeted_regions();
    uint64_t limit = 0;
    std::string why;
    const bool budgeted = pin_budget(limit, why);
    check(!budgeted || !why.empty(), "a pin budget names where it comes from");
    return failures ? 1 : 0;
}
