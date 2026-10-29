// tests/platform/region_test.cpp - platform::alloc_region and free_region. CPU only. Large pages are not assumed: the
// region has to come back usable whichever backing it got, and the note has to say which.
#include "basalt/platform/memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <unistd.h>
#endif

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("region %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    const uint64_t page = system_page_size();
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    check(page == info.dwPageSize, "the registration page size matches the OS");
    check(register_host_after_fill(), "registration follows the resident fill");
#else
    const long native_page = sysconf(_SC_PAGESIZE);
    check(native_page > 0 && page == (uint64_t) native_page, "the registration page size matches the OS");
    check(!register_host_after_fill(), "registration precedes the resident fill");
#endif
    check(page > 0 && (page & (page - 1)) == 0, "registration boundaries are powers of two");
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

    uint64_t limit = 123;
    std::string why = "stale reading";
    const bool budgeted = pin_budget(limit, why);
    check(!budgeted && limit == 0 && why.empty(), "an unidentified adapter has no fabricated budget");
    char missing[8];
    std::memset(missing, 0xff, sizeof missing);
    limit = 123;
    why = "stale reading";
    check(!pin_budget(limit, why, missing, 1) && limit == 0 && why.empty(),
          "an unknown adapter clears a stale budget reading");
    return failures ? 1 : 0;
}
