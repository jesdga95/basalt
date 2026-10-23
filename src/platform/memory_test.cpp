// src/platform/memory_test.cpp - lock_resident on 256 MiB (or what the locked-memory limit allows), the RAM floor, the
// file cache decisions and the read-ahead hint on a mapped file view (CPU only, no GPU, no model; one temp file).
#include "basalt/platform/memory.hpp"

#include "basalt/platform/file.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif


namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }

// The file tier's unbuffered choice (true = the file cache keeps its reads, so they stay buffered)
int file_cache_keeps_cases() {
    using basalt::platform::file_cache_keeps;
    struct Case { const char* what; double avail, arena, read; bool keeps; };
    const double experts = 71.7;   // routed experts of a large model
    const Case cases[] = {
        // passing the budget asked for and every shard's bytes (103.7 GiB) goes unbuffered although the experts
        // outside the copy would fit the cache
        {"96 GB, budget 72, all shards' bytes", 83.0, 72.0, 103.7, false},
        {"96 GB, budget 72, before the copy (arena = all experts)", 83.0, experts, 0.0, true},
        {"96 GB, budget 72, copy built (48.94 resident)", 83.0 - 48.94, 0.0, experts - 48.94, true},
        {"96 GB, budget 40, before the copy", 83.0, 40.0, experts - 40.0, true},
        {"96 GB, budget 40, copy built", 83.0 - 40.0, 0.0, experts - 40.0, true},
        // low-RAM machines stay unbuffered
        {"64 GB, budget 40, before the copy", 55.0, 40.0, experts - 40.0, false},
        {"64 GB, budget 40, copy built", 15.0, 0.0, experts - 40.0, false},
        {"32 GB, budget 16, before the copy", 26.0, 16.0, experts - 16.0, false},
        {"32 GB, budget 16, copy built", 10.0, 0.0, experts - 16.0, false},
        {"32 GB, Q2_0 experts (34 GiB), budget 8", 26.0, 8.0, 34.0 - 8.0, false},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 0.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read));
        std::printf("file_cache_keeps %-58s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    return fail;
}
// The RAM floor, the largest safe copy and the serve guard's decision, on injected MemAvailable readings
int ram_floor_cases() {
    using namespace basalt::platform;
    int fail = 0;
    auto check = [&](const char* what, double got, double want) {
        const bool ok = got > want - 0.01 && got < want + 0.01;
        std::printf("ram floor %-62s %8.2f GiB%s\n", what, got, ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    const double G = 1073741824.0;
    const uint64_t margin = gib(2.0);
    const uint64_t layer_q4 = 512ull * 3170304;   // every expert of a layer of a 4-bit pack (1.51 GiB)
    const uint64_t layer_q2 = 512ull * 1382400;   // the same of the 2-bit pack (0.66 GiB)
    check("nothing read from the file: the margin alone", ram_floor(margin, 0, layer_q4) / G, 2.0);
    check("4-bit pack, experts read from the file: margin + 3.5 + 2 layers", ram_floor(margin, gib(15), layer_q4) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q4 / G);
    check("2-bit pack: margin + 3.5 + 2 layers", ram_floor(margin, gib(1), layer_q2) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q2 / G);
    check("small layers: the 4 GiB file cache minimum applies", file_cache_floor(1, gib(0.1)) / G, 4.0);
    check("one byte from the file is enough to need the cache", file_cache_floor(1, layer_q4) / G,
          3.5 + 2.0 * (double) layer_q4 / G);
    // a 30 GiB machine with 27 GiB available and 2 GiB taken by the server later
    const uint64_t fl = ram_floor(margin, gib(15), layer_q4);
    check("largest safe copy, 27 available, 2 late", max_safe_copy(gib(27), gib(2), fl) / G, 27.0 - 2.0 - fl / G);
    check("largest safe copy when the floor exceeds the RAM", max_safe_copy(gib(8), gib(2), fl) / G, 0.0);
    const uint64_t guard = ram_guard_floor(margin, gib(15));
    check("serve guard, experts read from the file: margin + 3.5", guard / G, 5.5);
    check("serve guard, nothing read from the file: the margin", ram_guard_floor(margin, 0) / G, 2.0);
    struct Case { const char* what; double avail; bool low; };
    const Case cases[] = {
        {"well above the floor", 20.0, false},
        {"at the load floor (sized as asked)", fl / G, false},
        {"exactly at the guard line", guard / G, false},
        {"just below the guard line", guard / G - 0.01, true},
        {"the freeze's reading (2.5 GB)", 2.33, true},
        {"unknown (0)", 0.0, false},
    };
    for (const Case& c : cases) {
        const bool got = ram_below_floor(gib(c.avail), guard);
        std::printf("ram guard %-62s -> %s%s\n", c.what, got ? "stop" : "run", got == c.low ? "" : "   <-- WRONG");
        fail |= got != c.low;
    }
    return fail;
}
// The read-ahead seam on a real mapped file view. Whether the OS actually prefetched is not observable portably and
// is not what the seam promises, so what is checked is that every range shape returns without faulting and that the
// mapping's bytes stay exactly as written.
int read_ahead_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const bool on = read_ahead_enabled();
    std::printf("read-ahead %-58s -> %s%s\n", "read_ahead_enabled (both platforms)", on ? "on" : "off",
                on ? "" : "   <-- WRONG");
    fail |= !on;

    const uint64_t bytes = 256ull << 10;   // spans more than one 128 KiB hint window
    std::vector<uint8_t> pattern(bytes);
    for (uint64_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 13u + (i >> 8) * 7u);
    const std::string path = (std::filesystem::temp_directory_path() / "basalt_memory_test_view.bin").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) pattern.data(), (std::streamsize) bytes);
        if (!f) {
            std::printf("read-ahead %-58s   <-- cannot write %s\n", "temp file", path.c_str());
            return fail | 1;
        }
    }
    // mapped the way the engine maps its file views: whole file, read-only
    const uint8_t* base = nullptr;
#if defined(_WIN32)
    const std::wstring wpath = basalt::platform::to_wide(path);
    if (!wpath.empty()) {
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
            base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
            // the view holds the section and the file open, so both handles go now
            CloseHandle(m);
            CloseHandle(h);
        }
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (view != MAP_FAILED) base = (const uint8_t*) view;
    }
#endif
    if (base == nullptr) {
        std::printf("read-ahead %-58s   <-- cannot map %s\n", "read-only view", path.c_str());
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return fail | 1;
    }
    auto spots_intact = [&](const char* when) {
        bool ok = true;
        for (const uint64_t off : {(uint64_t) 0, (uint64_t) 1234, bytes / 2, bytes - 1}) ok &= base[off] == pattern[off];
        std::printf("read-ahead %-58s -> %s%s\n", when, ok ? "bytes intact" : "bytes CHANGED", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    spots_intact("view reads back as written, before the hints");
    struct HintCase { const char* what; const void* p; uint64_t n; };
    const HintCase cases[] = {
        {"whole view", base, bytes},
        {"page-unaligned interior range", base + 1234, 8ull << 10},
        {"one-byte range", base + bytes / 2, 1},
        {"range running past the end of the view", base + bytes - 100, 1ull << 20},
        {"null pointer", nullptr, 4096},
        {"zero length", base, 0},
    };
    for (const HintCase& c : cases) {
        // the hint is void and silent: reaching the print at all (no fault) and the identical read-back after the
        // whole sequence are what these cases assert
        advise_willneed(c.p, c.n);
        std::printf("read-ahead %-58s -> returned\n", c.what);
    }
    spots_intact("view reads back as written, after every hint");
#if defined(_WIN32)
    UnmapViewOfFile((void*) base);
#else
    munmap((void*) base, (size_t) bytes);
#endif
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return fail;
}
// drop_mapped_view on a real mapped view: ranges of every shape return, bad ranges are refused, and the bytes read
// back as written after the pages were given up (they are re-read from the file when touched again).
int drop_view_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const uint64_t bytes = 1ull << 20;
    std::vector<uint8_t> pattern(bytes);
    for (uint64_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 31u + (i >> 9) * 5u);
    const std::string path = (std::filesystem::temp_directory_path() / "basalt_memory_test_drop.bin").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) pattern.data(), (std::streamsize) bytes);
        if (!f) return 1;
    }
    const uint8_t* base = nullptr;
#if defined(_WIN32)
    const std::wstring wpath = basalt::platform::to_wide(path);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
        CloseHandle(h);
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (view != MAP_FAILED) base = (const uint8_t*) view;
    }
#endif
    if (base == nullptr) {
        std::printf("drop view %-58s   <-- cannot map\n", "read-only view");
        std::filesystem::remove(path);
        return 1;
    }
    auto intact = [&]() {
        for (uint64_t i = 0; i < bytes; i += 997) if (base[i] != pattern[i]) return false;
        return base[bytes - 1] == pattern[bytes - 1];
    };
    struct Case { const char* what; uint64_t off, n; bool want; };
    const Case cases[] = {
        {"whole view", 0, bytes, true},
        {"page-unaligned interior range", 1234, 300ull << 10, true},
        {"one byte", bytes / 2, 1, true},
        {"the last byte", bytes - 1, 1, true},
        {"zero length", 5, 0, true},
        {"range past the end of the view", bytes - 100, 200, false},
        {"offset past the end of the view", bytes + 1, 1, false},
    };
    for (const Case& c : cases) {
        volatile uint8_t sink = 0;
        for (uint64_t i = 0; i < bytes; i += 4096) sink = (uint8_t) (sink + base[i]);   // pages in, so there is something to give up
        const bool got = drop_mapped_view(base, bytes, c.off, c.n);
        const bool ok = got == c.want && intact();
        std::printf("drop view %-58s -> %s%s\n", c.what, got ? "ok" : "refused", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    const bool null_ok = drop_mapped_view(nullptr, 0, 0, 4096);
    std::printf("drop view %-58s -> %s%s\n", "null base", null_ok ? "ok" : "refused", null_ok ? "" : "   <-- WRONG");
    fail |= !null_ok;
#if defined(_WIN32)
    UnmapViewOfFile((void*) base);
#else
    munmap((void*) base, (size_t) bytes);
#endif
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return fail;
}
// The watchdog's line: the budget flag is named only when the resident copy is big enough for it to shrink.
int exhausted_message_cases() {
    using basalt::platform::ram_exhausted_message;
    int fail = 0;
    const uint64_t margin = gib(4.0);
    struct Case { const char* what; bool resident; double copy_gib; bool names_budget; bool names_mmap; };
    const Case cases[] = {
        {"mapped experts, nothing resident", false, 0.0, false, true},
        {"tiny resident copy", true, 0.008, false, false},
        {"empty resident copy", true, 0.0, false, false},
        {"large resident copy", true, 60.0, true, false},
    };
    for (const Case& c : cases) {
        const std::string m = ram_exhausted_message(gib(1.0), c.resident, gib(c.copy_gib), margin);
        const bool budget = m.find("--resident-budget-gib") != std::string::npos;
        const bool mmap = m.find("--mmap-experts") != std::string::npos;
        const bool ok = budget == c.names_budget && mmap == c.names_mmap && m.find("1024 MiB free") != std::string::npos;
        std::printf("ram message %-56s -> %s%s\n", c.what, m.c_str(), ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The size lock_resident is exercised with: the full 256 MiB where the locked-memory limit allows it, otherwise half
// of what the limit leaves. Zero means the limit is too low to lock anything meaningful.
uint64_t lock_test_bytes(bool& reduced) {
    const uint64_t want = 256ull << 20;
    reduced = false;
#if defined(_WIN32)
    return want;
#else
    struct rlimit rl {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || rl.rlim_cur == RLIM_INFINITY) return want;
    // headroom for locks the process already holds (the engine's own allocations, thread stacks)
    const uint64_t limit = (uint64_t) rl.rlim_cur;
    if (limit >= want + (want >> 2)) return want;
    reduced = true;
    const uint64_t page = (uint64_t) sysconf(_SC_PAGE_SIZE);
    const uint64_t half = limit / 2 / page * page;
    return half >= 16 * page ? half : 0;
#endif
}
}  // namespace

int main() {
    int fail_keeps = file_cache_keeps_cases() | ram_floor_cases() | read_ahead_cases() | drop_view_cases() |
                    exhausted_message_cases();
    bool reduced = false;
    const uint64_t bytes = lock_test_bytes(reduced);
    if (bytes == 0) {
        std::printf("lock_resident: SKIPPED, the locked-memory limit is too low to lock a region (raise ulimit -l)\n");
        std::printf("platform_memory_test: %s\n", fail_keeps ? "FAILED" : "OK (lock_resident skipped)");
        return fail_keeps;
    }
    if (reduced)
        std::printf("lock_resident: the 256 MiB case is SKIPPED, the locked-memory limit is lower; locking %llu KiB instead\n",
                    (unsigned long long) (bytes >> 10));
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const basalt::platform::LockResult r = basalt::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu KiB of %llu KiB (%s)\n", (int) r.ok,
                (unsigned long long) (r.locked_bytes >> 10), (unsigned long long) (bytes >> 10), r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps;
    basalt::platform::unlock_resident(p, bytes);
    // With a progress callback the lock runs in chunks: the counts only rise and the last one is the whole region.
    std::vector<uint64_t> seen;
    const basalt::platform::LockResult rp =
        basalt::platform::lock_resident(p, bytes, [&](uint64_t done) { seen.push_back(done); });
    bool rising = !seen.empty() && seen.back() == bytes;
    for (size_t i = 1; i < seen.size(); ++i) rising = rising && seen[i] > seen[i - 1];
    std::printf("lock_resident with progress: ok=%d %zu reports%s\n", (int) rp.ok, seen.size(),
                rising ? "" : "   <-- WRONG");
    fail |= !(rp.ok && rp.locked_bytes == bytes && rising);
    basalt::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
