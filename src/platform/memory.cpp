// src/platform/memory.cpp - see include/basalt/platform/memory.hpp.
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <malloc.h>   // _aligned_malloc, _aligned_free
#else
#include <fcntl.h>
#include <linux/mman.h>   // MAP_HUGE_2MB
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <cstring>

namespace basalt::platform {

#if defined(_WIN32)
namespace {
/// Enables one privilege in this process's token. A privilege can be assigned to the account and still be absent
/// from the token until the process asks for it, which is the normal state for an interactive one.
void enable_privilege(LPCWSTR name) {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    if (LookupPrivilegeValueW(nullptr, name, &tp.Privileges[0].Luid)) {
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        (void) AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
    }
    CloseHandle(tok);
}
}  // namespace
#endif

LockResult lock_resident(void* p, uint64_t bytes, const LockProgress& progress) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
#if defined(_WIN32)
    enable_privilege(L"SeIncreaseWorkingSetPrivilege");   // to raise this process's working set
    enable_privilege(L"SeLockMemoryPrivilege");          // and to lock pages in it
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    // Locked pages count against the minimum working set, so it has to grow by the region plus room for the rest of
    // the process first. Both limits are soft, so raising the minimum reserves without pinning the process to it.
    // A quota that cannot be raised is reported and the lock tried anyway: whatever the existing quota covers is
    // still worth having.
    std::string quota_note;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        quota_note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
    } else {
        const SIZE_T margin = (SIZE_T) 512 << 20;
        const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
        const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
        if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                        QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE))
            quota_note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                         " MiB) failed (error " + std::to_string(GetLastError()) + ")";
    }
    // Chunked, because one call over the whole region loses everything past the first page that does not fit.
    const uint64_t chunk = progress ? 256ull << 20 : 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = (quota_note.empty() ? "" : quota_note + "; ") + "VirtualLock stopped at " +
                     std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;   // what it did lock stays locked; the rest of the region is merely pageable
            return r;
        }
        r.locked_bytes = off + n;
        if (progress) progress(r.locked_bytes);
    }
    r.ok = true;
    const std::string locked = std::to_string((unsigned long long) (bytes >> 20)) + " MiB";
    r.note = quota_note.empty() ? "locked " + locked + " via working-set minimum + VirtualLock"
                                : quota_note + "; locked " + locked + " with the quota already in place";
    return r;
#else
    if (!progress) {
        if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
        r.ok = true;
        r.locked_bytes = bytes;
        r.note = "mlock";
        return r;
    }
    const uint64_t chunk = 256ull << 20;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (mlock((uint8_t*) p + off, (size_t) n) != 0) {
            r.note = "mlock failed (raise ulimit -l)";
            r.ok = off > 0;   // what it did lock stays locked; the rest of the region is merely pageable
            return r;
        }
        r.locked_bytes = off + n;
        progress(r.locked_bytes);
    }
    r.ok = true;
    r.note = "mlock";
    return r;
#endif
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
#if defined(_WIN32)
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
#else
    munlock(p, bytes);
#endif
}

uint64_t total_physical_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
#endif
}

#if defined(_WIN32)
namespace {
/// The two figures the sizing reads, from one query: available and total physical RAM in bytes. Both are 0 when the
/// query fails, which the callers read as unknown rather than as none.
void memory_status(uint64_t& avail, uint64_t& total) {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) { avail = 0; total = 0; return; }
    avail = (uint64_t) ms.ullAvailPhys;
    total = (uint64_t) ms.ullTotalPhys;
}
}  // namespace

uint64_t available_memory() {
    uint64_t avail = 0, total = 0;
    memory_status(avail, total);
    return avail;
}

uint64_t ram_margin() {
    uint64_t avail = 0, total = 0;
    memory_status(avail, total);
    return std::max<uint64_t>(2ull << 30, total / 20);
}
#else
namespace {
// One "<key>: <n> kB" line of /proc/meminfo in bytes (0 when absent).
uint64_t meminfo_bytes(const char* key) {
    std::ifstream in("/proc/meminfo");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(in, line))
        if (line.compare(0, k.size(), k) == 0) return std::strtoull(line.c_str() + k.size(), nullptr, 10) * 1024;
    return 0;
}
}  // namespace

uint64_t available_memory() { return meminfo_bytes("MemAvailable"); }

uint64_t ram_margin() {
    return std::max<uint64_t>(2ull << 30, meminfo_bytes("MemTotal") / 20);
}
#endif

uint64_t commit_headroom() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPageFile : 0;
#else
    return available_memory();
#endif
}

uint64_t ram_budget(bool* known) {
    const uint64_t avail = available_memory();
    if (known) *known = avail > 0;
    if (avail == 0) return UINT64_MAX;
    const uint64_t margin = ram_margin();
    return avail > margin ? avail - margin : 0;
}

namespace {
std::atomic<bool> g_watch_stop{false};
std::atomic<bool> g_watch_on{false};
std::atomic<bool> g_resident_mode{false};
std::atomic<uint64_t> g_resident_copy{0};
}  // namespace

std::string ram_exhausted_message(uint64_t avail, bool resident, uint64_t copy_bytes, uint64_t margin) {
    const unsigned long long mib = (unsigned long long) (avail >> 20);
    char buf[384];
    if (!resident) {
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); use "
                                       "--resident-experts or --mmap-experts", mib);
    } else if (copy_bytes < (1ull << 30)) {
        // a budget of 1 GiB or more would not shrink a copy this small, so it is not offered
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); the resident copy is only "
                                       "%llu MiB, so something else is holding the memory (other programs, or file "
                                       "pages read while loading)", mib, (unsigned long long) (copy_bytes >> 20));
    } else {
        // the copy shrunk by what the margin is short of leaves the margin free again
        const uint64_t keep = copy_bytes - std::min(copy_bytes, margin > avail ? margin - avail : 0);
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); use --resident-budget-gib %llu",
                      mib, (unsigned long long) std::max<uint64_t>(1, keep >> 30));
    }
    return buf;
}

void watch_resident_ram(bool resident, uint64_t copy_bytes) {
    g_resident_mode = resident;
    g_resident_copy = copy_bytes;
}

void start_memory_watchdog(uint64_t floor_bytes) {
    if (g_watch_on.exchange(true)) return;
    g_watch_stop = false;
    std::thread([floor_bytes] {
        while (!g_watch_stop.load()) {
            const uint64_t avail = available_memory();
            if (avail > 0 && avail < floor_bytes) {
                std::fprintf(stderr, "%s\n", ram_exhausted_message(avail, g_resident_mode.load(), g_resident_copy.load(),
                                                                   ram_margin()).c_str());
                std::fflush(stderr);
                _exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // never joined: the process may end while it sleeps
}

void stop_memory_watchdog() {
    g_watch_stop = true;
    g_watch_on = false;
}

namespace {
std::atomic<bool> g_guard_on{false};
std::atomic<bool> g_guard_low{false};
std::atomic<uint64_t> g_guard_avail{0};
std::atomic<uint64_t> g_guard_floor{0};
}  // namespace

void start_ram_guard(uint64_t floor_bytes) {
    g_guard_floor = floor_bytes;
    if (g_guard_on.exchange(true)) return;
    std::thread([] {
        bool was_low = false;
        for (;;) {
            const uint64_t avail = available_memory(), floor = g_guard_floor.load();
            const bool low = ram_below_floor(avail, floor);
            g_guard_avail = avail;
            g_guard_low = low;
            if (low && !was_low) {
                std::fprintf(stderr, "basalt warning: RAM low: %llu MiB free < %llu MiB guard\n",
                             (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20));
                std::fflush(stderr);
            }
            was_low = low;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // runs for the life of the process
}

bool ram_guard_low(uint64_t* avail, uint64_t* floor) {
    if (avail) *avail = g_guard_avail.load();
    if (floor) *floor = g_guard_floor.load();
    return g_guard_low.load(std::memory_order_relaxed);
}

#if defined(_WIN32)
namespace {
constexpr uint64_t kPrefetchStep = 128ull << 10;

/// The page size, so a prefetch window starts on a page boundary.
uintptr_t page_bytes() {
    static const uintptr_t ps = [] {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        return si.dwPageSize > 0 ? (uintptr_t) si.dwPageSize : (uintptr_t) 4096;
    }();
    return ps;
}
}  // namespace

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    // A view of a file mapping takes its pages one fault at a time as they are touched, so asking for the range
    // ahead of the access replaces those faults with large background reads.  The hint is best effort: a range the
    // system will not serve is skipped without an error, and pages already in memory cost nothing.  The range is
    // asked for in bounded windows, which keeps the reads in flight - and the memory they hold - bounded instead of
    // asking for a whole layer at once.
    const uintptr_t pg = page_bytes(), end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kPrefetchStep) {
        WIN32_MEMORY_RANGE_ENTRY entry{};
        entry.VirtualAddress = (PVOID) a;
        entry.NumberOfBytes = (SIZE_T) std::min<uintptr_t>(kPrefetchStep, end - a);
        (void) PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    // No descriptor-range hint on Windows: the cache manager takes no "will need" for an open handle.  Reads through
    // a descriptor are cached reads unless the caller opted out of the cache, and the cache manager already reads
    // ahead around the position a sequential reader is at.  Warming a range by reading it into a scratch buffer
    // would move the bytes twice and cost the file cache the room, so nothing is asked.  A caller holding a mapped
    // view of the same file gets the prefetch from the overload above.
    (void) fd; (void) offset; (void) bytes;
}
#else
namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}
#endif

void* alloc_aligned(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) return nullptr;
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
#endif
}

void free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

bool drop_mapped_view(const void* base, uint64_t view_bytes, uint64_t off, uint64_t bytes) {
    if (base == nullptr || bytes == 0) return true;
#if defined(_WIN32)
    // Unlocking a range that is not locked takes its pages out of this process's working set, which is the only
    // per-range release a view has: a clean page then waits on the standby list, counted as available, and is read
    // back from the file if touched again. The call reports ERROR_NOT_LOCKED for pages that were never locked even
    // though they were released, so its result carries nothing and a range the OS declines is simply left alone.
    if (off > view_bytes || bytes > view_bytes - off) return false;
    const uintptr_t pg = page_bytes();
    const uintptr_t first = ((uintptr_t) base + (uintptr_t) off) & ~(pg - 1);
    const uintptr_t last = ((uintptr_t) base + (uintptr_t) (off + bytes) + pg - 1) & ~(pg - 1);
    constexpr uintptr_t kStep = (uintptr_t) 1 << 30;
    for (uintptr_t a = first; a < last; a += kStep)
        (void) VirtualUnlock((LPVOID) a, (SIZE_T) std::min<uintptr_t>(kStep, last - a));
    return true;
#else
    // Only whole pages can be given back: a page the range shares with a neighbour still holds bytes the caller may
    // read. So the range is widened to the page boundaries and then clipped to the view.
    const long ps = sysconf(_SC_PAGE_SIZE);
    if (ps <= 0) return false;
    const uint64_t page = (uint64_t) ps;
    const uint64_t end = off + bytes;
    if (end > view_bytes) return false;
    const uint64_t start = off - off % page;
    const uint64_t remainder = end % page;
    const uint64_t extra = remainder == 0 ? 0 : page - remainder;
    const uint64_t wide = extra > view_bytes - end ? view_bytes : end + extra;
    if (wide <= start) return true;
    return madvise((void*) ((const uint8_t*) base + (size_t) start), (size_t) (wide - start), MADV_DONTNEED) == 0;
#endif
}


// ---------------------------------------------------------------------------------------------------- host regions

#if defined(_WIN32)
namespace {
// A size in GiB, for the notes that have to say how big something was.
std::string gib_size(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f GiB", (double) bytes / (double) (1ull << 30));
    return buf;
}

// What the OS could still commit, for the note that has to say why the arena did not fit.  The commit charge is the
// sum of RAM and the page file, so its free part is the headroom the arena has to fit into.  Empty when the OS
// gives no reading.
std::string commit_headroom_note() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return "; the commit headroom is unknown";
    return "; " + gib_size((uint64_t) ms.ullAvailPageFile) + " of commit charge free out of " +
           gib_size((uint64_t) ms.ullTotalPageFile) + " (RAM plus page file), " +
           gib_size((uint64_t) ms.ullTotalPhys) + " of RAM";
}
}  // namespace


static void* alloc_region_raw(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                              uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
    if (!shared_file.empty()) {
        note = "shared arena files are not available on Windows";
        return nullptr;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege, and the account having it assigned is not enough: the process
    // enables it in its own token first. An account without the assignment fails here and falls through to the
    // ordinary allocation below, which is the expected outcome rather than an error.
    enable_privilege(L"SeLockMemoryPrivilege");
    const SIZE_T large = GetLargePageMinimum();
    if (large > 0) {
        // The requested size must be an exact multiple of the large-page minimum; the slack is under one page.
        const SIZE_T lbytes = (SIZE_T) (((SIZE_T) bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p != nullptr) {
            got = RegionBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        // ERROR_NO_SYSTEM_RESOURCES is the page pool declining, ERROR_PRIVILEGE_NOT_HELD is the token and
        // ERROR_INVALID_PARAMETER a size that is not a multiple of the minimum. They need different fixes, so the
        // note carries the number.
        note = "large pages refused for " + std::to_string((unsigned long long) lbytes) +
               " B (GetLargePageMinimum=" + std::to_string((unsigned long long) large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long) GetLastError()) + "); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    {
        // Reserve first, then commit in chunks.  A single MEM_COMMIT over the whole arena only ever says that the
        // allocation failed, and whether it can succeed depends on RAM plus page file together, which is not known
        // before the commit reaches that far.  Committing a chunk at a time makes the failure name the size it
        // reached, and the region is released before anything is returned, so a half-committed arena is never
        // handed to the caller and nothing is reported as pinned that was not.
        void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE, PAGE_READWRITE);
        if (p == nullptr) {
            got = RegionBacking::NormalPages;
            note = "cannot reserve " + gib_size(bytes) + " of arena (VirtualAlloc error " +
                   std::to_string((unsigned long long) GetLastError()) + ")" + commit_headroom_note() +
                   "; the arena needs RAM plus page file; " + note;
            return nullptr;
        }
        constexpr uint64_t kCommitChunk = 1ull << 30;
        for (uint64_t off = 0; off < bytes; off += kCommitChunk) {
            const uint64_t n = bytes - off < kCommitChunk ? bytes - off : kCommitChunk;
            if (VirtualAlloc((uint8_t*) p + off, (SIZE_T) n, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                const DWORD err = GetLastError();
                VirtualFree(p, 0, MEM_RELEASE);
                got = RegionBacking::NormalPages;
                note = "cannot commit the arena: " + gib_size(off) + " of " + gib_size(bytes) +
                       " committed (VirtualAlloc error " + std::to_string((unsigned long long) err) + ")" +
                       commit_headroom_note() + "; the arena needs RAM plus page file; " + note;
                return nullptr;
            }
        }
        got = RegionBacking::NormalPages;
        return p;
    }
}

void free_region(void* p, uint64_t bytes) {
    if (!p) return;
    (void) bytes;   // one VirtualAlloc owns the whole region
    VirtualFree(p, 0, MEM_RELEASE);
}

bool pin_budget(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    const uint64_t ram = basalt::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    char buf[192];
    std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget, RAM/2 - 8 GiB of %.1f GiB RAM",
                  (double) ram / (double) GiB);
    why = buf;
    return true;
}

#else
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

namespace {
constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "BASALT-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t arena_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);
}  // namespace

static void* alloc_region_raw(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                              uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
    if (!shared_file.empty()) {
        if (shared_hash == 0) {
            note = "shared arena requires a nonzero model hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.arena_hash = shared_hash;
            const ssize_t written = ::pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t) sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            const ssize_t got_header = ::pread(fd, &hdr, sizeof hdr, 0);
            if (got_header != (ssize_t) sizeof hdr ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.arena_hash != shared_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for model hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.arena_hash,
                              (unsigned long long) shared_hash);
                note = b;
                return nullptr;
            }
        }

        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = RegionBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (model hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = RegionBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = RegionBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    if (madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
}

void free_region(void* p, uint64_t bytes) {
    if (!p) return;
    munmap(p, bytes);
}

bool pin_budget(uint64_t&, std::string&) { return false; }   // no OS-side budget below the driver's own limit
#endif

void* alloc_region(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                   uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    void* p = alloc_region_raw(bytes, got, note, shared_file, shared_hash, mapping_base, mapping_bytes);
    if (p != nullptr && mapping_base == nullptr) {   // not a file mapping: the region is what gets released
        mapping_base = p;
        mapping_bytes = bytes;
    }
    return p;
}


}  // namespace basalt::platform
