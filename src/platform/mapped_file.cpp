// src/platform/mapped_file.cpp - see include/basalt/platform/mapped_file.hpp.
#include "basalt/platform/mapped_file.hpp"

#include "basalt/platform/file.hpp"

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace basalt::platform {

#if defined(_WIN32)
// --------------------------------------------------------------------------------------------- Windows

std::optional<MappedFile> MappedFile::open_ro(const std::string& path, std::string* err, const MapOptions& opt) {
    auto fail = [&](const std::string& why) -> std::optional<MappedFile> {
        if (err) *err = why + " " + path;
        return std::nullopt;
    };
    // The section is taken on a file handle, so this open is the wide Win32 one: the file seam has no mapping call.
    HANDLE h = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return fail("cannot open");
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(h, &li)) {
        CloseHandle(h);
        return fail("cannot stat");
    }
    const uint64_t fsize = (uint64_t) li.QuadPart;
    if (opt.offset > fsize || (opt.bytes != 0 && opt.bytes > fsize - opt.offset)) {
        CloseHandle(h);
        return fail("cannot map a range past the end of");
    }
    // MapViewOfFile takes only allocation-granularity offsets, so the file is mapped whole and the pointer moved by
    // the offset. That is address space only: the pages arrive as they are touched. The view keeps the section and
    // the file alive, so both handles close here.
    HANDLE mapping = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(h);
        return fail("cannot map");
    }
    uint8_t* raw = (uint8_t*) MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(h);
    if (raw == nullptr) return fail("cannot map");
    MappedFile m;
    m.region_ = raw;
    m.region_bytes_ = fsize;
    m.data_ = raw + opt.offset;
    m.size_ = opt.bytes != 0 ? opt.bytes : fsize - opt.offset;
    m.file_size_ = fsize;
    return m;
}

void MappedFile::reset() {
    if (region_ != nullptr) UnmapViewOfFile(region_);   // from its real start, which precedes data_ by the offset
    data_ = nullptr;
    size_ = file_size_ = region_bytes_ = 0;
    region_ = nullptr;
}

void MappedFile::prefetch(uint64_t, uint64_t) const {}   // no per-range read-ahead here; see platform::advise_willneed

uint64_t MappedFile::release(uint64_t, uint64_t) const { return 0; }   // nothing is dropped, so nothing is reported

LockResult lock_view(void* p, uint64_t bytes, const LockProgress& progress) {
    // Locked pages count against the process working-set minimum, which lock_resident raises first. A view of a file
    // mapping can still refuse the lock; the caller then touches the pages to get them resident.
    return lock_resident(p, bytes, progress);
}

#else
// --------------------------------------------------------------------------------------------- Linux

std::optional<MappedFile> MappedFile::open_ro(const std::string& path, std::string* err, const MapOptions& opt) {
    auto fail = [&](const std::string& why) -> std::optional<MappedFile> {
        if (err) *err = why + " " + path;
        return std::nullopt;
    };
    const int fd = open_fd_utf8(path);
    if (fd < 0) return fail("cannot open");
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close_fd(fd);
        return fail("cannot stat");
    }
    const uint64_t fsize = (uint64_t) st.st_size;
    if (opt.offset > fsize || (opt.bytes != 0 && opt.bytes > fsize - opt.offset)) {
        close_fd(fd);
        return fail("cannot map a range past the end of");
    }
    const uint64_t len = opt.bytes != 0 ? opt.bytes : fsize - opt.offset;
    void* p = mmap(nullptr, (size_t) len, PROT_READ, opt.shared ? MAP_SHARED : MAP_PRIVATE, fd, (off_t) opt.offset);
    close_fd(fd);   // the mapping holds the file; the descriptor is not needed
    if (p == MAP_FAILED) return fail("cannot map");
    MappedFile m;
    m.region_ = p;
    m.region_bytes_ = len;
    m.data_ = (const uint8_t*) p;
    m.size_ = len;
    m.file_size_ = fsize;
    return m;
}

void MappedFile::reset() {
    if (region_ != nullptr) munmap(region_, (size_t) region_bytes_);
    data_ = nullptr;
    size_ = file_size_ = region_bytes_ = 0;
    region_ = nullptr;
}

void MappedFile::prefetch(uint64_t off, uint64_t len) const {
    if (data_ == nullptr || off >= size_) return;
    const uint64_t start = off & ~(uint64_t) 4095;
    madvise((uint8_t*) data_ + start, (size_t) (off + len - start), MADV_WILLNEED);
}

// Only the pages wholly inside the range: a page shared with a neighbour the CPU may still read stays. MADV_DONTNEED
// unmaps them from this process; they stay in the page cache as clean, unmapped pages - the first the kernel takes
// back under pressure, and a minor fault away when they are needed again. Dropping them from the cache too
// (POSIX_FADV_DONTNEED) would make every release a re-read from the disk.
uint64_t MappedFile::release(uint64_t off, uint64_t len) const {
    if (data_ == nullptr || off >= size_) return 0;
    const uint64_t start = (off + 4095) & ~(uint64_t) 4095;
    const uint64_t end = (off + len) & ~(uint64_t) 4095;
    if (end <= start) return 0;
    if (madvise((uint8_t*) data_ + start, (size_t) (end - start), MADV_DONTNEED) != 0) return 0;
    return end - start;
}

LockResult lock_view(void* p, uint64_t bytes, const LockProgress& progress) {
    if (p != nullptr && bytes != 0) madvise(p, (size_t) bytes, MADV_WILLNEED);
    return lock_resident(p, bytes, progress);
}

#endif

}  // namespace basalt::platform
