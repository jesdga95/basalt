// include/basalt/platform/mapped_file.hpp - a read-only view of a file, and the hints that go with one.
//
// The engine reads its model, expert and projector files through a view: the pages arrive as they are touched and
// nothing writes to them. Linux takes the view with mmap; Windows takes a file mapping and maps all of it, because
// a view may only start at the allocation granularity, and moves the pointer by the requested offset. Either way the
// bytes at `data()` are `[offset, offset + size())` of the file and the address is stable until the object is reset
// or destroyed.
#pragma once

#include "basalt/platform/memory.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace basalt::platform {

struct MapOptions {
    uint64_t offset = 0;   ///< first byte of the file to expose; on Linux it has to be a multiple of the page size
    uint64_t bytes = 0;    ///< bytes to expose; 0 is everything from `offset` to the end of the file
    bool shared = false;   ///< Linux: MAP_SHARED instead of MAP_PRIVATE (read-only either way); no effect on Windows
};

class MappedFile {
public:
    MappedFile() = default;
    MappedFile(MappedFile&& o) noexcept { take(o); }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) { reset(); take(o); }
        return *this;
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { reset(); }

    /// Maps the file at the UTF-8 `path`. nullopt on failure, with the reason in `*err` when it is given.
    static std::optional<MappedFile> open_ro(const std::string& path, std::string* err, const MapOptions& opt = {});

    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }
    /// The size of the whole file when it was mapped (`size()` is the exposed part of it).
    uint64_t file_size() const { return file_size_; }
    explicit operator bool() const { return data_ != nullptr; }

    /// Unmaps; the object is empty afterwards. Safe on an empty one.
    void reset();

    /// Asks the OS to start bringing the whole pages covering [off, off + len) of the view into memory. A hint.
    /// Windows asks nothing here; see `advise_willneed` in memory.hpp for the ranged prefetch it does have.
    void prefetch(uint64_t off, uint64_t len) const;

    /// Takes the whole pages inside [off, off + len) out of this process, which are re-read from the file when
    /// touched again. Returns the bytes released: 0 when the OS refused, and always 0 on Windows, whose view has
    /// no per-range release (its clean pages go back on pressure by themselves).
    uint64_t release(uint64_t off, uint64_t len) const;

private:
    void take(MappedFile& o) {
        data_ = o.data_; size_ = o.size_; file_size_ = o.file_size_; region_ = o.region_; region_bytes_ = o.region_bytes_;
        o.data_ = nullptr; o.size_ = 0; o.file_size_ = 0; o.region_ = nullptr; o.region_bytes_ = 0;
    }

    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
    uint64_t file_size_ = 0;
    void* region_ = nullptr;        ///< what the OS returned for the mapping, which may precede data_ (Windows)
    uint64_t region_bytes_ = 0;
};

/// Locks [p, p + bytes) of a mapped view into RAM after asking the OS to read it in, and reports how far it got
/// (see lock_resident). Used for a table that decode touches at random.
LockResult lock_view(void* p, uint64_t bytes, const LockProgress& progress = nullptr);

}  // namespace basalt::platform
