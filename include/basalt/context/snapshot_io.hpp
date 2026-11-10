// include/basalt/context/snapshot_io.hpp - context snapshots on disk: atomic publish and verified restore.
//
// A context's snapshots live in its own directory, one file per generation, `<generation as 16 hex digits>.bsnap`:
// the manifest, zero padded to kSnapshotAlign, then every stored inventory entry at its own aligned offset, zero
// padded. Every read and write is aligned, so the file goes around the OS cache where the file system allows it.
//
// A publish writes a temporary file, syncs it, reads it back and checks every digest, renames it into place and
// syncs the directory. A failed or interrupted step leaves either the complete new snapshot or none of it under a
// snapshot name, and never touches another generation: the previous one stays restorable until `prune` removes it.
// A publish refuses a generation at or below one already published (generation fencing).
//
// A restore checks the manifest against the request (owner, generation, model fingerprint, prefix, completeness) and
// the file's size against the manifest before it hands any bytes over, then streams each stored entry to the sink
// while checking it against its digest. Any refusal means the context is rebuilt from its transcript.
//
// The file calls go through `SnapshotFiles`; `PlatformSnapshotFiles` is the one over basalt::platform. Each call is
// self-contained, so threads may publish and restore different generations of one directory at once; one writer per
// generation is the caller's lease to keep.
#pragma once

#include "basalt/context/ids.hpp"
#include "basalt/context/manifest.hpp"
#include "basalt/platform/file.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::context {

inline constexpr uint64_t kSnapshotAlign = 4096;

/// Where each part of a snapshot file sits: `offsets[i]` is inventory entry i's (a recompute entry takes no bytes, so
/// its offset is the next entry's), `file_bytes` the whole file's size.
struct SnapshotLayout {
    uint64_t payload_offset = 0;
    std::vector<uint64_t> offsets;
    uint64_t file_bytes = 0;
};
bool snapshot_layout(const Manifest& m, SnapshotLayout& out, std::string& error);

std::string snapshot_file_name(Generation g);
/// The generation a snapshot file name stands for; false for any other name (a temporary file, generation 0).
bool parse_snapshot_file_name(const std::string& name, Generation& g);

struct SnapshotDirEntry {
    std::string name;
    uint64_t bytes = 0;
    int64_t mtime = 0;   ///< seconds on the file system's clock, the one `SnapshotFiles::now` reads
};

/// The file calls a snapshot store makes, in the shape of basalt::platform's: descriptors are ints, reads and writes
/// are positional and whole (a short transfer is a failure).
class SnapshotFiles {
public:
    virtual ~SnapshotFiles() = default;
    virtual int create(const std::string& path) = 0;   ///< creates or truncates for writing; -1 on failure
    virtual bool write(int fd, const void* src, uint64_t n, uint64_t off) = 0;
    virtual bool sync(int fd) = 0;                       ///< true once the written bytes are on the device
    virtual void close(int fd) = 0;
    virtual int open_read(const std::string& path) = 0;   ///< -1 on failure
    virtual bool read(int fd, void* dst, uint64_t n, uint64_t off) = 0;
    virtual bool size(const std::string& path, uint64_t& out) = 0;   ///< false unless a regular file
    virtual bool rename(const std::string& from, const std::string& to) = 0;   ///< replaces `to`
    virtual bool remove(const std::string& path) = 0;    ///< true when the file is gone afterwards
    virtual bool list(const std::string& dir, std::vector<SnapshotDirEntry>& out) = 0;   ///< regular files only
    virtual bool make_dirs(const std::string& dir) = 0;
    virtual bool sync_dir(const std::string& dir) = 0;   ///< true once its entries (a rename into it) are durable
    virtual int64_t now() = 0;
};

/// The running state a publish writes. `read` copies bytes [offset, offset + n) of a stored inventory entry into
/// `dst`; false aborts the publish. The digest covers the bytes as read.
class StateSource {
public:
    virtual ~StateSource() = default;
    virtual bool read(const InventoryEntry& entry, uint64_t offset, void* dst, size_t n) = 0;
};

/// Where a restore puts the running state. `begin` precedes the bytes of each stored entry (false refuses the
/// restore: no room, or a size the engine does not expect); `write` gets them in order. Bytes are unverified until
/// the restore returns `restored`: after any other result the sink discards everything it was given.
class StateSink {
public:
    virtual ~StateSink() = default;
    virtual bool begin(const InventoryEntry& entry) = 0;
    virtual bool write(const InventoryEntry& entry, uint64_t offset, const void* src, size_t n) = 0;
};

enum class PublishStatus : uint8_t {
    published,
    invalid,         ///< the draft manifest is malformed or incomplete for the requirements
    stale,           ///< this or a later generation is already published
    source_failed,   ///< the source refused a read, or gave bytes that differ from an expected digest
    io_failed,       ///< a file call failed
    verify_failed,   ///< the bytes read back differ from those written
};

struct PublishResult {
    PublishStatus status = PublishStatus::invalid;
    Manifest manifest;   ///< as published, digests filled in
    std::string path;
    std::string error;
    bool ok() const { return status == PublishStatus::published; }
};

enum class RestoreStatus : uint8_t {
    restored,
    invalid_request,
    missing,         ///< no snapshot of that generation
    unreadable,      ///< a file call failed
    corrupt,         ///< the file is not a well-formed snapshot: manifest, size, padding or an entry digest
    mismatch,        ///< a sound snapshot, but of another context, model fingerprint, prefix or format version
    stale,           ///< the file holds another generation than its name and the request say
    incomplete,      ///< the inventory lacks state the model needs, or holds state it cannot use
    sink_rejected,
};

struct RestoreRequest {
    ContextId context_id;
    Digest model_fingerprint{};
    Digest prefix_hash{};
    uint64_t prefix_tokens = 0;
    Generation generation = 0;   ///< the fenced generation the registry holds for the context
    StateRequirements requirements = kHybridState;
};

struct RestoreResult {
    RestoreStatus status = RestoreStatus::missing;
    Manifest manifest;   ///< valid when restored: it names the entries to rebuild (`recompute`)
    std::string error;
    /// The context is rebuilt from its transcript instead.
    bool replay() const { return status != RestoreStatus::restored; }
};

const char* publish_status_name(PublishStatus s);
const char* restore_status_name(RestoreStatus s);

class SnapshotStore {
public:
    /// `dir` is the context's own directory (created on the first publish); `files` outlives the store.
    SnapshotStore(std::string dir, SnapshotFiles& files) : dir_(std::move(dir)), files_(files) {}

    /// Publishes `draft` (digests left zero, or nonzero ones the bytes must match) with the bytes of `source`.
    PublishResult publish(const Manifest& draft, StateSource& source,
                          const StateRequirements& requirements = kHybridState);
    RestoreResult restore(const RestoreRequest& want, StateSink& sink);

    /// The published generations, by file name, ascending; empty when the directory cannot be read.
    std::vector<Generation> generations() const;
    /// The manifest of generation `g`, checked as a restore checks it before any payload byte.
    bool read_manifest(Generation g, Manifest& out, std::string& error) const;
    /// Removes the snapshot of generation `g` (one a restore found corrupt, say). True when it is gone.
    bool discard(Generation g);
    /// Removes the snapshots older than `keep_from`, once the registry holds that generation. Returns how many.
    size_t prune(Generation keep_from);
    /// Removes temporary files at least `min_age_seconds` old: the leftovers of a publish that died midway (a younger
    /// one may be another process's publish in flight). Returns how many.
    size_t remove_stale_temporaries(int64_t min_age_seconds);

    const std::string& dir() const { return dir_; }

private:
    std::string path_of(const std::string& name) const;
    bool list_generations(std::vector<Generation>& out) const;
    RestoreStatus open_snapshot(Generation g, uint8_t* buf, int& fd, Manifest& m, SnapshotLayout& layout,
                                std::string& error) const;

    std::string dir_;
    SnapshotFiles& files_;
};

/// SnapshotFiles over basalt::platform: unbuffered reads and writes where the file system gives them, otherwise the
/// OS cache is asked to drop a file's pages when it is closed, so snapshots do not grow the file cache.
/// A template so that only code constructing one instantiates it: the program that does links basalt::platform, and
/// the context library's own objects never reference it (an optimizer that sees a lone implementation of an
/// interface may inline calls against it).
template <class = void>
class BasicPlatformSnapshotFiles final : public SnapshotFiles {
public:
    int create(const std::string& path) override {
        const int fd = basalt::platform::open_write_unbuffered(path);
        return fd >= 0 ? fd : basalt::platform::open_write_utf8(path);
    }
    bool write(int fd, const void* src, uint64_t n, uint64_t off) override {
        return basalt::platform::pwrite_all(fd, src, n, off);
    }
    bool sync(int fd) override { return basalt::platform::sync_fd(fd); }
    void close(int fd) override {
        basalt::platform::advise_dontneed(fd, 0, 0);
        basalt::platform::close_fd(fd);
    }
    int open_read(const std::string& path) override {
        const int fd = basalt::platform::open_unbuffered(path);
        return fd >= 0 ? fd : basalt::platform::open_fd_utf8(path);
    }
    bool read(int fd, void* dst, uint64_t n, uint64_t off) override {
        return basalt::platform::pread_all(fd, dst, n, off);
    }
    bool size(const std::string& path, uint64_t& out) override { return basalt::platform::file_size(path, out); }
    bool rename(const std::string& from, const std::string& to) override {
        return basalt::platform::rename_replace(from, to);
    }
    bool remove(const std::string& path) override { return basalt::platform::remove_file(path); }
    bool list(const std::string& dir, std::vector<SnapshotDirEntry>& out) override {
        std::vector<basalt::platform::DirFile> files;
        out.clear();
        if (!basalt::platform::list_files(dir, files)) return false;
        for (auto& f : files) out.push_back({std::move(f.name), f.bytes, f.mtime});
        return true;
    }
    bool make_dirs(const std::string& dir) override { return basalt::platform::make_dirs(dir); }
    bool sync_dir(const std::string& dir) override { return basalt::platform::sync_dir(dir); }
    int64_t now() override { return basalt::platform::file_time_now(); }
};
using PlatformSnapshotFiles = BasicPlatformSnapshotFiles<>;

}  // namespace basalt::context
