// src/context/snapshot_io.cpp - see include/basalt/context/snapshot_io.hpp.
#include "basalt/context/snapshot_io.hpp"

#include "basalt/platform/host_allocator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <new>
#include <random>

namespace basalt::context {
namespace {

constexpr uint64_t kChunk = 4ull << 20;   // one read or write
static_assert(kChunk % kSnapshotAlign == 0, "every transfer stays aligned");
constexpr const char* kSnapshotExt = ".bsnap";
constexpr const char* kTempExt = ".tmp";
constexpr size_t kHexDigits = 16;

uint64_t align_up(uint64_t v) { return (v + kSnapshotAlign - 1) / kSnapshotAlign * kSnapshotAlign; }

// the aligned span the largest manifest and its padding take
constexpr uint64_t kHeadMax = (manifest_bytes(kMaxInventoryEntries) + kSnapshotAlign - 1) / kSnapshotAlign *
                              kSnapshotAlign;
static_assert(kHeadMax <= kChunk, "a manifest fits one transfer");

bool all_zero(const uint8_t* p, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i)
        if (p[i]) return false;
    return true;
}

std::string hex16(uint64_t v) {
    char b[kHexDigits + 1];
    for (int i = (int) kHexDigits - 1; i >= 0; --i, v >>= 4) b[i] = "0123456789abcdef"[v & 15];
    b[kHexDigits] = 0;
    return b;
}

bool hex_prefix(const std::string& name, uint64_t& v) {
    if (name.size() < kHexDigits) return false;
    v = 0;
    for (size_t i = 0; i < kHexDigits; ++i) {
        const char c = name[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        v = v << 4 | (uint64_t) d;
    }
    return true;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Unique per process and call: two publishes never share a temporary file, in one process or across several.
std::string temp_name(Generation g) {
    static const uint64_t nonce = [] {
        uint64_t v = 0;
        try {
            std::random_device rd;
            v = (uint64_t) rd() << 32 ^ rd();
        } catch (...) {
        }
        return v ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    }();
    static std::atomic<uint64_t> seq{0};
    return hex16(g) + "." + hex16(nonce) + "-" + std::to_string(seq.fetch_add(1)) + kTempExt;
}

bool is_temp_name(const std::string& name) {
    uint64_t g = 0;
    return hex_prefix(name, g) && name.size() > kHexDigits && name[kHexDigits] == '.' && ends_with(name, kTempExt);
}

std::string entry_name(const InventoryEntry& e) {
    return std::string(state_kind_name(e.kind)) + " part " + std::to_string(e.part);
}

// The buffer a publish or a restore moves bytes through, from the host pool when the engine runs one.
class Staging {
public:
    explicit Staging(uint64_t bytes) {
        try {
            p_ = static_cast<uint8_t*>(basalt::platform::allocate_host_buffer((size_t) bytes, kSnapshotAlign));
        } catch (const std::bad_alloc&) {
            p_ = nullptr;
        }
    }
    ~Staging() { basalt::platform::release_host_buffer(p_, kSnapshotAlign); }
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    uint8_t* get() const { return p_; }

private:
    uint8_t* p_ = nullptr;
};

class Descriptor {
public:
    Descriptor(SnapshotFiles& files, int fd) : files_(files), fd_(fd) {}
    ~Descriptor() {
        if (fd_ >= 0) files_.close(fd_);
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const { return fd_; }

private:
    SnapshotFiles& files_;
    int fd_;
};

std::string fail_text(const std::string& message) { return "context snapshot: " + message; }

enum class Read { ok, unreadable, corrupt, sink_rejected };

// Streams the stored entries of `m` from `fd` through `buf`, each checked against its digest and its padding against
// zero, and hands their bytes to `sink` when there is one.
Read read_entries(SnapshotFiles& files, int fd, const Manifest& m, const SnapshotLayout& layout, uint8_t* buf,
                  StateSink* sink, std::string& error) {
    for (size_t i = 0; i < m.inventory.size(); ++i) {
        const InventoryEntry& e = m.inventory[i];
        if (e.mode != EntryMode::stored) continue;
        if (sink && !sink->begin(e)) {
            error = fail_text("the sink refused " + entry_name(e));
            return Read::sink_rejected;
        }
        StateDigest digest(e.kind, e.part, e.bytes);
        const uint64_t padded = align_up(e.bytes);
        for (uint64_t off = 0; off < padded; off += kChunk) {
            const uint64_t len = std::min(kChunk, padded - off);
            if (!files.read(fd, buf, len, layout.offsets[i] + off)) {
                error = fail_text("cannot read " + entry_name(e));
                return Read::unreadable;
            }
            const uint64_t data = off < e.bytes ? std::min(len, e.bytes - off) : 0;
            digest.update(buf, (size_t) data);
            if (!all_zero(buf + data, len - data)) {
                error = fail_text("nonzero padding after " + entry_name(e));
                return Read::corrupt;
            }
            if (sink && data > 0 && !sink->write(e, off, buf, (size_t) data)) {
                error = fail_text("the sink refused " + entry_name(e));
                return Read::sink_rejected;
            }
        }
        if (digest.finish() != e.digest) {
            error = fail_text(entry_name(e) + " digest mismatch");
            return Read::corrupt;
        }
    }
    return Read::ok;
}

// Writes the stored entries of `m` from `source` and then `head` (the manifest, whose digests are filled in as the
// entries are written) to a new file at `path`, and syncs it.
PublishStatus write_snapshot(SnapshotFiles& files, const std::string& path, Manifest& m, const SnapshotLayout& layout,
                             StateSource& source, uint8_t* buf, std::vector<uint8_t>& head, std::string& error) {
    Descriptor fd(files, files.create(path));
    if (fd.get() < 0) {
        error = fail_text("cannot create " + path);
        return PublishStatus::io_failed;
    }
    for (size_t i = 0; i < m.inventory.size(); ++i) {
        InventoryEntry& e = m.inventory[i];
        if (e.mode != EntryMode::stored) continue;
        const Digest expected = e.digest;
        StateDigest digest(e.kind, e.part, e.bytes);
        for (uint64_t off = 0; off < e.bytes; off += kChunk) {
            const uint64_t n = std::min(kChunk, e.bytes - off), len = align_up(n);
            if (!source.read(e, off, buf, (size_t) n)) {
                error = fail_text("the source refused " + entry_name(e));
                return PublishStatus::source_failed;
            }
            digest.update(buf, (size_t) n);
            std::memset(buf + n, 0, (size_t) (len - n));
            if (!files.write(fd.get(), buf, len, layout.offsets[i] + off)) {
                error = fail_text("cannot write " + path);
                return PublishStatus::io_failed;
            }
        }
        e.digest = digest.finish();
        if (!is_zero(expected) && expected != e.digest) {
            error = fail_text(entry_name(e) + " differs from its expected digest");
            return PublishStatus::source_failed;
        }
    }
    m.payload_digest = payload_digest(m.inventory);
    if (!serialize_manifest(m, head, error)) return PublishStatus::invalid;
    std::memcpy(buf, head.data(), head.size());
    std::memset(buf + head.size(), 0, (size_t) (layout.payload_offset - head.size()));
    if (!files.write(fd.get(), buf, layout.payload_offset, 0) || !files.sync(fd.get())) {
        error = fail_text("cannot write " + path);
        return PublishStatus::io_failed;
    }
    return PublishStatus::published;
}

// Reads `path` back and checks it is exactly what `write_snapshot` meant to write.
PublishStatus verify_snapshot(SnapshotFiles& files, const std::string& path, const Manifest& m,
                              const SnapshotLayout& layout, const std::vector<uint8_t>& head, uint8_t* buf,
                              std::string& error) {
    uint64_t bytes = 0;
    if (!files.size(path, bytes)) {
        error = fail_text("cannot read back " + path);
        return PublishStatus::io_failed;
    }
    if (bytes != layout.file_bytes) {
        error = fail_text(path + " reads back " + std::to_string(bytes) + " bytes, " +
                          std::to_string(layout.file_bytes) + " written");
        return PublishStatus::verify_failed;
    }
    Descriptor fd(files, files.open_read(path));
    if (fd.get() < 0 || !files.read(fd.get(), buf, layout.payload_offset, 0)) {
        error = fail_text("cannot read back " + path);
        return PublishStatus::io_failed;
    }
    if (std::memcmp(buf, head.data(), head.size()) != 0 ||
        !all_zero(buf + head.size(), layout.payload_offset - head.size())) {
        error = fail_text(path + ": the manifest reads back changed");
        return PublishStatus::verify_failed;
    }
    switch (read_entries(files, fd.get(), m, layout, buf, nullptr, error)) {
        case Read::ok: return PublishStatus::published;
        case Read::unreadable: return PublishStatus::io_failed;
        default: return PublishStatus::verify_failed;
    }
}

PublishResult publish_refused(PublishStatus status, std::string error) {
    PublishResult r;
    r.status = status;
    r.error = std::move(error);
    return r;
}

RestoreResult restore_refused(RestoreStatus status, std::string error) {
    RestoreResult r;
    r.status = status;
    r.error = std::move(error);
    return r;
}

}  // namespace

bool snapshot_layout(const Manifest& m, SnapshotLayout& out, std::string& error) {
    out = {};
    if (m.inventory.empty() || m.inventory.size() > kMaxInventoryEntries) {
        error = fail_text("inventory size out of range");
        return false;
    }
    uint64_t at = align_up(manifest_bytes(m.inventory.size()));
    out.payload_offset = at;
    out.offsets.reserve(m.inventory.size());
    for (const auto& e : m.inventory) {
        out.offsets.push_back(at);
        if (e.mode != EntryMode::stored) continue;
        if (e.bytes > kMaxEntryBytes) {
            error = fail_text(entry_name(e) + " too large");
            return false;
        }
        at += align_up(e.bytes);
    }
    out.file_bytes = at;
    return true;
}

std::string snapshot_file_name(Generation g) { return hex16(g) + kSnapshotExt; }

bool parse_snapshot_file_name(const std::string& name, Generation& g) {
    uint64_t v = 0;
    if (name.size() != kHexDigits + std::strlen(kSnapshotExt) || !ends_with(name, kSnapshotExt) ||
        !hex_prefix(name, v) || v == 0)
        return false;
    g = v;
    return true;
}

const char* publish_status_name(PublishStatus s) {
    switch (s) {
        case PublishStatus::published: return "published";
        case PublishStatus::invalid: return "invalid";
        case PublishStatus::stale: return "stale";
        case PublishStatus::source_failed: return "source_failed";
        case PublishStatus::io_failed: return "io_failed";
        case PublishStatus::verify_failed: return "verify_failed";
    }
    return "unknown";
}

const char* restore_status_name(RestoreStatus s) {
    switch (s) {
        case RestoreStatus::restored: return "restored";
        case RestoreStatus::invalid_request: return "invalid_request";
        case RestoreStatus::missing: return "missing";
        case RestoreStatus::unreadable: return "unreadable";
        case RestoreStatus::corrupt: return "corrupt";
        case RestoreStatus::mismatch: return "mismatch";
        case RestoreStatus::stale: return "stale";
        case RestoreStatus::incomplete: return "incomplete";
        case RestoreStatus::sink_rejected: return "sink_rejected";
    }
    return "unknown";
}

std::string SnapshotStore::path_of(const std::string& name) const { return dir_.empty() ? name : dir_ + "/" + name; }

bool SnapshotStore::list_generations(std::vector<Generation>& out) const {
    out.clear();
    std::vector<SnapshotDirEntry> files;
    if (!files_.list(dir_, files)) return false;
    for (const auto& f : files) {
        Generation g = 0;
        if (parse_snapshot_file_name(f.name, g)) out.push_back(g);
    }
    std::sort(out.begin(), out.end());
    return true;
}

std::vector<Generation> SnapshotStore::generations() const {
    std::vector<Generation> out;
    list_generations(out);
    return out;
}

RestoreStatus SnapshotStore::open_snapshot(Generation g, uint8_t* buf, int& fd, Manifest& m, SnapshotLayout& layout,
                                           std::string& error) const {
    fd = -1;
    const std::string path = path_of(snapshot_file_name(g));
    uint64_t file_bytes = 0;
    if (!files_.size(path, file_bytes)) {
        error = fail_text("no generation " + std::to_string(g) + " in " + dir_);
        return RestoreStatus::missing;
    }
    if (file_bytes < kSnapshotAlign) {
        error = fail_text(path + " is truncated");
        return RestoreStatus::corrupt;
    }
    fd = files_.open_read(path);
    if (fd < 0 || !files_.read(fd, buf, kSnapshotAlign, 0)) {
        error = fail_text("cannot read " + path);
        return RestoreStatus::unreadable;
    }
    uint64_t bytes = 0;
    std::string why;
    ManifestParse parsed = manifest_extent(buf, kSnapshotAlign, bytes, why);
    if (parsed == ManifestParse::ok) {
        const uint64_t head = align_up(bytes);
        if (head > file_bytes) {
            error = fail_text(path + " is truncated");
            return RestoreStatus::corrupt;
        }
        if (head > kSnapshotAlign && !files_.read(fd, buf + kSnapshotAlign, head - kSnapshotAlign, kSnapshotAlign)) {
            error = fail_text("cannot read " + path);
            return RestoreStatus::unreadable;
        }
        parsed = parse_manifest(buf, (size_t) bytes, m, why);
    }
    if (parsed != ManifestParse::ok) {
        error = path + ": " + why;
        return parsed == ManifestParse::unsupported_version ? RestoreStatus::mismatch : RestoreStatus::corrupt;
    }
    if (!snapshot_layout(m, layout, why)) {
        error = path + ": " + why;
        return RestoreStatus::corrupt;
    }
    if (layout.file_bytes != file_bytes) {
        error = fail_text(path + " holds " + std::to_string(file_bytes) + " bytes, its manifest " +
                          std::to_string(layout.file_bytes));
        return RestoreStatus::corrupt;
    }
    if (!all_zero(buf + bytes, layout.payload_offset - bytes)) {
        error = fail_text(path + ": nonzero padding after the manifest");
        return RestoreStatus::corrupt;
    }
    return RestoreStatus::restored;
}

bool SnapshotStore::read_manifest(Generation g, Manifest& out, std::string& error) const {
    out = {};
    Staging buf(kHeadMax);
    if (!buf.get()) {
        error = fail_text("no memory to read a manifest");
        return false;
    }
    int fd = -1;
    Manifest m;
    SnapshotLayout layout;
    const RestoreStatus opened = open_snapshot(g, buf.get(), fd, m, layout, error);
    Descriptor guard(files_, fd);
    if (opened != RestoreStatus::restored) return false;
    if (m.generation != g) {
        error = fail_text("the file of generation " + std::to_string(g) + " holds generation " +
                          std::to_string(m.generation));
        return false;
    }
    out = std::move(m);
    return true;
}

PublishResult SnapshotStore::publish(const Manifest& draft, StateSource& source,
                                     const StateRequirements& requirements) {
    std::string why;
    SnapshotLayout layout;
    if (!validate_manifest_draft(draft, why) || !manifest_complete(draft, requirements, why) ||
        !snapshot_layout(draft, layout, why))
        return publish_refused(PublishStatus::invalid, why);
    std::vector<Generation> have;
    if (!list_generations(have) && (!files_.make_dirs(dir_) || !list_generations(have)))
        return publish_refused(PublishStatus::io_failed, fail_text("cannot use directory " + dir_));
    auto fenced = [&] {
        return fail_text("generation " + std::to_string(have.back()) + " is published, " +
                         std::to_string(draft.generation) + " is stale");
    };
    if (!have.empty() && have.back() >= draft.generation) return publish_refused(PublishStatus::stale, fenced());
    Staging buf(kChunk);
    if (!buf.get()) return publish_refused(PublishStatus::io_failed, fail_text("no memory for the staging buffer"));

    Manifest m = draft;
    std::vector<uint8_t> head;
    const std::string tmp = path_of(temp_name(m.generation)), path = path_of(snapshot_file_name(m.generation));
    PublishStatus status = write_snapshot(files_, tmp, m, layout, source, buf.get(), head, why);
    if (status == PublishStatus::published) status = verify_snapshot(files_, tmp, m, layout, head, buf.get(), why);
    if (status == PublishStatus::published && !list_generations(have)) {
        why = fail_text("cannot list " + dir_);
        status = PublishStatus::io_failed;
    }
    if (status == PublishStatus::published && !have.empty() && have.back() >= m.generation) {
        why = fenced();
        status = PublishStatus::stale;
    }
    if (status == PublishStatus::published && !files_.rename(tmp, path)) {
        why = fail_text("cannot rename " + tmp + " to " + path);
        status = PublishStatus::io_failed;
    }
    if (status != PublishStatus::published) {
        files_.remove(tmp);
        return publish_refused(status, why);
    }
    if (!files_.sync_dir(dir_)) {
        // the rename may or may not survive a crash: take it back so no caller records it
        files_.remove(path);
        return publish_refused(PublishStatus::io_failed, fail_text("cannot sync directory " + dir_));
    }
    PublishResult r;
    r.status = PublishStatus::published;
    r.manifest = std::move(m);
    r.path = path;
    return r;
}

RestoreResult SnapshotStore::restore(const RestoreRequest& want, StateSink& sink) {
    if (!valid_context_id(want.context_id) || want.generation == 0 || is_zero(want.model_fingerprint) ||
        is_zero(want.prefix_hash) || want.prefix_tokens == 0 || !valid_requirements(want.requirements))
        return restore_refused(RestoreStatus::invalid_request, fail_text("incomplete restore request"));
    Staging buf(kChunk);
    if (!buf.get()) return restore_refused(RestoreStatus::unreadable, fail_text("no memory for the staging buffer"));
    std::string why;
    int fd = -1;
    Manifest m;
    SnapshotLayout layout;
    const RestoreStatus opened = open_snapshot(want.generation, buf.get(), fd, m, layout, why);
    Descriptor guard(files_, fd);
    if (opened != RestoreStatus::restored) return restore_refused(opened, why);
    if (m.context_id != want.context_id)
        return restore_refused(RestoreStatus::mismatch, fail_text("a snapshot of context " + m.context_id));
    if (m.generation != want.generation)
        return restore_refused(RestoreStatus::stale, fail_text("the file of generation " +
                                                               std::to_string(want.generation) + " holds generation " +
                                                               std::to_string(m.generation)));
    if (m.model_fingerprint != want.model_fingerprint)
        return restore_refused(RestoreStatus::mismatch, fail_text("a snapshot of another model fingerprint"));
    if (m.prefix_tokens != want.prefix_tokens || m.prefix_hash != want.prefix_hash)
        return restore_refused(RestoreStatus::mismatch, fail_text("a snapshot of another prefix"));
    if (!manifest_complete(m, want.requirements, why)) return restore_refused(RestoreStatus::incomplete, why);
    switch (read_entries(files_, guard.get(), m, layout, buf.get(), &sink, why)) {
        case Read::ok: break;
        case Read::unreadable: return restore_refused(RestoreStatus::unreadable, why);
        case Read::corrupt: return restore_refused(RestoreStatus::corrupt, why);
        case Read::sink_rejected: return restore_refused(RestoreStatus::sink_rejected, why);
    }
    RestoreResult r;
    r.status = RestoreStatus::restored;
    r.manifest = std::move(m);
    return r;
}

bool SnapshotStore::discard(Generation g) { return g != 0 && files_.remove(path_of(snapshot_file_name(g))); }

size_t SnapshotStore::prune(Generation keep_from) {
    std::vector<Generation> have;
    size_t removed = 0;
    if (!list_generations(have)) return 0;
    for (Generation g : have)
        if (g < keep_from && files_.remove(path_of(snapshot_file_name(g)))) ++removed;
    return removed;
}

size_t SnapshotStore::remove_stale_temporaries(int64_t min_age_seconds) {
    std::vector<SnapshotDirEntry> files;
    if (!files_.list(dir_, files)) return 0;
    const int64_t now = files_.now();
    size_t removed = 0;
    for (const auto& f : files)
        if (is_temp_name(f.name) && now - f.mtime >= min_age_seconds && files_.remove(path_of(f.name))) ++removed;
    return removed;
}

}  // namespace basalt::context
