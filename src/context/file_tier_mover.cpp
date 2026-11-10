#include "basalt/context/file_tier_mover.hpp"

#include "basalt/context/hash.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>

namespace basalt::context {
namespace {

constexpr char kMagic[8] = {'B', 'C', 'P', 'G', 'F', 'I', 'L', '1'};
constexpr size_t kHeader = 8 + 3 * 8;

void put_be64(uint8_t* p, uint64_t v) { for (int i = 7; i >= 0; --i) { p[i] = (uint8_t) v; v >>= 8; } }
uint64_t get_be64(const uint8_t* p) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v = (v << 8) | p[i]; return v; }

bool write_all(int fd, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*) data;
    while (n) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += w; n -= (size_t) w;
    }
    return true;
}
bool read_all(int fd, void* data, size_t n) {
    uint8_t* p = (uint8_t*) data;
    while (n) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}
std::string err(const char* what, const std::string& path) { return std::string(what) + " " + path + ": " + std::strerror(errno); }

}  // namespace

Digest FileTierMover::combine(const std::vector<Digest>& pages) {
    Sha256 h;
    const char tag[] = "basalt.context.page_file.v1";
    h.update(tag, sizeof tag - 1);
    h.update_be64(pages.size());
    for (const auto& d : pages) h.update(d.data(), d.size());
    return h.finish();
}

size_t FileTierMover::sweep(const std::string& dir) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return 0;
    size_t removed = 0;
    while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.find(".pages.tmp.") != std::string::npos && ::unlink((dir + "/" + n).c_str()) == 0) ++removed;
    }
    ::closedir(d);
    return removed;
}

std::string FileTierMover::path_of(const ContextId& context, Generation g) const {
    return dir_ + "/" + context + "." + std::to_string(g) + ".pages";
}

void FileTierMover::release(const ContextId& context, Tier tier) {
    if (!valid_context_id(context)) return;   // the id becomes a file name: never a path
    if (tier == Tier::host) { store_.remove(context); return; }
    if (tier != Tier::disk) return;
    DIR* d = ::opendir(dir_.c_str());
    if (!d) return;
    const std::string prefix = context + ".";
    while (dirent* e = ::readdir(d)) {
        std::string n = e->d_name;
        if (n.compare(0, prefix.size(), prefix) == 0 && n.size() > 6 && n.compare(n.size() - 6, 6, ".pages") == 0)
            ::unlink((dir_ + "/" + n).c_str());
        else if (n.compare(0, prefix.size(), prefix) == 0 && n.find(".pages.tmp.") != std::string::npos)
            ::unlink((dir_ + "/" + n).c_str());
    }
    ::closedir(d);
}

TierMover::Outcome FileTierMover::copy(const TierMove& move, const std::atomic<bool>& cancel, Digest& digest,
                                       std::string& error) {
    const size_t pb = store_.page_bytes();
    if (pb == 0) { error = "page store is not open"; return Outcome::failed; }
    if (!valid_context_id(move.context) || move.generation == 0) { error = "invalid context id or generation"; return Outcome::failed; }
    if (move.from == Tier::host && move.to == Tier::disk) {
        if (move.bytes == 0 || move.bytes > (uint64_t) 1 << 48) { error = "bytes out of range"; return Outcome::failed; }
        const size_t pages = (size_t) ((move.bytes + pb - 1) / pb);
        const std::string final_path = path_of(move.context, move.generation);
        // unique per call: two copies of one generation never share a temporary file
        static std::atomic<uint64_t> serial{0};
        const std::string tmp = final_path + ".tmp." + std::to_string((long long) ::getpid()) + "." + std::to_string(serial.fetch_add(1));
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) { error = err("open", tmp); return Outcome::failed; }
        auto fail = [&](Outcome o, std::string msg) { ::close(fd); ::unlink(tmp.c_str()); error = std::move(msg); return o; };
        uint8_t head[kHeader];
        std::memcpy(head, kMagic, 8);
        put_be64(head + 8, pb); put_be64(head + 16, pages); put_be64(head + 24, move.generation);
        if (!write_all(fd, head, sizeof head)) return fail(Outcome::failed, err("write", tmp));
        std::vector<uint8_t> buf(pb);
        std::vector<Digest> digests;
        digests.reserve(pages);
        for (size_t i = 0; i < pages; ++i) {
            if (cancel.load()) return fail(Outcome::canceled, "canceled");
            PageResult r = store_.read(move.context, i, 0, buf.data(), pb);
            if (!r) return fail(r.status == PageStatus::digest_mismatch ? Outcome::corrupt : Outcome::failed,
                                std::string("host read: ") + r.reason);
            digests.push_back(page_digest(buf.data(), pb));
            if (!write_all(fd, buf.data(), pb)) return fail(Outcome::failed, err("write", tmp));
        }
        const Digest total = combine(digests);
        if (!write_all(fd, total.data(), total.size())) return fail(Outcome::failed, err("write", tmp));
        if (::fsync(fd) != 0) return fail(Outcome::failed, err("fsync", tmp));
        ::close(fd);
        if (::rename(tmp.c_str(), final_path.c_str()) != 0) { error = err("rename", final_path); ::unlink(tmp.c_str()); return Outcome::failed; }
        int dfd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd < 0 || ::fsync(dfd) != 0) {   // not durable: take the file back so a retry starts clean
            error = err("fsync dir", dir_);
            if (dfd >= 0) ::close(dfd);
            ::unlink(final_path.c_str());
            return Outcome::failed;
        }
        ::close(dfd);
        digest = total;
        return Outcome::done;
    }
    if (move.from == Tier::disk && move.to == Tier::host) {
        const std::string path = path_of(move.context, move.generation);
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { error = err("open", path); return errno == ENOENT ? Outcome::corrupt : Outcome::failed; }
        struct Closer { int fd; ~Closer() { ::close(fd); } } closer{fd};
        uint8_t head[kHeader];
        if (!read_all(fd, head, sizeof head) || std::memcmp(head, kMagic, 8) != 0) { error = "bad snapshot header: " + path; return Outcome::corrupt; }
        if (get_be64(head + 8) != pb) { error = "page size differs from the pool"; return Outcome::corrupt; }
        if (get_be64(head + 24) != move.generation) { error = "snapshot is another generation"; return Outcome::corrupt; }
        struct stat st;
        if (::fstat(fd, &st) != 0 || (uint64_t) st.st_size < kHeader + 32) { error = "snapshot size mismatch: " + path; return Outcome::corrupt; }
        const uint64_t body = (uint64_t) st.st_size - kHeader - 32;   // the header's count is checked against the real size
        const uint64_t declared = get_be64(head + 16);
        if (declared > body / pb || declared * pb != body) { error = "snapshot size mismatch: " + path; return Outcome::corrupt; }
        const size_t pages = (size_t) declared;
        if (declared * pb < move.bytes) { error = "snapshot is shorter than the context"; return Outcome::corrupt; }
        PageResult c = store_.create(move.context, pages);
        if (!c) { error = "host create: " + c.reason; return Outcome::failed; }
        auto drop = [&](Outcome o, std::string msg) { store_.remove(move.context); error = std::move(msg); return o; };
        std::vector<uint8_t> buf(pb);
        std::vector<Digest> digests;
        digests.reserve(pages);
        for (size_t i = 0; i < pages; ++i) {
            if (cancel.load()) return drop(Outcome::canceled, "canceled");
            if (!read_all(fd, buf.data(), pb)) return drop(Outcome::corrupt, err("read", path));
            digests.push_back(page_digest(buf.data(), pb));
            PageResult w = store_.write(move.context, i, 0, buf.data(), pb);
            if (!w) return drop(Outcome::failed, "host write: " + w.reason);
        }
        Digest stored{};
        if (!read_all(fd, stored.data(), stored.size())) return drop(Outcome::corrupt, err("read", path));
        const Digest total = combine(digests);
        if (total != stored || (move.verify && total != move.digest)) return drop(Outcome::corrupt, "digest mismatch: " + path);
        digest = total;
        return Outcome::done;
    }
    error = std::string("unsupported move ") + tier_name(move.from) + "->" + tier_name(move.to);
    return Outcome::failed;
}

}  // namespace basalt::context
