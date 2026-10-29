// src/core/conversation_store.cpp - see include/basalt/core/conversation_store.hpp.
//
// Three kinds of file share one container (little-endian, every offset a multiple of 4096):
//   header (4096 B)       magic, version, file kind, identity, keys, geometry, counts, table and hash-table places
//   section table         one Section per payload part: kind, owner (a checkpoint, the live state, a K/V layer),
//                         offset, bytes, its first block hash, kind fields (a K/V layer's format and extents)
//   block hashes          one per 1 MiB block of every section, in section order
//   payload               each section at its own aligned offset, zero padded to 4096
// A conversation file `<identity>-<key>.bcs` holds the live tokens, pictures and state, the checkpoint lengths with
// their state files' keys, the K/V layers' shapes, the K/V past its last whole block and the keys of the blocks below.
// A block file `<identity>-b<key>.bcs` holds kBlockCells cells of every K/V layer, a state file `<identity>-s<key>.bcs`
// one checkpoint's state.  Both are keyed by the token prefix they belong to and by their bytes, so a conversation
// written again a few turns on, or another one that starts the same way with the same bytes, refers to the files that
// are there and writes only the new ones.  Block and state files nobody refers to are removed between writes.
#include "basalt/core/conversation_store.hpp"

#include "basalt/kernels/qsa.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/direct_file.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <limits>

namespace basalt::core {

// one aligned buffer, freed on scope exit
struct ConversationStoreBuffer {
    void* p = nullptr;
    explicit ConversationStoreBuffer(size_t n) : p(basalt::platform::DirectFile::alloc_aligned(n)) {}
    ~ConversationStoreBuffer() { basalt::platform::DirectFile::free_aligned(p); }
    ConversationStoreBuffer(const ConversationStoreBuffer&) = delete;
    ConversationStoreBuffer& operator=(const ConversationStoreBuffer&) = delete;
    uint8_t* data() const { return static_cast<uint8_t*>(p); }
};

namespace {

constexpr char kMagic[8] = {'B', 'A', 'S', 'A', 'L', 'T', 'C', 'V'};
constexpr uint64_t kAlign = 4096;
constexpr uint64_t kBlock = 1ull << 20;    // one hash per block
constexpr uint64_t kStage = 8ull << 20;    // one write
constexpr uint64_t kPiece = 4ull << 20;    // one read
constexpr int kReaders = 8;                // reads in flight
constexpr uint64_t kSpaceFloor = 4ull << 30;   // a write leaves this much of the volume free
constexpr int64_t kBlockCells = 4096;      // K/V cells per block file
constexpr int64_t kOrphanAge = 60;         // seconds: a younger unreferenced file may be another process's write
constexpr int64_t kLiveSlack = 4096;       // a live end at most this far past the last checkpoint is stored as of it
constexpr uint32_t kLive = 0xFFFFFFFFu;
const char* const kExt = ".bcs";

enum FileKind : uint32_t { kConversationFile = 0, kBlockFile = 1, kStateFile = 2 };

enum Kind : uint32_t {
    kIds = 1, kImgs = 2, kCheck = 3, kRefs = 4,
    kGdn = 10, kPle = 11, kTails = 12, kDead = 13, kBlockPos = 14,
    kKvHead = 20, kK = 21, kV = 22, kKs = 23, kVs = 24, kPooled = 25, kKvRate = 26,
};

struct Header {
    char magic[8];
    uint32_t version, header_bytes;
    uint32_t file_kind, cvec;
    uint64_t identity, key, prefix_key, file_bytes;
    int64_t geometry[18];
    int64_t layer_lo, layer_hi;
    int64_t n_ids, n_imgs, n_checks, n_kv;
    int64_t block_cells, n_blocks, position;   // a block's index, a state's tokens
    uint32_t n_sections, pad;
    uint64_t table_offset, table_bytes, hash_offset, hash_count;
    uint64_t table_hash, hashes_hash, header_hash;
};
static_assert(sizeof(Header) <= kAlign, "the header fits its block");

struct Section {
    uint32_t kind, index;
    uint64_t offset, bytes, first_hash;
    int64_t meta[8];
};
static_assert(sizeof(Section) == 96, "a fixed section record");

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint64_t blocks_of(uint64_t bytes) { return (bytes + kBlock - 1) / kBlock; }

constexpr uint64_t P1 = 11400714785074694791ull, P2 = 14029467366897019727ull, P3 = 1609587929392839161ull,
                   P4 = 9650029242287828579ull, P5 = 2870177450012600261ull;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t round1(uint64_t acc, uint64_t in) { return rotl(acc + in * P2, 31) * P1; }
inline uint64_t merge(uint64_t acc, uint64_t v) { return (acc ^ round1(0, v)) * P1 + P4; }
inline uint64_t load64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t load32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

std::string hex16(uint64_t v) {
    char b[17];
    for (int i = 15; i >= 0; --i, v >>= 4) b[i] = "0123456789abcdef"[v & 15];
    b[16] = 0;
    return b;
}

using Aligned = ConversationStoreBuffer;

// an open file for positional reads: unbuffered when the platform gives that, else cached
struct ReadFd {
    int fd = -1;
    explicit ReadFd(const std::string& path) {
        fd = basalt::platform::open_unbuffered(path);
        if (fd < 0) fd = basalt::platform::open_fd_utf8(path);
    }
    ~ReadFd() { close(); }
    void close() {   // before the file is removed: an open file cannot be deleted everywhere
        basalt::platform::close_fd(fd);
        fd = -1;
    }
    ReadFd(const ReadFd&) = delete;
    ReadFd& operator=(const ReadFd&) = delete;
};

uint64_t key_of(const int32_t* ids, size_t n, const std::vector<ConversationImageKey>& imgs, bool cvec) {
    uint64_t h = conversation_store_hash(ids, n * sizeof(int32_t), cvec ? 1 : 2);
    for (const auto& k : imgs) {
        if (k.start >= (int64_t) n) continue;
        h = conversation_store_hash(&k.start, sizeof k.start, h);
        h = conversation_store_hash(&k.hash, sizeof k.hash, h);
    }
    return h;
}

uint64_t key_of(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec) {
    return key_of(ids.data(), ids.size(), imgs, cvec);
}

// the images of `all` that start before `tokens`
std::vector<ConversationImageKey> imgs_below(const std::vector<ConversationImageKey>& all, int64_t tokens) {
    std::vector<ConversationImageKey> v;
    for (const auto& k : all) if (k.start < tokens) v.push_back(k);
    return v;
}

std::string conversation_name(uint64_t identity, uint64_t key) { return hex16(identity) + "-" + hex16(key) + kExt; }
std::string part_name(uint64_t identity, char kind, uint64_t key) {
    return hex16(identity) + "-" + kind + hex16(key) + kExt;
}

// a part file's key: its token prefix and its bytes (the sections' shapes and their block hashes)
uint64_t content_key(const std::vector<Section>& table, const std::vector<uint64_t>& hashes, uint64_t prefix_key) {
    uint64_t h = 17;
    for (const Section& s : table) {
        const uint64_t shape[3] = {s.kind, s.index, s.bytes};
        h = conversation_store_hash(shape, sizeof shape, h);
    }
    h = conversation_store_hash(hashes.data(), hashes.size() * 8, h);
    return conversation_store_hash(&h, sizeof h, prefix_key);
}

// one section to write and where its bytes come from
struct Src {
    Section s{};
    std::function<void(uint64_t, uint8_t*, uint64_t)> copy;            // (offset, destination, count)
    std::function<const uint8_t*(uint64_t, uint64_t)> view;            // the bytes in place when contiguous, or null
};
using Srcs = std::vector<Src>;

Section& add(Srcs& v, uint32_t kind, uint32_t index, uint64_t n, std::function<void(uint64_t, uint8_t*, uint64_t)> copy,
             std::function<const uint8_t*(uint64_t, uint64_t)> view = nullptr) {
    Src s;
    std::memset(&s.s, 0, sizeof s.s);
    s.s.kind = kind;
    s.s.index = index;
    s.s.bytes = n;
    s.copy = std::move(copy);
    s.view = std::move(view);
    v.push_back(std::move(s));
    return v.back().s;
}

// a K/V buffer's bytes from `base` on
Section& add_kv(Srcs& v, uint32_t kind, uint32_t index, uint64_t n, const ConversationBuffer* buf, uint64_t base) {
    return add(v, kind, index, n, [buf, base](uint64_t o, uint8_t* d, uint64_t c) { buf->read(d, base + o, c); },
               [buf, base](uint64_t o, uint64_t c) -> const uint8_t* {
                   const uint8_t* r = nullptr;
                   buf->visit(base + o, c, [&](const uint8_t* p, size_t got, size_t) {
                       if (got == c) r = p;
                       return false;   // the first piece only: contiguous or not
                   });
                   return r;
               });
}

void add_state(Srcs& v, const ConversationCheckpoint& c, uint32_t index) {
    const CheckpointBytes* f[5] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
    for (int k = 0; k < 5; ++k) {
        const CheckpointBytes* b = f[k];
        add(v, kGdn + k, index, b->size(), [b](uint64_t o, uint8_t* d, uint64_t n) { std::memcpy(d, b->data() + o, n); },
            [b](uint64_t o, uint64_t) -> const uint8_t* { return b->data() + o; });
    }
}

// the table of `srcs` with each section's first block hash; the number of block hashes
uint64_t number_blocks(Srcs& srcs) {
    uint64_t n = 0;
    for (auto& s : srcs) { s.s.first_hash = n; n += blocks_of(s.s.bytes); }
    return n;
}

std::vector<Section> table_of(const Srcs& srcs) {
    std::vector<Section> t;
    for (const auto& s : srcs) t.push_back(s.s);
    return t;
}

// the block hashes of `srcs` (numbered): read in place where the source is contiguous, else copied to `stage`
void hash_srcs(const Srcs& srcs, Aligned& stage, std::vector<uint64_t>& hashes) {
    uint64_t n_hashes = 0;
    for (const auto& s : srcs) n_hashes += blocks_of(s.s.bytes);
    hashes.assign(n_hashes, 0);
    for (const auto& s : srcs)
        for (uint64_t from = 0; from < s.s.bytes; from += kBlock) {
            const uint64_t n = std::min(kBlock, s.s.bytes - from);
            const uint8_t* p = s.view ? s.view(from, n) : nullptr;
            if (p == nullptr) {
                s.copy(from, stage.data(), n);
                p = stage.data();
            }
            hashes[s.s.first_hash + from / kBlock] = conversation_store_hash(p, n, s.s.kind * 131 + s.s.index);
        }
}

// Writes `srcs` (numbered) under `h` (kind, identity, keys and counts set) to `path` through a temporary file in the
// same directory.  `hashes`: the block hashes when known, else computed on the way.  The file's size in `file_bytes`.
bool write_container(const std::string& path, Header h, Srcs& srcs, std::vector<uint64_t>* known, Aligned& stage,
                     uint64_t& file_bytes, std::string& error) {
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kConversationFormat;
    h.header_bytes = kAlign;
    h.n_sections = (uint32_t) srcs.size();
    h.table_offset = kAlign;
    h.table_bytes = (uint64_t) srcs.size() * sizeof(Section);
    h.hash_offset = align_up(h.table_offset + h.table_bytes, kAlign);
    h.hash_count = number_blocks(srcs);
    uint64_t at = align_up(h.hash_offset + h.hash_count * 8, kAlign);
    const uint64_t payload_at = at;
    for (auto& s : srcs) { s.s.offset = at; at = align_up(at + s.s.bytes, kAlign); }
    h.file_bytes = at;
    file_bytes = at;
    const std::string tmp = path + "." + std::to_string((long long) basalt::platform::process_id()) + ".tmp";
    int fd = basalt::platform::open_write_unbuffered(tmp);
    if (fd < 0) fd = basalt::platform::open_write_utf8(tmp);
    if (fd < 0) { error = "cannot create " + tmp; return false; }
    std::vector<uint64_t> own;
    std::vector<uint64_t>& hashes = known ? *known : own;
    if (!known) hashes.assign(h.hash_count, 0);
    bool ok = stage.p != nullptr && hashes.size() == h.hash_count;
    for (const auto& s : srcs) {
        for (uint64_t from = 0; ok && from < s.s.bytes; from += kStage) {
            const uint64_t n = std::min(kStage, s.s.bytes - from), len = align_up(n, kAlign);
            s.copy(from, stage.data(), n);
            if (len > n) std::memset(stage.data() + n, 0, len - n);
            if (!known)
                for (uint64_t b = 0; b < n; b += kBlock)
                    hashes[s.s.first_hash + (from + b) / kBlock] =
                        conversation_store_hash(stage.data() + b, std::min(kBlock, n - b), s.s.kind * 131 + s.s.index);
            ok = basalt::platform::pwrite_all(fd, stage.data(), len, s.s.offset + from);
        }
    }
    const std::vector<Section> table = table_of(srcs);
    h.table_hash = conversation_store_hash(table.data(), h.table_bytes, 11);
    h.hashes_hash = conversation_store_hash(hashes.data(), h.hash_count * 8, 13);
    h.header_hash = 0;
    h.header_hash = conversation_store_hash(&h, sizeof h, 7);
    if (ok) {
        Aligned meta(payload_at);
        ok = meta.p != nullptr;
        if (ok) {
            std::memset(meta.data(), 0, payload_at);
            std::memcpy(meta.data(), &h, sizeof h);
            std::memcpy(meta.data() + h.table_offset, table.data(), h.table_bytes);
            std::memcpy(meta.data() + h.hash_offset, hashes.data(), h.hash_count * 8);
            ok = basalt::platform::pwrite_all(fd, meta.data(), payload_at, 0);
        }
    }
    ok = ok && basalt::platform::sync_fd(fd);
    basalt::platform::close_fd(fd);
    if (ok) ok = basalt::platform::rename_replace(tmp, path);
    if (!ok) {
        basalt::platform::remove_file(tmp);
        error = "cannot write " + tmp;
    }
    return ok;
}

}  // namespace

uint64_t conversation_store_hash(const void* data, size_t n, uint64_t seed) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint8_t* const end = p + n;
    uint64_t h;
    if (n >= 32) {
        uint64_t a = seed + P1 + P2, b = seed + P2, c = seed, d = seed - P1;
        const uint8_t* const limit = end - 32;
        do {
            a = round1(a, load64(p)); b = round1(b, load64(p + 8));
            c = round1(c, load64(p + 16)); d = round1(d, load64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl(a, 1) + rotl(b, 7) + rotl(c, 12) + rotl(d, 18);
        h = merge(h, a); h = merge(h, b); h = merge(h, c); h = merge(h, d);
    } else {
        h = seed + P5;
    }
    h += (uint64_t) n;
    for (; p + 8 <= end; p += 8) h = rotl(h ^ round1(0, load64(p)), 27) * P1 + P4;
    if (p + 4 <= end) { h = rotl(h ^ (uint64_t) load32(p) * P1, 23) * P2 + P3; p += 4; }
    for (; p < end; ++p) h = rotl(h ^ (uint64_t) *p * P5, 11) * P1;
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

// ------------------------------------------------------------------------------------------------ index entries

struct ConversationStore::Job {
    SavedConversation image;
    std::string name;
    uint64_t key = 0, bytes = 0;
    std::vector<std::string> held;            ///< part files it refers to, counted while it is written
    std::vector<uint64_t> blocks, check_keys;
};

struct ConversationStore::Entry {
    std::string name;                         ///< file name in the directory
    uint64_t key = 0, bytes = 0, used = 0;
    bool mine = true;                         ///< written under this identity (others count for the budget only)
    bool cvec = true;
    std::vector<int32_t> ids;                 ///< live tokens
    std::vector<ConversationImageKey> imgs;   ///< live pictures
    std::vector<int64_t> checks;              ///< checkpoint lengths, in stored order
    std::vector<uint64_t> check_keys;         ///< each checkpoint's state file
    std::vector<uint64_t> blocks;             ///< the K/V block files, in cell order
    std::vector<std::string> refs;            ///< the part files counted for it
    std::vector<Section> table;               ///< the file's sections (empty while queued)
    Header head{};
    std::shared_ptr<Job> job;                 ///< queued, not yet written
};

// ------------------------------------------------------------------------------------------------ file reading

namespace {

bool read_exact(int fd, void* dst, uint64_t n, uint64_t off) {
    return basalt::platform::pread_all(fd, dst, n, off);
}

// header, table and hashes of an open file; false when any check fails
bool read_index(int fd, uint64_t file_bytes, Header& h, std::vector<Section>& table, std::vector<uint64_t>& hashes,
                std::string& error) {
    Aligned buf(kAlign);
    if (!buf.p || !read_exact(fd, buf.data(), kAlign, 0)) { error = "unreadable header"; return false; }
    std::memcpy(&h, buf.data(), sizeof h);
    Header z = h;
    z.header_hash = 0;
    if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kConversationFormat || h.header_bytes != kAlign ||
        conversation_store_hash(&z, sizeof z, 7) != h.header_hash) { error = "not a conversation file"; return false; }
    const bool conv = h.file_kind == kConversationFile;
    if (h.file_bytes != file_bytes || h.n_sections == 0 || h.n_sections > 100000 || h.file_kind > kStateFile ||
        h.table_offset != kAlign || h.table_bytes != (uint64_t) h.n_sections * sizeof(Section) ||
        h.hash_offset != align_up(h.table_offset + h.table_bytes, kAlign) || h.hash_count > file_bytes / 8 ||
        (conv && (h.n_ids <= 0 || h.n_ids > (int64_t) (file_bytes / 4))) || h.n_imgs < 0 || h.n_checks < 0 ||
        h.n_kv < 0 || h.n_blocks < 0 || h.block_cells < 0) {
        error = "inconsistent header";
        return false;
    }
    const uint64_t meta_end = align_up(h.hash_offset + h.hash_count * 8, kAlign);
    if (meta_end > file_bytes) { error = "truncated file"; return false; }
    const uint64_t span = meta_end - kAlign;
    Aligned meta(span);
    if (!meta.p || !read_exact(fd, meta.data(), span, kAlign)) { error = "unreadable section table"; return false; }
    table.resize(h.n_sections);
    std::memcpy(table.data(), meta.data(), h.table_bytes);
    hashes.resize(h.hash_count);
    std::memcpy(hashes.data(), meta.data() + (h.hash_offset - kAlign), h.hash_count * 8);
    if (conversation_store_hash(table.data(), h.table_bytes, 11) != h.table_hash ||
        conversation_store_hash(hashes.data(), h.hash_count * 8, 13) != h.hashes_hash) {
        error = "section table hash mismatch";
        return false;
    }
    uint64_t at = meta_end, next_hash = 0;
    for (const Section& s : table) {
        if (s.offset % kAlign || s.offset < at || s.bytes > file_bytes || s.offset > file_bytes - s.bytes ||
            s.first_hash != next_hash || blocks_of(s.bytes) > h.hash_count - next_hash) {
            error = "inconsistent section table";
            return false;
        }
        next_hash += blocks_of(s.bytes);
        at = align_up(s.offset + s.bytes, kAlign);
        if (at > file_bytes) { error = "truncated payload"; return false; }
    }
    if (next_hash != h.hash_count) { error = "inconsistent block hashes"; return false; }
    return true;
}

// an open file of a read: its descriptor and block hashes
struct Source {
    int fd = -1;
    const std::vector<uint64_t>* hashes = nullptr;
};

// One part to read: section bytes [0, need) of file `file` into `dst(offset, src, n)`, verified block by block.
struct Part {
    size_t file = 0;
    const Section* s = nullptr;
    uint64_t need = 0;
    std::function<void(uint64_t, const uint8_t*, uint64_t)> dst;
};

// Reads the parts with a few threads (each its own aligned buffer of kPiece bytes from `bufs`, allocated on first
// use and kept); false at the first failed read or block hash, with the file it was in.
bool read_parts(const std::vector<Source>& files, const std::vector<Part>& parts, uint64_t& bytes_read,
                std::string& error, size_t& bad_file, std::vector<std::unique_ptr<Aligned>>& bufs) {
    struct Piece { const Part* part; uint64_t from, to; };   // [from, to): whole blocks of one section
    std::vector<Piece> pieces;
    for (const Part& p : parts) {
        if (p.need == 0) continue;
        const uint64_t end = std::min(p.s->bytes, blocks_of(p.need) * kBlock);
        for (uint64_t from = 0; from < end; from += kPiece) pieces.push_back({&p, from, std::min(end, from + kPiece)});
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> got{0};
    std::mutex emu;
    const int n_threads = (int) std::min<size_t>(kReaders, std::max<size_t>(1, pieces.size()));
    while ((int) bufs.size() < n_threads) bufs.push_back(std::make_unique<Aligned>(kPiece));
    auto fail = [&](const char* why, size_t file) {
        failed = true;
        std::lock_guard<std::mutex> lk(emu);
        error = why;
        bad_file = file;
    };
    auto work = [&](int t) {
        Aligned& buf = *bufs[(size_t) t];
        if (!buf.p) { fail("no read buffer", SIZE_MAX); return; }
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (failed || i >= pieces.size()) break;
            const Piece& pc = pieces[i];
            const Section& s = *pc.part->s;
            const Source& f = files[pc.part->file];
            const uint64_t n = pc.to - pc.from, len = align_up(n, kAlign);
            if (!read_exact(f.fd, buf.data(), len, s.offset + pc.from)) { fail("read failed", pc.part->file); return; }
            for (uint64_t b = 0; b < n; b += kBlock) {
                const uint64_t bn = std::min(kBlock, n - b);
                const uint64_t idx = s.first_hash + (pc.from + b) / kBlock;
                if (conversation_store_hash(buf.data() + b, bn, s.kind * 131 + s.index) != (*f.hashes)[idx]) {
                    fail("payload hash mismatch", pc.part->file);
                    return;
                }
            }
            const uint64_t want = std::min(pc.to, pc.part->need);
            if (want > pc.from) pc.part->dst(pc.from, buf.data(), want - pc.from);
            got += len;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < n_threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& t : pool) t.join();
    bytes_read += got;
    return !failed;
}

// the K/V shapes a conversation file records: a layer's bytes per cell (k, v and their scales) and per pooled row
struct Rate {
    uint64_t cell[4] = {0, 0, 0, 0};
    uint64_t row = 0;
};

// Name parts: `<16 hex>-<16 hex>.bcs` a conversation, `<16 hex>-b<16 hex>.bcs` a block, `-s` a state.
enum class NameKind { kOther, kConversation, kBlock, kState };
NameKind name_kind(const std::string& name, const std::string& prefix) {
    if (name.rfind(prefix, 0) != 0) return NameKind::kOther;
    const size_t base = prefix.size();
    if (name.size() == base + 16 + 4) return NameKind::kConversation;
    if (name.size() == base + 17 + 4 && name[base] == 'b') return NameKind::kBlock;
    if (name.size() == base + 17 + 4 && name[base] == 's') return NameKind::kState;
    return NameKind::kOther;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ open and index

ConversationStore::ConversationStore() = default;

ConversationStore::~ConversationStore() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool ConversationStore::open(const std::string& dir, uint64_t identity, uint64_t budget, std::string& error) {
    if (open_) { error = "conversation store: already open"; return false; }
    if (dir.empty() || !basalt::platform::make_dirs(dir)) {
        error = "conversation store: cannot create the directory " + dir;
        return false;
    }
    dir_ = dir;
    identity_ = identity;
    budget_ = budget;
    std::vector<basalt::platform::DirFile> files;
    if (!basalt::platform::list_files(dir_, files)) {
        error = "conversation store: cannot list " + dir;
        return false;
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.mtime < b.mtime; });
    const int64_t now = basalt::platform::file_time_now();
    const std::string prefix = hex16(identity_) + "-";
    for (const auto& f : files) {
        const std::string path = dir_ + "/" + f.name;
        if (f.name.size() > 4 && f.name.compare(f.name.size() - 4, 4, ".tmp") == 0) {
            // a write cut short an hour ago or more (a newer one may be another process's write in progress)
            if (now - f.mtime >= 3600) basalt::platform::remove_file(path);
            continue;
        }
        if (f.name.size() <= 4 || f.name.compare(f.name.size() - 4, 4, kExt) != 0) continue;
        const NameKind nk = name_kind(f.name, prefix);
        if (nk == NameKind::kBlock || nk == NameKind::kState) {   // checked when a conversation reads it
            PartFile& p = parts_[f.name];
            p.bytes = f.bytes;
            p.mtime = f.mtime;
            continue;
        }
        auto e = std::make_unique<Entry>();
        e->name = f.name;
        e->bytes = f.bytes;
        e->used = ++clock_;
        e->mine = nk == NameKind::kConversation;
        if (e->mine) {
            std::string why;
            ReadFd rf(path);
            std::vector<uint64_t> hashes;
            bool ok = rf.fd >= 0 && read_index(rf.fd, f.bytes, e->head, e->table, hashes, why) &&
                      e->head.identity == identity_ && e->head.file_kind == kConversationFile;
            if (ok) {
                const Section* ids = nullptr;
                const Section* imgs = nullptr;
                const Section* refs = nullptr;
                for (const Section& s : e->table) {
                    if (s.kind == kIds) ids = &s;
                    if (s.kind == kImgs) imgs = &s;
                    if (s.kind == kRefs) refs = &s;
                    if (s.kind == kCheck) {
                        e->checks.push_back(s.meta[0]);
                        e->check_keys.push_back((uint64_t) s.meta[2]);
                    }
                }
                ok = ids && imgs && refs && ids->bytes == (uint64_t) e->head.n_ids * 4 &&
                     imgs->bytes == (uint64_t) e->head.n_imgs * sizeof(ConversationImageKey) &&
                     refs->bytes == (uint64_t) e->head.n_blocks * 8 &&
                     (int64_t) e->checks.size() == e->head.n_checks;
                if (ok) {
                    e->ids.resize((size_t) e->head.n_ids);
                    e->imgs.resize((size_t) e->head.n_imgs);
                    e->blocks.resize((size_t) e->head.n_blocks);
                    std::vector<Part> parts(3);
                    parts[0] = {0, ids, ids->bytes, [&](uint64_t at, const uint8_t* p, uint64_t n) {
                                    std::memcpy(reinterpret_cast<uint8_t*>(e->ids.data()) + at, p, n); }};
                    parts[1] = {0, imgs, imgs->bytes, [&](uint64_t at, const uint8_t* p, uint64_t n) {
                                    std::memcpy(reinterpret_cast<uint8_t*>(e->imgs.data()) + at, p, n); }};
                    parts[2] = {0, refs, refs->bytes, [&](uint64_t at, const uint8_t* p, uint64_t n) {
                                    std::memcpy(reinterpret_cast<uint8_t*>(e->blocks.data()) + at, p, n); }};
                    uint64_t read = 0;
                    size_t bad = 0;
                    ok = read_parts({Source{rf.fd, &hashes}}, parts, read, why, bad, read_bufs_);
                }
                ok = ok && key_of(e->ids, e->imgs, e->head.cvec != 0) == e->head.key;
                for (int64_t c : e->checks) ok = ok && c > 0 && c <= e->head.n_ids;
            }
            if (!ok) {   // fail closed: a file of this identity that does not check out is removed
                basalt::log::warn("basalt serve", "conversation cache: %s dropped (%s)", f.name.c_str(),
                                  why.empty() ? "its tokens do not match its key" : why.c_str());
                rf.close();
                basalt::platform::remove_file(path);
                ++stats_.dropped;
                continue;
            }
            e->key = e->head.key;
            e->cvec = e->head.cvec != 0;
            for (uint64_t k : e->blocks) e->refs.push_back(part_name(identity_, 'b', k));
            for (uint64_t k : e->check_keys) e->refs.push_back(part_name(identity_, 's', k));
        }
        entries_.push_back(std::move(e));
    }
    // a conversation whose block or state files are gone is gone too; the rest count their parts
    for (size_t i = 0; i < entries_.size();) {
        Entry& e = *entries_[i];
        bool whole = true;
        for (const auto& r : e.refs) whole = whole && parts_.count(r) != 0;
        if (!whole) {
            basalt::log::warn("basalt serve", "conversation cache: %s dropped (a block or state file is missing)",
                              e.name.c_str());
            basalt::platform::remove_file(dir_ + "/" + e.name);
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            ++stats_.dropped;
            continue;
        }
        for (const auto& r : e.refs) ++parts_[r].refs;
        ++i;
    }
    open_ = true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gc_locked(kOrphanAge);
        evict_locked();
    }
    thread_ = std::thread([this] { writer(); });
    return true;
}

// ------------------------------------------------------------------------------------------------ matching

ConversationStore::Match ConversationStore::best(const int32_t* prompt, size_t n,
                                                 const std::vector<ConversationImageKey>& images, bool cvec) const {
    Match best;
    if (!open_ || n < 2) return best;
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t best_used = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry& e = *entries_[i];
        if (!e.mine || e.cvec != cvec || e.ids.empty()) continue;
        // the common prefix, in bulk first
        const size_t lim = std::min(e.ids.size(), n - 1);
        size_t common = 0;
        constexpr size_t kChunk = 4096;
        while (common + kChunk <= lim && std::memcmp(e.ids.data() + common, prompt + common, kChunk * 4) == 0) common += kChunk;
        while (common < lim && e.ids[common] == prompt[common]) ++common;
        auto usable = [&](int64_t len) {
            if (len <= 0 || (size_t) len > common) return false;
            // the pictures below `len` must be the same ones at the same places
            size_t j = 0;
            for (const auto& im : images) {
                if (im.start >= len) continue;
                if (j == e.imgs.size() || !(e.imgs[j++] == im)) return false;
            }
            return j == imgs_below(e.imgs, len).size();
        };
        auto consider = [&](int64_t len) {
            if (usable(len) && (len > best.tokens || (len == best.tokens && e.used > best_used))) {
                best = {i, len};
                best_used = e.used;
            }
        };
        consider((int64_t) e.ids.size());
        for (int64_t c : e.checks) consider(c);
    }
    return best;
}

bool ConversationStore::holds(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                              bool cvec) const {
    if (!open_) return false;
    const uint64_t key = key_of(ids, images, cvec);
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& e : entries_)
        if (e->mine && e->key == key && e->ids == ids && e->imgs == images && e->cvec == cvec) return true;
    return false;
}

// ------------------------------------------------------------------------------------------------ loading

namespace {

// What a cut at `tokens` reads: the state there (a checkpoint's ordinal, else -1: the stored live state), the
// K/V up to it and the chain's root (the shortest checkpoint, the end of the system prompt in practice: a new chat or
// an edited history goes on from it).  The checkpoints between stay on disk for the prompts that match them.
struct Cut {
    int live = -1;
    int root = -1;
};

Cut plan_cut(const std::vector<int64_t>& checks, int64_t tokens) {
    Cut c;
    for (size_t j = 0; j < checks.size(); ++j)
        if (checks[j] == tokens) c.live = (int) j;
    int64_t shortest = tokens;
    for (size_t j = 0; j < checks.size(); ++j)
        if (checks[j] < shortest) { shortest = checks[j]; c.root = (int) j; }
    return c;
}

// bytes of a K/V buffer for `cells` cells (data and scales scale with the cells; pooled rows with the indexer blocks)
uint64_t kv_part_bytes(uint32_t kind, uint64_t stored, int64_t stored_cells, int64_t cells, int64_t rows,
                       int64_t idx_dim) {
    if (kind == kPooled) return rows > 0 ? (uint64_t) rows * (uint64_t) idx_dim * sizeof(float) : 0;
    if (stored_cells <= 0) return 0;
    return stored / (uint64_t) stored_cells * (uint64_t) cells;
}

}  // namespace

uint64_t ConversationStore::load_bytes(const Match& m) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (m.index >= entries_.size()) return 0;
    const Entry& e = *entries_[m.index];
    if (e.job) return e.job->image.bytes();   // an upper bound for a copy out of the queue
    uint64_t n = (uint64_t) m.tokens * 4 * 3, state = 0;
    for (const Section& s : e.table) {
        if (s.kind >= kGdn && s.kind <= kBlockPos && s.index == kLive) state += s.bytes;
        if (s.kind == kKvRate) {   // the cells up to the match, at most the stored ones
            const int64_t cells = std::min<int64_t>(m.tokens + 4, s.meta[5]);
            n += (s.meta[0] + s.meta[1] + s.meta[2] + s.meta[3]) * (uint64_t) cells +
                 (uint64_t) s.meta[4] * (uint64_t) (cells / std::max<int64_t>(1, basalt::kernels::qsa_real_shapes().idx_block) + 1);
        }
    }
    if (state == 0)   // stored as of a checkpoint: its state file's size
        for (uint64_t k : e.check_keys) {
            auto it = parts_.find(part_name(identity_, 's', k));
            if (it != parts_.end()) state = std::max(state, it->second.bytes);
        }
    return n + state * 3;   // the state at the match, once more as its checkpoint, and the root
}

bool ConversationStore::load(const Match& m, SavedConversation& out, std::string& error) {
    std::shared_ptr<Job> job;
    std::string name;
    Header head{};
    std::vector<Section> table;
    std::vector<int32_t> all_ids, ids;
    std::vector<ConversationImageKey> all_imgs, imgs;
    std::vector<int64_t> checks;
    std::vector<uint64_t> check_keys, blocks;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (m.index >= entries_.size() || m.tokens <= 0) { error = "conversation store: no such entry"; return false; }
        Entry& e = *entries_[m.index];
        e.used = ++clock_;
        job = e.job;
        name = e.name;
        head = e.head;
        table = e.table;
        all_ids = e.ids;
        all_imgs = e.imgs;
        checks = e.checks;
        check_keys = e.check_keys;
        blocks = e.blocks;
    }
    ids.assign(all_ids.begin(), all_ids.begin() + std::min<size_t>(all_ids.size(), (size_t) m.tokens));
    imgs = imgs_below(all_imgs, m.tokens);
    const int64_t P = m.tokens;
    if ((size_t) P != ids.size()) { error = "conversation store: the match is longer than the image"; return false; }
    SavedConversation img;
    img.live.ids = ids;
    img.live.imgs = imgs;
    if (job) {
        // still queued: the parts come out of the image in RAM (the writer only reads it too)
        const SavedConversation& src = job->image;
        const int64_t n_ids = (int64_t) src.live.ids.size();
        std::vector<int64_t> lens;
        for (const auto& c : src.checkpoints) lens.push_back((int64_t) c.ids.size());
        const Cut cut = plan_cut(lens, P);
        if (P != n_ids && cut.live < 0) { error = "conversation store: no state at the match"; return false; }
        const ConversationCheckpoint& state = cut.live < 0 ? src.live : src.checkpoints[(size_t) cut.live];
        auto copy_state = [&](const ConversationCheckpoint& from, ConversationCheckpoint& to) {
            to.gdn = from.gdn; to.ple = from.ple; to.tails = from.tails; to.dead = from.dead; to.block_pos = from.block_pos;
        };
        copy_state(state, img.live);
        img.geometry = src.geometry;
        img.layer_lo = src.layer_lo;
        img.layer_hi = src.layer_hi;
        img.cvec = src.cvec;
        if (cut.root >= 0) {
            ConversationCheckpoint root = src.checkpoints[(size_t) cut.root];
            root.used = 0;
            img.checkpoints.push_back(std::move(root));
        }
        if (cut.live >= 0) {
            ConversationCheckpoint at;
            at.ids = img.live.ids;
            at.imgs = img.live.imgs;
            copy_state(state, at);
            img.checkpoints.push_back(std::move(at));
        }
        const auto& shapes = basalt::kernels::qsa_real_shapes();
        for (const ConversationKv& kv : src.kv) {
            ConversationKv k;
            k.format = kv.format; k.heads = kv.heads; k.head_dim = kv.head_dim; k.page_size = kv.page_size;
            k.idx_dim = kv.idx_dim;
            k.cells = kv.page_size > 0 ? (P + kv.page_size - 1) / kv.page_size * kv.page_size : 0;
            k.pooled_rows = kv.pooled_rows > 0 ? P / shapes.idx_block + 1 : 0;
            const ConversationBuffer* from[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
            ConversationBuffer* to[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (int b = 0; b < 5; ++b) {
                const uint64_t n = kv_part_bytes((uint32_t) (kK + b), from[b]->size(), kv.cells, k.cells,
                                                 k.pooled_rows, kv.idx_dim);
                if (n > from[b]->size()) { error = "conversation store: K/V shorter than the match"; return false; }
                to[b]->resize_for_overwrite(n);
                to[b]->visit(0, n, [&](uint8_t* p, size_t c, size_t at) { return from[b]->read(p, at, c); });
            }
            img.kv.push_back(std::move(k));
        }
        out = std::move(img);
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.restored;
        return true;
    }
    // the files of the read: the conversation's, then each block and state file it needs
    struct Opened {
        std::string name;
        std::unique_ptr<ReadFd> rf;
        Header h{};
        std::vector<Section> tab;
        std::vector<uint64_t> hashes;
    };
    std::vector<Opened> opened;
    opened.reserve(3 + blocks.size());   // no reallocation: parts point into the tables
    auto fail = [&](const std::string& why, size_t bad) {
        for (auto& f : opened) f.rf->close();
        error = "conversation store: " + name + ": " + why;
        std::lock_guard<std::mutex> lk(mu_);
        if (bad != SIZE_MAX && bad > 0 && bad < opened.size()) {
            // a block or state file that does not check out goes; every conversation that refers to it fails its read
            basalt::platform::remove_file(dir_ + "/" + opened[bad].name);
            parts_.erase(opened[bad].name);
            error += " (" + opened[bad].name + ")";
        }
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i]->name == name && !entries_[i]->job) { drop_locked(i); break; }
        ++stats_.dropped;
        return false;
    };
    auto open_file = [&](const std::string& fname, uint32_t kind, uint64_t key) -> bool {
        Opened f;
        f.name = fname;
        const std::string path = dir_ + "/" + fname;
        f.rf = std::make_unique<ReadFd>(path);
        uint64_t file_bytes = 0;
        std::string why;
        const bool ok = f.rf->fd >= 0 && basalt::platform::file_size(path, file_bytes) &&
                        read_index(f.rf->fd, file_bytes, f.h, f.tab, f.hashes, why) && f.h.identity == identity_ &&
                        f.h.file_kind == kind && f.h.key == key;
        opened.push_back(std::move(f));
        if (!ok) error = why.empty() ? "a file of the read does not check out" : why;
        return ok;
    };
    if (!open_file(name, kConversationFile, head.key)) return fail(error, SIZE_MAX);
    const Header h = opened[0].h;
    if (std::memcmp(&h, &head, sizeof h) != 0) return fail("changed on disk", SIZE_MAX);
    const Cut cut = plan_cut(checks, P);
    if (P != h.n_ids && cut.live < 0) return fail("no state at the match", SIZE_MAX);
    std::copy(std::begin(h.geometry), std::end(h.geometry), img.geometry.begin());
    img.layer_lo = h.layer_lo;
    img.layer_hi = h.layer_hi;
    img.cvec = h.cvec != 0;
    const bool cvec = h.cvec != 0;
    // a block or state file checks out against the token prefix it belongs to as well as its bytes
    auto open_part = [&](char kind, uint64_t key, int64_t prefix_tokens, int64_t position) -> bool {
        if (!open_file(part_name(identity_, kind, key), kind == 'b' ? kBlockFile : kStateFile, key)) return false;
        const Opened& f = opened.back();
        const uint64_t prefix = key_of(all_ids.data(), (size_t) prefix_tokens, all_imgs, cvec);
        if (f.h.prefix_key != prefix || f.h.position != position ||
            content_key(f.tab, f.hashes, prefix) != key) {
            error = "its bytes or its tokens do not match its key";
            return false;
        }
        return true;
    };
    // the root, and the state at the match: the live one, and a checkpoint of the chain too
    ConversationCheckpoint root, at;
    if (cut.root >= 0) {
        root.ids.assign(ids.begin(), ids.begin() + checks[(size_t) cut.root]);
        root.imgs = imgs_below(imgs, checks[(size_t) cut.root]);
    }
    std::vector<Part> parts;
    auto state_field = [](ConversationCheckpoint& c, uint32_t kind) -> CheckpointBytes& {
        switch (kind) {
            case kGdn: return c.gdn;
            case kPle: return c.ple;
            case kTails: return c.tails;
            case kDead: return c.dead;
            default: return c.block_pos;
        }
    };
    auto read_state = [&](size_t file, uint32_t index, ConversationCheckpoint& c) {
        for (const Section& s : opened[file].tab) {
            if (s.kind < kGdn || s.kind > kBlockPos || s.index != index) continue;
            CheckpointBytes& d = state_field(c, s.kind);
            d.resize(s.bytes);
            parts.push_back({file, &s, s.bytes, [&d](uint64_t o, const uint8_t* p, uint64_t n) { std::memcpy(d.data() + o, p, n); }});
        }
    };
    if (cut.live < 0) {
        bool live = false;
        for (const Section& s : opened[0].tab) live = live || (s.kind >= kGdn && s.kind <= kBlockPos && s.index == kLive);
        if (!live) return fail("no state at the match", SIZE_MAX);
        read_state(0, kLive, img.live);
    }
    else {
        if (!open_part('s', check_keys[(size_t) cut.live], checks[(size_t) cut.live], checks[(size_t) cut.live]))
            return fail(error, opened.size() - 1);
        read_state(opened.size() - 1, 0, img.live);
    }
    if (cut.root >= 0) {
        if (!open_part('s', check_keys[(size_t) cut.root], checks[(size_t) cut.root], checks[(size_t) cut.root]))
            return fail(error, opened.size() - 1);
        read_state(opened.size() - 1, 0, root);
    }
    // the K/V: each layer's shapes and rates, then its bytes from the blocks below the tail and from the tail
    img.kv.resize((size_t) h.n_kv);
    std::vector<int64_t> kv_cells((size_t) h.n_kv, -1), kv_rows((size_t) h.n_kv, -1);
    std::vector<Rate> rates((size_t) h.n_kv);
    std::vector<uint8_t> rated((size_t) h.n_kv, 0);
    const auto& shapes = basalt::kernels::qsa_real_shapes();
    const int64_t B = h.block_cells, tail_from = h.n_blocks * h.block_cells;
    for (const Section& s : opened[0].tab) {
        if (s.kind == kKvHead) {
            if (s.index >= (uint32_t) h.n_kv) return fail("a K/V layer out of range", SIZE_MAX);
            ConversationKv& k = img.kv[s.index];
            k.format = (int) s.meta[0];
            k.heads = s.meta[2]; k.head_dim = s.meta[3]; k.page_size = s.meta[4]; k.idx_dim = s.meta[6];
            kv_cells[s.index] = s.meta[1];
            kv_rows[s.index] = s.meta[5];
            if (k.page_size <= 0 || s.meta[1] < tail_from || s.meta[5] < 0 || k.idx_dim < 0) return fail("bad K/V extents", SIZE_MAX);
            k.cells = (P + k.page_size - 1) / k.page_size * k.page_size;
            k.pooled_rows = s.meta[5] > 0 ? P / shapes.idx_block + 1 : 0;
            if (k.cells > s.meta[1] || k.pooled_rows > s.meta[5]) return fail("K/V shorter than the match", SIZE_MAX);
        } else if (s.kind == kKvRate) {
            if (s.index >= (uint32_t) h.n_kv) return fail("a K/V layer out of range", SIZE_MAX);
            Rate& r = rates[s.index];
            for (int b = 0; b < 4; ++b) r.cell[b] = (uint64_t) s.meta[b];
            r.row = (uint64_t) s.meta[4];
            rated[s.index] = 1;
        }
    }
    for (int64_t i = 0; i < h.n_kv; ++i)
        if (kv_cells[(size_t) i] < 0 || !rated[(size_t) i]) return fail("a K/V layer without its shapes", SIZE_MAX);
    if (B <= 0 && h.n_blocks > 0) return fail("blocks of no size", SIZE_MAX);
    if (h.n_blocks > 0 && (B % shapes.idx_block != 0 || (int64_t) blocks.size() != h.n_blocks))
        return fail("inconsistent blocks", SIZE_MAX);
    // the destination of each layer's buffer, sized for the match
    std::vector<std::array<ConversationBuffer*, 5>> dsts((size_t) h.n_kv);
    for (int64_t i = 0; i < h.n_kv; ++i) {
        ConversationKv& k = img.kv[(size_t) i];
        const Rate& r = rates[(size_t) i];
        dsts[(size_t) i] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (int b = 0; b < 4; ++b) dsts[(size_t) i][(size_t) b]->resize_for_overwrite(r.cell[b] * (uint64_t) k.cells);
        k.pooled.resize_for_overwrite(r.row * (uint64_t) k.pooled_rows);
    }
    // `kind`'s bytes of layer `i` in [from_unit, to_unit) (cells, or pooled rows) of the destination: the start of a
    // section of `file` that begins at unit `from_unit`
    auto take = [&](size_t file, const Section& s, int64_t i, int kind, int64_t from_unit, int64_t to_unit) -> bool {
        const Rate& r = rates[(size_t) i];
        const uint64_t unit = kind < 4 ? r.cell[kind] : r.row;
        if (to_unit <= from_unit || unit == 0) return true;
        const uint64_t need = (uint64_t) (to_unit - from_unit) * unit;
        if (need > s.bytes) return false;
        ConversationBuffer* d = dsts[(size_t) i][(size_t) kind];
        const uint64_t base = (uint64_t) from_unit * unit;
        parts.push_back({file, &s, need, [d, base](uint64_t o, const uint8_t* p, uint64_t c) {
                             d->visit(base + o, c, [&](uint8_t* q, size_t m2, size_t at) {
                                 std::memcpy(q, p + (at - base - o), m2);
                                 return true;
                             });
                         }});
        return true;
    };
    const int64_t ib = std::max<int64_t>(1, shapes.idx_block);
    for (int64_t b = 0; b < h.n_blocks; ++b) {
        // whether the match reaches into this block (cells, or the pooled row past the last whole one)
        const int64_t c0 = b * B;
        bool needed = false;
        for (int64_t i = 0; i < h.n_kv; ++i) {
            const ConversationKv& k = img.kv[(size_t) i];
            needed = needed || c0 < k.cells || (k.pooled_rows > 0 && c0 / ib < k.pooled_rows);
        }
        if (!needed) break;
        if (!open_part('b', blocks[(size_t) b], (b + 1) * B, b)) return fail(error, opened.size() - 1);
        const size_t file = opened.size() - 1;
        const Opened& f = opened[file];
        if (f.h.n_kv != h.n_kv || f.h.block_cells != B) return fail("a block of another shape", file);
        for (const Section& s : f.tab) {
            if (s.kind < kK || s.kind > kPooled || s.index >= (uint32_t) h.n_kv) return fail("a block of another shape", file);
            const int64_t i = s.index;
            const int kind = (int) (s.kind - kK);
            const ConversationKv& k = img.kv[(size_t) i];
            const Rate& r = rates[(size_t) i];
            const bool ok = kind < 4 ? s.bytes == r.cell[kind] * (uint64_t) B &&
                                           take(file, s, i, kind, c0, std::min<int64_t>(k.cells, c0 + B))
                                     : s.bytes == r.row * (uint64_t) (B / ib) &&
                                           take(file, s, i, kind, c0 / ib, std::min<int64_t>(k.pooled_rows, (c0 + B) / ib));
            if (!ok) return fail("a block of another shape", file);
        }
    }
    for (const Section& s : opened[0].tab) {
        if (s.kind < kK || s.kind > kPooled) continue;
        if (s.index >= (uint32_t) h.n_kv) return fail("K/V data before its layer", SIZE_MAX);
        const int64_t i = s.index;
        const int kind = (int) (s.kind - kK);
        const ConversationKv& k = img.kv[(size_t) i];
        const bool ok = kind < 4 ? take(0, s, i, kind, tail_from, k.cells) : take(0, s, i, kind, tail_from / ib, k.pooled_rows);
        if (!ok) return fail("K/V sizes do not line up", SIZE_MAX);
    }
    uint64_t read = 0;
    size_t bad = SIZE_MAX;
    std::vector<Source> sources;
    for (const auto& f : opened) sources.push_back({f.rf->fd, &f.hashes});
    if (!read_parts(sources, parts, read, error, bad, read_bufs_)) return fail(error, bad);
    if (cut.root >= 0) img.checkpoints.push_back(std::move(root));
    if (cut.live >= 0) {
        at.ids = img.live.ids;
        at.imgs = img.live.imgs;
        at.gdn = img.live.gdn; at.ple = img.live.ple; at.tails = img.live.tails; at.dead = img.live.dead;
        at.block_pos = img.live.block_pos;
        img.checkpoints.push_back(std::move(at));
    }
    out = std::move(img);
    for (auto& f : opened) {
        f.rf->close();
        basalt::platform::touch_file(dir_ + "/" + f.name);
    }
    std::lock_guard<std::mutex> lk(mu_);
    ++stats_.restored;
    stats_.read_bytes += read;
    return true;
}

// ------------------------------------------------------------------------------------------------ writing

bool ConversationStore::put(SavedConversation&& image, uint64_t queue_cap) {
    if (!open_ || image.live.ids.empty() || !image.stage_images.empty()) return false;
    for (const auto& c : image.checkpoints)
        if (c.ids.size() > image.live.ids.size() || !std::equal(c.ids.begin(), c.ids.end(), image.live.ids.begin()) ||
            c.imgs != imgs_below(image.live.imgs, (int64_t) c.ids.size()) || !c.stage_parts.empty())
            return false;
    // The live end a reply past the last checkpoint: stored as of that checkpoint (the next turn goes on from there; a
    // reply sent back as it was is read again), whose state file then stands in for the live state.
    size_t deepest = 0;
    for (const auto& c : image.checkpoints) deepest = std::max(deepest, c.ids.size());
    if (deepest >= 2 && deepest < image.live.ids.size() && image.live.ids.size() - deepest <= (size_t) kLiveSlack) {
        image.live.ids.resize(deepest);
        image.live.imgs = imgs_below(image.live.imgs, (int64_t) deepest);
        image.live.gdn = {};
        image.live.ple = {};
        image.live.tails = {};
        image.live.dead = {};
        image.live.block_pos = {};
    }
    const uint64_t bytes = image.bytes();
    std::lock_guard<std::mutex> lk(mu_);
    if (queue_bytes_ + bytes > queue_cap) return false;
    auto job = std::make_shared<Job>();
    job->key = key_of(image.live.ids, image.live.imgs, image.cvec);
    job->name = conversation_name(identity_, job->key);
    job->bytes = bytes;
    // the same conversation a turn back: its deepest checkpoint is one this chain holds
    auto held = [&](const std::vector<int32_t>& eids, int64_t len) {
        if ((size_t) len > image.live.ids.size() ||
            std::memcmp(eids.data(), image.live.ids.data(), (size_t) len * 4) != 0) return false;
        if ((size_t) len == image.live.ids.size()) return true;
        for (const auto& c : image.checkpoints) if ((int64_t) c.ids.size() == len) return true;
        return false;
    };
    for (size_t i = 0; i < entries_.size();) {
        Entry& e = *entries_[i];
        int64_t deepest = 0;
        for (int64_t c : e.checks) deepest = std::max(deepest, c);
        const bool same = e.mine && e.key == job->key;
        const bool superseded = e.mine && e.cvec == image.cvec && deepest > 0 && held(e.ids, deepest) &&
                                imgs_below(e.imgs, deepest) == imgs_below(image.live.imgs, deepest);
        if (same || superseded) {
            if (!same) ++stats_.dropped;
            drop_locked(i);   // its block and state files stay until the writer has used them for this one
            continue;
        }
        ++i;
    }
    job->image = std::move(image);
    auto e = std::make_unique<Entry>();
    e->name = job->name;
    e->key = job->key;
    e->bytes = bytes;
    e->used = ++clock_;
    e->cvec = job->image.cvec;
    e->ids = job->image.live.ids;
    e->imgs = job->image.live.imgs;
    for (const auto& c : job->image.checkpoints) e->checks.push_back((int64_t) c.ids.size());
    e->job = job;
    entries_.push_back(std::move(e));
    queue_.push_back(job);
    queue_bytes_ += bytes;
    cv_.notify_all();
    return true;
}

void ConversationStore::release_locked(const std::vector<std::string>& refs) {
    for (const auto& r : refs) {
        auto it = parts_.find(r);
        if (it != parts_.end() && it->second.refs > 0) --it->second.refs;
    }
}

void ConversationStore::drop_locked(size_t index) {
    Entry& e = *entries_[index];
    if (e.job) {   // queued: it never reaches the disk
        auto it = std::find(queue_.begin(), queue_.end(), e.job);
        if (it != queue_.end()) {
            queue_bytes_ -= (*it)->bytes;
            queue_.erase(it);
        }
        // a write in progress finishes into a file nobody indexes: the writer removes it (see writer)
    } else {
        basalt::platform::remove_file(dir_ + "/" + e.name);
        release_locked(e.refs);
    }
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

void ConversationStore::gc_locked(int64_t min_age) {
    const int64_t now = min_age > 0 ? basalt::platform::file_time_now() : 0;
    for (auto it = parts_.begin(); it != parts_.end();) {
        if (it->second.refs > 0 || (min_age > 0 && now - it->second.mtime < min_age)) { ++it; continue; }
        const std::string path = dir_ + "/" + it->first;
        uint64_t bytes = 0;
        if (basalt::platform::remove_file(path) || !basalt::platform::file_size(path, bytes)) it = parts_.erase(it);
        else ++it;   // open somewhere (a read in progress): the next pass
    }
}

void ConversationStore::evict_locked() {
    auto total = [&]() {
        uint64_t t = 0;
        for (const auto& e : entries_) t += e->bytes;
        for (const auto& [n, p] : parts_) t += p.bytes;
        return t;
    };
    uint64_t have = total();
    if (have > budget_) {   // what nobody uses goes first
        gc_locked(0);
        have = total();
    }
    while (have > budget_) {
        size_t victim = SIZE_MAX;
        for (size_t i = 0; i < entries_.size(); ++i)
            if (!entries_[i]->job && (victim == SIZE_MAX || entries_[i]->used < entries_[victim]->used)) victim = i;
        if (victim == SIZE_MAX) break;
        drop_locked(victim);
        gc_locked(0);
        ++stats_.evicted;
        have = total();
    }
}

void ConversationStore::writer() {
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;   // stop_ with nothing left
            job = queue_.front();
            busy_ = true;
        }
        std::string why;
        uint64_t wrote = 0, whole = 0;
        const uint64_t free_bytes = basalt::platform::disk_free_bytes(dir_);
        const bool room = free_bytes >= job->bytes + kSpaceFloor;
        const bool ok = room && write_job(*job, wrote, whole, why);
        if (!room)
            basalt::log::warn("basalt serve", "conversation cache: %s has %.1f GiB free, not written (%.1f GiB "
                              "image, %.0f GiB kept free)", dir_.c_str(), (double) free_bytes / 1073741824.0,
                              (double) job->bytes / 1073741824.0, (double) kSpaceFloor / 1073741824.0);
        else if (!ok)
            basalt::log::warn("basalt serve", "conversation cache: %s", why.c_str());
        std::lock_guard<std::mutex> lk(mu_);
        if (!queue_.empty() && queue_.front() == job) {
            queue_.pop_front();
            queue_bytes_ -= job->bytes;
        }
        busy_ = false;
        stats_.written_bytes += wrote;
        Entry* mine = nullptr;
        for (auto& e : entries_) if (e->job == job) mine = e.get();
        if (mine == nullptr) {
            // dropped while it was being written: the file is nobody's
            if (ok) basalt::platform::remove_file(dir_ + "/" + job->name);
            release_locked(job->held);
        } else if (!ok) {
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].get() == mine) { entries_.erase(entries_.begin() + (std::ptrdiff_t) i); break; }
            release_locked(job->held);
            ++stats_.dropped;
        } else {
            // the index switches to the file: header and table as written
            std::string why2;
            ReadFd rf(dir_ + "/" + job->name);
            uint64_t file_bytes = 0;
            std::vector<uint64_t> hashes;
            if (rf.fd >= 0 && basalt::platform::file_size(dir_ + "/" + job->name, file_bytes) &&
                read_index(rf.fd, file_bytes, mine->head, mine->table, hashes, why2)) {
                mine->bytes = file_bytes;
                mine->blocks = job->blocks;
                mine->check_keys = job->check_keys;
                mine->refs = std::move(job->held);
                mine->job.reset();
                ++stats_.written;
                stats_.whole_bytes += whole;
            } else {
                for (size_t i = 0; i < entries_.size(); ++i)
                    if (entries_[i].get() == mine) { entries_.erase(entries_.begin() + (std::ptrdiff_t) i); break; }
                rf.close();
                basalt::platform::remove_file(dir_ + "/" + job->name);
                release_locked(job->held);
                ++stats_.dropped;
            }
        }
        if (queue_.empty()) gc_locked(0);   // between bursts: the files the writes left behind
        evict_locked();
        idle_.notify_all();
    }
}

bool ConversationStore::write_job(Job& job, uint64_t& wrote, uint64_t& whole, std::string& error) {
    const SavedConversation& img = job.image;
    const int64_t n_ids = (int64_t) img.live.ids.size();
    const auto& shapes = basalt::kernels::qsa_real_shapes();
    const int64_t ib = std::max<int64_t>(1, shapes.idx_block);
    // each layer's bytes per cell and per pooled row; whole blocks need every layer's buffers to divide evenly
    std::vector<Rate> rates(img.kv.size());
    bool blocky = !img.kv.empty() && kBlockCells % ib == 0;
    for (size_t i = 0; i < img.kv.size(); ++i) {
        const ConversationKv& kv = img.kv[i];
        const ConversationBuffer* b[4] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale};
        for (int k = 0; k < 4; ++k) {
            rates[i].cell[k] = kv.cells > 0 ? b[k]->size() / (uint64_t) kv.cells : 0;
            if (kv.cells <= 0 || b[k]->size() % (uint64_t) kv.cells) blocky = false;
        }
        rates[i].row = kv.pooled_rows > 0 ? kv.pooled.size() / (uint64_t) kv.pooled_rows : 0;
        if ((kv.pooled_rows > 0 && kv.pooled.size() % (uint64_t) kv.pooled_rows) ||
            (kv.pooled_rows == 0 && kv.pooled.size() != 0))
            blocky = false;
        if (kv.page_size <= 0 || kBlockCells % kv.page_size || kv.cells < n_ids ||
            (kv.pooled_rows > 0 && kv.pooled_rows < n_ids / ib))
            blocky = false;
    }
    const int64_t nb = blocky ? n_ids / kBlockCells : 0;
    const int64_t tail_from = nb * kBlockCells;
    Aligned stage(kStage);
    if (!stage.p) { error = "no write buffer"; return false; }
    // A part file (block or state): named by its prefix and bytes; written unless it is there already.
    auto part = [&](char kind, Srcs& srcs, Header h, uint64_t& key) -> bool {
        number_blocks(srcs);
        std::vector<uint64_t> hashes;
        hash_srcs(srcs, stage, hashes);
        key = content_key(table_of(srcs), hashes, h.prefix_key);
        const std::string name = part_name(identity_, kind, key);
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = parts_.find(name);
            if (it != parts_.end()) {   // there (or being written by this job): one more reference
                ++it->second.refs;
                job.held.push_back(name);
                whole += it->second.bytes;
                return true;
            }
            PartFile& p = parts_[name];
            p.refs = 1;
            p.mtime = basalt::platform::file_time_now();
            job.held.push_back(name);
        }
        h.identity = identity_;
        h.key = key;
        h.cvec = img.cvec ? 1 : 0;
        std::copy(img.geometry.begin(), img.geometry.end(), h.geometry);
        h.layer_lo = img.layer_lo;
        h.layer_hi = img.layer_hi;
        uint64_t file_bytes = 0;
        if (!write_container(dir_ + "/" + name, h, srcs, &hashes, stage, file_bytes, error)) {
            std::lock_guard<std::mutex> lk(mu_);
            parts_.erase(name);
            job.held.pop_back();
            return false;
        }
        wrote += file_bytes;
        whole += file_bytes;
        std::lock_guard<std::mutex> lk(mu_);
        parts_[name].bytes = file_bytes;
        return true;
    };
    // the K/V blocks below the tail
    job.blocks.clear();
    for (int64_t b = 0; b < nb; ++b) {
        Srcs srcs;
        const int64_t c0 = b * kBlockCells;
        for (size_t i = 0; i < img.kv.size(); ++i) {
            const ConversationKv& kv = img.kv[i];
            const ConversationBuffer* bufs[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
            for (int k = 0; k < 5; ++k) {
                const ConversationBuffer* buf = bufs[k];
                const uint64_t unit = k < 4 ? rates[i].cell[k] : rates[i].row;
                const uint64_t base = unit * (uint64_t) (k < 4 ? c0 : c0 / ib);
                const uint64_t n = unit * (uint64_t) (k < 4 ? kBlockCells : kBlockCells / ib);
                add_kv(srcs, kK + k, (uint32_t) i, n, buf, base);
            }
        }
        Header h;
        std::memset(&h, 0, sizeof h);
        h.file_kind = kBlockFile;
        h.prefix_key = key_of(img.live.ids.data(), (size_t) (c0 + kBlockCells), img.live.imgs, img.cvec);
        h.n_kv = (int64_t) img.kv.size();
        h.block_cells = kBlockCells;
        h.position = b;
        uint64_t key = 0;
        if (!part('b', srcs, h, key)) return false;
        job.blocks.push_back(key);
    }
    // the checkpoints' states
    job.check_keys.clear();
    for (const auto& c : img.checkpoints) {
        Srcs srcs;
        add_state(srcs, c, 0);
        Header h;
        std::memset(&h, 0, sizeof h);
        h.file_kind = kStateFile;
        h.prefix_key = key_of(img.live.ids.data(), c.ids.size(), img.live.imgs, img.cvec);
        h.n_ids = 0;
        h.position = (int64_t) c.ids.size();
        uint64_t key = 0;
        if (!part('s', srcs, h, key)) return false;
        job.check_keys.push_back(key);
    }
    // the conversation file: tokens, pictures, the live state, the checkpoints, the K/V shapes and its tail
    Srcs srcs;
    srcs.reserve(16 + img.checkpoints.size() + img.kv.size() * 7);
    add(srcs, kIds, 0, img.live.ids.size() * 4, [&](uint64_t o, uint8_t* d, uint64_t n) {
        std::memcpy(d, reinterpret_cast<const uint8_t*>(img.live.ids.data()) + o, n); });
    add(srcs, kImgs, 0, img.live.imgs.size() * sizeof(ConversationImageKey), [&](uint64_t o, uint8_t* d, uint64_t n) {
        std::memcpy(d, reinterpret_cast<const uint8_t*>(img.live.imgs.data()) + o, n); });
    bool live_checked = false;   // a checkpoint at the live end: its state file holds the live state
    for (const auto& c : img.checkpoints) live_checked = live_checked || (int64_t) c.ids.size() == n_ids;
    if (!live_checked) add_state(srcs, img.live, kLive);
    for (size_t j = 0; j < img.checkpoints.size(); ++j) {
        Section& s = add(srcs, kCheck, (uint32_t) j, 0, nullptr);
        s.meta[0] = (int64_t) img.checkpoints[j].ids.size();
        s.meta[1] = (int64_t) img.checkpoints[j].imgs.size();
        s.meta[2] = (int64_t) job.check_keys[j];
    }
    for (size_t i = 0; i < img.kv.size(); ++i) {
        const ConversationKv& kv = img.kv[i];
        Section& s = add(srcs, kKvHead, (uint32_t) i, 0, nullptr);
        const int64_t meta[7] = {kv.format, kv.cells, kv.heads, kv.head_dim, kv.page_size, kv.pooled_rows, kv.idx_dim};
        std::copy(meta, meta + 7, s.meta);
        Section& r = add(srcs, kKvRate, (uint32_t) i, 0, nullptr);
        for (int k = 0; k < 4; ++k) r.meta[k] = (int64_t) rates[i].cell[k];
        r.meta[4] = (int64_t) rates[i].row;
        r.meta[5] = kv.cells;
        const ConversationBuffer* bufs[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        for (int k = 0; k < 5; ++k) {
            const ConversationBuffer* buf = bufs[k];
            // with no blocks the rates are not needed: the whole buffer is the tail
            const uint64_t base = nb == 0 ? 0 : (k < 4 ? rates[i].cell[k] * (uint64_t) tail_from
                                                       : rates[i].row * (uint64_t) (tail_from / ib));
            const uint64_t n = buf->size() > base ? buf->size() - base : 0;
            add_kv(srcs, kK + k, (uint32_t) i, n, buf, base);
        }
    }
    add(srcs, kRefs, 0, job.blocks.size() * 8, [&](uint64_t o, uint8_t* d, uint64_t n) {
        std::memcpy(d, reinterpret_cast<const uint8_t*>(job.blocks.data()) + o, n); });
    Header h;
    std::memset(&h, 0, sizeof h);
    h.file_kind = kConversationFile;
    h.identity = identity_;
    h.key = job.key;
    std::copy(img.geometry.begin(), img.geometry.end(), h.geometry);
    h.layer_lo = img.layer_lo;
    h.layer_hi = img.layer_hi;
    h.cvec = img.cvec ? 1 : 0;
    h.n_ids = n_ids;
    h.n_imgs = (int64_t) img.live.imgs.size();
    h.n_checks = (int64_t) img.checkpoints.size();
    h.n_kv = (int64_t) img.kv.size();
    h.block_cells = nb > 0 ? kBlockCells : 0;
    h.n_blocks = nb;
    uint64_t conv_bytes = 0;
    if (!write_container(dir_ + "/" + job.name, h, srcs, nullptr, stage, conv_bytes, error)) return false;
    wrote += conv_bytes;
    whole += conv_bytes;
    return true;
}

void ConversationStore::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    idle_.wait(lk, [&] { return queue_.empty() && !busy_; });
}

ConversationStore::Stats ConversationStore::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    s.files = 0;
    s.bytes = 0;
    for (const auto& e : entries_)
        if (e->mine) { ++s.files; s.bytes += e->bytes; }
    for (const auto& [n, p] : parts_) s.bytes += p.bytes;
    return s;
}

uint64_t ConversationStore::queued_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_bytes_;
}

}  // namespace basalt::core
