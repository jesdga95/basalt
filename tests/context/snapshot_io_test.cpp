// Context snapshots on disk (snapshot_io.hpp): a publish read back byte for byte by a restore (an entry larger than one
// transfer, empty and recompute entries, the layout and the file name); generation fencing on publish; every refusal
// a restore makes before it hands a byte over (owner, generation, fingerprint, prefix, completeness, request);
// corruption of a published file (every manifest and padding byte, payload bytes, truncation, extension, an older
// generation under a newer name) always refused; a fault at every file call of a publish - an ordinary failure, a crash
// that stops every later call, a write that lies - leaving the previous generation restorable and never a partial
// snapshot under a snapshot name; faults in a restore; temporaries and pruning; publishes and restores in parallel.
//
// The files go through the standard library; built with BASALT_SNAPSHOT_TEST_PLATFORM (and linked with the platform
// sources) the same checks run over PlatformSnapshotFiles.
#include "basalt/context/snapshot_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace basalt::context;
namespace fs = std::filesystem;

namespace {
int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr StateKind kKv = StateKind::kv, kGdn = StateKind::gdn, kPle = StateKind::ple, kMtp = StateKind::mtp_draft,
                    kTail = StateKind::tail;

fs::path fs_path(const std::string& utf8) { return fs::path(std::u8string(utf8.begin(), utf8.end())); }
std::string utf8_of(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}
int64_t seconds_of(fs::file_time_type t) {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

// SnapshotFiles over the standard library: every call a test needs, without the durability no test can observe.
class StdFiles final : public SnapshotFiles {
public:
    int create(const std::string& path) override {
        return open(path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    }
    bool write(int fd, const void* src, uint64_t n, uint64_t off) override {
        std::fstream* f = get(fd);
        if (!f) return false;
        f->seekp((std::streamoff) off);
        f->write(static_cast<const char*>(src), (std::streamsize) n);
        return (bool) *f;
    }
    bool sync(int fd) override {
        std::fstream* f = get(fd);
        return f && f->flush();
    }
    void close(int fd) override {
        std::lock_guard<std::mutex> lk(mu_);
        files_.erase(fd);
    }
    int open_read(const std::string& path) override { return open(path, std::ios::in | std::ios::binary); }
    bool read(int fd, void* dst, uint64_t n, uint64_t off) override {
        std::fstream* f = get(fd);
        if (!f) return false;
        f->clear();
        f->seekg((std::streamoff) off);
        f->read(static_cast<char*>(dst), (std::streamsize) n);
        return f->gcount() == (std::streamsize) n;
    }
    bool size(const std::string& path, uint64_t& out) override {
        std::error_code ec;
        if (!fs::is_regular_file(fs_path(path), ec)) return false;
        out = (uint64_t) fs::file_size(fs_path(path), ec);
        return !ec;
    }
    bool rename(const std::string& from, const std::string& to) override {
        std::error_code ec;
        fs::rename(fs_path(from), fs_path(to), ec);
        return !ec;
    }
    bool remove(const std::string& path) override {
        std::error_code ec;
        fs::remove(fs_path(path), ec);
        return !fs::exists(fs_path(path), ec);
    }
    bool list(const std::string& dir, std::vector<SnapshotDirEntry>& out) override {
        out.clear();
        std::error_code ec;
        fs::directory_iterator it(fs_path(dir), ec), end;
        if (ec) return false;
        for (; it != end; it.increment(ec)) {
            if (ec) return false;
            std::error_code fe;
            if (!it->is_regular_file(fe)) continue;
            SnapshotDirEntry e;
            e.name = utf8_of(it->path().filename());
            e.bytes = (uint64_t) it->file_size(fe);
            e.mtime = seconds_of(it->last_write_time(fe));
            out.push_back(e);
        }
        return true;
    }
    bool make_dirs(const std::string& dir) override {
        std::error_code ec;
        fs::create_directories(fs_path(dir), ec);
        return fs::is_directory(fs_path(dir), ec);
    }
    bool sync_dir(const std::string& dir) override {
        std::error_code ec;
        return fs::is_directory(fs_path(dir), ec);
    }
    int64_t now() override { return seconds_of(fs::file_time_type::clock::now()); }

private:
    int open(const std::string& path, std::ios::openmode mode) {
        auto f = std::make_unique<std::fstream>(fs_path(path), mode);
        if (!f->is_open()) return -1;
        std::lock_guard<std::mutex> lk(mu_);
        const int fd = next_++;
        files_[fd] = std::move(f);
        return fd;
    }
    std::fstream* get(int fd) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = files_.find(fd);
        return it == files_.end() ? nullptr : it->second.get();
    }
    std::mutex mu_;
    std::map<int, std::unique_ptr<std::fstream>> files_;
    int next_ = 3;
};

#ifdef BASALT_SNAPSHOT_TEST_PLATFORM
using BaseFiles = PlatformSnapshotFiles;
#else
using BaseFiles = StdFiles;
#endif

enum class Fault { none, fail, crash, flip, drop };

// Passes every call to `base`, counting them and the descriptors open, and fails the armed one: `fail` that call
// alone, `crash` it and every call after (descriptors still close, as a dead process's do), `flip` / `drop` a write
// that then reports success with one byte changed / nothing written.
class FaultFiles final : public SnapshotFiles {
public:
    explicit FaultFiles(SnapshotFiles& base) : base_(base) {}
    void arm(Fault mode, int at) {
        mode_ = mode;
        at_ = at;
        calls_ = 0;
        dead_ = tripped_ = false;
        ops_.clear();
    }
    int calls() const { return calls_; }
    bool tripped() const { return tripped_; }
    const std::string& ops() const { return ops_; }   // one letter per call, in order
    size_t open_count() {
        std::lock_guard<std::mutex> lk(mu_);
        return open_.size();
    }

    int create(const std::string& path) override { return opened(stop('c') ? -1 : base_.create(path)); }
    bool write(int fd, const void* src, uint64_t n, uint64_t off) override {
        const int index = calls_;
        if (stop('w')) return false;
        if (index == at_ && mode_ == Fault::drop) return true;
        if (index == at_ && mode_ == Fault::flip && n > 0) {
            // aligned: the descriptor may be an unbuffered one
            uint8_t* copy = static_cast<uint8_t*>(::operator new((size_t) n, std::align_val_t(kSnapshotAlign)));
            std::memcpy(copy, src, (size_t) n);
            copy[n / 2] ^= 0x10;
            const bool ok = base_.write(fd, copy, n, off);
            ::operator delete(copy, std::align_val_t(kSnapshotAlign));
            return ok;
        }
        return base_.write(fd, src, n, off);
    }
    bool sync(int fd) override { return !stop('s') && base_.sync(fd); }
    void close(int fd) override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            open_.erase(fd);
        }
        base_.close(fd);
    }
    int open_read(const std::string& path) override { return opened(stop('o') ? -1 : base_.open_read(path)); }
    bool read(int fd, void* dst, uint64_t n, uint64_t off) override { return !stop('r') && base_.read(fd, dst, n, off); }
    bool size(const std::string& path, uint64_t& out) override { return !stop('z') && base_.size(path, out); }
    bool rename(const std::string& from, const std::string& to) override {
        return !stop('n') && base_.rename(from, to);
    }
    bool remove(const std::string& path) override { return !stop('x') && base_.remove(path); }
    bool list(const std::string& dir, std::vector<SnapshotDirEntry>& out) override {
        return !stop('l') && base_.list(dir, out);
    }
    bool make_dirs(const std::string& dir) override { return !stop('m') && base_.make_dirs(dir); }
    bool sync_dir(const std::string& dir) override { return !stop('d') && base_.sync_dir(dir); }
    int64_t now() override { return base_.now(); }

private:
    bool stop(char op) {
        std::lock_guard<std::mutex> lk(mu_);
        const int index = calls_++;
        ops_.push_back(op);
        if (dead_) return true;
        if (index != at_ || mode_ == Fault::none) return false;
        tripped_ = true;
        if (mode_ == Fault::crash) dead_ = true;
        return mode_ == Fault::fail || mode_ == Fault::crash;
    }
    int opened(int fd) {
        if (fd >= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            open_.insert(fd);
        }
        return fd;
    }
    SnapshotFiles& base_;
    std::mutex mu_;
    std::set<int> open_;
    std::string ops_;
    Fault mode_ = Fault::none;
    int at_ = -1, calls_ = 0;
    bool dead_ = false, tripped_ = false;
};

using Key = std::pair<StateKind, uint32_t>;
using StateBytes = std::map<Key, std::vector<uint8_t>>;

class MemorySource final : public StateSource {
public:
    explicit MemorySource(const StateBytes& state) : state_(state) {}
    bool read(const InventoryEntry& e, uint64_t offset, void* dst, size_t n) override {
        if (reads++ == fail_at) return false;
        const auto it = state_.find({e.kind, e.part});
        if (it == state_.end() || offset + n > it->second.size()) return false;
        std::memcpy(dst, it->second.data() + offset, n);
        return true;
    }
    int reads = 0, fail_at = -1;

private:
    const StateBytes& state_;
};

class MemorySink final : public StateSink {
public:
    bool begin(const InventoryEntry& e) override {
        ++begins;
        if (refuse_begin) return false;
        got[{e.kind, e.part}].assign((size_t) e.bytes, 0xEE);
        return true;
    }
    bool write(const InventoryEntry& e, uint64_t offset, const void* src, size_t n) override {
        ++writes;
        if (refuse_write) return false;
        auto& v = got[{e.kind, e.part}];
        if (offset + n > v.size()) return false;
        std::memcpy(v.data() + offset, src, n);
        delivered[{e.kind, e.part}] += n;
        return true;
    }
    StateBytes got;
    std::map<Key, uint64_t> delivered;
    int begins = 0, writes = 0;
    bool refuse_begin = false, refuse_write = false;
};

Digest digest_of(const std::string& s) { return sha256(s.data(), s.size()); }

struct Fixture {
    Manifest draft;
    StateBytes state;
};

// A hybrid model's state at one position: three QSA layers of K/V (the first larger than one transfer with `big`), the
// GDN state, an empty PLE entry, the draft layer's K/V (marked recompute without `draft`) and three indexer tails.
// No byte is zero, so a write that never reached the file always reads back different.
Fixture make_fixture(Generation g, uint64_t tokens, uint32_t seed, bool draft = true, bool big = false) {
    std::mt19937_64 rng(seed);
    Fixture f;
    auto add = [&](StateKind kind, uint32_t part, size_t n) {
        std::vector<uint8_t> v(n);
        for (auto& b : v) b = (uint8_t) (rng() % 255 + 1);
        f.state[{kind, part}] = std::move(v);
        f.draft.inventory.push_back({kind, EntryMode::stored, part, n, {}});
    };
    f.draft.context_id = "ctx-test_1";
    f.draft.model_fingerprint = digest_of("model");
    f.draft.prefix_hash = digest_of("prefix " + std::to_string(tokens));
    f.draft.prefix_tokens = tokens;
    f.draft.generation = g;
    add(kKv, 0, big ? (4u << 20) + 3 * 4096 + 123 : 70000);
    add(kKv, 1, 4096);
    add(kKv, 2, 1);
    add(kGdn, 0, 300001);
    add(kPle, 0, 0);
    if (draft)
        add(kMtp, 0, 8192 + 5);
    else
        f.draft.inventory.push_back({kMtp, EntryMode::recompute, 0, 0, {}});
    add(kTail, 0, 4095);
    add(kTail, 1, 4097);
    add(kTail, 2, 77);
    return f;
}

RestoreRequest request_for(const Manifest& m) {
    RestoreRequest r;
    r.context_id = m.context_id;
    r.model_fingerprint = m.model_fingerprint;
    r.prefix_hash = m.prefix_hash;
    r.prefix_tokens = m.prefix_tokens;
    r.generation = m.generation;
    return r;
}

PublishResult publish(SnapshotStore& store, const Fixture& f) {
    MemorySource source(f.state);
    return store.publish(f.draft, source);
}

// Restores `f`'s generation and compares every stored entry's bytes, delivered once each, with what was published.
bool restores(SnapshotStore& store, const Fixture& f) {
    MemorySink sink;
    const RestoreResult r = store.restore(request_for(f.draft), sink);
    if (r.status != RestoreStatus::restored || r.replay()) return false;
    int stored = 0;
    for (const auto& e : r.manifest.inventory) {
        const Key k{e.kind, e.part};
        if (e.mode == EntryMode::recompute) {
            if (sink.got.count(k)) return false;
            continue;
        }
        ++stored;
        const auto it = f.state.find(k);
        if (it == f.state.end() || sink.got[k] != it->second || sink.delivered[k] != e.bytes) return false;
    }
    return stored == sink.begins && r.manifest.inventory.size() == f.draft.inventory.size();
}

std::string root_dir;

std::string fresh_dir(const std::string& name) {
    const std::string dir = root_dir + "/" + name;
    std::error_code ec;
    fs::remove_all(fs_path(dir), ec);
    return dir;
}

std::vector<std::string> names_in(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(fs_path(dir), ec), end; !ec && it != end; it.increment(ec))
        out.push_back(utf8_of(it->path().filename()));
    std::sort(out.begin(), out.end());
    return out;
}

size_t temporaries_in(const std::string& dir) {
    size_t n = 0;
    for (const auto& name : names_in(dir))
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) ++n;
    return n;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream in(fs_path(path), std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void spit(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(fs_path(path), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
}

// One byte of a file overwritten in place.
void patch(const std::string& path, uint64_t at, uint8_t byte) {
    std::fstream f(fs_path(path), std::ios::in | std::ios::out | std::ios::binary);
    f.seekp((std::streamoff) at);
    f.put((char) byte);
}

using Generations = std::vector<Generation>;

void names_and_layout() {
    CHECK(snapshot_file_name(1) == "0000000000000001.bsnap");
    CHECK(snapshot_file_name(0xabcdef) == "0000000000abcdef.bsnap");
    Generation g = 0;
    CHECK(parse_snapshot_file_name("00000000000000ff.bsnap", g) && g == 255);
    CHECK(parse_snapshot_file_name(snapshot_file_name(~0ull), g) && g == ~0ull);
    for (const char* bad : {"0000000000000000.bsnap", "00000000000000FF.bsnap", "00000000000000ff.bsnap.tmp",
                            "0ff.bsnap", "00000000000000ff.bsnaq", "00000000000000fg.bsnap", "000000000000000ff.bsnap",
                            "00000000000000ff.deadbeef-0.tmp", ""})
        CHECK(!parse_snapshot_file_name(bad, g));

    Manifest m;
    m.inventory = {{kKv, EntryMode::stored, 0, 10, {}}, {kGdn, EntryMode::stored, 0, 4096, {}},
                   {kPle, EntryMode::stored, 0, 0, {}}, {kMtp, EntryMode::recompute, 0, 0, {}},
                   {kTail, EntryMode::stored, 0, 4097, {}}};
    SnapshotLayout layout;
    std::string error;
    CHECK(snapshot_layout(m, layout, error));
    CHECK(layout.payload_offset == 4096);
    CHECK((layout.offsets == std::vector<uint64_t>{4096, 8192, 12288, 12288, 12288}));
    CHECK(layout.file_bytes == 12288 + 8192);
    m.inventory.clear();
    CHECK(!snapshot_layout(m, layout, error));
    CHECK(std::string(publish_status_name(PublishStatus::verify_failed)) == "verify_failed");
    CHECK(std::string(restore_status_name(RestoreStatus::sink_rejected)) == "sink_rejected");
}

void round_trip(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("round") + "/nested/context";   // created on the first publish
    SnapshotStore store(dir, files);
    const Fixture f = make_fixture(1, 5000, 1, true, true);
    const PublishResult p = publish(store, f);
    CHECK(p.ok());
    CHECK(p.path == dir + "/" + snapshot_file_name(1));
    std::string error;
    CHECK(validate_manifest(p.manifest, error));
    CHECK(p.manifest.context_id == f.draft.context_id && p.manifest.generation == 1);
    SnapshotLayout layout;
    uint64_t bytes = 0;
    CHECK(snapshot_layout(p.manifest, layout, error) && base.size(p.path, bytes) && bytes == layout.file_bytes);
    for (uint64_t off : layout.offsets) CHECK(off % kSnapshotAlign == 0);
    CHECK(layout.payload_offset == kSnapshotAlign);
    CHECK(restores(store, f));
    Manifest m;
    CHECK(store.read_manifest(1, m, error) && m == p.manifest);
    CHECK((store.generations() == Generations{1}));
    CHECK((names_in(dir) == std::vector<std::string>{snapshot_file_name(1)}));   // no temporary left

    // the draft layer's K/V marked recompute: nothing delivered for it, the result names it
    const Fixture r = make_fixture(2, 6000, 2, false);
    const PublishResult pr = publish(store, r);
    CHECK(pr.ok());
    MemorySink sink;
    const RestoreResult rr = store.restore(request_for(r.draft), sink);
    CHECK(rr.status == RestoreStatus::restored);
    CHECK(rr.manifest.inventory[5].kind == kMtp && rr.manifest.inventory[5].mode == EntryMode::recompute);
    CHECK(!sink.got.count({kMtp, 0}));
    CHECK(restores(store, r));

    // digests the state's owner already knows are checked against the bytes
    Fixture known = r;
    known.draft.generation = 3;
    known.draft.inventory = pr.manifest.inventory;
    CHECK(publish(store, known).ok());
    Fixture wrong = known;
    wrong.draft.generation = 4;
    wrong.draft.inventory[3].digest[0] ^= 1;
    const PublishResult pw = publish(store, wrong);
    CHECK(pw.status == PublishStatus::source_failed && !pw.ok());
    CHECK((store.generations() == Generations{1, 2, 3}));
    CHECK(temporaries_in(dir) == 0);
    CHECK(files.open_count() == 0);
}

void publish_rules(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("fencing");
    SnapshotStore store(dir, files);
    CHECK(store.generations().empty());
    CHECK(publish(store, make_fixture(5, 100, 5)).ok());
    // the same generation again, and an older lease's writer: refused before a byte of state is read
    for (const Fixture& late : {make_fixture(5, 200, 6), make_fixture(4, 300, 7)}) {
        MemorySource source(late.state);
        const PublishResult r = store.publish(late.draft, source);
        CHECK(r.status == PublishStatus::stale && !r.error.empty());
        CHECK(source.reads == 0);
    }
    CHECK(publish(store, make_fixture(6, 400, 8)).ok());
    CHECK((store.generations() == Generations{5, 6}));
    CHECK(temporaries_in(dir) == 0);
    CHECK(restores(store, make_fixture(5, 100, 5)));   // a refused publish touched nothing
    CHECK(restores(store, make_fixture(6, 400, 8)));

    // drafts refused before any file is touched
    const std::string untouched = fresh_dir("refused");
    SnapshotStore other(untouched, files);
    Fixture f = make_fixture(1, 100, 9);
    f.draft.generation = 0;
    CHECK(publish(other, f).status == PublishStatus::invalid);
    f = make_fixture(1, 100, 9);
    f.draft.inventory.erase(f.draft.inventory.begin() + 3);   // no GDN state: K/V alone cannot resume the model
    CHECK(publish(other, f).status == PublishStatus::invalid);
    f = make_fixture(1, 100, 9);
    f.draft.inventory.resize(3);   // K/V only
    CHECK(publish(other, f).status == PublishStatus::invalid);
    f = make_fixture(1, 100, 9);
    f.draft.context_id = "not/an/id";
    CHECK(publish(other, f).status == PublishStatus::invalid);
    std::error_code ec;
    CHECK(!fs::exists(fs_path(untouched), ec));

    // a pure-attention consumer's K/V-only snapshot is complete for it, and incomplete for the hybrid model
    f = make_fixture(1, 100, 9);
    f.draft.inventory.resize(3);
    MemorySource source(f.state);
    CHECK(other.publish(f.draft, source, StateRequirements{state_bit(kKv), 0}).ok());
    MemorySink sink;
    CHECK(other.restore(request_for(f.draft), sink).status == RestoreStatus::incomplete);
    CHECK(sink.begins == 0);
    RestoreRequest kv_only = request_for(f.draft);
    kv_only.requirements = {state_bit(kKv), 0};
    CHECK(other.restore(kv_only, sink).status == RestoreStatus::restored);

    // a source that refuses a read
    f = make_fixture(2, 100, 10);
    MemorySource failing(f.state);
    failing.fail_at = 3;
    CHECK(other.publish(f.draft, failing).status == PublishStatus::source_failed);
    CHECK((other.generations() == Generations{1}));
    CHECK(temporaries_in(untouched) == 0);

    CHECK(store.prune(6) == 1);
    CHECK((store.generations() == Generations{6}));
    MemorySink gone;
    CHECK(store.restore(request_for(make_fixture(5, 100, 5).draft), gone).status == RestoreStatus::missing);
    CHECK(restores(store, make_fixture(6, 400, 8)));
    CHECK(store.discard(6));
    CHECK(store.generations().empty());
    CHECK(files.open_count() == 0);
}

// A source that, at its first read, lets a writer of a later generation publish first.
class RacingSource final : public StateSource {
public:
    RacingSource(const StateBytes& state, std::function<void()> race) : inner_(state), race_(std::move(race)) {}
    bool read(const InventoryEntry& e, uint64_t offset, void* dst, size_t n) override {
        if (race_) std::exchange(race_, nullptr)();
        return inner_.read(e, offset, dst, n);
    }

private:
    MemorySource inner_;
    std::function<void()> race_;
};

void publish_race(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("race");
    SnapshotStore store(dir, files);
    CHECK(publish(store, make_fixture(1, 10, 61)).ok());
    const Fixture slow = make_fixture(2, 20, 62), fast = make_fixture(3, 30, 63);
    RacingSource source(slow.state, [&] {
        SnapshotStore other(dir, files);
        CHECK(publish(other, fast).ok());
    });
    const PublishResult r = store.publish(slow.draft, source);
    CHECK(r.status == PublishStatus::stale);   // the later generation, published while this one was written, wins
    CHECK((store.generations() == Generations{1, 3}));
    CHECK(temporaries_in(dir) == 0);
    CHECK(restores(store, fast));
    CHECK(files.open_count() == 0);
}

void restore_refusals(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("refusals");
    SnapshotStore store(dir, files);
    const Fixture f = make_fixture(3, 1234, 3);
    CHECK(publish(store, f).ok());
    const RestoreRequest good = request_for(f.draft);
    const struct {
        const char* what;
        RestoreStatus expect;
        std::function<void(RestoreRequest&)> edit;
    } cases[] = {
        {"another context", RestoreStatus::mismatch, [](auto& r) { r.context_id = "ctx-test_2"; }},
        {"another model", RestoreStatus::mismatch, [](auto& r) { r.model_fingerprint[0] ^= 1; }},
        {"another prefix", RestoreStatus::mismatch, [](auto& r) { r.prefix_hash[31] ^= 1; }},
        {"another length", RestoreStatus::mismatch, [](auto& r) { r.prefix_tokens += 1; }},
        {"a later generation", RestoreStatus::missing, [](auto& r) { r.generation = 4; }},
        {"an earlier generation", RestoreStatus::missing, [](auto& r) { r.generation = 2; }},
        {"a model without a drafter", RestoreStatus::incomplete,
         [](auto& r) { r.requirements.needed &= ~state_bit(kMtp); r.requirements.recomputable = 0; }},
        {"a model without PLE", RestoreStatus::incomplete, [](auto& r) { r.requirements.needed &= ~state_bit(kPle); }},
        {"no context", RestoreStatus::invalid_request, [](auto& r) { r.context_id.clear(); }},
        {"generation 0", RestoreStatus::invalid_request, [](auto& r) { r.generation = 0; }},
        {"no fingerprint", RestoreStatus::invalid_request, [](auto& r) { r.model_fingerprint = {}; }},
        {"no prefix", RestoreStatus::invalid_request, [](auto& r) { r.prefix_hash = {}; }},
        {"no tokens", RestoreStatus::invalid_request, [](auto& r) { r.prefix_tokens = 0; }},
        {"invalid requirements", RestoreStatus::invalid_request, [](auto& r) { r.requirements = {0, 0}; }},
    };
    for (const auto& t : cases) {
        RestoreRequest r = good;
        t.edit(r);
        MemorySink sink;
        const RestoreResult got = store.restore(r, sink);
        if (got.status != t.expect)
            std::fprintf(stderr, "case \"%s\": %s (%s)\n", t.what, restore_status_name(got.status), got.error.c_str());
        CHECK(got.status == t.expect);
        CHECK(got.replay() && !got.error.empty());
        CHECK(sink.begins == 0 && sink.writes == 0);   // refused before a byte was handed over
    }
    MemorySink refuses;
    refuses.refuse_begin = true;
    CHECK(store.restore(good, refuses).status == RestoreStatus::sink_rejected);
    MemorySink refuses_bytes;
    refuses_bytes.refuse_write = true;
    CHECK(store.restore(good, refuses_bytes).status == RestoreStatus::sink_rejected);
    CHECK(restores(store, f));
    CHECK(files.open_count() == 0);
}

void corruption(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("corruption");
    SnapshotStore store(dir, files);
    const Fixture f = make_fixture(1, 777, 4);
    const PublishResult p = publish(store, f);
    CHECK(p.ok());
    const std::vector<uint8_t> clean = slurp(p.path);
    SnapshotLayout layout;
    std::string error;
    CHECK(snapshot_layout(p.manifest, layout, error) && clean.size() == layout.file_bytes);

    auto restore_status = [&](int* begins = nullptr) {
        MemorySink sink;
        const RestoreStatus s = store.restore(request_for(f.draft), sink).status;
        if (begins) *begins = sink.begins;
        return s;
    };
    // the restore's answer with byte `at` set to `byte`; the file is put back after
    auto with_byte = [&](uint64_t at, uint8_t byte, int* begins = nullptr) {
        patch(p.path, at, byte);
        const RestoreStatus s = restore_status(begins);
        patch(p.path, at, clean[(size_t) at]);
        return s;
    };
    // every byte of the manifest and its padding: refused before any byte is handed over. The version field reads as
    // another format (a sound file this engine does not read); every other byte as damage.
    int wrong = 0, delivered = 0;
    for (uint64_t i = 0; i < layout.payload_offset; ++i) {
        int begins = 0;
        const RestoreStatus s = with_byte(i, (uint8_t) (clean[(size_t) i] ^ (1u << (i % 8))), &begins);
        if (s != (i >= 8 && i < 12 ? RestoreStatus::mismatch : RestoreStatus::corrupt)) ++wrong;
        delivered += begins;
    }
    CHECK(wrong == 0);
    CHECK(delivered == 0);
    // payload bytes at the start, middle and end of every stored entry, and the padding after it
    for (size_t i = 0; i < p.manifest.inventory.size(); ++i) {
        const InventoryEntry& e = p.manifest.inventory[i];
        if (e.mode != EntryMode::stored || e.bytes == 0) continue;
        for (uint64_t at : {uint64_t(0), e.bytes / 2, e.bytes - 1}) {
            const uint64_t where = layout.offsets[i] + at;
            CHECK(with_byte(where, (uint8_t) (clean[(size_t) where] ^ 0x01)) == RestoreStatus::corrupt);
        }
        const uint64_t pad = (kSnapshotAlign - e.bytes % kSnapshotAlign) % kSnapshotAlign;
        if (pad) CHECK(with_byte(layout.offsets[i] + e.bytes + pad - 1, 0x5A) == RestoreStatus::corrupt);
    }
    CHECK(restores(store, f));
    // truncation anywhere, and bytes past the end
    Manifest m;
    for (uint64_t cut : {uint64_t(0), uint64_t(1), uint64_t(4095), uint64_t(4096), layout.payload_offset + 1,
                         layout.file_bytes / 2, layout.file_bytes - 4096, layout.file_bytes - 1,
                         layout.file_bytes + 1, layout.file_bytes + kSnapshotAlign}) {
        std::error_code ec;
        fs::resize_file(fs_path(p.path), cut, ec);
        CHECK(!ec);
        CHECK(restore_status() == RestoreStatus::corrupt);
        CHECK(!store.read_manifest(1, m, error));
        spit(p.path, clean);
    }
    CHECK(restores(store, f));
    CHECK(store.read_manifest(1, m, error) && m == p.manifest);
    CHECK(files.open_count() == 0);
}

void stale_generation(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("stale");
    SnapshotStore store(dir, files);
    const Fixture one = make_fixture(1, 4000, 11), two = make_fixture(2, 5000, 12);
    CHECK(publish(store, one).ok() && publish(store, two).ok());
    const std::string path_one = dir + "/" + snapshot_file_name(1), path_two = dir + "/" + snapshot_file_name(2);
    const auto saved = slurp(path_two);
    spit(path_two, slurp(path_one));   // an older snapshot under the newer generation's name
    MemorySink sink;
    const RestoreResult r = store.restore(request_for(two.draft), sink);
    CHECK(r.status == RestoreStatus::stale && sink.begins == 0);
    Manifest m;
    std::string error;
    CHECK(!store.read_manifest(2, m, error) && !error.empty());
    RestoreRequest as_one = request_for(one.draft);
    as_one.generation = 2;   // even with the old snapshot's own identity, the name's generation is fenced
    CHECK(store.restore(as_one, sink).status == RestoreStatus::stale);
    CHECK(restores(store, one));
    spit(path_two, saved);
    CHECK(restores(store, two));
    CHECK(files.open_count() == 0);
}

// The publish of generation 2 over a directory holding generation 1, with a fault at call `at`.
struct FaultRun {
    PublishResult result;
    bool tripped = false;
};

FaultRun faulted_publish(BaseFiles& base, const std::string& dir, const Fixture& one, const Fixture& two, Fault mode,
                         int at) {
    {
        FaultFiles clean(base);
        SnapshotStore store(dir, clean);
        CHECK(publish(store, one).ok());
    }
    FaultFiles files(base);
    files.arm(mode, at);
    FaultRun run;
    {
        SnapshotStore store(dir, files);
        run.result = publish(store, two);
    }
    run.tripped = files.tripped();
    CHECK(files.open_count() == 0);
    return run;
}

void publish_faults(BaseFiles& base) {
    const Fixture one = make_fixture(1, 3000, 21), two = make_fixture(2, 3500, 22);
    std::string ops;
    {
        const std::string dir = fresh_dir("faults");
        FaultFiles clean(base);
        SnapshotStore store(dir, clean);
        CHECK(publish(store, one).ok());
        clean.arm(Fault::none, -1);
        CHECK(publish(store, two).ok());
        ops = clean.ops();
    }
    CHECK(ops.size() > 20);
    CHECK(ops.back() == 'd');   // the directory sync is the last step
    int published_after_fault = 0;
    for (Fault mode : {Fault::fail, Fault::crash}) {
        for (int at = 0; at < (int) ops.size(); ++at) {
            const std::string dir = fresh_dir("faults");
            const FaultRun run = faulted_publish(base, dir, one, two, mode, at);
            CHECK(run.tripped);
            // as after a restart: a new store over the plain files
            FaultFiles files(base);
            SnapshotStore after(dir, files);
            CHECK(restores(after, one));   // the previous generation is untouched
            const Generations have = after.generations();
            if (run.result.ok()) {
                ++published_after_fault;
                CHECK((have == Generations{1, 2}) && restores(after, two));
            } else if (mode == Fault::fail) {
                CHECK((have == Generations{1}));   // a failed publish takes itself back
                CHECK(temporaries_in(dir) == 0);
            } else {
                // a crash after the rename may leave the new snapshot, but only whole
                CHECK((have == Generations{1}) || ((have == Generations{1, 2}) && restores(after, two)));
            }
            after.remove_stale_temporaries(0);
            CHECK(temporaries_in(dir) == 0);
            if (!run.result.ok() && run.result.error.empty()) std::fprintf(stderr, "a failure without a reason\n");
            CHECK(run.result.ok() || !run.result.error.empty());
        }
    }
    // only an unreadable directory listing can be retried past: the publish then creates the directory and goes on
    CHECK(published_after_fault == 1);
    // writes that report success but put the wrong bytes, or none, down: the read-back catches them
    for (Fault mode : {Fault::flip, Fault::drop})
        for (int at = 0; at < (int) ops.size(); ++at) {
            if (ops[(size_t) at] != 'w') continue;
            const std::string dir = fresh_dir("faults");
            const FaultRun run = faulted_publish(base, dir, one, two, mode, at);
            CHECK(run.tripped);
            CHECK(run.result.status == PublishStatus::verify_failed);
            FaultFiles files(base);
            SnapshotStore after(dir, files);
            CHECK((after.generations() == Generations{1}));
            CHECK(restores(after, one));
            CHECK(temporaries_in(dir) == 0);
        }
}

void restore_faults(BaseFiles& base) {
    const std::string dir = fresh_dir("restore_faults");
    const Fixture f = make_fixture(1, 900, 31);
    std::string ops;
    {
        FaultFiles clean(base);
        SnapshotStore store(dir, clean);
        CHECK(publish(store, f).ok());
        clean.arm(Fault::none, -1);
        CHECK(restores(store, f));
        ops = clean.ops();
    }
    for (int at = 0; at < (int) ops.size(); ++at) {
        FaultFiles files(base);
        files.arm(Fault::fail, at);
        SnapshotStore store(dir, files);
        MemorySink sink;
        const RestoreStatus s = store.restore(request_for(f.draft), sink).status;
        CHECK(files.tripped());
        CHECK(s == (ops[(size_t) at] == 'z' ? RestoreStatus::missing : RestoreStatus::unreadable));
        CHECK(files.open_count() == 0);
    }
    FaultFiles files(base);
    SnapshotStore store(dir, files);
    CHECK(restores(store, f));
}

void temporaries_and_foreign_files(BaseFiles& base) {
    FaultFiles files(base);
    const std::string dir = fresh_dir("temporaries");
    SnapshotStore store(dir, files);
    CHECK(publish(store, make_fixture(1, 50, 41)).ok());
    spit(dir + "/0000000000000002.0123456789abcdef-0.tmp", {1, 2, 3});   // a publish that died midway
    spit(dir + "/notes.tmp", {4});
    spit(dir + "/README", {5});
    CHECK(store.remove_stale_temporaries(3600) == 0);   // young: may be another process's publish in flight
    CHECK(store.remove_stale_temporaries(0) == 1);
    CHECK((names_in(dir) == std::vector<std::string>{snapshot_file_name(1), "README", "notes.tmp"}));
    CHECK((store.generations() == Generations{1}));
    CHECK(store.prune(100) == 1);
    CHECK((names_in(dir) == std::vector<std::string>{"README", "notes.tmp"}));
    CHECK(files.open_count() == 0);
}

void parallel(BaseFiles& base) {
    // writers of separate contexts, and a reader of one context while its next generations are published
    FaultFiles files(base);
    const std::string dir = fresh_dir("parallel");
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            SnapshotStore store(dir + "/c" + std::to_string(t), files);
            for (Generation g = 1; g <= 3; ++g) {
                const Fixture f = make_fixture(g, 100 * g + (uint64_t) t, (uint32_t) (t * 10 + g));
                if (!publish(store, f).ok() || !restores(store, f)) ++bad;
            }
        });
    const std::string shared = dir + "/shared";
    SnapshotStore store(shared, files);
    const Fixture first = make_fixture(1, 10, 51);
    CHECK(publish(store, first).ok());
    threads.emplace_back([&] {
        for (Generation g = 2; g <= 5; ++g)
            if (!publish(store, make_fixture(g, 10 + g, (uint32_t) (51 + g))).ok()) ++bad;
    });
    threads.emplace_back([&] {
        for (int i = 0; i < 20; ++i)
            if (!restores(store, first)) ++bad;
    });
    for (auto& t : threads) t.join();
    CHECK(bad.load() == 0);
    CHECK((store.generations() == Generations{1, 2, 3, 4, 5}));
    CHECK(temporaries_in(shared) == 0);
    CHECK(files.open_count() == 0);
}

}  // namespace

int main() {
    std::mt19937_64 rng(std::random_device{}());
    std::error_code ec;
    root_dir = utf8_of(fs::temp_directory_path(ec) / ("basalt_snapshot_io_test_" + std::to_string(rng())));
    BaseFiles base;
    names_and_layout();
    round_trip(base);
    publish_rules(base);
    publish_race(base);
    restore_refusals(base);
    corruption(base);
    stale_generation(base);
    publish_faults(base);
    restore_faults(base);
    temporaries_and_foreign_files(base);
    parallel(base);
    fs::remove_all(fs_path(root_dir), ec);
    if (failures) {
        std::fprintf(stderr, "snapshot_io_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("snapshot_io_test: ok\n");
    return 0;
}
