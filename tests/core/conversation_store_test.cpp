// The disk tier of the conversation cache (conversation_store.hpp): a parked image written, matched by its live tokens
// and checkpoints, read back cut at the match (from the write queue and from the files, in a fresh store as after a
// restart; inside a K/V block, on a block's edge and in the tail), refused when a payload byte of a shared block changed
// or the identity differs, written again a turn on with only its new state, stored as of its last checkpoint when its
// live end is a reply past it, the directory kept to its budget, and only a chat's root and last three turn-end states
// written (a cut at a state not written goes back to the nearest one before it, refcounts of a shared root kept).  A
// conversation written from its source a buffer at a time (as from the session's device buffers) leaves the same files
// as the queued write of its image, and a read that leaves the K/V in its files gives the same bytes through its sink as
// a whole read.
#include "basalt/core/conversation_store.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace basalt::core;

namespace {
int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

std::mt19937_64 rng(7);

void fill(CheckpointBytes& b, size_t n) {
    b.resize(n);
    for (auto& x : b) x = (uint8_t) rng();
}
void fill(ConversationBuffer& b, size_t n) {
    b.resize(n);
    b.visit(0, n, [](uint8_t* p, size_t c, size_t) { for (size_t i = 0; i < c; ++i) p[i] = (uint8_t) rng(); return true; });
}
ConversationCheckpoint state(const std::vector<int32_t>& ids, size_t len) {
    ConversationCheckpoint c;
    c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) len);
    fill(c.gdn, (3u << 20) + 123);
    fill(c.ple, 1024);
    fill(c.tails, 12 * 3 * 512);
    fill(c.dead, 12 * 512);
    fill(c.block_pos, 12 * 4);
    return c;
}

constexpr int64_t kPage = 64, kHeads = 2, kDim = 128, kIdx = 128;

ConversationKv kv_layer(int64_t tokens, bool index) {
    ConversationKv k;
    k.format = 1;
    k.cells = (tokens + kPage - 1) / kPage * kPage;
    k.heads = kHeads;
    k.head_dim = kDim;
    k.page_size = kPage;
    k.idx_dim = kIdx;
    k.pooled_rows = index ? tokens / basalt::kernels::qsa_real_shapes().idx_block + 1 : 0;
    fill(k.k, (size_t) (k.cells * kHeads * kDim));
    fill(k.v, (size_t) (k.cells * kHeads * kDim));
    fill(k.k_scale, (size_t) (k.cells * kHeads * 4));
    fill(k.v_scale, (size_t) (k.cells * kHeads * 4));
    if (index) fill(k.pooled, (size_t) (k.pooled_rows * kIdx * 4));
    return k;
}

SavedConversation image(const std::vector<int32_t>& ids, const std::vector<size_t>& checks) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = (int64_t) i + 1;
    s.layer_lo = 0;
    s.layer_hi = 48;
    s.live = state(ids, ids.size());
    for (size_t c : checks) s.checkpoints.push_back(state(ids, c));
    s.kv.push_back(kv_layer((int64_t) ids.size(), true));
    s.kv.push_back(kv_layer((int64_t) ids.size(), true));
    s.kv.push_back(kv_layer((int64_t) ids.size(), false));   // the draft layer: no indexer
    return s;
}

bool prefix_of(const ConversationBuffer& part, const ConversationBuffer& whole) {
    if (part.size() > whole.size()) return false;
    std::vector<uint8_t> a(part.size()), b(part.size());
    return part.read(a.data(), 0, a.size()) && whole.read(b.data(), 0, b.size()) && a == b;
}

uint64_t dir_bytes(const std::string& dir, size_t* count = nullptr) {
    std::vector<basalt::platform::DirFile> files;
    uint64_t n = 0;
    if (basalt::platform::list_files(dir, files))
        for (const auto& f : files) n += f.bytes;
    if (count) *count = files.size();
    return n;
}

// bytes a source only copies out (as from a device): never in place; `budget` copies at most, then a failure
template <class Bytes>
ConversationBytes copy_only(const Bytes& b, int* budget = nullptr, uint64_t* largest = nullptr) {
    ConversationBytes r;
    r.size = b.size();
    r.read = [&b, budget, largest](uint64_t o, uint8_t* d, uint64_t n) {
        if (budget && (*budget)-- <= 0) return false;
        if (largest) *largest = std::max(*largest, n);
        std::vector<uint8_t> tmp(n);   // through a bounce buffer
        if constexpr (std::is_same_v<Bytes, ConversationBuffer>) {
            if (!b.read(tmp.data(), o, n)) return false;
        } else {
            std::memcpy(tmp.data(), b.data() + o, n);
        }
        std::memcpy(d, tmp.data(), n);
        return true;
    };
    return r;
}

ConversationSource source(const SavedConversation& s, int* budget = nullptr, uint64_t* largest = nullptr) {
    ConversationSource r;
    r.geometry = s.geometry;
    r.layer_lo = s.layer_lo;
    r.layer_hi = s.layer_hi;
    r.cvec = s.cvec;
    r.ids = &s.live.ids;
    r.imgs = &s.live.imgs;
    r.checkpoints = &s.checkpoints;
    r.live = {copy_only(s.live.gdn, budget, largest), copy_only(s.live.ple, budget, largest),
              copy_only(s.live.tails, budget, largest), copy_only(s.live.dead, budget, largest),
              copy_only(s.live.block_pos, budget, largest)};
    for (const ConversationKv& kv : s.kv) {
        ConversationKvSource k;
        k.format = kv.format;
        k.cells = kv.cells;
        k.heads = kv.heads;
        k.head_dim = kv.head_dim;
        k.page_size = kv.page_size;
        k.pooled_rows = kv.pooled_rows;
        k.idx_dim = kv.idx_dim;
        k.parts = {copy_only(kv.k, budget, largest), copy_only(kv.v, budget, largest),
                   copy_only(kv.k_scale, budget, largest), copy_only(kv.v_scale, budget, largest),
                   copy_only(kv.pooled, budget, largest)};
        r.kv.push_back(std::move(k));
    }
    return r;
}

// every file of a directory, by name
std::map<std::string, std::vector<uint8_t>> dir_files(const std::string& dir) {
    std::map<std::string, std::vector<uint8_t>> out;
    std::vector<basalt::platform::DirFile> files;
    if (!basalt::platform::list_files(dir, files)) return out;
    for (const auto& f : files) {
        std::vector<uint8_t>& b = out[f.name];
        b.resize((size_t) f.bytes);
        std::FILE* fp = basalt::platform::open_file_utf8(dir + "/" + f.name, "rb");
        if (!fp || std::fread(b.data(), 1, b.size(), fp) != b.size()) b.clear();
        if (fp) std::fclose(fp);
    }
    return out;
}

void clear_dir(const std::string& dir) {
    std::vector<basalt::platform::DirFile> left;
    if (basalt::platform::list_files(dir, left))
        for (const auto& f : left) basalt::platform::remove_file(dir + "/" + f.name);
    basalt::platform::remove_file(dir);
}

// the K/V of `kv` read through its sink (from several threads) into buffers shaped like `shapes`
bool read_kv(ConversationKvRead& kv, const std::vector<ConversationKv>& shapes, std::vector<ConversationKv>& out,
             std::string& err) {
    out = shapes;
    if (kv.layers() != out.size()) return false;
    std::vector<std::array<ConversationBuffer*, 5>> bufs;
    std::vector<std::vector<uint8_t>> seen(out.size() * 5);
    for (size_t i = 0; i < out.size(); ++i) {
        ConversationKv& k = out[i];
        bufs.push_back({&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled});
        for (int b = 0; b < 5; ++b) {
            if (!bufs[i][(size_t) b]->empty()) return false;   // shapes only
            bufs[i][(size_t) b]->resize((size_t) kv.bytes(i, b));
            seen[i * 5 + (size_t) b].assign((size_t) kv.bytes(i, b), 0);
        }
    }
    std::mutex mu;
    bool once = true;
    const bool ok = kv.read([&](size_t layer, int part, uint64_t off, const uint8_t* src, uint64_t n) {
        if (layer >= out.size() || part < 0 || part >= 5) return false;
        ConversationBuffer* d = bufs[layer][(size_t) part];
        {
            std::lock_guard<std::mutex> lk(mu);
            std::vector<uint8_t>& s = seen[layer * 5 + (size_t) part];
            if (off + n > s.size()) return false;
            for (uint64_t j = off; j < off + n; ++j) once = once && s[j]++ == 0;
        }
        return d->visit(off, n, [&](uint8_t* q, size_t c, size_t at) { std::memcpy(q, src + (at - off), c); return true; });
    }, err);
    for (const auto& s : seen)
        for (uint8_t c : s) once = once && c == 1;   // every byte exactly once
    return ok && once;
}

bool same_kv(const std::vector<ConversationKv>& a, const std::vector<ConversationKv>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(a[i].k == b[i].k && a[i].v == b[i].v && a[i].k_scale == b[i].k_scale && a[i].v_scale == b[i].v_scale &&
              a[i].pooled == b[i].pooled && a[i].cells == b[i].cells && a[i].pooled_rows == b[i].pooled_rows &&
              a[i].format == b[i].format))
            return false;
    return true;
}

bool same_state(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails && a.dead == b.dead && a.block_pos == b.block_pos;
}

// the checks every cut image passes: live = the state at `at`, K/V the prefix up to it
void check_cut(const SavedConversation& got, const SavedConversation& src, int64_t at, const ConversationCheckpoint& want) {
    CHECK((int64_t) got.live.ids.size() == at);
    CHECK(std::equal(got.live.ids.begin(), got.live.ids.end(), src.live.ids.begin()));
    CHECK(same_state(got.live, want));
    CHECK(got.geometry == src.geometry && got.layer_lo == src.layer_lo && got.layer_hi == src.layer_hi);
    CHECK(got.kv.size() == src.kv.size());
    for (size_t i = 0; i < got.kv.size() && i < src.kv.size(); ++i) {
        const ConversationKv& k = got.kv[i];
        CHECK(k.cells == (at + kPage - 1) / kPage * kPage);
        CHECK(k.k.size() == (size_t) (k.cells * kHeads * kDim) && prefix_of(k.k, src.kv[i].k));
        CHECK(k.v.size() == (size_t) (k.cells * kHeads * kDim) && prefix_of(k.v, src.kv[i].v));
        CHECK(prefix_of(k.k_scale, src.kv[i].k_scale) && prefix_of(k.v_scale, src.kv[i].v_scale));
        const int64_t rows = src.kv[i].pooled_rows > 0 ? at / basalt::kernels::qsa_real_shapes().idx_block + 1 : 0;
        CHECK(k.pooled_rows == rows && k.pooled.size() == (size_t) (rows * kIdx * 4) && prefix_of(k.pooled, src.kv[i].pooled));
    }
    // the chain: the root, then the state at the match once more when it is a checkpoint
    CHECK(!got.checkpoints.empty() && got.checkpoints.front().ids.size() == src.checkpoints.front().ids.size() &&
          same_state(got.checkpoints.front(), src.checkpoints.front()));
    if (at == (int64_t) src.live.ids.size()) {
        CHECK(got.checkpoints.size() == 1);
    } else {
        CHECK(got.checkpoints.size() == 2 && (int64_t) got.checkpoints[1].ids.size() == at &&
              same_state(got.checkpoints[1], want));
    }
}

using Src = ConversationCheckpoint::Source;

// a chat of `ids` with a checkpoint of the given source at each length
SavedConversation chat(const std::vector<int32_t>& ids, const std::vector<std::pair<size_t, Src>>& checks) {
    SavedConversation s = image(ids, {});
    for (const auto& c : checks) {
        s.checkpoints.push_back(state(ids, c.first));
        s.checkpoints.back().source = c.second;
    }
    return s;
}

void copy_state_bytes(ConversationCheckpoint& to, const ConversationCheckpoint& from) {
    to.gdn = from.gdn; to.ple = from.ple; to.tails = from.tails; to.dead = from.dead; to.block_pos = from.block_pos;
}

// state files in `dir` ("<identity>-s<key>.bcs")
size_t state_files(const std::string& dir) {
    std::vector<basalt::platform::DirFile> files;
    size_t n = 0;
    if (basalt::platform::list_files(dir, files))
        for (const auto& f : files) n += f.name.size() > 18 && f.name[17] == 's';
    return n;
}

void remove_dir(const std::string& dir) {
    std::vector<basalt::platform::DirFile> left;
    if (basalt::platform::list_files(dir, left))
        for (const auto& f : left) basalt::platform::remove_file(dir + "/" + f.name);
    basalt::platform::remove_file(dir);
}
}  // namespace

int main() {
    const std::string dir = basalt::platform::temp_dir() + "/basalt-store-test-" +
                            std::to_string((long long) basalt::platform::process_id());
    std::vector<int32_t> ids(10000);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) (i * 7 + 3);
    const SavedConversation src = [&] {   // its live end far past the last checkpoint
        SavedConversation c = image(ids, {2048, 4096, 5000});
        c.checkpoints[0].source = ConversationCheckpoint::kRoot;
        return c;
    }();
    const uint64_t id = 42;
    {
        ConversationStore st;
        std::string err;
        CHECK(st.open(dir, id, 1ull << 30, err));
        SavedConversation copy = src;
        CHECK(st.put(std::move(copy), 1ull << 30));
        CHECK(st.holds(src.live.ids, src.live.imgs, src.cvec));
        // a prompt past the turn checkpoint but off the live tokens: the checkpoint
        std::vector<int32_t> p(ids.begin(), ids.begin() + 9500);
        p.push_back(-5);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens == 5000);
        SavedConversation got;
        CHECK(st.load(m, got, err));   // possibly still queued: out of RAM
        check_cut(got, src, 5000, src.checkpoints[2]);
        st.flush();
        CHECK(st.stats().written == 1);
        SavedConversation got2;
        CHECK(st.load(m, got2, err));   // from the files: a whole block and part of the next
        check_cut(got2, src, 5000, src.checkpoints[2]);
        // a cut on the first block's edge: its cells from one block, the pooled row past them from the next
        std::vector<int32_t> q(ids.begin(), ids.begin() + 4500);
        q.push_back(-5);
        auto m2 = st.best(q.data(), q.size(), {}, true);
        CHECK(m2.tokens == 4096);
        SavedConversation got3;
        CHECK(st.load(m2, got3, err));
        check_cut(got3, src, 4096, src.checkpoints[1]);
        // another control-vector state or another picture never matches
        CHECK(st.best(p.data(), p.size(), {}, false).tokens == 0);
        CHECK(st.best(p.data(), p.size(), {ConversationImageKey{4500, 9}}, true).tokens == 4096);
    }
    {
        // a fresh store on the same directory: the index comes from the file (as after a restart)
        ConversationStore st;
        std::string err;
        CHECK(st.open(dir, id, 1ull << 30, err));
        CHECK(st.stats().files == 1);
        std::vector<int32_t> p = ids;
        p.push_back(1);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens == 10000);
        SavedConversation got;
        CHECK(st.load(m, got, err));
        check_cut(got, src, 10000, src.live);
        // another identity sees nothing
        ConversationStore other;
        CHECK(other.open(dir, id + 1, 1ull << 30, err));
        CHECK(other.best(p.data(), p.size(), {}, true).tokens == 0);
    }
    {
        // the same conversation a turn on, with the bytes it had: only its new state and its tail are written; its live
        // end a reply past the turn's checkpoint, it is stored as of that checkpoint
        ConversationStore st;
        std::string err;
        CHECK(st.open(dir, id, 1ull << 30, err));
        std::vector<int32_t> ids2 = ids;
        for (int i = 0; i < 300; ++i) ids2.push_back(91 + i);
        SavedConversation next = src;
        next.live = state(ids2, ids2.size());
        next.checkpoints.push_back(state(ids2, ids.size()));
        next.checkpoints.back().gdn = src.live.gdn;   // the turn's checkpoint: the state the conversation had
        next.checkpoints.back().ple = src.live.ple;
        next.checkpoints.back().tails = src.live.tails;
        next.checkpoints.back().dead = src.live.dead;
        next.checkpoints.back().block_pos = src.live.block_pos;
        next.live.ids = ids2;
        SavedConversation longer = image(ids2, {});
        for (size_t i = 0; i < next.kv.size(); ++i) {   // the old cells' bytes, then new ones
            ConversationKv k = longer.kv[i];
            const ConversationKv& old = src.kv[i];
            ConversationBuffer* to[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            const ConversationBuffer* from[5] = {&old.k, &old.v, &old.k_scale, &old.v_scale, &old.pooled};
            for (int b = 0; b < 5; ++b) {
                std::vector<uint8_t> tmp(from[b]->size());
                from[b]->read(tmp.data(), 0, tmp.size());
                to[b]->visit(0, tmp.size(), [&](uint8_t* q, size_t c, size_t at) { std::memcpy(q, tmp.data() + at, c); return true; });
            }
            next.kv[i] = std::move(k);
        }
        const uint64_t before = dir_bytes(dir);
        const SavedConversation keep = next;
        CHECK(st.put(std::move(next), 1ull << 30));
        st.flush();
        const auto s = st.stats();
        CHECK(s.files == 1 && s.written == 1);
        // the turn's checkpoint and its tail; no live state, no block and no old checkpoint again
        CHECK(s.written_bytes < (uint64_t) (keep.live.gdn.size() + 3 * (2u << 20)));
        CHECK(dir_bytes(dir) < before + s.written_bytes);   // the old conversation file is gone
        std::vector<int32_t> p = ids2;
        p.push_back(1);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens == (int64_t) ids.size());
        SavedConversation got;
        CHECK(st.load(m, got, err));
        for (size_t i = 0; i < got.kv.size(); ++i) CHECK(prefix_of(got.kv[i].k, keep.kv[i].k) && prefix_of(got.kv[i].pooled, keep.kv[i].pooled));
        CHECK(got.live.ids.size() == ids.size() && same_state(got.live, src.live));
        // the chain keeps the checkpoint the state came from
        CHECK(got.checkpoints.size() == 2 && got.checkpoints.back().ids.size() == ids.size() &&
              same_state(got.checkpoints.back(), src.live));
        // the old turn's prefix still reads back from the shared files
        std::vector<int32_t> q(ids.begin(), ids.begin() + 9500);
        q.push_back(-5);
        auto m2 = st.best(q.data(), q.size(), {}, true);
        CHECK(m2.tokens == 5000);
        SavedConversation got2;
        CHECK(st.load(m2, got2, err));
        check_cut(got2, src, 5000, src.checkpoints[2]);
    }
    {
        // a fresh store again: the conversation stored as of its checkpoint reads back from the checkpoint's state file
        ConversationStore st;
        std::string err;
        CHECK(st.open(dir, id, 1ull << 30, err));
        std::vector<int32_t> p = ids;
        p.push_back(1);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens == (int64_t) ids.size());
        SavedConversation got;
        CHECK(st.load(m, got, err));
        CHECK(got.live.ids.size() == ids.size() && same_state(got.live, src.live));
    }
    {
        // one changed payload byte of a shared block: the read fails, the conversation and the block go
        std::vector<basalt::platform::DirFile> files;
        CHECK(basalt::platform::list_files(dir, files));
        std::string block;
        uint64_t block_bytes = 0;
        for (const auto& f : files)
            if (f.name.size() > 18 && f.name[17] == 'b') { block = f.name; block_bytes = f.bytes; }
        CHECK(!block.empty());
        const std::string path = dir + "/" + block;
        std::FILE* f = block.empty() ? nullptr : basalt::platform::open_file_utf8(path, "r+b");
        CHECK(f != nullptr);
        if (f) {
            basalt::platform::fseek64(f, (int64_t) block_bytes - 8192, SEEK_SET);
            const int c = std::fgetc(f);
            basalt::platform::fseek64(f, (int64_t) block_bytes - 8192, SEEK_SET);
            std::fputc(c ^ 0x5a, f);
            std::fclose(f);
        }
        ConversationStore st;
        std::string err;
        CHECK(st.open(dir, id, 1ull << 30, err));
        CHECK(st.stats().files == 1);
        std::vector<int32_t> p = ids;
        p.push_back(1);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens > 0);
        SavedConversation got;
        CHECK(!st.load(m, got, err));
        CHECK(st.stats().files == 0);
        uint64_t sz = 0;
        CHECK(!basalt::platform::file_size(path, sz));
    }
    {
        // the budget: a second conversation pushes the least recently used one out
        ConversationStore st;
        std::string err;
        SavedConversation a = src;
        const uint64_t one = a.bytes();
        CHECK(st.open(dir, id, one + one / 2, err));
        CHECK(st.put(std::move(a), 1ull << 30));
        st.flush();
        std::vector<int32_t> ids2(ids.size());
        for (size_t i = 0; i < ids2.size(); ++i) ids2[i] = (int32_t) (i * 5 + 1);
        SavedConversation b = image(ids2, {4096});
        CHECK(st.put(std::move(b), 1ull << 30));
        st.flush();
        const auto s = st.stats();
        CHECK(s.files == 1 && s.evicted == 1);
        std::vector<int32_t> p = ids2;
        p.push_back(0);
        CHECK(st.best(p.data(), p.size(), {}, true).tokens == 10000);
        // the same conversation a turn later supersedes it
        std::vector<int32_t> ids3 = ids2;
        for (int i = 0; i < 300; ++i) ids3.push_back(77);
        SavedConversation c = image(ids3, {4096, 10000});
        CHECK(st.put(std::move(c), 1ull << 30));
        st.flush();
        CHECK(st.stats().files == 1);
        p = ids3;
        p.push_back(0);
        CHECK(st.best(p.data(), p.size(), {}, true).tokens == 10000);   // as of its checkpoint, 300 tokens back
    }
    clear_dir(dir);
    // written from a source a buffer at a time: the same files as the queued write; read with the K/V left in its
    // files: the same bytes through the sink as a whole read, out of the files and out of the write queue
    struct Case { size_t tokens; std::vector<size_t> checks; size_t gdn; };
    const Case cases[] = {
        {9001, {1000}, 0},                  // the live state stored, a tail past two whole blocks
        {13001, {2048, 4096, 9500}, 0},     // stored as of its last checkpoint
        {4096, {}, (17u << 20) + 5},        // one whole block, no tail; a live state over two write buffers
        {8191, {4096, 8191}, 0},            // a checkpoint at the live end
        {300, {7}, 0},                      // no whole block
        {13001, {1000, 2048, 4096, 6000, 9500}, 0},   // only the last three turn ends' states written
    };
    for (const Case& c : cases) {
        const std::string qdir = dir + "-q", ddir = dir + "-d";
        std::vector<int32_t> cids(c.tokens);
        for (size_t i = 0; i < cids.size(); ++i) cids[i] = (int32_t) (i * 13 + c.tokens);
        SavedConversation img = image(cids, c.checks);
        if (c.gdn) fill(img.live.gdn, c.gdn);
        std::string err;
        {
            ConversationStore q;
            CHECK(q.open(qdir, id, 1ull << 30, err));
            SavedConversation copy = img;
            CHECK(q.put(std::move(copy), 1ull << 30));
            q.flush();
            CHECK(q.stats().written == 1);
        }
        ConversationStore d;
        CHECK(d.open(ddir, id, 1ull << 30, err));
        SavedConversation refused = img;
        CHECK(!d.put(std::move(refused), 0));
        refused = {};
        uint64_t largest = 0;
        CHECK(d.write(source(img, nullptr, &largest), err));
        CHECK(largest > 0 && largest <= (8ull << 20));
        CHECK(d.queued_bytes() == 0);
        size_t deepest = 0;
        for (size_t k : c.checks) deepest = std::max(deepest, k);
        const bool trimmed = deepest >= 2 && deepest < c.tokens && c.tokens - deepest <= 4096;   // stored as of it
        CHECK(d.stats().written == 1 && d.holds(img.live.ids, img.live.imgs, img.cvec) == !trimmed);
        const auto qf = dir_files(qdir), df = dir_files(ddir);
        CHECK(!qf.empty() && qf == df);
        {   // a disk-only park survives a restart and the unchanged whole-image read path
            ConversationStore restarted, baseline;
            CHECK(restarted.open(ddir, id, 1ull << 30, err) && baseline.open(qdir, id, 1ull << 30, err));
            std::vector<int32_t> p = cids;
            p.push_back(-7);
            const auto m = restarted.best(p.data(), p.size(), {}, true);
            SavedConversation got, want;
            CHECK(m.tokens == (int64_t) (trimmed ? deepest : c.tokens) && restarted.load(m, got, err));
            const auto qm = baseline.best(p.data(), p.size(), {}, true);
            CHECK(qm.tokens == m.tokens && baseline.load(qm, want, err));
            CHECK(got.live.ids == want.live.ids && same_state(got.live, want.live) && same_kv(got.kv, want.kv));
            CHECK(got.checkpoints.size() == want.checkpoints.size());
            for (size_t i = 0; i < got.checkpoints.size() && i < want.checkpoints.size(); ++i)
                CHECK(got.checkpoints[i].ids == want.checkpoints[i].ids &&
                      same_state(got.checkpoints[i], want.checkpoints[i]));
        }
        // every match: the live end and each checkpoint, whole and with the K/V in its files
        std::vector<int64_t> ends = {(int64_t) c.tokens};
        for (size_t k : c.checks) ends.push_back((int64_t) k);
        for (int64_t at : ends) {
            std::vector<int32_t> p(cids.begin(), cids.begin() + at);
            p.push_back(-7);
            const auto m = d.best(p.data(), p.size(), {}, true);
            if (m.tokens != at) continue;   // a live end stored as of its checkpoint
            SavedConversation whole, part;
            std::unique_ptr<ConversationKvRead> kv;
            CHECK(d.load(m, whole, err));
            CHECK(d.load(m, part, err, &kv) && kv != nullptr);
            CHECK(same_state(part.live, whole.live) && part.checkpoints.size() == whole.checkpoints.size());
            std::vector<ConversationKv> got;
            CHECK(kv && read_kv(*kv, part.kv, got, err) && same_kv(got, whole.kv));
            CHECK(kv && !kv->read([](size_t, int, uint64_t, const uint8_t*, uint64_t) { return true; }, err));   // once
        }
        {   // out of the write queue: the image in RAM through the sink
            ConversationStore q;
            const std::string q2 = dir + "-q2";
            CHECK(q.open(q2, id, 1ull << 30, err));
            SavedConversation copy = img;
            CHECK(q.put(std::move(copy), 1ull << 30));
            std::vector<int32_t> p = cids;
            p.push_back(-7);
            const auto m = q.best(p.data(), p.size(), {}, true);
            SavedConversation whole, part;
            std::unique_ptr<ConversationKvRead> kv;
            CHECK(m.tokens > 0 && q.load(m, whole, err) && q.load(m, part, err, &kv) && kv != nullptr);
            std::vector<ConversationKv> got;
            CHECK(kv && read_kv(*kv, part.kv, got, err) && same_kv(got, whole.kv));
            q.flush();
            clear_dir(q2);
        }
        clear_dir(qdir);
        clear_dir(ddir);
    }
    {
        // a conversation dropped (superseded by its next turn) while a read holds its files open: the files stay until
        // the read has closed them, the read gives the bytes of the whole read, then they go
        std::vector<int32_t> xids(9001);
        for (size_t i = 0; i < xids.size(); ++i) xids[i] = (int32_t) (i * 5 + 2);
        const SavedConversation x = image(xids, {1000});
        std::string err;
        ConversationStore d;
        CHECK(d.open(dir, id, 1ull << 30, err));
        CHECK(d.write(source(x), err));
        std::vector<int32_t> p = xids;
        p.push_back(-7);
        const auto m = d.best(p.data(), p.size(), {}, true);
        SavedConversation whole, part;
        std::unique_ptr<ConversationKvRead> kv;
        CHECK(m.tokens == 9001 && d.load(m, whole, err) && d.load(m, part, err, &kv) && kv != nullptr);
        std::vector<basalt::platform::DirFile> before;
        CHECK(basalt::platform::list_files(dir, before));
        std::string xname;
        for (const auto& f : before)
            if (f.name.size() == 16 + 1 + 16 + 4) xname = f.name;
        std::vector<int32_t> yids = xids;
        for (int i = 0; i < 5000; ++i) yids.push_back(5 + i);   // past the slack: stored with its own name
        SavedConversation y = image(yids, {1000, 9001});
        CHECK(d.write(source(y), err));
        uint64_t sz = 0;
        CHECK(!xname.empty() && d.stats().files == 1 && basalt::platform::file_size(dir + "/" + xname, sz));
        std::vector<ConversationKv> got;
        CHECK(kv && read_kv(*kv, part.kv, got, err) && same_kv(got, whole.kv));
        kv.reset();
        CHECK(!basalt::platform::file_size(dir + "/" + xname, sz));
        clear_dir(dir);
    }
    {
        // A failed direct replacement leaves the prior file and its referenced data readable, including after reopen.
        const std::string d2 = dir + "-direct-failure";
        std::vector<int32_t> old_ids(300);
        for (size_t i = 0; i < old_ids.size(); ++i) old_ids[i] = (int32_t) (i * 3 + 1);
        const SavedConversation old = chat(old_ids, {{old_ids.size(), ConversationCheckpoint::kTurnEnd}});
        const SavedConversation same = image(old_ids, {});
        std::vector<int32_t> longer_ids = old_ids;
        longer_ids.resize(5000, 19);
        const SavedConversation longer = chat(longer_ids, {{old_ids.size(), ConversationCheckpoint::kTurnEnd}});
        std::vector<int32_t> prompt = old_ids;
        prompt.push_back(-1);
        std::string err;
        {
            ConversationStore d;
            CHECK(d.open(d2, id, 1ull << 30, err));
            CHECK(d.write(source(old), err));
            const auto before = dir_files(d2);
            for (const SavedConversation* replacement : {&same, &longer}) {
                int budget = 0;
                CHECK(!d.write(source(*replacement, &budget), err));
                CHECK(d.stats().files == 1 && dir_files(d2) == before);
                const auto match = d.best(prompt.data(), prompt.size(), {}, true);
                SavedConversation got;
                CHECK(match.tokens == (int64_t) old_ids.size() && d.load(match, got, err));
                CHECK(same_state(got.live, old.checkpoints[0]) && same_kv(got.kv, old.kv));
            }
            // A newly queued image changes the index during the direct copy; neither entry is discarded.
            std::vector<int32_t> other_ids(300, 777);
            SavedConversation other = image(other_ids, {});
            ConversationSource concurrent = source(same);
            auto read = concurrent.kv[0].parts[0].read;
            bool queued = false;
            concurrent.kv[0].parts[0].read = [&](uint64_t off, uint8_t* dst, uint64_t n) {
                if (!queued) {
                    queued = true;
                    CHECK(d.put(std::move(other), 1ull << 30));
                }
                return read(off, dst, n);
            };
            CHECK(!d.write(concurrent, err) && queued);
            d.flush();
            CHECK(d.stats().files == 2);
            const auto match = d.best(prompt.data(), prompt.size(), {}, true);
            SavedConversation got;
            CHECK(match.tokens == (int64_t) old_ids.size() && d.load(match, got, err));
            CHECK(same_state(got.live, old.checkpoints[0]) && same_kv(got.kv, old.kv));
        }
        {
            ConversationStore d;
            CHECK(d.open(d2, id, 1ull << 30, err));
            const auto match = d.best(prompt.data(), prompt.size(), {}, true);
            SavedConversation got;
            CHECK(match.tokens == (int64_t) old_ids.size() && d.load(match, got, err));
            CHECK(same_state(got.live, old.checkpoints[0]) && same_kv(got.kv, old.kv));
            CHECK(d.write(source(same), err));
            const auto updated = d.best(prompt.data(), prompt.size(), {}, true);
            SavedConversation replacement;
            CHECK(updated.tokens == (int64_t) old_ids.size() && d.load(updated, replacement, err));
            CHECK(same_state(replacement.live, same.live) && same_kv(replacement.kv, same.kv));
        }
        clear_dir(d2);
    }
    {
        // a source that fails partway: nothing indexed, nothing left behind; a K/V read that fails a block hash drops
        // the conversation
        std::vector<int32_t> cids(9001);
        for (size_t i = 0; i < cids.size(); ++i) cids[i] = (int32_t) (i * 3 + 1);
        const SavedConversation img = image(cids, {1000});
        std::string err;
        ConversationStore d;
        CHECK(d.open(dir, id, 1ull << 30, err));
        int budget = 20;
        CHECK(!d.write(source(img, &budget), err));
        CHECK(d.stats().files == 0 && dir_bytes(dir) == 0);
        CHECK(d.write(source(img), err) && d.stats().files == 1);
        std::vector<basalt::platform::DirFile> files;
        CHECK(basalt::platform::list_files(dir, files));
        std::string block;
        for (const auto& f : files)
            if (f.name.size() > 18 && f.name[17] == 'b') block = f.name;
        std::FILE* f = block.empty() ? nullptr : basalt::platform::open_file_utf8(dir + "/" + block, "r+b");
        CHECK(f != nullptr);
        if (f) {
            basalt::platform::fseek64(f, 3 * 4096 + 100, SEEK_SET);
            const int ch = std::fgetc(f);
            basalt::platform::fseek64(f, 3 * 4096 + 100, SEEK_SET);
            std::fputc(ch ^ 0x33, f);
            std::fclose(f);
        }
        std::vector<int32_t> p = cids;
        p.push_back(-7);
        const auto m = d.best(p.data(), p.size(), {}, true);
        SavedConversation part;
        std::unique_ptr<ConversationKvRead> kv;
        CHECK(m.tokens == 9001 && d.load(m, part, err, &kv) && kv != nullptr);
        std::vector<ConversationKv> got;
        CHECK(kv && !read_kv(*kv, part.kv, got, err));
        kv.reset();
        CHECK(d.stats().files == 0);
        clear_dir(dir);
    }
    {
        // only the root and the last three turn ends are written; strides and older turn ends are not
        const std::string d2 = dir + "-keep";
        std::vector<int32_t> a(12000);
        for (size_t i = 0; i < a.size(); ++i) a[i] = (int32_t) (i * 11 + 5);
        const Src T = ConversationCheckpoint::kTurnEnd, P = ConversationCheckpoint::kPeriodic, R = ConversationCheckpoint::kRoot;
        const SavedConversation sa = chat(a, {{1000, R}, {3000, P}, {4000, T}, {5000, P}, {6000, T}, {7000, T},
                                              {8000, T}, {9000, T}, {9500, P}});
        std::string err;
        {
            ConversationStore st;
            CHECK(st.open(d2, id, 1ull << 30, err));
            SavedConversation copy = sa;
            CHECK(st.put(std::move(copy), 1ull << 30));
            st.flush();
            CHECK(st.stats().written == 1);
            CHECK(state_files(d2) == 4);   // root 1000, turn ends 7000, 8000, 9000
            // a cut at a state that was not written goes back to the nearest one before it
            std::vector<int32_t> p(a.begin(), a.begin() + 6500);
            p.push_back(-5);
            auto m = st.best(p.data(), p.size(), {}, true);
            CHECK(m.tokens == 1000);
            SavedConversation got;
            CHECK(st.load(m, got, err));
            CHECK(got.live.ids.size() == 1000 && same_state(got.live, sa.checkpoints[0]));
            CHECK(got.kv.size() == sa.kv.size());
            for (size_t i = 0; i < got.kv.size(); ++i)
                CHECK(prefix_of(got.kv[i].k, sa.kv[i].k) && prefix_of(got.kv[i].pooled, sa.kv[i].pooled));
            // past a stride: the turn end before it, with the root beside
            p.assign(a.begin(), a.begin() + 9700);
            p.push_back(-5);
            m = st.best(p.data(), p.size(), {}, true);
            CHECK(m.tokens == 9000);
            SavedConversation g2;
            CHECK(st.load(m, g2, err));
            CHECK(same_state(g2.live, sa.checkpoints[7]));
            CHECK(g2.checkpoints.size() == 2 && g2.checkpoints[0].ids.size() == 1000 &&
                  g2.checkpoints[0].source == R && same_state(g2.checkpoints[0], sa.checkpoints[0]) &&
                  g2.checkpoints[1].source == T);
            // the live end is a reply past the last turn end: stored as of it
            p = a;
            p.push_back(-5);
            CHECK(st.best(p.data(), p.size(), {}, true).tokens == 9000);
        }
        {
            ConversationStore st;   // after a restart the kinds come from the file
            CHECK(st.open(d2, id, 1ull << 30, err));
            std::vector<int32_t> p(a.begin(), a.begin() + 8200);
            p.push_back(-5);
            auto m = st.best(p.data(), p.size(), {}, true);
            CHECK(m.tokens == 8000);
            SavedConversation got;
            CHECK(st.load(m, got, err));
            CHECK(same_state(got.live, sa.checkpoints[6]) && got.checkpoints.size() == 2 &&
                  got.checkpoints[0].source == R && got.checkpoints[1].source == T);
        }
        remove_dir(d2);
        // two chats that share a root: its state file is shared, and stays while either chat is stored
        const std::string d3 = dir + "-share";
        std::vector<int32_t> b = a;
        for (size_t i = 1500; i < b.size(); ++i) b[i] = (int32_t) (i * 13 + 2);
        SavedConversation sb = chat(b, {{1000, R}, {7000, T}, {8000, T}, {9000, T}});
        copy_state_bytes(sb.checkpoints[0], sa.checkpoints[0]);
        {
            SavedConversation first = sa;
            ConversationStore probe;
            CHECK(probe.open(d3, id, 1ull << 30, err));
            CHECK(probe.put(std::move(first), 1ull << 30));
            probe.flush();
        }
        const uint64_t one = dir_bytes(d3);
        ConversationStore st;
        CHECK(st.open(d3, id, one + one / 4, err));
        CHECK(st.stats().files == 1 && state_files(d3) == 4);
        CHECK(st.put(SavedConversation(sb), 1ull << 30));
        st.flush();
        CHECK(st.stats().files == 1 && st.stats().evicted == 1);   // the first chat went out of the budget
        CHECK(state_files(d3) == 4);   // its three turn ends are gone; the shared root stays with the second
        std::vector<int32_t> p(b.begin(), b.begin() + 7500);
        p.push_back(-5);
        auto m = st.best(p.data(), p.size(), {}, true);
        CHECK(m.tokens == 7000);
        SavedConversation got;
        CHECK(st.load(m, got, err));
        CHECK(same_state(got.live, sb.checkpoints[1]) && same_state(got.checkpoints[0], sa.checkpoints[0]));
        remove_dir(d3);
    }
    std::vector<basalt::platform::DirFile> left;
    if (basalt::platform::list_files(dir, left))
        for (const auto& f : left) basalt::platform::remove_file(dir + "/" + f.name);
    basalt::platform::remove_file(dir);
    if (failures) {
        std::fprintf(stderr, "conversation_store_test: %d failures\n", failures);
        return 1;
    }
    std::printf("conversation_store_test: ok\n");
    return 0;
}
