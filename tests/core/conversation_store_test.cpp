// The disk tier of the conversation cache (conversation_store.hpp): a parked image written, matched by its live tokens
// and checkpoints, read back cut at the match (from the write queue and from the files, in a fresh store as after a
// restart; inside a K/V block, on a block's edge and in the tail), refused when a payload byte of a shared block changed
// or the identity differs, written again a turn on with only its new state, stored as of its last checkpoint when its
// live end is a reply past it, and the directory kept to its budget.
#include "basalt/core/conversation_store.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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
}  // namespace

int main() {
    const std::string dir = basalt::platform::temp_dir() + "/basalt-store-test-" +
                            std::to_string((long long) basalt::platform::process_id());
    std::vector<int32_t> ids(10000);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) (i * 7 + 3);
    const SavedConversation src = image(ids, {2048, 4096, 5000});   // its live end far past the last checkpoint
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
