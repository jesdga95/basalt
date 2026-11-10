#include "basalt/context/page_store.hpp"
#include "basalt/context/hash.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace basalt::context;

namespace {
void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
void ok(const PageResult& r, const char* message) {
    if (!r) { std::fprintf(stderr, "FAIL: %s: %s %s\n", message, page_status_name(r.status), r.reason.c_str()); std::abort(); }
}
// A caller's range as its pool hands it out: page-aligned.
struct Region {
    explicit Region(size_t n) : base(static_cast<uint8_t*>(::operator new(n, std::align_val_t(4096)))), bytes(n) {
        std::memset(base, 0, n);
    }
    ~Region() { ::operator delete(base, std::align_val_t(4096)); }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;
    uint8_t* data() const { return base; }
    size_t size() const { return bytes; }
    uint8_t& operator[](size_t i) { return base[i]; }
    uint8_t* base;
    size_t bytes;
};
std::vector<uint8_t> pattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937 r(seed);
    for (auto& b : v) b = static_cast<uint8_t>(r());
    return v;
}
std::vector<uint8_t> page_of(const PageStore& s, const ContextId& id, size_t page) {
    std::vector<uint8_t> v(s.page_bytes());
    ok(s.read(id, page, 0, v.data(), v.size()), "read page");
    return v;
}

void digests() {
    check(to_hex(page_digest("abc", 3)) == "ea29555ec7d8ffd05e99a85faeb8f4dd0739f204b718d94c4f743eb07f1b29c1", "short digest");
    std::vector<uint8_t> data(1000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
    check(to_hex(page_digest(data.data(), data.size())) ==
              "2857fbfab5ee6da605fbfeca2bbf9c9e91c1c96138b6b62c5d570121847f9e58", "multi-block digest");
    check(to_hex(page_digest(nullptr, 0)) == "70f00a9d6a9482286335a9a567cac49722d4bf81a16934469051dd5cfe73e30a", "empty digest");
}

void lazy_admission() {
    constexpr size_t page = 4096;
    Region region(4 * page);
    PageStore s;
    check(s.create("a", 1).status == PageStatus::not_open, "closed store rejects");
    check(s.open(nullptr, region.size(), page).status == PageStatus::invalid_argument, "no range");
    check(s.open(region.data(), page - 1, page).status == PageStatus::invalid_argument, "range under one page");
    ok(s.open(region.data(), region.size(), page), "open");
    check(s.create("bad id", 4).status == PageStatus::invalid_argument, "context id checked");
    ok(s.create("a", 64), "a table larger than the range is lazy");
    check(s.stats().used == 0 && s.stats().pages == 0, "nothing backed before a write");
    check(s.create("a", 1).status == PageStatus::context_exists, "duplicate id");
    const auto zero = page_of(s, "a", 63);
    for (uint8_t b : zero) check(b == 0, "unbacked slot reads as zeros");
    const uint8_t x = 7;
    for (size_t i = 0; i < 4; ++i) ok(s.write("a", i, 100, &x, 1), "lazy backing within budget");
    const auto r = s.write("a", 4, 0, &x, 1);
    check(r.status == PageStatus::budget && r.reason.find("budget exhausted") != std::string::npos &&
              r.reason.find(std::to_string(region.size())) != std::string::npos, "explicit budget rejection");
    check(s.stats().rejections == 1 && s.stats().used == region.size(), "rejection counted, nothing grew");
    const auto p0 = page_of(s, "a", 0);
    check(p0[100] == 7 && p0[99] == 0 && p0[101] == 0, "partial write on a fresh page zero-fills the rest");
    check(s.write("a", 64, 0, &x, 1).status == PageStatus::out_of_range, "slot past the table");
    check(s.write("a", 0, page, &x, 1).status == PageStatus::out_of_range, "byte past the page");
    ok(s.write("a", 0, page, &x, 0), "empty write at the end is a no-op");
    ok(s.remove("a"), "remove");
    check(s.stats().used == 0 && s.stats().pages == 0 && s.stats().contexts == 0, "remove frees every page");
    ok(s.create("b", 4), "create b");
    for (size_t i = 0; i < 4; ++i) ok(s.write("b", i, 0, &x, 1), "freed space is reused");
    check(s.open(region.data(), region.size(), page).status == PageStatus::invalid_argument, "no reopen while live");
    ok(s.remove("b"), "remove b");
}

void cow_fork_lease() {
    constexpr size_t page = 1024, n = 8;
    Region region(32 * page);
    PageStore s;
    ok(s.open(region.data(), region.size(), page), "open");
    ok(s.create("p", n), "parent");
    std::vector<std::vector<uint8_t>> want(n);
    for (size_t i = 0; i < n; ++i) {
        want[i] = pattern(page, uint32_t(i + 1));
        ok(s.write("p", i, 0, want[i].data(), page), "fill parent");
    }
    ok(s.fork("p", "c"), "fork");
    check(s.fork("p", "c").status == PageStatus::context_exists, "fork id unique");
    check(s.fork("none", "d").status == PageStatus::unknown_context, "fork of an unknown context");
    check(s.stats().pages == n, "a fork copies nothing");
    const uint8_t y = 0xAB;
    ok(s.write("c", 2, 5, &y, 1), "child write");
    check(page_of(s, "p", 2) == want[2], "parent page untouched by the child's write");
    auto c2 = want[2];
    c2[5] = y;
    check(page_of(s, "c", 2) == c2, "child sees its write over the parent's bytes");
    ok(s.write("p", 3, 0, &y, 1), "parent write to a shared page");
    check(page_of(s, "c", 3) == want[3], "child untouched by the parent's write");
    want[3][0] = y;
    check(s.stats().cow_copies == 2 && s.stats().pages == n + 2, "each side copied once");
    ok(s.write("c", 2, 6, &y, 1), "private page written in place");
    check(s.stats().cow_copies == 2, "no copy for a private page");

    ok(s.lease("p", "l"), "lease");
    check(s.write("l", 0, 0, &y, 1).status == PageStatus::read_only, "a lease is read-only");
    const auto held = s.stats().used;
    ok(s.remove("p"), "parent removed under a child and a lease");
    check(s.stats().used == held, "pages held by the lease and the child stay");
    for (size_t i = 0; i < n; ++i) check(page_of(s, "l", i) == want[i], "lease reads the parent's pages");
    check(s.read("p", 0, 0, nullptr, 0).status == PageStatus::unknown_context, "removed parent is gone");
    ok(s.remove("c"), "child removed");
    check(s.stats().pages == n, "only the lease's pages remain");
    ok(s.remove("l"), "lease released");
    check(s.stats().used == 0 && s.stats().pages == 0, "last reference reclaims everything");
}

void seal_dedup_verify() {
    constexpr size_t page = 512, n = 4;
    Region region(16 * page);
    PageStore s;
    ok(s.open(region.data(), region.size(), page), "open");
    for (const char* id : {"x", "y"}) {
        ok(s.create(id, n + 1), "create");
        for (size_t i = 0; i < n; ++i) ok(s.write(id, i, 0, pattern(page, uint32_t(40 + i)).data(), page), "fill");
    }
    std::vector<Digest> dx, dy;
    ok(s.seal("x", &dx), "seal x");
    ok(s.seal("y", &dy), "seal y");
    check(dx == dy && dx[n] == Digest{} && dx[0] == page_digest(pattern(page, 40).data(), page), "content keys");
    check(s.stats().dedup_hits == n && s.stats().pages == n && s.stats().sealed == n, "equal sealed pages shared");
    ok(s.verify("x"), "verify");
    const uint8_t y = 1;
    ok(s.write("y", 1, 0, &y, 1), "write to a sealed shared page");
    check(page_of(s, "x", 1) == pattern(page, 41), "sealed page stays immutable");
    ok(s.verify("x"), "verify after the copy");
    ok(s.verify("y"), "the copy is unsealed and not checked");
    region[3] ^= 0x5A;  // the first page placed sits at the start of the range
    const auto bad = s.verify("x");
    check(bad.status == PageStatus::digest_mismatch && bad.reason.find("page 0 of x") != std::string::npos,
          "corruption is reported with its page");
    ok(s.create("z", n), "create z");
    ok(s.write("z", 0, 0, pattern(page, 40).data(), page), "fill z with the corrupted page's original bytes");
    check(s.seal("z").status == PageStatus::digest_mismatch, "no sharing with a page that fails its key");
    for (const char* id : {"x", "y", "z"}) ok(s.remove(id), "remove");
    check(s.stats().used == 0 && s.stats().sealed == 0, "sealed index empties with its pages");
}

// Threads fork the parent and their own children, write, read, seal, lease and remove at random while another thread
// keeps reading and verifying the parent; the parent is sealed in the middle of it.
void concurrent_divergence() {
    constexpr size_t page = 1024, n = 16, threads = 8, kids_max = 4;
    Region region((n + threads * (kids_max + 1) * n + 16) * page);
    PageStore s;
    ok(s.open(region.data(), region.size(), page), "open");
    ok(s.create("root", n), "root");
    std::vector<std::vector<uint8_t>> want(n);
    for (size_t i = 0; i < n; ++i) {
        want[i] = pattern(page, uint32_t(900 + i));
        ok(s.write("root", i, 0, want[i].data(), page), "fill root");
    }
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> ops{0};
    std::thread checker([&] {
        while (!stop.load()) {
            for (size_t i = 0; i < n; ++i) check(page_of(s, "root", i) == want[i], "parent bytes never change");
            ok(s.verify("root"), "parent keys hold under load");
        }
    });
    std::vector<std::thread> workers;
    for (size_t t = 0; t < threads; ++t)
        workers.emplace_back([&, t] {
            std::mt19937 r(uint32_t(77 + t));
            struct Child { std::string id; std::vector<std::vector<uint8_t>> shadow; };
            std::vector<Child> kids;
            uint64_t serial = 0;
            auto name = [&] { return "t" + std::to_string(t) + "-" + std::to_string(serial++); };
            for (int i = 0; i < 1500; ++i, ++ops) {
                const unsigned op = r() % 10;
                if (kids.empty() || (op < 2 && kids.size() < kids_max)) {
                    Child k{name(), {}};
                    if (kids.empty() || r() % 2) {
                        ok(s.fork("root", k.id), "fork the parent");
                        k.shadow = want;
                    } else {
                        const Child& from = kids[r() % kids.size()];
                        ok(s.fork(from.id, k.id), "fork a child");
                        k.shadow = from.shadow;
                    }
                    kids.push_back(std::move(k));
                } else if (op < 6) {
                    Child& k = kids[r() % kids.size()];
                    const size_t p = r() % n, off = r() % page, len = 1 + r() % (page - off);
                    const auto bytes = pattern(len, uint32_t(r()));
                    ok(s.write(k.id, p, off, bytes.data(), len), "child write");
                    std::memcpy(k.shadow[p].data() + off, bytes.data(), len);
                } else if (op < 8) {
                    const Child& k = kids[r() % kids.size()];
                    const size_t p = r() % n;
                    check(page_of(s, k.id, p) == k.shadow[p], "a child reads its own branch");
                } else if (op == 8) {
                    const Child& k = kids[r() % kids.size()];
                    if (r() % 2) {
                        ok(s.seal(k.id), "seal a child");
                        ok(s.verify(k.id), "verify a child");
                    } else {
                        const std::string lease = name();
                        ok(s.lease(k.id, lease), "lease a child");
                        const size_t p = r() % n;
                        check(page_of(s, lease, p) == k.shadow[p], "a lease reads what it leased");
                        ok(s.remove(lease), "release the lease");
                    }
                } else {
                    const size_t at = r() % kids.size();
                    ok(s.remove(kids[at].id), "remove a child");
                    kids.erase(kids.begin() + std::ptrdiff_t(at));
                }
            }
            for (const auto& k : kids) ok(s.remove(k.id), "drain");
        });
    while (ops.load() < 2000) std::this_thread::yield();
    std::vector<Digest> keys;
    ok(s.seal("root", &keys), "seal the parent under load");
    for (size_t i = 0; i < n; ++i) check(keys[i] == page_digest(want[i].data(), page), "parent keys match its bytes");
    for (auto& w : workers) w.join();
    stop = true;
    checker.join();
    ok(s.verify("root"), "parent page hashes unchanged");
    for (size_t i = 0; i < n; ++i) check(page_of(s, "root", i) == want[i], "parent bytes unchanged");
    const auto st = s.stats();
    check(st.contexts == 1 && st.pages == n && st.used == n * page, "every child page reclaimed");
    check(st.rejections == 0 && st.cow_copies > 0, "the run diverged within its budget");
    ok(s.remove("root"), "remove the parent");
    const auto end = s.stats();
    check(end.used == 0 && end.pages == 0 && end.sealed == 0 && end.contexts == 0, "refcount GC empties the store");
    std::printf("page_store concurrent: %llu ops, %llu copies, %llu shared by content\n",
                (unsigned long long) ops.load(), (unsigned long long) st.cow_copies, (unsigned long long) st.dedup_hits);
}

}  // namespace

int main() {
    digests();
    lazy_admission();
    cow_fork_lease();
    seal_dedup_verify();
    concurrent_divergence();
    std::printf("page_store_test: ok\n");
    return 0;
}
