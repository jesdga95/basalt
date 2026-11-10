// tests/context/page_store_test.cpp - sharing, copy-on-write, seal dedup and the hard budget.
#include "basalt/context/page_store.hpp"

#include <cstdio>
#include <cstring>

namespace {
using namespace basalt::context;
int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) { ++failures; std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

constexpr uint64_t kPage = 4096;

void fork_shares_for_free() {
    PageStore s(kPage, 100 * kPage);
    const uint32_t parent = s.create_segment("parent", StateKind::kv, 0);
    for (int i = 0; i < 4; ++i) CHECK(s.append_page(parent));
    for (int i = 0; i < 4; ++i) std::memset(s.write_page(parent, i), 0x10 + i, kPage);
    Digest d;
    CHECK(s.seal_page(parent, 0, d));
    CHECK(s.seal_page(parent, 1, d));
    const uint64_t before = s.used_pages();
    CHECK(before == 4);

    const uint32_t child = s.create_segment("child", StateKind::kv, 0);
    CHECK(s.share_prefix(child, parent, 2));
    CHECK(s.used_pages() == before);          // the fork touched no bytes
    CHECK(s.shared_pages() == 2);
    CHECK(s.segment_pages(child) == 2);
    CHECK(s.page_shared(child, 0) && s.page_sealed(child, 0));
    // the child reads exactly the parent's bytes
    CHECK(std::memcmp(s.read_page(child, 0), s.read_page(parent, 0), kPage) == 0);
    // sealed prefix pages are immutable for both holders
    CHECK(s.write_page(parent, 0) == nullptr);
    CHECK(s.write_page(child, 0) == nullptr);
    // sharing only grows from the parent's head
    CHECK(s.share_prefix(child, parent, 4));
    CHECK(s.segment_pages(child) == 4);
    CHECK(!s.share_prefix(child, parent, 3));
    const uint32_t foreign = s.create_segment("f", StateKind::gdn, 0);
    CHECK(s.append_page(foreign));
    CHECK(!s.share_prefix(child, foreign, 1));   // foreign kind
    const uint32_t stranger = s.create_segment("s", StateKind::kv, 0);
    CHECK(s.append_page(stranger));
    CHECK(!s.share_prefix(stranger, parent, 1));  // its head is not the parent's
}

void cow_keeps_the_parent() {
    PageStore s(kPage, 100 * kPage);
    const uint32_t parent = s.create_segment("p", StateKind::kv, 0);
    CHECK(s.append_page(parent));
    uint8_t* p0 = s.write_page(parent, 0);
    CHECK(p0 != nullptr);
    std::memset(p0, 0xAB, kPage);
    const uint32_t child = s.create_segment("c", StateKind::kv, 0);
    CHECK(s.share_prefix(child, parent, 1));
    CHECK(s.page_shared(child, 0) && !s.page_sealed(child, 0));

    uint8_t* c0 = s.write_page(child, 0);      // copy-on-write
    CHECK(c0 != nullptr && c0 != p0);
    CHECK(s.used_pages() == 2);
    CHECK(std::memcmp(p0, c0, kPage) == 0);    // the copy starts identical
    std::memset(c0, 0xCD, kPage);
    for (uint64_t i = 0; i < kPage; ++i) CHECK(p0[i] == 0xAB);   // the parent never changed
    // the child now holds its private copy; writing again does not copy
    CHECK(s.write_page(child, 0) == c0);
    CHECK(s.used_pages() == 2);
}

void seal_dedups() {
    PageStore s(kPage, 100 * kPage);
    const uint32_t a = s.create_segment("a", StateKind::kv, 0);
    const uint32_t b = s.create_segment("b", StateKind::kv, 0);
    CHECK(s.append_page(a) && s.append_page(b));
    std::memset(s.write_page(a, 0), 0x5A, kPage);
    std::memset(s.write_page(b, 0), 0x5A, kPage);   // the same bytes, arrived at twice
    Digest da, db;
    CHECK(s.seal_page(a, 0, da));
    CHECK(s.seal_page(b, 0, db));
    CHECK(da == db);
    CHECK(s.used_pages() == 1);                     // held once
    CHECK(s.shared_pages() == 1);
    CHECK(s.read_page(a, 0) == s.read_page(b, 0));
    // a third copy of the same bytes also collapses; different bytes do not
    CHECK(s.append_page(b));
    std::memset(s.write_page(b, 1), 0x5A, kPage);
    Digest dd;
    CHECK(s.seal_page(b, 1, dd) && dd == da);
    CHECK(s.used_pages() == 1);
    CHECK(s.append_page(b));
    std::memset(s.write_page(b, 2), 0x99, kPage);
    Digest de;
    CHECK(s.seal_page(b, 2, de) && de != da);
    CHECK(s.used_pages() == 2);
    // releasing one holder keeps the shared page for the other
    s.release_segment(a);
    CHECK(s.used_pages() == 2);
    Digest again;
    CHECK(s.page_digest(b, 0, again) && again == da);
}

void budget_is_hard() {
    PageStore s(kPage, 3 * kPage);
    const uint32_t a = s.create_segment("a", StateKind::kv, 0);
    CHECK(s.append_page(a) && s.append_page(a) && s.append_page(a));
    CHECK(!s.append_page(a));                       // the budget is full
    CHECK(s.used_bytes() == 3 * kPage);
    const uint32_t b = s.create_segment("b", StateKind::kv, 0);
    CHECK(s.share_prefix(b, a, 3));
    CHECK(s.write_page(b, 0) == nullptr);           // no room to copy: the write changes nothing
    CHECK(s.page_shared(b, 0));
    s.release_segment(a);                           // b still holds the pages
    CHECK(s.used_pages() == 3);
    s.release_segment(b);
    CHECK(s.used_pages() == 0);
    CHECK(!s.has_segment(b));
}

void release_frees_last_holder() {
    PageStore s(kPage, 10 * kPage);
    const uint32_t a = s.create_segment("a", StateKind::kv, 3);
    CHECK(s.append_page(a));
    Digest d;
    CHECK(s.seal_page(a, 0, d));
    const uint32_t b = s.create_segment("b", StateKind::kv, 3);
    CHECK(s.share_prefix(b, a, 1));
    s.release_segment(a);
    CHECK(s.used_pages() == 1);                     // b still reads it
    Digest d2;
    CHECK(s.seal_page(b, 0, d2) && d2 == d);
    s.release_segment(b);
    CHECK(s.used_pages() == 0);
    ContextId owner;
    StateKind kind;
    uint32_t part = 0;
    CHECK(!s.segment_info(a, owner, kind, part));
}

}  // namespace

int main() {
    fork_shares_for_free();
    cow_keeps_the_parent();
    seal_dedups();
    budget_is_hard();
    release_frees_last_holder();
    if (failures) { std::printf("%d page_store test failures\n", failures); return 1; }
    std::printf("page_store tests ok\n");
    return 0;
}
