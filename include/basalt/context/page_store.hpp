// include/basalt/context/page_store.hpp - the pooled RAM pages a context's running state lives in.
//
// A segment is the ordered page list of one (context, state kind, part): the RAM image of one manifest entry.
// Pages are fixed-size and refcounted, so a fork shares the parent's prefix for free: the child's segment aliases
// the parent's pages and touches no bytes. Two rules make the sharing safe:
//
//   sealed pages are immutable - sealing computes a page's digest and freezes it; a write to a sealed page is
//       refused, so a shared prefix never changes under its readers, and
//   writing a page that is shared but unsealed copies it first (copy-on-write): the writer keeps a private page
//       with the old bytes and the other holders keep theirs untouched.
//
// Sealing also deduplicates: a page whose bytes equal an already-sealed page's becomes an alias of it, which is how
// two contexts that prefilled the same prefix end up holding it once.
//
// The store enforces one hard byte budget over its physical pages: an append or a COW copy that would cross it
// fails, and the caller sheds state (cool a context, park a segment) instead. Freeing happens when the last segment
// reference to a page drops; no separate GC pass is needed.
//
// Thread-safe: every call takes the store's lock. Page pointers are only valid while the caller holds a reference
// to the segment (a lease, in registry terms); the engine must not cache them across releases.
#pragma once

#include "basalt/context/ids.hpp"
#include "basalt/context/manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace basalt::context {

class PageStore {
public:
    /// Pages are allocated at this alignment, so a page size is any positive multiple of it.
    static constexpr uint64_t kPageAlign = 4096;

    /// `budget_bytes` is the hard cap on physical page bytes. Zero budget is allowed: an empty store that refuses
    /// every append, useful for tests.
    PageStore(uint64_t page_bytes, uint64_t budget_bytes);
    ~PageStore();

    PageStore(const PageStore&) = delete;
    PageStore& operator=(const PageStore&) = delete;

    uint64_t page_bytes() const { return page_bytes_; }
    uint64_t budget_bytes() const { return budget_bytes_; }

    /// A new empty segment owned by `owner`, holding state `kind` part `part`. Segment ids are never reused.
    uint32_t create_segment(const ContextId& owner, StateKind kind, uint32_t part);
    bool has_segment(uint32_t segment) const;
    /// The owner, kind and part the segment was created with; false for an unknown id.
    bool segment_info(uint32_t segment, ContextId& owner, StateKind& kind, uint32_t& part) const;

    /// Appends a fresh zeroed private page. False when the budget is full or the segment is gone.
    bool append_page(uint32_t segment);

    /// The first `n` pages of `parent` become the head of `child`, aliasing them (each reference counted).
    /// `child` must be empty, or already a shared head of `parent` extended to `n` (sharing only ever grows).
    /// False (no change) for unknown segments, a foreign kind or part, `n` past the parent's length, or a child
    /// whose existing head is not `parent`'s.
    bool share_prefix(uint32_t child, uint32_t parent, uint64_t n);

    uint64_t segment_pages(uint32_t segment) const;
    /// Whether the page at `index` is shared with another segment or sealed.
    bool page_shared(uint32_t segment, uint64_t index) const;
    bool page_sealed(uint32_t segment, uint64_t index) const;
    /// The sealed page's digest; false while the page is unsealed.
    bool page_digest(uint32_t segment, uint64_t index, Digest& out) const;

    /// Readable bytes of the page; null for an unknown segment or index. Valid until the segment or page is freed.
    const uint8_t* read_page(uint32_t segment, uint64_t index) const;
    /// Writable bytes: the page as it stands when private and unsealed, a private copy when shared and unsealed
    /// (copy-on-write). Null - changing nothing - when the page is sealed or the copy would cross the budget.
    uint8_t* write_page(uint32_t segment, uint64_t index);

    /// Freezes the page at `index` and records its digest, deduplicating it against the sealed pages: when another
    /// sealed page holds the same bytes, this slot aliases that one and the private page is freed. A page already
    /// sealed returns its digest unchanged. False for an unknown segment or index.
    bool seal_page(uint32_t segment, uint64_t index, Digest& out_digest);

    /// Drops the segment and every reference it holds; pages it was the last holder of are freed.
    void release_segment(uint32_t segment);

    /// Physical pages held and the bytes they account for (pages * page_bytes).
    uint64_t used_pages() const;
    uint64_t used_bytes() const;
    /// Pages held by more than one segment (the sharing the forks and dedup bought).
    uint64_t shared_pages() const;

private:
    struct Page;
    struct Segment;
    struct DigestHash {
        size_t operator()(const Digest& d) const {
            // FNV-1a over the digest: its bytes are already well mixed, this only buckets them.
            size_t h = 1469598103934665603ull;
            for (uint8_t b : d) { h ^= b; h *= 1099511628211ull; }
            return h;
        }
    };
    struct DigestEqual {
        bool operator()(const Digest& a, const Digest& b) const { return a == b; }
    };

    uint64_t take_page_locked();            // fresh zeroed page id, 0 at budget (id 0 is never handed out)
    void drop_slot_locked(uint64_t page_id);

    const uint64_t page_bytes_;
    const uint64_t budget_bytes_;
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, Segment> segments_;
    uint32_t next_segment_ = 1;
    std::unordered_map<uint64_t, Page> pages_;
    uint64_t next_page_ = 1;
    uint64_t used_pages_ = 0;
    // sealed page bytes -> the page holding them; content-addressed dedup
    std::unordered_map<Digest, uint64_t, DigestHash, DigestEqual> sealed_;
};

}  // namespace basalt::context
