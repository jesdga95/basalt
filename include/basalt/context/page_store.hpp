// Shared, copy-on-write state pages of agent contexts. Host-only; no CUDA headers.
//
// Every page lives inside one range the caller hands over (carved from its existing pinned host pool) and is placed
// by a basalt::core::HostArena over it: exhaustion is a rejection, never a second allocation. A context is a fixed
// table of page slots, backed lazily on first write. A fork shares every page of its parent; a page referenced more
// than once, or sealed, is immutable and a write to it copies it first, so neither side ever sees the other's writes.
// Sealing keys a page by its content digest; an equal sealed page is shared instead of kept twice. A lease is a
// read-only fork: it keeps the pages it saw alive after their context is removed. A page is reclaimed with its last
// reference.
#pragma once

#include "basalt/context/ids.hpp"
#include "basalt/core/host_arena.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace basalt::context {

enum class PageStatus : uint8_t {
    ok,
    not_open,
    invalid_argument,
    unknown_context,
    context_exists,
    out_of_range,
    read_only,
    budget,           ///< no free page within the caller's range
    digest_mismatch,  ///< a page's bytes differ from its content key
};
const char* page_status_name(PageStatus s);

struct PageResult {
    PageStatus status = PageStatus::ok;
    std::string reason;  ///< why, with the figures that decided it; empty when ok
    explicit operator bool() const { return status == PageStatus::ok; }
};

/// SHA-256 of a page's bytes, domain-separated from every other digest of the context store.
Digest page_digest(const void* data, size_t bytes);

class PageStore {
public:
    struct Stats {
        uint64_t capacity = 0, used = 0;  ///< bytes of the caller's range, and those pages hold
        size_t pages = 0, sealed = 0, contexts = 0;
        uint64_t cow_copies = 0, dedup_hits = 0, rejections = 0;
    };

    PageStore() = default;
    PageStore(const PageStore&) = delete;
    PageStore& operator=(const PageStore&) = delete;

    /// Places pages of `page_bytes` bytes in [base, base + bytes) only. Fails while pages or contexts exist.
    PageResult open(void* base, size_t bytes, size_t page_bytes);
    size_t page_bytes() const { return page_bytes_; }

    /// A context of `pages` unbacked slots; an unbacked slot reads as zeros and takes no space.
    PageResult create(const ContextId& id, size_t pages);
    /// `child` references every page `parent` holds now.
    PageResult fork(const ContextId& parent, const ContextId& child);
    /// A read-only fork named `lease`; remove it to let its pages go.
    PageResult lease(const ContextId& source, const ContextId& lease);
    /// Drops the context's references; a page shared with a child or lease stays.
    PageResult remove(const ContextId& id);

    /// Writes `n` bytes at `offset` of page slot `page`, backing the slot or copying a shared page first.
    PageResult write(const ContextId& id, size_t page, size_t offset, const void* src, size_t n);
    PageResult read(const ContextId& id, size_t page, size_t offset, void* dst, size_t n) const;

    /// Makes every backed page immutable and keys it by content, sharing an equal sealed page instead of keeping a
    /// copy. `digests` (when given) gets one digest per slot, all zeros for an unbacked one.
    PageResult seal(const ContextId& id, std::vector<Digest>* digests = nullptr);
    /// Re-hashes the context's sealed pages against their keys.
    PageResult verify(const ContextId& id) const;

    Stats stats() const;

private:
    using PageId = uint64_t;  // 0: unbacked slot
    struct Page {
        uint8_t* data = nullptr;
        uint32_t refs = 0;
        bool sealed = false;
        Digest digest{};
    };
    struct Ctx;

    std::shared_ptr<Ctx> find(const ContextId& id) const;
    PageResult add(const ContextId& source, const ContextId& child, bool frozen);
    PageResult fail(PageStatus s, std::string reason) const;
    uint8_t* allocate_locked(PageResult& why);
    void unref_locked(PageId id);

    mutable std::mutex mu_;  // after a context's own lock, never before it
    core::HostArena arena_;
    size_t page_bytes_ = 0, slot_bytes_ = 0;
    bool open_ = false;
    PageId next_ = 1;
    std::unordered_map<PageId, Page> pages_;
    std::map<Digest, PageId> sealed_;
    std::unordered_map<ContextId, std::shared_ptr<Ctx>> contexts_;
    Stats stats_;
};

}  // namespace basalt::context
