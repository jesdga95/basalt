#include "basalt/context/page_store.hpp"
#include "basalt/context/fingerprint.hpp"

#include <algorithm>
#include <cstring>
#include <shared_mutex>

namespace basalt::context {
namespace {

std::string page_name(const ContextId& id, size_t page) { return "page " + std::to_string(page) + " of " + id; }

}  // namespace

const char* page_status_name(PageStatus s) {
    switch (s) {
        case PageStatus::ok: return "ok";
        case PageStatus::not_open: return "not_open";
        case PageStatus::invalid_argument: return "invalid_argument";
        case PageStatus::unknown_context: return "unknown_context";
        case PageStatus::context_exists: return "context_exists";
        case PageStatus::out_of_range: return "out_of_range";
        case PageStatus::read_only: return "read_only";
        case PageStatus::budget: return "budget";
        case PageStatus::digest_mismatch: return "digest_mismatch";
    }
    return "unknown";
}

Digest page_digest(const void* data, size_t bytes) {
    return DomainHash("basalt/context/page/v1").u64(bytes).bytes(data, bytes).finish();
}

// `mu` excludes the context's writers from its readers, forks and seals; the table itself changes under the store's.
struct PageStore::Ctx {
    mutable std::shared_mutex mu;
    std::vector<PageId> table;
    bool frozen = false;
    bool dead = false;
};

PageResult PageStore::fail(PageStatus s, std::string reason) const { return PageResult{s, std::move(reason)}; }

std::shared_ptr<PageStore::Ctx> PageStore::find(const ContextId& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = contexts_.find(id);
    return it == contexts_.end() ? nullptr : it->second;
}

PageResult PageStore::open(void* base, size_t bytes, size_t page_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!pages_.empty() || !contexts_.empty()) return fail(PageStatus::invalid_argument, "pages or contexts are live");
    // Whole slots on the arena's alignment: freed slots tile, so a free slot always has a page-sized range.
    const size_t slot = page_bytes > SIZE_MAX - 255 ? 0 : (page_bytes + 255) & ~size_t(255);
    const size_t skip = base ? (256 - reinterpret_cast<uintptr_t>(base) % 256) % 256 : 0;
    const size_t usable = !slot || bytes < skip ? 0 : (bytes - skip) / slot * slot;
    if (!base || usable == 0)
        return fail(PageStatus::invalid_argument, "range of " + std::to_string(bytes) + " B cannot hold a page of " +
                                                      std::to_string(page_bytes) + " B");
    if (!arena_.reset(static_cast<uint8_t*>(base) + skip, usable))
        return fail(PageStatus::invalid_argument, "range still holds pages");
    page_bytes_ = page_bytes;
    slot_bytes_ = slot;
    open_ = true;
    stats_ = {};
    return {};
}

PageResult PageStore::create(const ContextId& id, size_t pages) {
    if (!valid_context_id(id)) return fail(PageStatus::invalid_argument, "invalid context id");
    if (pages == 0) return fail(PageStatus::invalid_argument, "a context needs at least one page slot");
    auto ctx = std::make_shared<Ctx>();
    ctx->table.assign(pages, 0);
    std::lock_guard<std::mutex> lk(mu_);
    if (!open_) return fail(PageStatus::not_open, "page store is not open");
    if (!contexts_.emplace(id, std::move(ctx)).second) return fail(PageStatus::context_exists, id + " exists");
    return {};
}

PageResult PageStore::add(const ContextId& source, const ContextId& child, bool frozen) {
    if (!valid_context_id(child)) return fail(PageStatus::invalid_argument, "invalid context id");
    const auto src = find(source);
    if (!src) return fail(PageStatus::unknown_context, "no context " + source);
    std::shared_lock<std::shared_mutex> sl(src->mu);  // no write of the source lands mid-fork
    std::lock_guard<std::mutex> lk(mu_);
    if (src->dead) return fail(PageStatus::unknown_context, "no context " + source);
    if (contexts_.count(child)) return fail(PageStatus::context_exists, child + " exists");
    auto ctx = std::make_shared<Ctx>();
    ctx->table = src->table;
    ctx->frozen = frozen;
    contexts_.emplace(child, std::move(ctx));
    for (PageId p : src->table)
        if (p) ++pages_.at(p).refs;
    return {};
}

PageResult PageStore::fork(const ContextId& parent, const ContextId& child) { return add(parent, child, false); }
PageResult PageStore::lease(const ContextId& source, const ContextId& lease) { return add(source, lease, true); }

void PageStore::unref_locked(PageId id) {
    const auto it = pages_.find(id);
    if (--it->second.refs) return;
    if (it->second.sealed) sealed_.erase(it->second.digest);  // a sealed page is the one its digest keys
    arena_.release(it->second.data);
    pages_.erase(it);
}

PageResult PageStore::remove(const ContextId& id) {
    const auto ctx = find(id);
    if (!ctx) return fail(PageStatus::unknown_context, "no context " + id);
    std::unique_lock<std::shared_mutex> xl(ctx->mu);  // waits out the context's reads and writes
    std::lock_guard<std::mutex> lk(mu_);
    if (ctx->dead) return fail(PageStatus::unknown_context, "no context " + id);
    ctx->dead = true;
    contexts_.erase(id);
    for (PageId p : ctx->table)
        if (p) unref_locked(p);
    ctx->table.clear();
    return {};
}

uint8_t* PageStore::allocate_locked(PageResult& why) {
    if (void* p = arena_.allocate(slot_bytes_)) return static_cast<uint8_t*>(p);
    ++stats_.rejections;
    const auto s = arena_.stats();
    why = fail(PageStatus::budget, "page store budget exhausted: need " + std::to_string(slot_bytes_) + " B, " +
                                       std::to_string(s.used) + " of " + std::to_string(s.capacity) + " B in use");
    return nullptr;
}

PageResult PageStore::write(const ContextId& id, size_t page, size_t offset, const void* src, size_t n) {
    const auto ctx = find(id);
    if (!ctx) return fail(PageStatus::unknown_context, "no context " + id);
    std::unique_lock<std::shared_mutex> xl(ctx->mu);
    uint8_t* dst = nullptr;
    const uint8_t* shared_bytes = nullptr;
    PageId shared = 0;  // the page copied from, still referenced until the copy is done
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx->dead) return fail(PageStatus::unknown_context, "no context " + id);
        if (ctx->frozen) return fail(PageStatus::read_only, id + " is a read-only lease");
        if (page >= ctx->table.size() || offset > page_bytes_ || n > page_bytes_ - offset)
            return fail(PageStatus::out_of_range, page_name(id, page) + ": write past the page or table");
        if (n == 0) return {};
        if (!src) return fail(PageStatus::invalid_argument, "no source bytes");
        PageId& slot = ctx->table[page];
        if (slot && pages_.at(slot).refs == 1) {
            Page& own = pages_.at(slot);
            if (own.sealed) { sealed_.erase(own.digest); own.sealed = false; }  // nobody else can see it
            dst = own.data;
        } else {
            PageResult why;
            dst = allocate_locked(why);
            if (!dst) return why;
            if (slot) {
                shared = slot;
                shared_bytes = pages_.at(slot).data;
                ++stats_.cow_copies;
            } else {
                fresh = true;
            }
            slot = next_++;
            pages_.emplace(slot, Page{dst, 1, false, {}});
        }
    }
    // Only this context references `dst`, and its lock keeps forks and readers out until the bytes are in place.
    const bool whole = offset == 0 && n == page_bytes_;
    if (shared && !whole) std::memcpy(dst, shared_bytes, page_bytes_);
    else if (fresh && !whole) std::memset(dst, 0, page_bytes_);
    std::memcpy(dst + offset, src, n);
    if (shared) {
        std::lock_guard<std::mutex> lk(mu_);
        unref_locked(shared);
    }
    return {};
}

PageResult PageStore::read(const ContextId& id, size_t page, size_t offset, void* dst, size_t n) const {
    const auto ctx = find(id);
    if (!ctx) return fail(PageStatus::unknown_context, "no context " + id);
    std::shared_lock<std::shared_mutex> sl(ctx->mu);
    const uint8_t* from = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx->dead) return fail(PageStatus::unknown_context, "no context " + id);
        if (page >= ctx->table.size() || offset > page_bytes_ || n > page_bytes_ - offset)
            return fail(PageStatus::out_of_range, page_name(id, page) + ": read past the page or table");
        if (n && !dst) return fail(PageStatus::invalid_argument, "no destination");
        if (const PageId p = ctx->table[page]) from = pages_.at(p).data;
    }
    if (!n) return {};
    if (from) std::memcpy(dst, from + offset, n);
    else std::memset(dst, 0, n);
    return {};
}

PageResult PageStore::seal(const ContextId& id, std::vector<Digest>* digests) {
    const auto ctx = find(id);
    if (!ctx) return fail(PageStatus::unknown_context, "no context " + id);
    std::unique_lock<std::shared_mutex> xl(ctx->mu);
    struct Item { size_t slot; PageId page; const uint8_t* data; Digest digest; };
    std::vector<Item> todo;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx->dead) return fail(PageStatus::unknown_context, "no context " + id);
        for (size_t i = 0; i < ctx->table.size(); ++i)
            if (const PageId p = ctx->table[i]; p && !pages_.at(p).sealed) todo.push_back({i, p, pages_.at(p).data, {}});
    }
    for (auto& t : todo) t.digest = page_digest(t.data, page_bytes_);  // stable: private under our lock, or shared
    PageResult out;
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& t : todo) {
        Page& pg = pages_.at(t.page);
        if (pg.sealed) continue;  // a co-owner sealed this shared page meanwhile
        const auto hit = sealed_.find(t.digest);
        if (hit == sealed_.end()) {
            pg.sealed = true;
            pg.digest = t.digest;
            sealed_.emplace(t.digest, t.page);
            continue;
        }
        Page& same = pages_.at(hit->second);
        if (std::memcmp(same.data, pg.data, page_bytes_) != 0) {  // corruption: share nothing
            if (out) out = fail(PageStatus::digest_mismatch, page_name(id, t.slot) + " collides with a sealed page");
            continue;
        }
        ++same.refs;
        ctx->table[t.slot] = hit->second;
        ++stats_.dedup_hits;
        unref_locked(t.page);
    }
    if (digests) {
        digests->assign(ctx->table.size(), Digest{});
        for (size_t i = 0; i < ctx->table.size(); ++i)
            if (const PageId p = ctx->table[i]) (*digests)[i] = pages_.at(p).digest;
    }
    return out;
}

PageResult PageStore::verify(const ContextId& id) const {
    const auto ctx = find(id);
    if (!ctx) return fail(PageStatus::unknown_context, "no context " + id);
    std::shared_lock<std::shared_mutex> sl(ctx->mu);
    struct Item { size_t slot; const uint8_t* data; Digest digest; };
    std::vector<Item> items;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx->dead) return fail(PageStatus::unknown_context, "no context " + id);
        for (size_t i = 0; i < ctx->table.size(); ++i)
            if (const PageId p = ctx->table[i]; p && pages_.at(p).sealed)
                items.push_back({i, pages_.at(p).data, pages_.at(p).digest});
    }
    for (const auto& it : items)
        if (page_digest(it.data, page_bytes_) != it.digest)
            return fail(PageStatus::digest_mismatch, page_name(id, it.slot) + " no longer matches its content key");
    return {};
}

PageStore::Stats PageStore::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    const auto a = arena_.stats();
    s.capacity = a.capacity;
    s.used = a.used;
    s.pages = pages_.size();
    s.sealed = sealed_.size();
    s.contexts = contexts_.size();
    return s;
}

}  // namespace basalt::context
