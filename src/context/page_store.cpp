// src/context/page_store.cpp - see include/basalt/context/page_store.hpp.
#include "basalt/context/page_store.hpp"

#include "basalt/context/hash.hpp"
#include "basalt/platform/host_allocator.hpp"

#include <cstring>

namespace basalt::context {

struct PageStore::Page {
    uint8_t* data = nullptr;
    Digest digest{};
    bool sealed = false;
    uint32_t refs = 0;   // segment slots holding this page
};

struct PageStore::Segment {
    ContextId owner;
    StateKind kind = StateKind::kv;
    uint32_t part = 0;
    std::vector<uint64_t> pages;
};

PageStore::PageStore(uint64_t page_bytes, uint64_t budget_bytes)
    : page_bytes_(page_bytes), budget_bytes_(budget_bytes) {}

PageStore::~PageStore() {
    for (auto& [id, page] : pages_) basalt::platform::release_host_buffer(page.data, kPageAlign);
}

uint64_t PageStore::take_page_locked() {
    const uint64_t cap = budget_bytes_ / page_bytes_;
    if (used_pages_ >= cap) return 0;
    void* p = basalt::platform::allocate_host_buffer(page_bytes_, kPageAlign);
    if (p == nullptr) return 0;
    std::memset(p, 0, page_bytes_);
    const uint64_t id = next_page_++;
    pages_[id] = Page{static_cast<uint8_t*>(p), {}, false, 1};
    ++used_pages_;
    return id;
}

void PageStore::drop_slot_locked(uint64_t page_id) {
    auto it = pages_.find(page_id);
    if (it == pages_.end()) return;
    if (--it->second.refs > 0) return;
    if (it->second.sealed) {
        const auto sit = sealed_.find(it->second.digest);
        if (sit != sealed_.end() && sit->second == page_id) sealed_.erase(sit);
    }
    basalt::platform::release_host_buffer(it->second.data, kPageAlign);
    pages_.erase(it);
    --used_pages_;
}

uint32_t PageStore::create_segment(const ContextId& owner, StateKind kind, uint32_t part) {
    std::lock_guard<std::mutex> lock(mu_);
    const uint32_t id = next_segment_++;
    segments_[id] = Segment{owner, kind, part, {}};
    return id;
}

bool PageStore::has_segment(uint32_t segment) const {
    std::lock_guard<std::mutex> lock(mu_);
    return segments_.count(segment) > 0;
}

bool PageStore::segment_info(uint32_t segment, ContextId& owner, StateKind& kind, uint32_t& part) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = segments_.find(segment);
    if (it == segments_.end()) return false;
    owner = it->second.owner;
    kind = it->second.kind;
    part = it->second.part;
    return true;
}

bool PageStore::append_page(uint32_t segment) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end()) return false;
    const uint64_t id = take_page_locked();
    if (id == 0) return false;
    sit->second.pages.push_back(id);
    return true;
}

bool PageStore::share_prefix(uint32_t child, uint32_t parent, uint64_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto cit = segments_.find(child);
    const auto pit = segments_.find(parent);
    if (cit == segments_.end() || pit == segments_.end()) return false;
    if (cit->second.kind != pit->second.kind || cit->second.part != pit->second.part) return false;
    if (n > pit->second.pages.size()) return false;
    Segment& c = cit->second;
    const Segment& p = pit->second;
    if (n < c.pages.size()) return false;                  // a shared head never shrinks
    // The child's existing pages must be the first pages of the parent, alias for alias: sharing only grows.
    for (size_t i = 0; i < c.pages.size(); ++i) {
        if (i >= p.pages.size() || c.pages[i] != p.pages[i]) return false;
    }
    for (uint64_t i = c.pages.size(); i < n; ++i) {
        c.pages.push_back(p.pages[i]);
        ++pages_[p.pages[i]].refs;
    }
    return true;
}

uint64_t PageStore::segment_pages(uint32_t segment) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = segments_.find(segment);
    return it == segments_.end() ? 0 : it->second.pages.size();
}

bool PageStore::page_shared(uint32_t segment, uint64_t index) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return false;
    return pages_.at(sit->second.pages[index]).refs > 1;
}

bool PageStore::page_sealed(uint32_t segment, uint64_t index) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return false;
    return pages_.at(sit->second.pages[index]).sealed;
}

bool PageStore::page_digest(uint32_t segment, uint64_t index, Digest& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return false;
    const Page& page = pages_.at(sit->second.pages[index]);
    if (!page.sealed) return false;
    out = page.digest;
    return true;
}

const uint8_t* PageStore::read_page(uint32_t segment, uint64_t index) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return nullptr;
    return pages_.at(sit->second.pages[index]).data;
}

uint8_t* PageStore::write_page(uint32_t segment, uint64_t index) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return nullptr;
    uint64_t& slot = sit->second.pages[index];
    Page& page = pages_.at(slot);
    if (page.sealed) return nullptr;                       // a frozen prefix page never changes
    if (page.refs > 1) {                                   // shared and writable: copy first, then write
        uint8_t* const old_data = page.data;               // take_page_locked may rehash pages_ and drop `page`
        const uint64_t fresh = take_page_locked();
        if (fresh == 0) return nullptr;                    // no room to copy: the caller sheds state
        std::memcpy(pages_[fresh].data, old_data, page_bytes_);
        drop_slot_locked(slot);
        slot = fresh;
        return pages_[fresh].data;
    }
    return page.data;
}

bool PageStore::seal_page(uint32_t segment, uint64_t index, Digest& out_digest) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto sit = segments_.find(segment);
    if (sit == segments_.end() || index >= sit->second.pages.size()) return false;
    uint64_t& slot = sit->second.pages[index];
    Page& page = pages_.at(slot);
    if (page.sealed) { out_digest = page.digest; return true; }
    const Digest digest = sha256(page.data, page_bytes_);
    const auto existing = sealed_.find(digest);
    if (existing != sealed_.end()) {                       // the bytes are already held once: alias them
        ++pages_[existing->second].refs;
        drop_slot_locked(slot);
        slot = existing->second;
    } else {
        page.sealed = true;
        page.digest = digest;
        sealed_[digest] = slot;
    }
    out_digest = digest;
    return true;
}

void PageStore::release_segment(uint32_t segment) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = segments_.find(segment);
    if (it == segments_.end()) return;
    for (const uint64_t id : it->second.pages) drop_slot_locked(id);
    segments_.erase(it);
}

uint64_t PageStore::used_pages() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_pages_;
}

uint64_t PageStore::used_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_pages_ * page_bytes_;
}

uint64_t PageStore::shared_pages() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t n = 0;
    for (const auto& [id, page] : pages_)
        if (page.refs > 1) ++n;
    return n;
}

}  // namespace basalt::context
