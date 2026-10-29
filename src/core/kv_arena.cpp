// src/core/kv_arena.cpp - the elastic KV pools.  Read the header first.
#include "basalt/core/kv_arena.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>

namespace basalt::core {

KvArena::~KvArena() { close(); }

bool KvArena::open(const std::vector<uint64_t>& page_bytes, int64_t pages, int64_t page_cells, std::string& err,
                   const std::vector<uint64_t>& tail_bytes) {
    close();
    if (pages <= 0 || page_cells <= 0 || page_bytes.empty()) {
        err = "KvArena: empty layout";
        return false;
    }
    if (!vmm::supported() || (gran_ = vmm::granularity()) == 0) {
        err = "KvArena: the elastic KV needs the driver's virtual memory management, which this GPU or driver does "
              "not offer";
        return false;
    }
    pages_ = pages;
    page_cells_ = page_cells;
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t pb = page_bytes[i];
        Array a;
        a.page_bytes = pb;
        a.tail_bytes = i < tail_bytes.size() ? tail_bytes[i] : 0;
        a.reserved = (pb * (uint64_t) pages + a.tail_bytes + gran_ - 1) / gran_ * gran_;
        a.base = vmm::reserve(a.reserved);
        if (a.base == nullptr) {
            err = "KvArena: cannot reserve the address range of a KV pool";
            close();
            return false;
        }
        arrays_.push_back(a);
    }
    return true;
}

void KvArena::set_contents(size_t i, std::vector<uint8_t> bytes) {
    if (i < arrays_.size()) arrays_[i].contents = std::move(bytes);
}

void KvArena::close() {
    if (!arrays_.empty()) cudaDeviceSynchronize();
    for (Array& a : arrays_) {
        for (const Extent& x : a.extents) vmm::unmap(a.base + x.off, x.bytes, x.handle);
        vmm::release_range(a.base, a.reserved);
    }
    arrays_.clear();
    cells_ = 0;
    pages_ = 0;
}

uint64_t KvArena::need(const Array& a, int64_t cells) const {
    const int64_t pages = std::min<int64_t>(pages_, (std::max<int64_t>(cells, 0) + page_cells_ - 1) / page_cells_);
    if (pages == 0) return 0;
    return std::min(a.reserved, ((uint64_t) pages * a.page_bytes + a.tail_bytes + gran_ - 1) / gran_ * gran_);
}

void KvArena::recount() {
    int64_t pages = pages_;
    for (const Array& a : arrays_)
        pages = std::min<int64_t>(pages, (int64_t) ((a.mapped() > a.tail_bytes ? a.mapped() - a.tail_bytes : 0) / a.page_bytes));
    cells_ = pages * page_cells_;
}

uint64_t KvArena::mapped_bytes() const {
    uint64_t b = 0;
    for (const Array& a : arrays_) b += a.mapped();
    return b;
}

uint64_t KvArena::bytes_for(int64_t cells) const {
    uint64_t b = 0;
    for (const Array& a : arrays_) b += need(a, cells);
    return b;
}

uint64_t KvArena::grow_bytes(int64_t cells) const {
    uint64_t b = 0;
    for (const Array& a : arrays_) b += std::max(need(a, cells), a.mapped()) - a.mapped();
    return b;
}

bool KvArena::map(int64_t cells, std::string& err) {
    bool any = false;
    for (Array& a : arrays_) {
        const uint64_t want = need(a, cells);
        for (uint64_t at = a.mapped(); at < want; at = a.mapped()) {
            // this mapping ends at the first grid line past `at`
            uint64_t end = want;
            for (int64_t c = extent_cells_; c > 0 && c < max_cells(); c += extent_cells_)
                if (need(a, c) > at) { end = std::min(end, need(a, c)); break; }
            const unsigned long long h = vmm::map(a.base + at, end - at);
            if (h == 0) {
                (void) cudaGetLastError();
                recount();
                err = "the driver has no VRAM for the KV pools";
                return false;
            }
            a.extents.push_back({at, end - at, h});
            const uint64_t have = at < a.contents.size() ? std::min<uint64_t>(end - at, a.contents.size() - at) : 0;
            if (cudaMemset(a.base + at, 0, (size_t) (end - at)) != cudaSuccess ||
                (have > 0 && cudaMemcpy(a.base + at, a.contents.data() + at, (size_t) have, cudaMemcpyHostToDevice) != cudaSuccess)) {
                recount();
                err = std::string("zeroing new KV pages failed: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            any = true;
        }
    }
    recount();
    // the zeroing is ordered before work on any stream that follows
    if (any && cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("zeroing new KV pages failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool KvArena::unmap(int64_t cells, std::string& err) {
    bool synced = false;
    for (Array& a : arrays_) {
        const uint64_t want = need(a, cells);
        while (!a.extents.empty() && a.extents.back().off >= want) {
            if (!synced) {
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    err = std::string("the device failed before the KV pools shrank: ") +
                          cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                synced = true;
            }
            const Extent x = a.extents.back();
            if (!vmm::unmap(a.base + x.off, x.bytes, x.handle)) {
                recount();
                err = "the driver refused to release a KV pool mapping";
                return false;
            }
            a.extents.pop_back();
        }
    }
    recount();
    return true;
}

}  // namespace basalt::core
