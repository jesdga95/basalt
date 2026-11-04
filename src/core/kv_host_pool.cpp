// src/core/kv_host_pool.cpp - a request's K/V in pinned host RAM.  Read the header first.
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/session.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <system_error>
#include <thread>
#include <utility>

namespace basalt::core {

struct KvHostPool::Ahead {
    std::thread writer;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> written{0};
    double ms = 0;                      // the writer's own time, set before it returns
    std::vector<std::pair<uint8_t*, uint64_t>> committed;   // per registration, in its order
    const QsaState* first = nullptr;    // the layers it was laid out for
    int64_t layers = 0;
    int64_t max_cells = 0;
};

namespace {
// the arrays of one layer's host copy, in carve order
std::vector<uint8_t*> arrays_of(const basalt::kernels::KvHostPools& h) {
    if (h.k_q4 != nullptr) return {h.k_q4, h.v_q4};
    if (h.k_q != nullptr) return {(uint8_t*) h.k_q, (uint8_t*) h.v_q, (uint8_t*) h.k_scale, (uint8_t*) h.v_scale};
    return {(uint8_t*) h.k_pool, (uint8_t*) h.v_pool};
}

std::vector<uint64_t> page_runs(const QsaState& st, const ModelGeometry& g) {
    const basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    const uint64_t rows = (uint64_t) g.n_head_kv * s.page_size;
    if (st.kv_q4) {
        const uint64_t q4 = rows * basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim);
        return {q4, q4};
    }
    if (st.kv_int8) {
        const uint64_t code = rows * g.head_dim, scale = rows * (g.head_dim / basalt::kernels::KV_Q8_GROUP) * 2;
        return {code, code, scale, scale};
    }
    return {rows * g.head_dim * 2, rows * g.head_dim * 2};
}

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

void unregister_range(void* p) {
    const cudaError_t e = cudaHostUnregister(p);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "host K/V: cannot unregister RAM: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool register_identity(void* base, uint64_t bytes, uint64_t& registered, std::string& err) {
    if (cudaHostRegister(base, (size_t) bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "host K/V: cannot register the RAM";
        return false;
    }
    registered = bytes;
    void* device = nullptr;
    if (cudaHostGetDevicePointer(&device, base, 0) != cudaSuccess || device != base) {
        (void) cudaGetLastError();
        err = "host K/V: the registered RAM has a different device address";
        return false;
    }
    return true;
}

int64_t page_cells() { return basalt::kernels::qsa_real_shapes().page_size; }

// A write backs each page (a read would map the shared zero page). The pages are new, so a zero keeps their bytes.
void write_pages(std::atomic<bool>* stop, std::atomic<uint64_t>* written, double* ms,
                 std::vector<std::pair<uint8_t*, uint64_t>> ranges, uint64_t page) {
    const auto t0 = std::chrono::steady_clock::now();
    constexpr uint64_t kStep = 2ull << 20;   // the stop flag is read between steps
    for (const auto& [base, bytes] : ranges)
        for (uint64_t off = 0; off < bytes && !stop->load(std::memory_order_relaxed); off += kStep) {
            const uint64_t end = std::min(bytes, off + kStep);
            for (uint64_t at = off; at < end; at += page) ((volatile uint8_t*) base)[at] = 0;
            written->fetch_add(end - off, std::memory_order_relaxed);
        }
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

KvHostPool::KvHostPool() = default;

KvHostPool::~KvHostPool() { release(); }

KvHostPool::KvHostPool(KvHostPool&& o) noexcept { *this = std::move(o); }

KvHostPool& KvHostPool::operator=(KvHostPool&& o) noexcept {
    if (this == &o) return *this;
    release();
    host_ = std::exchange(o.host_, nullptr);
    dev_ = std::exchange(o.dev_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
    reserved_bytes_ = std::exchange(o.reserved_bytes_, 0);
    page_bytes_ = std::exchange(o.page_bytes_, 0);
    max_cells_ = std::exchange(o.max_cells_, 0);
    registrations_ = std::move(o.registrations_);
    o.registrations_.clear();
    ahead_ = std::move(o.ahead_);
    cells_ = std::exchange(o.cells_, 0);
    layers_ = std::move(o.layers_);
    run_bytes_ = std::move(o.run_bytes_);
    o.layers_.clear();
    return *this;
}

uint64_t KvHostPool::bytes_for(const SessionState& s, const ModelGeometry& g, int64_t cells) {
    if (s.qsa_states == nullptr || s.qsa_alloc <= 0) return 0;
    const uint64_t per = (qsa_kv_host_layer_bytes(s.qsa_states[s.qsa_ord0], g, cells) + 255) & ~255ull;
    return per * (uint64_t) s.qsa_alloc;
}

bool KvHostPool::reserve(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err) {
    const bool laid_out = ahead_ != nullptr && s.qsa_states != nullptr && ahead_->first == s.qsa_states + s.qsa_ord0 &&
                          ahead_->layers == s.qsa_alloc && ahead_->max_cells == s.max_cells;
    std::vector<std::pair<uint8_t*, uint64_t>> committed;
    if (laid_out) {
        (void) prefault_wait();
        committed = std::move(ahead_->committed);
        ahead_.reset();
    } else {
        release();
    }
    if (s.qsa_states == nullptr || s.qsa_alloc <= 0 || cells <= 0) {
        release();
        err = "host K/V: no QSA layers or no cells";
        return false;
    }
    const int64_t ps = page_cells();
    const int64_t c = (cells + ps - 1) / ps * ps;
    if (s.max_cells > 0 && c <= (int64_t) align_up((uint64_t) s.max_cells, (uint64_t) ps) &&
        (laid_out ? grow(c, err) : reserve_registered(s, g, c, err))) {
        // prefaulted pages past the registered prefixes go back
        for (size_t i = 0; i < committed.size() && i < registrations_.size(); ++i) {
            const Registration& r = registrations_[i];
            if (committed[i].second > r.bytes)
                (void) basalt::platform::vm_decommit(r.base + r.bytes, committed[i].second - r.bytes);
        }
        return true;
    }
    // A refused reservation or non-identity device alias uses the allocated pool.
    release();
    err.clear();
    const QsaState& q0 = s.qsa_states[s.qsa_ord0];
    const uint64_t per = (qsa_kv_host_layer_bytes(q0, g, c) + 255) & ~255ull;
    const uint64_t total = per * (uint64_t) s.qsa_alloc;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaHostAlloc(&h, (size_t) total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "host K/V: cannot pin " + std::to_string(total >> 20) + " MiB of RAM";
        return false;
    }
    if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFreeHost(h);
        err = "host K/V: the pinned RAM has no device address";
        return false;
    }
    host_ = (uint8_t*) h;
    dev_ = (uint8_t*) d;
    bytes_ = total;
    cells_ = c;
    run_bytes_ = page_runs(q0, g);
    layers_.clear();
    for (int64_t j = 0; j < s.qsa_alloc; ++j)
        layers_.push_back(qsa_kv_host_carve(s.qsa_states[s.qsa_ord0 + j], g, dev_ + (uint64_t) j * per, c));
    return true;
}

bool KvHostPool::reserve_registered(const SessionState& s, const ModelGeometry& g, int64_t cells,
                                    std::string& err) {
    return lay_out(s, g, err) && grow(cells, err);
}

bool KvHostPool::prefault(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err) {
    release();
    if (s.qsa_states == nullptr || s.qsa_alloc <= 0 || cells <= 0 || s.max_cells <= 0) {
        err = "host K/V: no QSA layers or no cells";
        return false;
    }
    if (!lay_out(s, g, err)) {
        release();
        if (err.empty()) err = "host K/V: registered pools are unavailable";
        return false;
    }
    const uint64_t pages = (uint64_t) ((std::min(cells, max_cells_) + page_cells() - 1) / page_cells());
    std::vector<std::pair<uint8_t*, uint64_t>> ranges;
    for (size_t i = 0; i < registrations_.size(); ++i) {
        const uint64_t bytes = align_up(pages * run_bytes_[i % run_bytes_.size()], page_bytes_);
        if (!basalt::platform::vm_commit(registrations_[i].base, bytes)) {
            release();
            err = "host K/V: cannot commit the RAM";
            return false;
        }
        ranges.emplace_back(registrations_[i].base, bytes);
    }
    ahead_ = std::make_unique<Ahead>();
    ahead_->first = s.qsa_states + s.qsa_ord0;
    ahead_->layers = s.qsa_alloc;
    ahead_->max_cells = s.max_cells;
    ahead_->committed = ranges;
    try {
        ahead_->writer = std::thread(write_pages, &ahead_->stop, &ahead_->written, &ahead_->ms, std::move(ranges),
                                     page_bytes_);
    } catch (const std::system_error&) {
        ahead_.reset();
        release();
        err = "host K/V: cannot start the prefault";
        return false;
    }
    return true;
}

uint64_t KvHostPool::prefaulted_bytes() const {
    return ahead_ != nullptr ? ahead_->written.load(std::memory_order_relaxed) : 0;
}

double KvHostPool::prefault_wait() {
    if (ahead_ == nullptr) return 0;
    if (ahead_->writer.joinable()) ahead_->writer.join();
    return ahead_->ms;
}

bool KvHostPool::lay_out(const SessionState& s, const ModelGeometry& g, std::string& err) {
    int device = 0, identity = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&identity, cudaDevAttrCanUseHostPointerForRegisteredMem, device) != cudaSuccess ||
        !identity) {
        (void) cudaGetLastError();
        return false;
    }
    page_bytes_ = basalt::platform::vm_page_size();
    if (page_bytes_ == 0) return false;
    const QsaState& q0 = s.qsa_states[s.qsa_ord0];
    run_bytes_ = page_runs(q0, g);
    // Widen the carve so every array starts at a host-page boundary, including a partial context page.
    uint64_t quantum = 1;
    for (uint64_t run : run_bytes_)
        quantum = std::lcm(quantum, page_bytes_ / std::gcd(page_bytes_, run));
    const uint64_t pages = align_up(((uint64_t) s.max_cells + page_cells() - 1) / page_cells(), quantum);
    const int64_t carve_cells = (int64_t) pages * page_cells();
    const uint64_t per = align_up(qsa_kv_host_layer_bytes(q0, g, carve_cells), page_bytes_);
    reserved_bytes_ = per * (uint64_t) s.qsa_alloc;
    host_ = (uint8_t*) basalt::platform::vm_reserve(reserved_bytes_);
    if (host_ == nullptr) { reserved_bytes_ = 0; return false; }
    dev_ = host_;
    max_cells_ = (int64_t) align_up((uint64_t) s.max_cells, (uint64_t) page_cells());
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        const QsaState& st = s.qsa_states[s.qsa_ord0 + j];
        if (page_runs(st, g) != run_bytes_) { err = "host K/V: different layer formats"; return false; }
        layers_.push_back(qsa_kv_host_carve(st, g, host_ + (uint64_t) j * per, carve_cells));
        for (uint8_t* base : arrays_of(layers_.back())) registrations_.push_back({base, 0});
    }
    return true;
}

uint64_t KvHostPool::growth_bytes(int64_t cells) const {
    if (!grows_in_place() || cells <= cells_ || cells > max_cells_) return 0;
    const uint64_t pages = ((uint64_t) cells + page_cells() - 1) / page_cells();
    const uint64_t old_pages = (uint64_t) cells_ / page_cells();
    uint64_t bytes = 0;
    for (uint64_t run : run_bytes_)
        bytes += align_up(pages * run, page_bytes_) - align_up(old_pages * run, page_bytes_);
    return bytes * layers_.size();
}

bool KvHostPool::grow(int64_t cells, std::string& err) {
    if (!grows_in_place() || cells <= 0 || cells > max_cells_) {
        err = "host K/V: no registered capacity for the cells";
        return false;
    }
    const int64_t c = (cells + page_cells() - 1) / page_cells() * page_cells();
    if (c <= cells_) return true;
    const uint64_t pages = (uint64_t) c / page_cells();
    struct Change { size_t index; uint64_t old_bytes; uint64_t new_bytes; };
    std::vector<Change> changed;
    changed.reserve(registrations_.size());
    bool ok = true;
    for (size_t i = 0; i < registrations_.size(); ++i) {
        auto& r = registrations_[i];
        const uint64_t wanted = align_up(pages * run_bytes_[i % run_bytes_.size()], page_bytes_);
        if (wanted == r.bytes) continue;
        if (!basalt::platform::vm_commit(r.base + r.bytes, wanted - r.bytes)) {
            err = "host K/V: cannot commit the new RAM";
            ok = false;
            break;
        }
        changed.push_back({i, r.bytes, wanted});
        // Keep one registration per array so whole-array DMA never crosses registration boundaries.
        if (r.bytes != 0) unregister_range(r.base);
        r.bytes = 0;
        if (!register_identity(r.base, wanted, r.bytes, err)) { ok = false; break; }
    }
    if (!ok) {
        for (auto it = changed.rbegin(); it != changed.rend(); ++it) {
            auto& r = registrations_[it->index];
            if (r.bytes != 0) unregister_range(r.base);
            r.bytes = 0;
            if (it->old_bytes != 0) {
                std::string restore_error;
                if (!register_identity(r.base, it->old_bytes, r.bytes, restore_error)) {
                    basalt::log::error("basalt", "host K/V: cannot restore registered RAM: %s", restore_error.c_str());
                    std::exit(1);
                }
            }
            (void) basalt::platform::vm_decommit(r.base + it->old_bytes, it->new_bytes - it->old_bytes);
        }
        return false;
    }
    bytes_ += growth_bytes(c);
    cells_ = c;
    return true;
}

bool KvHostPool::copy_from(const KvHostPool& from, int64_t cells, std::string& err) {
    if (!valid() || !from.valid() || from.layers_.size() != layers_.size() || from.run_bytes_ != run_bytes_) {
        err = "host K/V: the two pools are not of one session";
        return false;
    }
    const int64_t ps = page_cells();
    const int64_t pages = (std::max<int64_t>(cells, 0) + ps - 1) / ps;
    if (pages * ps > cells_ || pages * ps > from.cells_) {
        err = "host K/V: more cells to move than a pool holds";
        return false;
    }
    for (size_t j = 0; j < layers_.size(); ++j) {
        const std::vector<uint8_t*> dst = arrays_of(layers_[j]), src = arrays_of(from.layers_[j]);
        for (size_t a = 0; a < dst.size() && a < run_bytes_.size(); ++a)
            std::memcpy(host_ + (dst[a] - dev_), from.host_ + (src[a] - from.dev_), (size_t) (pages * run_bytes_[a]));
    }
    return true;
}

void KvHostPool::release() {
    if (ahead_ != nullptr) {
        ahead_->stop.store(true, std::memory_order_relaxed);
        (void) prefault_wait();
        ahead_.reset();
    }
    if (grows_in_place()) {
        for (auto r = registrations_.rbegin(); r != registrations_.rend(); ++r)
            if (r->bytes != 0) unregister_range(r->base);
        basalt::platform::vm_release(host_, reserved_bytes_);
    } else if (host_ != nullptr) {
        (void) cudaFreeHost(host_);
    }
    host_ = dev_ = nullptr;
    bytes_ = reserved_bytes_ = page_bytes_ = 0;
    cells_ = max_cells_ = 0;
    registrations_.clear();
    layers_.clear();
    run_bytes_.clear();
}

}  // namespace basalt::core
