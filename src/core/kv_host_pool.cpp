// src/core/kv_host_pool.cpp - a request's K/V in pinned host RAM.  Read the header first.
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/session.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace basalt::core {
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

int64_t page_cells() { return basalt::kernels::qsa_real_shapes().page_size; }
}  // namespace

KvHostPool::~KvHostPool() { release(); }

KvHostPool::KvHostPool(KvHostPool&& o) noexcept { *this = std::move(o); }

KvHostPool& KvHostPool::operator=(KvHostPool&& o) noexcept {
    if (this == &o) return *this;
    release();
    host_ = std::exchange(o.host_, nullptr);
    dev_ = std::exchange(o.dev_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
    layer_bytes_ = std::exchange(o.layer_bytes_, 0);
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
    release();
    if (s.qsa_states == nullptr || s.qsa_alloc <= 0 || cells <= 0) {
        err = "host K/V: no QSA layers or no cells";
        return false;
    }
    const int64_t ps = page_cells();
    const int64_t c = (cells + ps - 1) / ps * ps;
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
    layer_bytes_ = per;
    cells_ = c;
    run_bytes_ = page_runs(q0, g);
    layers_.clear();
    for (int64_t j = 0; j < s.qsa_alloc; ++j)
        layers_.push_back(qsa_kv_host_carve(s.qsa_states[s.qsa_ord0 + j], g, dev_ + (uint64_t) j * per, c));
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
    if (host_ != nullptr) cudaFreeHost(host_);
    host_ = dev_ = nullptr;
    bytes_ = layer_bytes_ = 0;
    cells_ = 0;
    layers_.clear();
}

}  // namespace basalt::core
