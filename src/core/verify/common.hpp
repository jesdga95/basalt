#pragma once

#include "basalt/core/verify.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/qsa.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

namespace basalt::core::verify_detail {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

inline bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;   // multi-GPU: portable only with a peer - the peer card writes its rows into them
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

inline basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

inline const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

inline bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively";
        return false;
    }
    return true;
}

// The commit graph is two launches, its GDN part from 32 persistent blocks so it trickles beside the draft instead of
// filling the card
constexpr int kCommitBlocks = 32;
extern bool g_commit_async;
void note_fallback(const char* what, int a, int b);
void dense_proj(int bit, int type, const void* w, const void* xq, float* y, int n_in, int n_out, int n, void* s);
double card_bandwidth(int dev);

}  // namespace basalt::core::verify_detail
