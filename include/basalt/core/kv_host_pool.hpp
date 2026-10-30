// include/basalt/core/kv_host_pool.hpp - a request's K/V in pinned host RAM.
//
// A session whose QSA states can stream (qsa_set_kv_host) keeps its authoritative K/V here while a request runs
// streamed: one pinned, device-mapped allocation for every QSA layer of the session, each layer laid out as
// qsa_state_init lays a host copy out, for the request's cells rather than the whole context.  It is taken at a
// request's admission and kept while the session stays streamed (the next request of the conversation resumes from
// it); a deeper request takes a larger one and the cells move over in host memory.
#pragma once

#include "basalt/core/layer.hpp"
#include "basalt/kernels/kv_stream.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

struct SessionState;

class KvHostPool {
public:
    KvHostPool() = default;
    ~KvHostPool();
    KvHostPool(const KvHostPool&) = delete;
    KvHostPool& operator=(const KvHostPool&) = delete;
    KvHostPool(KvHostPool&& o) noexcept;
    KvHostPool& operator=(KvHostPool&& o) noexcept;

    /// Bytes a pool of `cells` cells takes for the QSA layers `s` allocates.
    static uint64_t bytes_for(const SessionState& s, const ModelGeometry& g, int64_t cells);
    /// Pins a pool of `cells` cells (whole pages) for `s`'s QSA layers; what this pool held before is released first.
    /// False with `err`, holding nothing, when the driver cannot pin it.
    bool reserve(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err);
    /// Copies the first `cells` cells of every layer of `from` (same session) into this pool, in host memory.
    bool copy_from(const KvHostPool& from, int64_t cells, std::string& err);
    /// Frees the pinned RAM; the device must be done with it (no state points at it any more).
    void release();

    bool valid() const { return host_ != nullptr; }
    int64_t cells() const { return cells_; }
    uint64_t bytes() const { return bytes_; }
    /// The host copy of the session's QSA layer `j` (its allocated ordinal, 0 .. qsa_alloc).
    const basalt::kernels::KvHostPools& layer(int64_t j) const { return layers_[(size_t) j]; }
    int64_t layers() const { return (int64_t) layers_.size(); }

private:
    uint8_t* host_ = nullptr;   // the host address of the allocation
    uint8_t* dev_ = nullptr;    // its device alias
    uint64_t bytes_ = 0;
    uint64_t layer_bytes_ = 0;
    int64_t cells_ = 0;
    std::vector<basalt::kernels::KvHostPools> layers_;
    std::vector<uint64_t> run_bytes_;   // per array of a layer: bytes per page
};

}  // namespace basalt::core
