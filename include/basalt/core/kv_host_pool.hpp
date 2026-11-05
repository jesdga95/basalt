// include/basalt/core/kv_host_pool.hpp - a request's K/V in pinned host RAM.
//
// A session whose QSA states can stream (qsa_set_kv_host) keeps its authoritative K/V here while a request runs
// streamed. Each array reserves the context's address space and pins only its used prefix, so growth preserves
// addresses. If registered host pointers cannot be used by the device, a pool uses one mapped allocation and grows
// by copying into a replacement. A pool can be prefaulted ahead of its first pin: its pages are committed and written
// on a background thread, so the pin registers pages already in RAM.
#pragma once

#include "basalt/core/layer.hpp"
#include "basalt/kernels/kv_stream.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {

struct SessionState;

class KvHostPool {
public:
    KvHostPool();
    ~KvHostPool();
    KvHostPool(const KvHostPool&) = delete;
    KvHostPool& operator=(const KvHostPool&) = delete;
    KvHostPool(KvHostPool&& o) noexcept;
    KvHostPool& operator=(KvHostPool&& o) noexcept;

    /// Bytes a pool of `cells` cells takes for the QSA layers `s` allocates.
    static uint64_t bytes_for(const SessionState& s, const ModelGeometry& g, int64_t cells);
    /// Pins a pool of `cells` cells (whole pages) for `s`'s QSA layers; what this pool held before is released first,
    /// except a prefault for the same layers, which is waited for and registered in place.
    /// False with `err`, holding nothing, when the driver cannot pin it.
    bool reserve(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err);
    /// Lays out a registered pool for `s` and writes the pages of its first `cells` cells on a background thread,
    /// registering nothing; what this pool held before is released first. False with `err`, holding nothing, where
    /// registered pools are unavailable. No device call runs on the thread.
    bool prefault(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err);
    /// Whether this holds a prefault (laid out, not registered).
    bool prefaulting() const { return ahead_ != nullptr; }
    /// RAM a prefault has written so far.
    uint64_t prefaulted_bytes() const;
    /// RAM a prefault writes in all (its committed pages); 0 without a prefault.
    uint64_t prefault_bytes() const;
    /// Waits for a prefault's writer; its own time in ms (0 without a prefault).
    double prefault_wait();
    /// Re-registers longer array prefixes at their existing addresses, without moving bytes.
    /// On refusal the old registrations and extent are restored; a failed restoration stops the engine.
    /// The device must be drained before growth. False also for an allocated pool, which needs replacement.
    bool grow(int64_t cells, std::string& err);
    bool grows_in_place() const { return reserved_bytes_ != 0; }
    /// Additional page-rounded RAM a registered growth pins (zero when already covered).
    uint64_t growth_bytes(int64_t cells) const;
    /// Copies the first `cells` cells of every layer of `from` (same session) into this pool, in host memory.
    bool copy_from(const KvHostPool& from, int64_t cells, std::string& err);
    /// Frees the pinned RAM; the device must be done with it (no state points at it any more).
    void release();

    bool valid() const { return host_ != nullptr && cells_ > 0; }
    int64_t cells() const { return cells_; }
    uint64_t bytes() const { return bytes_; }
    /// The host copy of the session's QSA layer `j` (its allocated ordinal, 0 .. qsa_alloc).
    const basalt::kernels::KvHostPools& layer(int64_t j) const { return layers_[(size_t) j]; }
    int64_t layers() const { return (int64_t) layers_.size(); }

private:
    uint8_t* host_ = nullptr;   // the host address of the allocation
    uint8_t* dev_ = nullptr;    // its device alias
    uint64_t bytes_ = 0;
    uint64_t reserved_bytes_ = 0;
    uint64_t page_bytes_ = 0;
    int64_t max_cells_ = 0;
    struct Registration { uint8_t* base; uint64_t bytes; };
    std::vector<Registration> registrations_;
    struct Ahead;                       // a prefault's writer thread and progress
    std::unique_ptr<Ahead> ahead_;
    bool lay_out(const SessionState& s, const ModelGeometry& g, std::string& err);
    bool reserve_registered(const SessionState& s, const ModelGeometry& g, int64_t cells, std::string& err);
    int64_t cells_ = 0;
    std::vector<basalt::kernels::KvHostPools> layers_;
    std::vector<uint64_t> run_bytes_;   // per array of a layer: bytes per page
};

}  // namespace basalt::core
