// src/prefill/timing.hpp - --prefill-timing: where a prompt read's time goes.
//
// The compute stream gets an event at each stage boundary of a chunk; each mark names the stage of the interval it
// closes, so the intervals add up to the stream's whole timeline, idle gaps included.  The events are read and reused
// after syncs the read makes anyway (each layer's routing read-back, the read's end), so no wait is added and the
// events in use are one layer's.  Host waits and the streamed bytes are counted beside them, per chunk.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

namespace basalt::prefill::detail {

/// The stage of the interval a mark closes.
enum class Span : int {
    Inputs,     ///< the embeddings, the step records, the stream's set-up
    Ple,        ///< layer 1's n-gram block: its rows' wait and upload, its products
    Hc,         ///< the hyper-connection reads and writes around each half
    Gdn,        ///< the GDN mixers
    Qsa,        ///< the attention mixers, without their K/V appends
    Kv,         ///< the K/V appends (VRAM, the host copy) and a streamed layer's K/V staged from the host copy
    Router,     ///< the router and the next layer's guessed routing
    Shared,     ///< the shared expert
    Routing,    ///< the routing's read-back and the GPU idle while the host groups the rows
    Experts,    ///< the routed experts on the GPU
    Ring,       ///< the GPU waiting for a streamed expert's copy
    Cpu,        ///< the GPU waiting for the CPU share's rows, and their upload
    Combine,    ///< the combine and the copies ahead of the next layer's routing
    Callback,   ///< the chunk's end: the last wait and the caller's callback (draft layer K/V, checkpoints)
    Count
};

/// Host time counted per chunk.
enum class Host : int {
    Sync,       ///< the host waiting for the compute stream (the routing read-backs, the chunk's end)
    Callback,   ///< the caller's per-chunk callback
    DraftKv,    ///< the draft layer's K/V from the chunk's rows (inside the callback)
    Stager,     ///< the copy issuer waiting for an unpinned expert's host copy
    CpuPool,    ///< the CPU share's experts on the pool
    CpuWait,    ///< the launching thread waiting for the CPU share
    Events,     ///< the timeline's own event reads
    Count
};

/// Where a streamed expert's bytes came from.
enum class Src : int {
    Pinned,     ///< DMA from the page-locked arena
    Ram,        ///< copied to a pinned buffer from pageable memory
    File,       ///< read from the model file into a pinned buffer
    Card,       ///< another GPU's memory
    Count
};

class Timeline {
public:
    using Clock = std::chrono::steady_clock;
    Timeline() = default;
    ~Timeline();
    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    bool on() const { return on_; }
    /// A read of `n` tokens from `pos0` starts; `on` false makes every call until the next `open` do nothing.
    void open(bool on, cudaStream_t cs, int64_t pos0, int64_t n);
    /// Closes the interval since the last mark as `s`.
    void mark(Span s, cudaStream_t cs) {
        if (on_) record(s, cs);
    }
    void chunk_begin(int64_t p0, int64_t T, bool stream_all);
    void chunk_end();
    void add(Host h, double ms) {
        if (on_ && !chunks_.empty()) chunks_.back().host[(size_t) h] += ms;
    }
    void streamed(Src s, uint64_t bytes, int64_t copies = 1) {
        if (!on_ || chunks_.empty()) return;
        chunks_.back().bytes[(size_t) s] += bytes;
        chunks_.back().copies += copies;
    }
    void resident(uint64_t bytes) {
        if (!on_ || chunks_.empty()) return;
        chunks_.back().resident_bytes += bytes;
        ++chunks_.back().resident;
    }
    void cpu_experts(int64_t n) {
        if (on_ && !chunks_.empty()) chunks_.back().cpu_experts += n;
    }
    /// The compute stream is about to wait for the copy `copied`: true (and the interval so far closed as `before`)
    /// when that copy is still running, so the wait can stall the stream.
    bool wait_begins(cudaEvent_t copied, Span before, cudaStream_t cs);
    /// After the wait `wait_begins` said could stall.
    void wait_ends(cudaStream_t cs) {
        mark(Span::Ring, cs);
        if (!chunks_.empty()) ++chunks_.back().waits;
    }
    /// The compute stream is synchronized: the marks so far into their chunks' sums, and their events free again.
    void fold() {
        if (on_ && pending_ > 1) fold_pending();
    }
    /// Both streams are synchronized: the read's line, and one line per chunk into --prefill-timing-log.
    void close(double wall_ms);

private:
    struct Chunk {
        int64_t p0 = 0, T = 0;
        bool stream_all = false;
        Clock::time_point t0{};
        double wall_ms = 0;
        double unix_s = 0;                 // when it started, for lining it up with other logs
        std::array<double, (size_t) Host::Count> host{};
        std::array<uint64_t, (size_t) Src::Count> bytes{};
        int64_t copies = 0, resident = 0, cpu_experts = 0, waits = 0;
        uint64_t resident_bytes = 0;
        std::array<double, (size_t) Span::Count> gpu{};
    };
    struct Mark {
        Span span;                         // the stage of the interval it closes (Count: none)
        int chunk;                         // the chunk that interval belongs to (-1: none)
    };
    void record(Span s, cudaStream_t cs);
    void fold_pending();

    bool on_ = false;
    bool in_chunk_ = false;
    int64_t pos0_ = 0, n_ = 0;
    std::vector<cudaEvent_t> ev_;          // grows to the most marks between two folds; pending mark i uses ev_[i]
    std::vector<Mark> mark_;               // the pending marks; mark 0 is the last one folded
    size_t pending_ = 0;
    std::vector<Chunk> chunks_;
};

/// The compute stream waits for a streamed copy; with --prefill-timing, a wait that can stall is marked as Ring.
inline void wait_copy(Timeline& tl, cudaStream_t cs, cudaEvent_t copied) {
    const bool marked = tl.on() && tl.wait_begins(copied, Span::Experts, cs);
    cudaStreamWaitEvent(cs, copied, 0);
    if (marked) tl.wait_ends(cs);
}

/// Adds the host time of its scope to `h` while the timeline is on.
class HostTimer {
public:
    HostTimer(Timeline& tl, Host h) : tl_(tl.on() ? &tl : nullptr), h_(h) {
        if (tl_) t0_ = Timeline::Clock::now();
    }
    ~HostTimer() {
        if (tl_) tl_->add(h_, std::chrono::duration<double, std::milli>(Timeline::Clock::now() - t0_).count());
    }
    HostTimer(const HostTimer&) = delete;
    HostTimer& operator=(const HostTimer&) = delete;

private:
    Timeline* tl_;
    Host h_;
    Timeline::Clock::time_point t0_{};
};

}  // namespace basalt::prefill::detail
