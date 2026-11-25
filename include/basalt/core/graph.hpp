// include/basalt/core/graph.hpp - the GraphRegistry.
//
// `GraphRegistry`: capture/replay wrappers keyed by `(layer_type, n_tokens)`, with static input/output
// buffers at fixed addresses.
//
// THREE THINGS ARE BAKED INTO THIS INTERFACE, and each one is a measurement rather than a preference.
//
// 1. **THE WAIT MUST POLL THE DRIVER, NOT MEMORY.**  A host loop that spins on
//    doorbell_seq == expected with no driver call does not work: a spin that only reads memory can NEVER
//    RUN THE KERNEL, because the datum only flips once something calls into the driver (the driver batches
//    command submission).  Hence `wait_ms`, which polls an event - a QUERY and not a blocking sync, so there
//    are still no synchronization calls in the layer loop, and it is cheap.
//
// 2. **A GRAPH RE-READS ITS INPUT BUFFERS AT REPLAY, BUT NOT ITS KERNEL ARGUMENTS.**  Verified: eight
//    replays with a different pinned input each time all tracked correctly, so the data path is live; but a
//    kernel argument is copied into the node at capture and is never re-read.  **Anything that changes per
//    step - a position, a page-table base, a token id - must be DATA in a device buffer, never an argument.**
//    This is the phase's "graph capture with pointers that change between steps silently replays stale
//    addresses" hazard, and it now has a measurement behind it.
//
// 3. **FIXED ADDRESSES ARE THE CALLER'S OBLIGATION.**  The registry cannot check them.  Every buffer a
//    captured body touches must be allocated before `record` and never reallocated; the registry holds the
//    graph, not the memory.
//
// What the registry is FOR: a layer's ops are the same every token, so recording them once and replaying
// removes the per-call host cost of submitting them individually.  Each submission is a driver call, so a layer with
// many of them pays a round trip per call.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace basalt::core {

/// The two attention families.  the model definition: 36 GDN layers and 12 QSA, at indices 3, 7, ... 47.
enum class LayerType : int { GDN = 0, QSA = 1 };
inline const char* to_string(LayerType t) { return t == LayerType::GDN ? "GDN" : "QSA"; }

/// One recorded graph.  Move-only: it owns a `cudaGraphExec_t` and an event, and two owners would
/// double-destroy.
class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    CapturedGraph(CapturedGraph&& o) noexcept { *this = std::move(o); }
    CapturedGraph& operator=(CapturedGraph&& o) noexcept;

    /// Start recording everything the caller launches into `stream`.
    bool begin(void* stream, std::string& err);

    /// Stop recording, instantiate, and record the completion event.  A capture that produced ZERO nodes is
    /// reported as an error: it means the body did nothing, which is a wiring mistake and NOT an empty graph
    /// to be replayed happily forever.
    bool end(void* stream, std::string& err);

    /// Replay.  Does not synchronise.
    bool launch(void* stream, std::string& err) const;

    /// The host loop's wait: poll `done_` until it completes or `timeout_ms` elapses.  Returns false on
    /// timeout, which the caller must treat as a hang and not as "probably fine".  See NOTE 1 above for why
    /// this exists instead of a memory spin.
    bool wait_ms(int timeout_ms) const;

    size_t nodes() const { return nodes_; }
    bool valid() const { return exec_ != nullptr; }
    /// The CUDA status of the call `begin`/`end` last failed in (cudaSuccess when none did); a failed `end` has released its
    /// partial graph, so what it held is free again.
    cudaError_t status() const { return status_; }

private:
    void reset();
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t done_ = nullptr;
    size_t nodes_ = 0;
    cudaError_t status_ = cudaSuccess;
};

/// Capture-once, replay-many, keyed by `(layer_type, n_tokens)`.
///
/// `record` is given a CALLABLE that launches the layer's ops.  On the first call for a key it captures;
/// afterwards it returns the cached graph and does NOT run the callable - so a caller that mutates state
/// inside the body will be surprised, which is why the body must only launch.
class GraphRegistry {
public:
    explicit GraphRegistry(void* stream) : stream_(stream) {}

    /// Capture `body` for `key` if it is not already recorded.  Returns false and fills `err` on failure.
    /// Recording the same key twice is a no-op and is NOT an error - that is the point of a registry, and a
    /// regression test asserts that the body runs exactly once across many calls.
    bool record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err);

    const CapturedGraph* find(LayerType type, int n_tokens) const;

    /// Launch and wait, in that order, the two things the host loop does per layer.
    bool launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const;

    size_t size() const { return graphs_.size(); }
    /// How many times a BODY was actually captured.  With a working cache this equals `size()`.
    size_t captures() const { return captures_; }

private:
    using Key = std::pair<int, int>;
    void* stream_ = nullptr;
    std::map<Key, CapturedGraph> graphs_;
    size_t captures_ = 0;
};

}  // namespace basalt::core
