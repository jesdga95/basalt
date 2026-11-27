// src/models/inkling2/cpu_experts.hpp - the RAM experts of a window computed on the CPU, beside the card's experts.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::inkling2 {

class InklingModel;

/// The experts that spilled to RAM (and have no card copy) run on the CPU pool instead of being read over the bus: once a
/// layer's MLP input is written a kernel copies its rows to mapped memory, which the host quantizes while the router runs;
/// after the router a kernel writes its choices and rings a doorbell when any pair's expert is in RAM; a host thread
/// spinning on the doorbell has the pool compute those pairs while the card computes the rest, then sets the completion
/// ticket, which the card's merge matches before taking the pool's rows. A visit without a RAM pair neither rings nor
/// waits. Sessions share one hand-off per model and queue visits in order; a cached graph takes the next ticket when it
/// runs. A call with such windows holds admission (enter, leave): the abort hook closes it, releases stalled waits and gives
/// the calls in flight a bounded time to finish; a window it interrupted is discarded. The card's blob table and the host's
/// RAM blob table agree on placement and change only between windows.
///
/// With gpu-mode peer the experts on the second card (InklingModel::peer_held) are off the card for the card's kernels too:
/// the doorbell's thread has the second card compute those pairs (core::PeerExperts, straight into the pool's rows) while
/// the pool computes the rest, and the same completion ticket covers both.
class CpuExperts {
public:
    ~CpuExperts();
    /// Null with `err` empty when no expert spilled.
    static std::unique_ptr<CpuExperts> make(const InklingModel& model, std::string& err);
    /// Once layer l's MLP input [rows][n_embd] is written, before its router: the rows to RAM, where the host quantizes
    /// them.
    void rows(int l, int rows, const float* input, cudaStream_t stream);
    /// After layer l's router, after rows: their choices [rows][n_used] to RAM, then the pool computes the pairs whose
    /// expert is in RAM.
    void start(int l, int rows, const int32_t* ids, cudaStream_t stream);
    /// After the card's experts: once the pool is done, each RAM pair's row into `out` (row r * n_used + slot) and its lookup
    /// into `tally`.
    void finish(int l, int rows, const int32_t* ids, float* out, uint32_t* tally, cudaStream_t stream);
    /// Before a window that calls start: the doorbell's thread spins from here instead of napping.
    void wake();
    /// Before a call's first window that calls start: false once the abort hook ran (the call submits nothing).
    bool enter();
    /// After the call's closing sync (`synced`: its status; `joined`: no start of the call is left without its finish):
    /// false once the abort hook ran, and the call's results are discarded.
    bool leave(cudaError_t synced, bool joined);
    /// Whether the abort hook ran: a call in flight starts no further window.
    bool stopped();

private:
    struct Impl;
    CpuExperts();
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::inkling2
