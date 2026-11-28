// src/models/ernie55/decode_moe.hpp - the decode windows' routed experts without a host round trip per layer.
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::ernie55 {

class ErnieModel;

/// The main card computes the pairs whose expert it holds from a device table of blobs ([layer][expert], 0: off the
/// card), planned on the card from the router's choices. The other pairs go to the host through a doorbell in mapped
/// memory: after a layer's router a kernel writes its rows and choices there and rings when any pair is off the card;
/// a host thread spinning on the bell has the second card compute the pairs it holds and some of the RAM ones it fetches
/// over its own link (straight into the mapped rows) and the CPU pool the other RAM ones (or those in the mapped file),
/// then sets the completion ticket, which the card's merge
/// matches before it takes those rows. A window without such a pair neither rings nor waits.
///
/// The placement is core::ExpertTier's and the tables follow it a window late: after window k its choices (logged by
/// the plan) go through the tier's lookup and promotion, whose copies wait; window k + 1 still reads the experts where
/// they were (an evicted home in its slot, the incoming expert in RAM) and each layer's copies go out behind that
/// layer's experts in it (an event the window records); after window k + 1 they land and the tables take the new
/// places. So no copy sits between two windows. The tier is shared by the sessions, and so is this.
///
/// Between its visits the host thread warms: a layer's predicted RAM experts (hint) are read by the pool's workers, each
/// L3 group the rows it takes in the layer's batch, so the visit finds most of its bytes in the caches; the memory reads
/// them while the cards work on the layer before. The warming stops when the visit starts and resumes after it if it
/// was for a later layer, and ends with the window (after_window waits for that).
class DecodeMoe {
public:
    ~DecodeMoe();
    /// `rows`: the mapped rows the tier gave its second cards (open), kMoeDecPairs of n_latent floats at least; the
    /// host's pairs land there.
    static std::unique_ptr<DecodeMoe> make(const ErnieModel& m, core::ExpertTier& tier, float* rows, std::string& err);

    /// Layer l's device table (n_expert blob addresses on the main card, 0: off it).
    const unsigned long long* tab(int l) const;
    /// Layer l's log of the window's choices (device, kMoeDecPairs).
    int32_t* log(int l) const;

    /// Before a window: when the placement may have changed outside the decode windows (`dirty`: a prompt), the tier
    /// settles and the tables are written again on `stream`. The doorbell's thread spins from here.
    bool begin_window(cudaStream_t stream, bool dirty, std::string& err);
    /// Recorded on the window's stream after layer l's merge (captured as an external event record).
    cudaEvent_t layer_done(int l) const;
    /// Once the window is launched: each layer's queued copies go out behind its layer_done.
    void issue_copies();
    /// After layer l's router: the rows and choices of the pairs off the card to the host.
    void ring(int l, int rows, const float* x, const int32_t* ids, cudaStream_t stream);
    /// On a stream of the window's that forks after layer l - 1's router: layer l's choices predicted from that router's
    /// input (`ids`, rows * kMoeHintK, the likeliest first) to the host, which reads the predicted RAM experts into the
    /// pool's caches until layer l rings (each L3 group the rows it computes, ExpertPool::split_by_groups).
    void hint(int l, int rows, const int32_t* ids, cudaStream_t stream);
    /// After the card's own pairs: the host's rows of layer l into `out` (row r * k + slot).
    void merge(int l, int rows, const int32_t* ids, float* out, cudaStream_t stream);
    /// At the window's end (before its sync): the choices' logs to the host.
    void end_window(int rows, cudaStream_t stream);
    /// After the window's sync: the copies it carried land and the tables take the places (on `stream`, before the
    /// next window); then its choices go through the tier's lookup and promotion, whose copies wait for the next.
    bool after_window(cudaStream_t stream, std::string& err);
    /// Whether the host thread hit an error (the second card failed): the bell is aborted for good.
    bool failed(std::string& err) const;

private:
    struct Impl;
    DecodeMoe();
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::ernie55
