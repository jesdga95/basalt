// src/models/gemma4/experts.hpp - the routed experts where core::ExpertTier keeps them: the main card's from a device table
// of blobs, the others computed by the CPU pool beside the card's.
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/models/gemma4/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::gemma4 {

class Gemma4Model;

/// The placement is the tier's: each layer's device table holds the card address of every expert on the card and 0 for
/// the others. A decode window's grouped kernel computes the pairs whose expert has an address (moe_plan leaves the others
/// out); when any pair's expert is off the card the layer rings a doorbell in mapped memory (moe_ring) and a host thread
/// spinning on it has the CPU pool compute those pairs from the RAM tier into mapped rows, which the card's merge
/// (moe_merge) takes once the ticket is done. A layer without such a pair neither rings nor waits; with every expert on the
/// card (off_card() false) the windows hold no hand-off at all.
///
/// The tables follow the placement a window late: a window's choices go through the tier's lookups and promotions when
/// the next window begins (begin_window), the promotions' copies go out behind each layer's experts in that window
/// (issue_copies, after the layer's recorded event) and land before the one after it, so no copy sits between two
/// windows and a window never reads a slot being written. A prompt settles every copy first (prompt_begin).
class TierExperts {
public:
    ~TierExperts();
    /// `rows`: mapped host rows [kMoePairs][n_embd] the host's pairs land in.
    static std::unique_ptr<TierExperts> make(const Gemma4Model& m, core::ExpertTier& tier, float* rows, std::string& err);

    /// Some expert is (or may be) off the card: windows hand those pairs off and prompts stage them.
    bool off_card() const;
    /// Layer l's device table: n_expert blob addresses on the card, 0 off it.
    const unsigned long long* tab(int l) const;

    /// Before a window: the last window's choices through the tier's lookups and promotions, the copies the one before it
    /// carried landed and the tables written again (on `stream`) when the placement moved. The doorbell's thread spins
    /// from here.
    bool begin_window(cudaStream_t stream, std::string& err);
    /// After layer l's router: the rows and choices of the pairs off the card to the host (`xr` the router's unweighted
    /// normed rows, `w` the experts' norm weights); and the window's choices into the log.
    void ring(int l, int rows, const float* xr, const float* w, const int32_t* ids, cudaStream_t stream);
    /// After the card's own pairs: the host's rows of layer l into `out` (row r * k + slot); then the layer's event.
    void merge(int l, int rows, const int32_t* ids, float* out, cudaStream_t stream);
    /// Once the window is launched: each layer's queued copies go out behind its event.
    void issue_copies();
    /// At the window's end (before its sync): the choices' log to the host, for the next begin_window.
    void end_window(int rows, cudaStream_t stream);
    /// Before a prompt: every queued copy lands and the tables follow (the prompt reads the places as they are).
    bool prompt_begin(cudaStream_t stream, std::string& err);
    /// Whether the doorbell's thread hit an error: the bell is aborted for good.
    bool failed(std::string& err) const;

private:
    struct Impl;
    TierExperts();
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::gemma4
