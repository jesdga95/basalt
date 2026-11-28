// include/basalt/model/split_hooks.hpp - gpu-mode split for any model (docs/architecture/multi-gpu.md): the layers are divided
// between the cards in order, each layer's KV and state on its card. Whatever runs a model's split derives SplitHooks and
// writes the hooks below: what each layer and each card holds in bytes and the room each card has, then its part on each
// card once cut. The driver does the rest, the same for every model: the cut by the Split shares (or --diag-gpu-cut), the
// refusal of a card that would hold nothing or whose part does not fit, the --diag-gpu-same reference (every part on the
// main card) and each card's part of the start line after the load. How a model runs its parts and hands rows between
// them stays the model's.
#pragma once

#include "basalt/model/multi_gpu.hpp"

#include <string>
#include <vector>

namespace basalt::model {

class SplitHooks {
public:
    virtual ~SplitHooks() = default;

    /// Cuts the layers, the same for every model: the model's bytes (split_spec) cut by model::split_cut, a part that
    /// cannot run refused with its numbers, then the model's part on each card (split_place). With --diag-gpu-same every
    /// part runs on the main card at the --diag-gpu-cut cut and no card's room is checked: the model's own fit decides.
    /// False with `err`.
    bool split_open(const GpuCards& c, std::string& err);
    /// Card `card`'s part once open (0 = the main card): "layers 0-37" and the model's own words, for the start line after
    /// the load; empty when the split is not open.
    std::string split_report(size_t card) const;
    /// The cut (empty until split_open).
    const SplitCut& split_placement() const { return cut_; }
    /// Per card, the CUDA device that runs its layers (the main card's for every part under --diag-gpu-same).
    const std::vector<int>& split_devices() const { return devices_; }

protected:
    // ---- the hooks a model writes (docs/architecture/multi-gpu.md, "Split")

    /// Per layer its weights plus its KV and state at the context and slots, per card what it holds whatever its layers,
    /// and per card the bytes it has for them past its reserves.
    virtual SplitSpec split_spec(const GpuCards& c) const = 0;
    /// The model's part on each card as cut: card i runs layers [cut.first[i], cut.first[i + 1]) on `devices[i]`.
    virtual bool split_place(const SplitCut& cut, const std::vector<int>& devices, std::string& err) = 0;
    /// Card `card`'s own part beside its layers once loaded ("8600 experts (22.1 GiB)"); empty: nothing more to say.
    virtual std::string split_part(size_t card) const = 0;

private:
    SplitCut cut_;
    std::vector<int> devices_;
    int n_layers_ = 0;
    bool open_ = false;
};

}  // namespace basalt::model
