// src/models/qwen4exp/load/split.hpp - Flash-Next's gpu-mode split (model::SplitHooks, docs/architecture/multi-gpu.md): each card
// after the main one runs a block of later layers as a stage with its own dense weights, session and expert cache. The
// shared driver cuts the layers by the shares over the bytes stated here (or at --diag-gpu-cut) and writes each card's
// part of the start line; this states the bytes and takes the cut and the stages' devices.
#pragma once

#include "basalt/core/weights.hpp"
#include "basalt/model/split_hooks.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace basalt::models::qwen4exp::engine_local {

class LayerSplit final : public basalt::model::SplitHooks {
public:
    /// What the bytes are counted over, once the geometry is known.
    struct Open {
        const basalt::core::PackFiles* pack = nullptr;
        const std::vector<std::string>* native_shards = nullptr;
        const std::set<std::string>* skip = nullptr;    ///< tensors no card loads
        const basalt::core::ModelGeometry* g = nullptr;
        int64_t k = 0, max_context = 0;
        std::function<uint64_t(size_t card)> room;      ///< per card: its free memory past its reserves
        std::vector<int64_t>* split_at = nullptr;       ///< out: each later stage's first layer
        std::vector<int>* split_devs = nullptr;         ///< out: each later stage's device
    };
    /// Set before split_open.
    void set_open(const Open& in) { in_ = in; }
    /// Each card's part beside its layers, once its expert cache is filled.
    std::function<std::string(size_t card)> part;

protected:
    basalt::model::SplitSpec split_spec(const basalt::model::GpuCards& c) const override;
    bool split_place(const basalt::model::SplitCut& cut, const std::vector<int>& devices, std::string& err) override;
    std::string split_part(size_t card) const override { return part ? part(card) : std::string(); }

private:
    Open in_;
};

}  // namespace basalt::models::qwen4exp::engine_local
