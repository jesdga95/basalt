// src/models/qwen4exp/load/split.cpp - see split.hpp.
#include "split.hpp"

#include "basalt/core/weights.hpp"
#include "basalt/models/qwen4exp/decode/native_dense.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"

#include <cstdlib>

namespace basalt::models::qwen4exp::engine_local {

// What a card holds before its experts, by block: its layers' dense weights (the arena, the native projections, the 8-bit
// hyper-connection copies) and session; and what every card holds whatever its layers. Each card's expert cache fills what
// is left, so the cut divides these bytes.
basalt::model::SplitSpec LayerSplit::split_spec(const basalt::model::GpuCards& c) const {
    constexpr uint64_t kMainOtherBytes = 100ull << 20;   // what CUDA0 holds beside its dense weights before the experts: buffers and tables
    const basalt::core::ModelGeometry& g = *in_.g;
    basalt::model::SplitSpec spec;
    std::vector<uint64_t> pool_l, nat_l;
    uint64_t pool_sh = 0, nat_sh = 0;
    std::string err;
    if (!basalt::core::WeightTable::pool_bytes_by_layer(*in_.pack, g.n_layers, pool_l, pool_sh, err, in_.skip) ||
        !basalt::core::NativeDense::plan_bytes_by_layer(*in_.native_shards, true, g.n_layers, nat_l, nat_sh, err))
        return spec;   // (no layers: split_cut refuses)
    auto block_of = [](const std::string& n) -> long {
        return n.rfind("blk.", 0) == 0 ? std::strtol(n.c_str() + 4, nullptr, 10) : -1;
    };
    const uint64_t hc8_sh = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) < 0; }, g);
    const uint64_t sess0 = basalt::core::session_bytes(g, in_.max_context, in_.k, 0, 0);   // a session's part outside any layer
    for (int64_t l = 0; l < g.n_layers; ++l)
        spec.layer.push_back(pool_l[(size_t) l] + nat_l[(size_t) l] +
                             basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) == l; }, g) +
                             basalt::core::session_bytes(g, in_.max_context, in_.k, l, l + 1) - sess0);
    for (size_t i = 0; i < c.cards.size(); ++i) {
        spec.fixed.push_back(pool_sh + nat_sh + hc8_sh + sess0 + (i == 0 ? kMainOtherBytes : 0));
        spec.room.push_back(in_.room ? in_.room(i) : 0);
    }
    return spec;
}

bool LayerSplit::split_place(const basalt::model::SplitCut& cut, const std::vector<int>& devices, std::string& err) {
    (void) err;
    in_.split_at->assign(cut.first.begin() + 1, cut.first.end());
    in_.split_devs->assign(devices.begin() + 1, devices.end());
    return true;
}

}  // namespace basalt::models::qwen4exp::engine_local
