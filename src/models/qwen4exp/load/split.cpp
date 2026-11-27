// src/models/qwen4exp/load/split.cpp - see split.hpp.
#include "split.hpp"

#include "basalt/core/weights.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/decode/native_dense.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"

#include <algorithm>
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
    const std::set<std::string> empty_skip;
    const std::set<std::string>* base_skip = in_.skip ? in_.skip : &empty_skip;
    if (in_.shared_blk != nullptr && !in_.shared_blk->empty()) {
        std::set<std::string> skip_with_shared = *base_skip;
        skip_with_shared.insert(in_.shared_blk->begin(), in_.shared_blk->end());
        uint64_t pool_all = 0, pool_wo = 0;
        if (!basalt::core::WeightTable::pool_bytes_by_layer(*in_.pack, g.n_layers, pool_l, pool_sh, err, &skip_with_shared) ||
            !basalt::core::WeightTable::pool_bytes(*in_.pack, pool_all, err, base_skip) ||
            !basalt::core::WeightTable::pool_bytes(*in_.pack, pool_wo, err, &skip_with_shared))
            return spec;   // (no layers: split_cut refuses)
        if (pool_all > pool_wo) pool_sh += pool_all - pool_wo;
    } else if (!basalt::core::WeightTable::pool_bytes_by_layer(*in_.pack, g.n_layers, pool_l, pool_sh, err, in_.skip)) {
        return spec;   // (no layers: split_cut refuses)
    }
    if (!basalt::core::NativeDense::plan_bytes_by_layer(*in_.native_shards, true, g.n_layers, nat_l, nat_sh, err))
        return spec;   // (no layers: split_cut refuses)
    auto block_of = [](const std::string& n) -> long {
        return n.rfind("blk.", 0) == 0 ? std::strtol(n.c_str() + 4, nullptr, 10) : -1;
    };
    const uint64_t hc8_sh = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) < 0; }, g);
    // session_bytes(0, 0) already includes one primary QSA state when the model has QSA layers (q_n = max(..., 1)), so
    // session_bytes(l, l+1) - sess0 cancels out QSA KV to 0. Split the primary QSA state's per-layer body out into each
    // QSA layer and leave only the RoPE table + shared session scratch in fixed, scaled by all sessions per card.
    const uint64_t n_sess = (uint64_t) std::max<int64_t>(in_.n_sess, 1);
    const uint64_t sess0 = basalt::core::session_bytes(g, in_.max_context, in_.k, 0, 0);
    const uint64_t qsa_layer = g.n_qsa_layers() > 0 ? basalt::core::qsa_state_bytes(g, in_.max_context, false) : 0;
    const uint64_t sess_sh = sess0 > qsa_layer ? sess0 - qsa_layer : 0;
    const int64_t qsa_interval = std::max<int64_t>(g.qsa_interval, 1);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool is_qsa = g.n_qsa_layers() > 0 && ((l + 1) / qsa_interval - l / qsa_interval) > 0;
        uint64_t sess_l = qsa_layer;
        if (!is_qsa) {
            const uint64_t s1 = basalt::core::session_bytes(g, in_.max_context, in_.k, l, l + 1);
            sess_l = s1 > sess0 ? s1 - sess0 : 0;
        }
        spec.layer.push_back(pool_l[(size_t) l] + nat_l[(size_t) l] +
                             basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) == l; }, g) +
                             n_sess * sess_l);
    }
    for (size_t i = 0; i < c.cards.size(); ++i) {
        spec.fixed.push_back(pool_sh + nat_sh + hc8_sh + n_sess * sess_sh + (i == 0 ? kMainOtherBytes : 0));
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
