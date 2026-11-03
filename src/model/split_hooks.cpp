// src/model/split_hooks.cpp - see include/basalt/model/split_hooks.hpp.
#include "basalt/model/split_hooks.hpp"

#include <cstdint>
#include <limits>

namespace basalt::model {

bool SplitHooks::split_open(const GpuCards& c, std::string& err) {
    if (c.mode != program::GpuMode::Split || c.cards.size() < 2) return true;
    SplitSpec s = split_spec(c);
    if (c.diag_same) {
        if (c.cards.size() != 2 || c.diag_cut <= 0) {
            err = "--diag-gpu-same needs two cards and --diag-gpu-cut L, the layer the second part starts at";
            return false;
        }
        s.room.assign(c.cards.size(), std::numeric_limits<uint64_t>::max());   // one card holds both parts: the model's fit decides
    }
    cut_ = split_cut(c, s);
    if (!cut_.error.empty()) {
        err = cut_.error;
        return false;
    }
    devices_.clear();
    for (const GpuCards::Card& card : c.cards) devices_.push_back(c.diag_same ? c.cards[0].device : card.device);
    n_layers_ = (int) s.layer.size();
    open_ = true;
    return split_place(cut_, devices_, err);
}

std::string SplitHooks::split_report(size_t card) const {
    if (!open_ || card >= cut_.first.size()) return {};
    const std::string part = split_part(card);
    return cut_layers(cut_, card, n_layers_) + (part.empty() ? "" : ", " + part);
}

}  // namespace basalt::model
