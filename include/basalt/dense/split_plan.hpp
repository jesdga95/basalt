// include/basalt/dense/split_plan.hpp - where a dense model's layers go when it does not fit its main card: the main card keeps
// layers [0, cut) with the head, the drafter and the shared scratch, a second card layers [cut, n) with their cells and states.
// A pure function of the byte counts and the cards' rates (no GPU), so the choice is tested on the CPU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::dense {

struct SplitCard {
    uint64_t free = 0;              ///< bytes free for the model (after the reserve)
    double mem_gbps = 0;            ///< memory bandwidth (0: unknown, counted as equal)
};

struct SplitInputs {
    std::vector<uint64_t> layer;    ///< per layer: its weights' bytes
    std::vector<uint64_t> per_slot; ///< per layer: a slot's cells and recurrent state there
    int slots = 1;
    uint64_t main_fixed = 0;        ///< the main card's own: head, norms, drafter, scratch, per-slot logits and drafter state
    uint64_t peer_fixed = 0;        ///< the second card's own: its scratch
    SplitCard main, peer;
    int forced = 0;                 ///< a cut to take as given (0: plan one)
};

struct SplitPlan {
    int cut = 0;                    ///< 0: the main card holds everything; else the first layer on the second card
    uint64_t main_bytes = 0, peer_bytes = 0;
    std::string error;              ///< non-empty: neither one card nor a split fits
};

/// Cards within 10% of each other's memory bandwidth are alike.
bool split_cards_alike(const SplitCard& a, const SplitCard& b);

/// The split for `in`: none when everything fits the main card; else on unlike cards the most layers the faster card holds
/// (the main card when its bandwidth is not lower), on alike cards the cut nearest half the layers' bytes that fits both.
SplitPlan plan_split(const SplitInputs& in);

/// A layer split across `cards` (the main card first). `bounds` has one more entry than `cards`: layers [bounds[i], bounds[i + 1])
/// run on card i. The main card's fixed cost is `in.main_fixed` (it already holds one outgoing hand-off). The last card's is
/// `in.peer_fixed`. A card between them is `in.peer_fixed` plus `stage_bytes` (its own outgoing hand-off). Among the cuts that
/// fit, the one whose heaviest card (layer bytes over memory bandwidth) is the lightest.
struct StageSplit {
    std::vector<int> bounds;
    std::vector<uint64_t> bytes;
    std::string error;
};

StageSplit plan_stage_split(const SplitInputs& in, const std::vector<SplitCard>& cards, uint64_t stage_bytes);

}  // namespace basalt::dense
