#pragma once

#include <functional>
#include <optional>
#include <vector>
#include "basalt/program/gpu_select.hpp"

namespace basalt::program { struct CommonOptions; }

namespace basalt::program::cli {

// An exit code for --gpu-plan or a failed choice; otherwise selected cards and roles are applied.  `layer_stage` (with
// `prompt_line`): asked with the main and the second card's bandwidth once the cards are visible, whether the model splits its
// layers across them.
std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, int max_expert_tiers, bool prompt_line = false,
                               const std::function<bool(double, double)>& layer_stage = {});

}  // namespace basalt::program::cli
