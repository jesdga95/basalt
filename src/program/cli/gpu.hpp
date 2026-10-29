#pragma once

#include <optional>
#include <vector>
#include "basalt/program/gpu_select.hpp"

namespace basalt::program { struct Options; }

namespace basalt::program::cli {

// An exit code for --gpu-plan or a failed choice; otherwise selected cards and roles are applied.
std::optional<int> select_gpus(Options& o, std::vector<GpuCard>& chosen_cards, int max_expert_tiers);

}  // namespace basalt::program::cli
