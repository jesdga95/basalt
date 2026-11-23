#pragma once

#include <optional>
#include <vector>
#include "basalt/program/gpu_select.hpp"

namespace basalt::model { class ModelAdapter; }
namespace basalt::program { struct CommonOptions; }

namespace basalt::program::cli {

// An exit code for --gpu-plan or a failed choice; otherwise selected cards and roles are applied.  `adapter` (null before a
// model is known): the modes and card counts its model runs, refused here before any load.  The mode comes from
// --gpu-mode and sets each helper card's role: layer stages in split mode, else prompt splits for a model that reads prompts
// on its second card (`prompt_split`), else expert tiers.
std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, const model::ModelAdapter* adapter);

}  // namespace basalt::program::cli
