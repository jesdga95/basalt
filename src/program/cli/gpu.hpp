#pragma once

#include <optional>
#include <vector>
#include "basalt/program/gpu_select.hpp"

namespace basalt::model { class ModelAdapter; }
namespace basalt::program { struct CommonOptions; }

namespace basalt::program::cli {

// An exit code for --gpu-plan or a failed choice; otherwise the chosen cards are applied (the CUDA view narrowed to them),
// o.gpus set (each card's free memory, the mode and the shares) and the start line before the load printed.  `adapter`
// (null before a model is known): the modes and card counts its model runs, refused here before any load.
std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, const model::ModelAdapter* adapter);

}  // namespace basalt::program::cli
