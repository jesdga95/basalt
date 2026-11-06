#include "run.hpp"
#include "session_model.hpp"
#include "../cli/gpu.hpp"
#include "../cli/startup.hpp"

#include "basalt/model/adapter.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/options.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace basalt::program::generate {

namespace cli = basalt::program::cli;

int run(int argc, char** argv) {
    basalt::models::register_builtin_models();
    std::unique_ptr<basalt::program::CommonOptions> options;
    const basalt::model::ModelAdapter* adapter = nullptr;
    if (const auto code = cli::start(argc, argv, options, adapter)) return *code;
    basalt::program::CommonOptions& common = *options;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<basalt::program::GpuCard> chosen_cards;
    if (const auto code = cli::select_gpus(common, chosen_cards, (adapter != nullptr ? adapter->max_expert_cards() : basalt::model::max_expert_cards()) + 1)) return *code;
    // ---- where every loader reads from: the sections of the .basalt file
    if (const auto code = cli::prepare_model_process(common)) return *code;
    if (const std::optional<int> code = adapter->run(common, chosen_cards)) return *code;
    return run_session_model(common, *adapter);
}

}
