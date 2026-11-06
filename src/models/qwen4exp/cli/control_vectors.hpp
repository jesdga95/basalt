// Load the control-vector files selected by the CLI.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include <string>

namespace basalt::core {
struct ModelGeometry;
}

namespace basalt::models::qwen4exp {
struct Options;
}

namespace basalt::models::qwen4exp::cli {

bool load_control_vectors(const Options& o, const basalt::core::ModelGeometry& g, std::string& summary,
                          std::string& error);

}  // namespace basalt::models::qwen4exp::cli
