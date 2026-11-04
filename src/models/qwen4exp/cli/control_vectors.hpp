// Load the control-vector files selected by the CLI.
#pragma once

#include <string>

namespace basalt::core {
struct ModelGeometry;
}

namespace basalt::program {
struct Options;
}

namespace basalt::program::cli {

bool load_control_vectors(const Options& o, const basalt::core::ModelGeometry& g, std::string& summary,
                          std::string& error);

}  // namespace basalt::program::cli
