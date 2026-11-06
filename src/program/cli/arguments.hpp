#pragma once

#include <optional>
#include <string>

namespace basalt::program { struct CommonOptions; }
namespace basalt::model { class ModelAdapter; }

namespace basalt::program::cli {

// An exit code for help or an invalid argument; otherwise startup can continue.
std::optional<int> parse_options(int argc, char** argv, CommonOptions& o, const model::ModelAdapter* adapter);

}  // namespace basalt::program::cli
