#pragma once

#include <optional>
#include <string>

namespace basalt::program { struct Options; }

namespace basalt::program::cli {

std::optional<int> start(int argc, char** argv, Options& o, std::string& ppl_base, std::string& ppl_out);
std::optional<int> prepare_model_process(const Options& o);

}  // namespace basalt::program::cli
