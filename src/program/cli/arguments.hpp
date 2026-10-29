#pragma once

#include <optional>
#include <string>

namespace basalt::program { struct Options; }

namespace basalt::program::cli {

// An exit code for help or an invalid argument; otherwise startup can continue.
std::optional<int> parse_options(int argc, char** argv, Options& o, std::string& ppl_base, std::string& ppl_out);

}  // namespace basalt::program::cli
