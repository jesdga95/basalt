#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::program {
struct Options;
}

namespace basalt::models::qwen4exp {
std::unique_ptr<model::ModelAdapter> make_adapter();
bool parse_flag(const std::string& name, model::FlagArgs& args, program::Options& o, int& code);
std::optional<int> finish_flags(program::Options& o);
}
