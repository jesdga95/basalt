#pragma once
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/model/adapter.hpp"

namespace basalt::models::qwen4exp {
struct Options;
}

namespace basalt::models::qwen4exp {
std::unique_ptr<model::ModelAdapter> make_adapter();
int run_engine(Options& o, std::vector<program::GpuCard>& cards);
bool parse_flag(const std::string& name, model::FlagArgs& args, Options& o, int& code);
std::optional<int> finish_flags(Options& o);
}
