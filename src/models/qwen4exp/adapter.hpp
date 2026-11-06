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

namespace basalt::models::qwen4exp::tokenizer {
/// The qwen35 pre-tokenizer: `ends` gets where each piece of the text ends.
void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends);
}  // namespace basalt::models::qwen4exp::tokenizer
