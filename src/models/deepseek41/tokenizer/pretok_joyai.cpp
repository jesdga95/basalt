#include "models/deepseek41/adapter.hpp"

#include "basalt/program/tokenizer/pretok_joyai.hpp"

namespace basalt::models::deepseek41 {

void pretok_joyai(const char* text, size_t size, std::vector<size_t>& ends) {
    program::tokenizer::pretok_joyai(text, size, ends);
}

}  // namespace basalt::models::deepseek41
