#include "basalt/models/deepseek4/tokenizer.hpp"

#include "basalt/program/tokenizer/pretok_joyai.hpp"

namespace basalt::models::deepseek4 {

void pretok_deepseek4(const char* text, size_t size, std::vector<size_t>& ends) {
    program::tokenizer::pretok_joyai(text, size, ends);
}

}  // namespace basalt::models::deepseek4
