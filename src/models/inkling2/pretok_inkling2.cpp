// src/models/inkling2/pretok_inkling2.cpp - the inkling2 pre-tokenizer: the checkpoint's tokenizer.json splits with
// OpenAI's o200k expression (basalt/program/pretok_o200k.hpp).
#include "models/inkling2/adapter.hpp"

#include "basalt/program/pretok_o200k.hpp"

namespace basalt::models::inkling2 {

void pretok_inkling2(const char* text, size_t size, std::vector<size_t>& ends) {
    program::tokenizer::pretok_o200k(text, size, ends, {true, 3});
}

}  // namespace basalt::models::inkling2
