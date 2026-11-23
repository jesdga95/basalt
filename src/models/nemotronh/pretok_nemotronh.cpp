// src/models/nemotronh/pretok_nemotronh.cpp - the nemotronh pre-tokenizer: the checkpoint's tokenizer.json splits with
// Mistral's tekken expression, o200k's without the contractions and one digit a piece (basalt/program/pretok_o200k.hpp).
#include "models/nemotronh/adapter.hpp"

#include "basalt/program/pretok_o200k.hpp"

namespace basalt::models::nemotronh {

void pretok_nemotronh(const char* text, size_t size, std::vector<size_t>& ends) {
    program::tokenizer::pretok_o200k(text, size, ends, {false, 1});
}

}  // namespace basalt::models::nemotronh
