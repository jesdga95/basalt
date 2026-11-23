// include/basalt/program/pretok_o200k.hpp - the o200k family of pre-tokenizers: OpenAI's o200k expression and
// Mistral's tekken one derived from it (see src/program/tokenizer/pretok_o200k.cpp).
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::program::tokenizer {

struct O200kForm {
    bool contractions = true;   ///< (?i:'s|'t|'re|'ve|'m|'ll|'d)? after the letter alternatives (o200k; not tekken)
    int digits = 3;             ///< \p{N}{1,digits}: 3 for o200k, 1 for tekken
};

/// `ends` gets where each piece of the text ends.
void pretok_o200k(const char* text, size_t size, std::vector<size_t>& ends, const O200kForm& form);

}  // namespace basalt::program::tokenizer
