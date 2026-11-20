// src/models/inkling2/adapter.hpp - the Inkling 2 adapter's entry points.
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::models::inkling2 {

/// The inkling2 pre-tokenizer (OpenAI's o200k split): `ends` gets where each piece of the text ends.
void pretok_inkling2(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::models::inkling2
