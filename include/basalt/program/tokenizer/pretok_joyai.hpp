#pragma once

#include <cstddef>
#include <vector>

namespace basalt::program::tokenizer {

/// Splits text with the GGUF `joyai-llm` pre-tokenizer rules.
void pretok_joyai(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::program::tokenizer
