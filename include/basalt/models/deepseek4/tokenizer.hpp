#pragma once

#include <cstddef>
#include <vector>

namespace basalt::models::deepseek4 {

/// Splits text using the checkpoint tokenizer.json pre-tokenizer (numeric groups, CJK runs, then its GPT-style regex).
void pretok_deepseek4(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::models::deepseek4
