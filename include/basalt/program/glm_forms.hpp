// include/basalt/program/glm_forms.hpp - what every model of the GLM family shares at the text level: the glm4
// pre-tokenizer. The checkpoint's `tokenizer.ggml.pre` is glm4 for GLM-5.3-Flash and for GLM-5.3 alike.
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::program::tokenizer {
/// The glm4 pre-tokenizer: `ends` gets where each piece of the text ends.
void pretok_glm4(const char* text, size_t size, std::vector<size_t>& ends);
}  // namespace basalt::program::tokenizer
