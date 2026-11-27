// src/models/inkling2/adapter.hpp - the Inkling 2 adapter's entry points.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace basalt::model {
class ModelAdapter;
}

namespace basalt::models::inkling2 {

std::unique_ptr<model::ModelAdapter> make_adapter();

/// The inkling2 pre-tokenizer (OpenAI's o200k split): `ends` gets where each piece of the text ends.
void pretok_inkling2(const char* text, size_t size, std::vector<size_t>& ends);

/// The checkpoint's tokenizer.json sets ignore_merges: a piece that is a vocabulary entry is that one token (337 entries
/// cannot be built from the merges).
inline constexpr bool kIgnoreMerges = true;

}  // namespace basalt::models::inkling2
