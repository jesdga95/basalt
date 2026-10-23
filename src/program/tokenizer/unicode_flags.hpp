// src/program/tokenizer/unicode_flags.hpp - Unicode category flags of a code point, for the qwen35 pre-tokenizer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::program::tokenizer {

enum CptFlag : uint16_t {
    kNumber = 0x0002,       // \p{N}
    kLetter = 0x0004,       // \p{L}
    kMark = 0x0010,         // \p{M}
    kWhitespace = 0x0100,   // \s
};

/// The flags of every code point, 0x110000 entries; a code point that belongs to no category has a non-zero entry too
/// (undefined = 1), so "flags == 0" means out of range.
const uint16_t* cpt_flag_table();

/// Splits `text` (UTF-8) into the pieces the qwen35 pre-tokenizer yields and appends each piece's end byte offset.
void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::program::tokenizer
