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

inline // A stray or truncated byte stands for U+FFFD and consumes one byte.
uint32_t decode_cp(const uint8_t* s, size_t n, size_t i, size_t& len) {
    const uint8_t c = s[i];
    len = 1;
    if (c < 0x80) return c;
    int extra = 0;
    uint32_t v = 0, lo = 0x80;
    if ((c & 0xE0) == 0xC0) { extra = 1; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { extra = 2; v = c & 0x0F; lo = 0x800; }
    else if ((c & 0xF8) == 0xF0) { extra = 3; v = c & 0x07; lo = 0x10000; }
    else return 0xFFFD;
    if (i + (size_t) extra >= n) return 0xFFFD;
    for (int k = 1; k <= extra; ++k) {
        if ((s[i + (size_t) k] & 0xC0) != 0x80) return 0xFFFD;
        v = (v << 6) | (s[i + (size_t) k] & 0x3F);
    }
    if (v < lo || v > 0x10FFFF) return 0xFFFD;
    len = (size_t) extra + 1;
    return v;
}

/// The flags of every code point, 0x110000 entries; a code point that belongs to no category has a non-zero entry too
/// (undefined = 1), so "flags == 0" means out of range.
const uint16_t* cpt_flag_table();

/// Splits `text` (UTF-8) into the pieces the qwen35 pre-tokenizer yields and appends each piece's end byte offset.

}  // namespace basalt::program::tokenizer
