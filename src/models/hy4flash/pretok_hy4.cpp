// src/models/hy4flash/pretok_hy4.cpp - the hy_v4 pre-tokenizer: the text split into the pieces BPE merges.
//
// llama.cpp's HY_V4 pre-type applies three expressions in turn, each to the pieces the previous left:
//   \p{N}{1,3}
//   [CJK ideographs, hiragana, katakana]+
//   [ASCII punctuation][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// The shared Unicode table has letters, numbers, marks and whitespace; punctuation and symbols are taken as the code points
// that are none of those, nor controls.
#include "models/hy4flash/adapter.hpp"

#include "unicode_flags.hpp"

#include <functional>

namespace basalt::models::hy4flash {

using program::tokenizer::cpt_flag_table;
using program::tokenizer::decode_cp;
using program::tokenizer::kLetter;
using program::tokenizer::kMark;
using program::tokenizer::kNumber;
using program::tokenizer::kWhitespace;

namespace {

struct Text {
    std::vector<uint32_t> cp;
    std::vector<uint16_t> fl;
    std::vector<size_t> off;   // byte offset of each code point, then the size
    bool letter(size_t i) const { return (fl[i] & kLetter) != 0; }
    bool mark(size_t i) const { return (fl[i] & kMark) != 0; }
    bool number(size_t i) const { return (fl[i] & kNumber) != 0; }
    bool space(size_t i) const { return (fl[i] & kWhitespace) != 0; }
    bool newline(size_t i) const { return cp[i] == '\r' || cp[i] == '\n'; }
    bool control(size_t i) const { return cp[i] < 0x20 || (cp[i] >= 0x7f && cp[i] < 0xa0); }
    bool punct_symbol(size_t i) const { return !letter(i) && !number(i) && !mark(i) && !space(i) && !control(i); }
    bool ascii_punct(size_t i) const {
        const uint32_t c = cp[i];
        return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
    }
    bool ascii_letter(size_t i) const { return (cp[i] | 0x20) >= 'a' && (cp[i] | 0x20) <= 'z'; }
    bool cjk(size_t i) const {
        const uint32_t c = cp[i];
        return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
    }
};

// A matcher returns the code points it takes at `i` (0: no match), over [i, end).
using Matcher = std::function<size_t(const Text&, size_t, size_t)>;

size_t digits(const Text& t, size_t i, size_t end) {
    size_t n = 0;
    while (i + n < end && n < 3 && t.number(i + n)) ++n;
    return n;
}

size_t cjk_run(const Text& t, size_t i, size_t end) {
    size_t n = 0;
    while (i + n < end && t.cjk(i + n)) ++n;
    return n;
}

size_t word_piece(const Text& t, size_t i, size_t end) {
    // [ASCII punctuation][A-Za-z]+
    if (t.ascii_punct(i)) {
        size_t n = 1;
        while (i + n < end && t.ascii_letter(i + n)) ++n;
        if (n > 1) return n;
    }
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    for (size_t skip : {(size_t) 1, (size_t) 0}) {
        if (skip && (t.newline(i) || t.letter(i) || t.punct_symbol(i))) continue;
        size_t n = skip;
        while (i + n < end && (t.letter(i + n) || t.mark(i + n))) ++n;
        if (n > skip) return n;
    }
    // ?[\p{P}\p{S}]+[\r\n]*
    {
        size_t n = t.cp[i] == ' ' ? 1 : 0;
        const size_t start = n;
        while (i + n < end && t.punct_symbol(i + n)) ++n;
        if (n > start) {
            while (i + n < end && t.newline(i + n)) ++n;
            return n;
        }
    }
    // \s*[\r\n]+
    {
        size_t n = 0, last_nl = 0;
        while (i + n < end && t.space(i + n)) {
            ++n;
            if (t.newline(i + n - 1)) last_nl = n;
        }
        if (last_nl) return last_nl;
    }
    // \s+(?!\S), then \s+
    if (t.space(i)) {
        size_t n = 0;
        while (i + n < end && t.space(i + n)) ++n;
        if (i + n == end) return n;
        return n > 1 ? n - 1 : 1;
    }
    return 0;
}

// Splits every piece [a, b) by `m`: what it matches become pieces, and so does what lies between.
std::vector<std::pair<size_t, size_t>> split(const Text& t, const std::vector<std::pair<size_t, size_t>>& in, const Matcher& m) {
    std::vector<std::pair<size_t, size_t>> out;
    for (auto [a, b] : in) {
        size_t rest = a, i = a;
        while (i < b) {
            const size_t n = m(t, i, b);
            if (n == 0) { ++i; continue; }
            if (rest < i) out.push_back({rest, i});
            out.push_back({i, i + n});
            i += n;
            rest = i;
        }
        if (rest < b) out.push_back({rest, b});
    }
    return out;
}

}  // namespace

void pretok_hy4(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    const uint8_t* s = (const uint8_t*) text;
    const uint16_t* table = cpt_flag_table();
    Text t;
    for (size_t i = 0; i < size;) {
        size_t len = 0;
        const uint32_t c = decode_cp(s, size, i, len);
        t.cp.push_back(c);
        t.fl.push_back(c < 0x110000 ? table[c] : 0);
        t.off.push_back(i);
        i += len;
    }
    t.off.push_back(size);
    std::vector<std::pair<size_t, size_t>> pieces = {{0, t.cp.size()}};
    pieces = split(t, pieces, digits);
    pieces = split(t, pieces, cjk_run);
    pieces = split(t, pieces, word_piece);
    for (auto [a, b] : pieces) ends.push_back(t.off[b]);
}

}  // namespace basalt::models::hy4flash
