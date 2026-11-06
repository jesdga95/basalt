// src/models/deepseek41/tokenizer/pretok_joyai.cpp - the joyai-llm pre-tokenizer: the text split into the pieces BPE merges.
//
// Three regexes apply in turn, each splitting the pieces the previous one left; text a regex does not match stays a
// piece for the next:
//   \p{N}{1,3}
//   [\x{4e00}-\x{9fa5}\x{3040}-\x{309f}\x{30a0}-\x{30ff}]+
//   [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// by hand, without backtracking. tools/tokenizer.py runs the regexes themselves; the parity test holds the two together.
#include "models/deepseek41/adapter.hpp"

#include "unicode_flags.hpp"

#include <cstdint>
#include <string_view>

namespace basalt::models::deepseek41 {

namespace {

using program::tokenizer::cpt_flag_table;
using program::tokenizer::decode_cp;
using program::tokenizer::kLetter;
using program::tokenizer::kMark;
using program::tokenizer::kNumber;
using program::tokenizer::kWhitespace;

constexpr uint16_t kPunctuation = 0x0020, kSymbol = 0x0040;

struct Span { size_t a, b; };

bool is_cjk(uint32_t c) {
    return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x309f) || (c >= 0x30a0 && c <= 0x30ff);
}

bool ascii_punct(uint32_t c) {
    return c < 0x80 && c != 0 && std::string_view("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~").find((char) c) != std::string_view::npos;
}

bool ascii_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

/// The end of the third regex's match at `p` in [p, end); `p` when it does not match there.
size_t match_words(const std::vector<uint32_t>& cps, const std::vector<uint16_t>& fl, size_t p, size_t end) {
    const auto word = [&](size_t i) { return i < end && (fl[i] & (kLetter | kMark)) != 0; };
    const auto mark = [&](size_t i) { return i < end && (fl[i] & (kPunctuation | kSymbol)) != 0; };
    const auto space = [&](size_t i) { return i < end && (fl[i] & kWhitespace) != 0; };
    const auto newline = [&](size_t i) { return i < end && (cps[i] == '\r' || cps[i] == '\n'); };
    const uint32_t c0 = cps[p];
    if (ascii_punct(c0) && p + 1 < end && ascii_letter(cps[p + 1])) {
        size_t q = p + 1;
        while (q < end && ascii_letter(cps[q])) ++q;
        return q;
    }
    const bool lead = c0 != '\r' && c0 != '\n' && (fl[p] & (kLetter | kPunctuation | kSymbol)) == 0;
    if (lead && word(p + 1)) {
        size_t q = p + 1;
        while (word(q)) ++q;
        return q;
    }
    if (word(p)) {
        size_t q = p;
        while (word(q)) ++q;
        return q;
    }
    const size_t m = c0 == ' ' && mark(p + 1) ? p + 1 : p;
    if (mark(m)) {
        size_t q = m;
        while (mark(q)) ++q;
        while (newline(q)) ++q;
        return q;
    }
    if (!space(p)) return p;
    size_t run = p;
    size_t last_newline = end;
    while (space(run)) {
        if (newline(run)) last_newline = run;
        ++run;
    }
    if (last_newline != end) return last_newline + 1;
    if (run == end) return run;
    return run - p >= 2 ? run - 1 : run;
}

}  // namespace

void pretok_joyai(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    const uint8_t* s = (const uint8_t*) text;
    const uint16_t* table = cpt_flag_table();
    std::vector<uint32_t> cps;
    std::vector<uint16_t> fl;
    std::vector<size_t> off;
    cps.reserve(size);
    fl.reserve(size);
    off.reserve(size + 1);
    for (size_t i = 0; i < size;) {
        size_t len = 0;
        const uint32_t c = decode_cp(s, size, i, len);
        off.push_back(i);
        cps.push_back(c);
        fl.push_back(table[c]);
        i += len;
    }
    off.push_back(size);
    if (cps.empty()) return;

    // one regex over the pieces: matches and the text between them become pieces
    const auto split = [](const std::vector<Span>& in, auto match_at) {
        std::vector<Span> out;
        out.reserve(in.size());
        for (const Span& piece : in) {
            size_t gap = piece.a;
            for (size_t p = piece.a; p < piece.b;) {
                const size_t q = match_at(p, piece.b);
                if (q == p) { ++p; continue; }
                if (p > gap) out.push_back({gap, p});
                out.push_back({p, q});
                p = gap = q;
            }
            if (gap < piece.b) out.push_back({gap, piece.b});
        }
        return out;
    };
    std::vector<Span> pieces{{0, cps.size()}};
    pieces = split(pieces, [&](size_t p, size_t end) {
        size_t q = p;
        while (q < end && q - p < 3 && (fl[q] & kNumber) != 0) ++q;
        return q;
    });
    pieces = split(pieces, [&](size_t p, size_t end) {
        size_t q = p;
        while (q < end && is_cjk(cps[q])) ++q;
        return q;
    });
    pieces = split(pieces, [&](size_t p, size_t end) { return match_words(cps, fl, p, end); });
    ends.reserve(pieces.size());
    for (const Span& piece : pieces) ends.push_back(off[piece.b]);
}

}  // namespace basalt::models::deepseek41
