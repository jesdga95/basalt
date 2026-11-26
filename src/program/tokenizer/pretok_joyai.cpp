#include "basalt/program/tokenizer/pretok_joyai.hpp"

#include "unicode_flags.hpp"

#include <cstdint>
#include <utility>

namespace basalt::program::tokenizer {
namespace {

using Span = std::pair<size_t, size_t>;
constexpr uint16_t kPunctuation = 0x0020;
constexpr uint16_t kSymbol = 0x0040;

bool is_cjk(uint32_t c) {
    return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x309f) || (c >= 0x30a0 && c <= 0x30ff);
}

bool ascii_punct(uint32_t c) {
    if (c >= '!' && c <= '/') return true;
    if (c >= ':' && c <= '@') return true;
    if (c >= '[' && c <= '`') return true;
    return c >= '{' && c <= '~';
}

bool ascii_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

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

template <class Match>
std::vector<Span> split(const std::vector<Span>& in, Match match_at) {
    std::vector<Span> out;
    out.reserve(in.size());
    for (const Span& piece : in) {
        size_t gap = piece.first;
        for (size_t p = piece.first; p < piece.second;) {
            const size_t q = match_at(p, piece.second);
            if (q == p) {
                ++p;
                continue;
            }
            if (p > gap) out.emplace_back(gap, p);
            out.emplace_back(p, q);
            p = gap = q;
        }
        if (gap < piece.second) out.emplace_back(gap, piece.second);
    }
    return out;
}

}  // namespace

void pretok_joyai(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (!text || size == 0) return;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(text);
    const uint16_t* table = cpt_flag_table();
    std::vector<uint32_t> cps;
    std::vector<uint16_t> fl;
    std::vector<size_t> offsets;
    cps.reserve(size);
    fl.reserve(size);
    offsets.reserve(size + 1);
    for (size_t i = 0; i < size;) {
        size_t len = 0;
        const uint32_t cp = decode_cp(bytes, size, i, len);
        offsets.push_back(i);
        cps.push_back(cp);
        fl.push_back(table[cp]);
        i += len;
    }
    offsets.push_back(size);
    if (cps.empty()) return;

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
    for (const Span& piece : pieces) ends.push_back(offsets[piece.second]);
}

}  // namespace basalt::program::tokenizer
