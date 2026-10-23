// src/program/tokenizer/pretok_qwen35.cpp - the qwen35 pre-tokenizer: the text split into the pieces BPE merges.
//
// The splitting rules are those of llama.cpp's unicode_regex_split_custom_qwen35 (src/unicode.cpp at commit f5027fb21,
// MIT licence, written by Kabir Potdar for llama.cpp pull request 22110; see THIRD_PARTY_LICENSES). They implement
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// by hand, without backtracking. Here the code points are decoded once, with their flags, and the pieces come back as
// byte offsets into the UTF-8 text.
#include "unicode_flags.hpp"

namespace basalt::program::tokenizer {

namespace {

constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// A stray or truncated byte stands for U+FFFD and consumes one byte.
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

}  // namespace

void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends) {
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
    const size_t n = cps.size();

    auto cpt_at = [&](size_t pos) -> uint32_t { return pos < n ? cps[pos] : kOutOfRange; };
    auto flags_at = [&](size_t pos) -> uint16_t { return pos < n ? fl[pos] : (uint16_t) 0; };
    auto lower = [](uint32_t c) -> uint32_t { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    const uint16_t kLetterOrMark = kLetter | kMark;

    size_t prev_end = 0;
    auto add = [&](size_t end) {
        if (end > prev_end) ends.push_back(off[end]);
        prev_end = end;
    };

    for (size_t pos = 0; pos < n;) {
        const uint32_t cpt = cps[pos];
        const uint16_t flags = fl[pos];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            const uint32_t c1 = lower(cpt_at(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                add(pos);
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t c2 = lower(cpt_at(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    add(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' || cpt == '\n' || (flags & kNumber))) {
            if ((flags & kLetterOrMark) || (flags_at(pos + 1) & kLetterOrMark)) {
                ++pos;
                while (flags_at(pos) & kLetterOrMark) ++pos;
                add(pos);
                continue;
            }
        }

        // \p{N}
        if (flags & kNumber) {
            ++pos;
            add(pos);
            continue;
        }

        // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        uint16_t flags2 = (cpt == ' ' ? flags_at(pos + 1) : flags);
        const uint16_t kNotPunct = kWhitespace | kLetter | kMark | kNumber;
        if (!(flags2 & kNotPunct) && flags) {
            pos += (cpt == ' ');
            while (!(flags2 & kNotPunct) && flags2) flags2 = flags_at(++pos);
            uint32_t c2 = cpt_at(pos);
            while (c2 == '\r' || c2 == '\n') c2 = cpt_at(++pos);
            add(pos);
            continue;
        }

        size_t num_ws = 0, last_end_rn = 0;
        while (flags_at(pos + num_ws) & kWhitespace) {
            const uint32_t c2 = cpt_at(pos + num_ws);
            if (c2 == '\r' || c2 == '\n') last_end_rn = pos + num_ws + 1;
            ++num_ws;
        }

        // \s*[\r\n]+
        if (last_end_rn > 0) {
            pos = last_end_rn;
            add(pos);
            continue;
        }

        // \s+(?!\S)
        if (num_ws > 1 && cpt_at(pos + num_ws) != kOutOfRange) {
            pos += num_ws - 1;
            add(pos);
            continue;
        }

        // \s+
        if (num_ws > 0) {
            pos += num_ws;
            add(pos);
            continue;
        }

        add(++pos);
    }
}

}  // namespace basalt::program::tokenizer
