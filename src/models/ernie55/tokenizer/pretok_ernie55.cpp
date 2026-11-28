// src/models/ernie55/tokenizer/pretok_ernie55.cpp - ERNIE 5.5 pre-tokenizer matching tokenizer.json:
//   1. Isolate contiguous decimal digit runs (\p{Nd}+), chunking each run right-aligned into groups of 3
//      (e.g. "12345" -> "12", "345"; "123456" -> "123", "456").
//   2. Isolate CJK/kana runs: [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+
//   3. Split remaining text with:
//      [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
#include "models/ernie55/adapter.hpp"

#include "unicode_flags.hpp"

#include <cstdint>
#include <string_view>

namespace basalt::models::ernie55 {
namespace {

using program::tokenizer::cpt_flag_table;
using program::tokenizer::decode_cp;
using program::tokenizer::kLetter;
using program::tokenizer::kMark;
using program::tokenizer::kNumber;
using program::tokenizer::kWhitespace;

constexpr uint16_t kPunctuation = 0x0020, kSymbol = 0x0040;

bool is_cjk(uint32_t c) {
    return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x309f) || (c >= 0x30a0 && c <= 0x30ff);
}

bool is_decimal_digit(uint32_t c, uint16_t flag) {
    if (c >= '0' && c <= '9') return true;
    if (c >= 0xFF10 && c <= 0xFF19) return true;
    return (flag & kNumber) != 0;
}

bool ascii_punct(uint32_t c) {
    return c < 0x80 && c != 0 &&
           std::string_view("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~").find((char) c) != std::string_view::npos;
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

void split_cjk_and_words(const std::vector<uint32_t>& cps, const std::vector<uint16_t>& fl,
                         const std::vector<size_t>& off, size_t a, size_t b, std::vector<size_t>& ends) {
    size_t p = a;
    while (p < b) {
        if (is_cjk(cps[p])) {
            size_t q = p + 1;
            while (q < b && is_cjk(cps[q])) ++q;
            ends.push_back(off[q]);
            p = q;
            continue;
        }
        size_t non_cjk_end = p;
        while (non_cjk_end < b && !is_cjk(cps[non_cjk_end])) ++non_cjk_end;
        size_t cur = p;
        while (cur < non_cjk_end) {
            const size_t nxt = match_words(cps, fl, cur, non_cjk_end);
            if (nxt > cur) {
                ends.push_back(off[nxt]);
                cur = nxt;
            } else {
                ends.push_back(off[cur + 1]);
                ++cur;
            }
        }
        p = non_cjk_end;
    }
}

}  // namespace

void pretok_ernie55(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (!text || size == 0) return;

    const uint16_t* table = cpt_flag_table();
    const auto* bytes = reinterpret_cast<const uint8_t*>(text);
    std::vector<uint32_t> cps;
    std::vector<uint16_t> fl;
    std::vector<size_t> off;
    cps.reserve(size);
    fl.reserve(size);
    off.reserve(size + 1);

    size_t byte_i = 0;
    while (byte_i < size) {
        off.push_back(byte_i);
        size_t len = 1;
        const uint32_t cp = decode_cp(bytes, size, byte_i, len);
        cps.push_back(cp);
        fl.push_back(cp < 0x110000 ? table[cp] : 0);
        byte_i += len;
    }
    off.push_back(size);
    const size_t n = cps.size();

    size_t i = 0;
    while (i < n) {
        if (is_decimal_digit(cps[i], fl[i])) {
            size_t j = i + 1;
            while (j < n && is_decimal_digit(cps[j], fl[j])) ++j;
            const size_t run = j - i;
            size_t pos = i;
            const size_t lead = run % 3;
            if (lead != 0) {
                pos += lead;
                ends.push_back(off[pos]);
            }
            while (pos < j) {
                pos += 3;
                ends.push_back(off[pos]);
            }
            i = j;
        } else {
            size_t j = i + 1;
            while (j < n && !is_decimal_digit(cps[j], fl[j])) ++j;
            split_cjk_and_words(cps, fl, off, i, j, ends);
            i = j;
        }
    }
}

}  // namespace basalt::models::ernie55
