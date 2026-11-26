// src/program/tokenizer/pretok_glm4.cpp - the glm4 pre-tokenizer: the text split into the pieces BPE merges.
//
// The splitting rules are those of llama.cpp's LLAMA_VOCAB_PRE_TYPE_CHATGLM4 (src/llama-vocab.cpp, MIT licence; see
// THIRD_PARTY_LICENSES), which tools/tokenizer.py ships as one GLM4_PATTERNS expression. It is
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// and is done here by hand, without backtracking, exactly as pretok_qwen35.cpp does the qwen35 one: the code points are
// decoded once with their flags and the pieces come back as byte offsets into the UTF-8 text.
//
// Against qwen35 there are three differences, and each one changes real text. A word run is \p{L} alone, so a combining
// mark starts a piece of its own ("e" + U+0301 is "e" then the mark); a number run takes at most three digits, so a long
// run of them is several pieces; and the punctuation class is [^\s\p{L}\p{N}], which holds marks that the qwen35 class
// leaves out. The contractions are spelled with explicit case classes rather than (?i:...).
#include "basalt/program/glm_forms.hpp"
#include "unicode_flags.hpp"

namespace basalt::program::tokenizer {

namespace {

constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

}  // namespace

void pretok_glm4(const char* text, size_t size, std::vector<size_t>& ends) {
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
    // The classes of the expression: a word run is letters only, and everything that is neither whitespace, letter nor
    // number (marks included) is punctuation.
    const uint16_t kNotPunct = kWhitespace | kLetter | kNumber;

    size_t prev_end = 0;
    auto add = [&](size_t end) {
        if (end > prev_end) ends.push_back(off[end]);
        prev_end = end;
    };

    for (size_t pos = 0; pos < n;) {
        const uint32_t cpt = cps[pos];
        const uint16_t flags = fl[pos];

        // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
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

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || (flags & kNumber))) {
            if ((flags & kLetter) || (flags_at(pos + 1) & kLetter)) {
                ++pos;
                while (flags_at(pos) & kLetter) ++pos;
                add(pos);
                continue;
            }
        }

        // \p{N}{1,3}
        if (flags & kNumber) {
            size_t run = 1;
            while (pos + run < n && run < 3 && (flags_at(pos + run) & kNumber)) ++run;
            pos += run;
            add(pos);
            continue;
        }

        // ?[^\s\p{L}\p{N}]+[\r\n]*
        uint16_t flags2 = (cpt == ' ' ? flags_at(pos + 1) : flags);
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
