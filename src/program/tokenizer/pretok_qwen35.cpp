// src/program/tokenizer/pretok_qwen35.cpp - the qwen35 pre-tokenizer: the text split into the pieces BPE merges.
//
// The splitting rules are those of llama.cpp's unicode_regex_split_custom_qwen35 (src/unicode.cpp at commit f5027fb21,
// MIT licence, written by Kabir Potdar for llama.cpp pull request 22110; see THIRD_PARTY_LICENSES). They implement
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// by hand, without backtracking. Here the code points are decoded once, with their flags, and the pieces come back as
// byte offsets into the UTF-8 text.
#include "basalt/program/qwen_forms.hpp"
#include "unicode_flags.hpp"

namespace basalt::program::tokenizer {

namespace {

constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// ASCII (0x00..0x7F) subset of cpt_flag_table(), kept inline in L1 cache.
constexpr uint16_t kAsciiFlags[128] = {
    0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,   // 0x00..0x07
    0x0080, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0080, 0x0080,   // 0x08..0x0F (\t..\r)
    0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,   // 0x10..0x17
    0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,   // 0x18..0x1F
    0x0108, 0x0020, 0x0020, 0x0020, 0x0040, 0x0020, 0x0020, 0x0020,   // 0x20..0x27 (' ', !, ", #, $, %, &, ')
    0x0020, 0x0020, 0x0020, 0x0040, 0x0020, 0x0020, 0x0020, 0x0020,   // 0x28..0x2F ((, ), *, +, ,, -, ., /)
    0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,   // 0x30..0x37 (0..7)
    0x0002, 0x0002, 0x0020, 0x0020, 0x0040, 0x0040, 0x0040, 0x0020,   // 0x38..0x3F (8, 9, :, ;, <, =, >, ?)
    0x0020, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x40..0x47 (@, A..G)
    0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x48..0x4F (H..O)
    0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x50..0x57 (P..W)
    0x0004, 0x0004, 0x0004, 0x0020, 0x0020, 0x0020, 0x0040, 0x0020,   // 0x58..0x5F (X..Z, [, \, ], ^, _)
    0x0040, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x60..0x67 (`, a..g)
    0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x68..0x6F (h..o)
    0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,   // 0x70..0x77 (p..w)
    0x0004, 0x0004, 0x0004, 0x0020, 0x0040, 0x0020, 0x0040, 0x0080,   // 0x78..0x7F (x..z, {, |, }, ~, DEL)
};

struct Cp {
    uint32_t cp;
    uint16_t fl;
    size_t next;
};

inline Cp read_cp(const uint8_t* s, size_t size, const uint16_t* table, size_t i) {
    if (i >= size) return {kOutOfRange, 0, size};
    const uint8_t b = s[i];
    if (b < 0x80) return {b, kAsciiFlags[b], i + 1};
    size_t len = 0;
    const uint32_t c = decode_cp(s, size, i, len);
    return {c, table[c], i + len};
}

inline uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

}  // namespace

void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (size == 0) return;
    if (ends.capacity() < size / 4 + 16) ends.reserve(size / 4 + 16);
    const uint8_t* s = (const uint8_t*) text;
    const uint16_t* table = cpt_flag_table();
    constexpr uint16_t kLetterOrMark = kLetter | kMark;
    constexpr uint16_t kNotPunct = kWhitespace | kLetter | kMark | kNumber;

    size_t prev_end = 0;
    auto add = [&](size_t end) {
        if (end > prev_end) ends.push_back(end);
        prev_end = end;
    };

    for (size_t pos = 0; pos < size;) {
        const Cp cur = read_cp(s, size, table, pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cur.cp == '\'' && cur.next < size) {
            const uint32_t c1 = ascii_lower(s[cur.next]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos = cur.next + 1;
                add(pos);
                continue;
            }
            if (cur.next + 1 < size) {
                const uint32_t c2 = ascii_lower(s[cur.next + 1]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos = cur.next + 2;
                    add(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cur.cp == '\r' || cur.cp == '\n' || (cur.fl & kNumber))) {
            size_t run = 0;
            if (cur.fl & kLetterOrMark) {
                run = cur.next;
            } else {
                const Cp nxt = read_cp(s, size, table, cur.next);
                if (nxt.fl & kLetterOrMark) run = nxt.next;
            }
            if (run != 0) {
                pos = run;
                while (pos < size) {
                    const uint8_t b = s[pos];
                    if (b < 0x80) {
                        if (!(kAsciiFlags[b] & kLetterOrMark)) break;
                        ++pos;
                    } else {
                        size_t len = 0;
                        const uint32_t c = decode_cp(s, size, pos, len);
                        if (!(table[c] & kLetterOrMark)) break;
                        pos += len;
                    }
                }
                add(pos);
                continue;
            }
        }

        // \p{N}
        if (cur.fl & kNumber) {
            pos = cur.next;
            add(pos);
            continue;
        }

        // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        const Cp probe = (cur.cp == ' ') ? read_cp(s, size, table, cur.next) : cur;
        if (!(probe.fl & kNotPunct)) {
            pos = probe.fl ? probe.next : cur.next;
            if (probe.fl) {
                while (pos < size) {
                    const uint8_t b = s[pos];
                    if (b < 0x80) {
                        if (kAsciiFlags[b] & kNotPunct) break;
                        ++pos;
                    } else {
                        size_t len = 0;
                        const uint32_t c = decode_cp(s, size, pos, len);
                        if (table[c] & kNotPunct) break;
                        pos += len;
                    }
                }
                while (pos < size && (s[pos] == '\r' || s[pos] == '\n')) ++pos;
            }
            add(pos);
            continue;
        }

        size_t ws_pos = pos;
        size_t prev_ws_pos = pos;
        size_t last_end_rn = 0;
        size_t num_ws = 0;
        while (ws_pos < size) {
            const Cp w = (ws_pos == pos) ? cur : read_cp(s, size, table, ws_pos);
            if (!(w.fl & kWhitespace)) break;
            prev_ws_pos = ws_pos;
            ws_pos = w.next;
            if (w.cp == '\r' || w.cp == '\n') last_end_rn = ws_pos;
            ++num_ws;
        }

        // \s*[\r\n]+
        if (last_end_rn > 0) {
            pos = last_end_rn;
            add(pos);
            continue;
        }

        // \s+(?!\S)
        if (num_ws > 1 && ws_pos < size) {
            pos = prev_ws_pos;
            add(pos);
            continue;
        }

        // \s+
        if (num_ws > 0) {
            pos = ws_pos;
            add(pos);
            continue;
        }

        pos = cur.next;
        add(pos);
    }
}

}  // namespace basalt::program::tokenizer
