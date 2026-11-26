#include "basalt/models/deepseek4/tokenizer.hpp"

#include "unicode_flags.hpp"

#include <cstdint>

namespace basalt::models::deepseek4 {
namespace {

using basalt::program::tokenizer::cpt_flag_table;
using basalt::program::tokenizer::decode_cp;
using basalt::program::tokenizer::kLetter;
using basalt::program::tokenizer::kMark;
using basalt::program::tokenizer::kNumber;
using basalt::program::tokenizer::kWhitespace;

constexpr uint16_t kPunctuation = 0x20;
constexpr uint16_t kSymbol = 0x40;
constexpr uint32_t kCjkHanFirst = 0x4E00, kCjkHanLast = 0x9FA5;
constexpr uint32_t kCjkKanaFirst = 0x3040, kCjkKanaLast = 0x30FF;

struct CodePoint {
    uint32_t value;
    uint16_t flags;
    size_t end;
};

CodePoint at(const uint8_t* text, size_t size, size_t pos, const uint16_t* flags) {
    if (pos >= size) return {0, 0, size};
    size_t bytes = 1;
    const uint32_t value = decode_cp(text, size, pos, bytes);
    return {value, flags[value], pos + bytes};
}

bool ascii_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

bool ascii_punctuation(uint32_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

bool cjk(uint32_t c) {
    return (c >= kCjkHanFirst && c <= kCjkHanLast) || (c >= kCjkKanaFirst && c <= kCjkKanaLast);
}

bool letter_mark(uint16_t flags) { return (flags & (kLetter | kMark)) != 0; }
bool punct_symbol(uint16_t flags) { return (flags & (kPunctuation | kSymbol)) != 0; }

}  // namespace

void pretok_deepseek4(const char* input, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (!input || size == 0) return;
    if (ends.capacity() < size / 4 + 16) ends.reserve(size / 4 + 16);

    const auto* text = reinterpret_cast<const uint8_t*>(input);
    const uint16_t* flags = cpt_flag_table();
    size_t pos = 0;
    const auto add = [&](size_t end) {
        if (end > pos) ends.push_back(end);
        pos = end;
    };

    while (pos < size) {
        CodePoint cur = at(text, size, pos, flags);

        // tokenizer.json isolates Unicode number runs in groups of at most three code points.
        if (cur.flags & kNumber) {
            size_t end = pos;
            int count = 0;
            while (end < size && count < 3) {
                const CodePoint n = at(text, size, end, flags);
                if (!(n.flags & kNumber)) break;
                end = n.end;
                ++count;
            }
            add(end);
            continue;
        }

        // Its CJK split is intentionally limited to the ranges declared in tokenizer.json.
        if (cjk(cur.value)) {
            size_t end = cur.end;
            while (end < size) {
                const CodePoint n = at(text, size, end, flags);
                if (!cjk(n.value)) break;
                end = n.end;
            }
            add(end);
            continue;
        }

        // [!"#$%&'()*+,\\-./:;<=>?@[\\]^_`{|}~][A-Za-z]+
        if (ascii_punctuation(cur.value) && cur.end < size) {
            size_t end = cur.end;
            CodePoint n = at(text, size, end, flags);
            if (ascii_letter(n.value)) {
                end = n.end;
                while (end < size) {
                    n = at(text, size, end, flags);
                    if (!ascii_letter(n.value)) break;
                    end = n.end;
                }
                add(end);
                continue;
            }
        }

        // [^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+
        size_t letters = pos;
        if (!letter_mark(cur.flags)) {
            const bool optional = cur.value != '\r' && cur.value != '\n' && !(cur.flags & kLetter) &&
                                  !punct_symbol(cur.flags);
            const CodePoint n = optional ? at(text, size, cur.end, flags) : CodePoint{0, 0, size};
            if (!optional || cjk(n.value) || !letter_mark(n.flags)) letters = size;
            else letters = cur.end;
        }
        if (letters < size) {
            size_t end = letters;
            while (end < size) {
                const CodePoint n = at(text, size, end, flags);
                if (cjk(n.value) || !letter_mark(n.flags)) break;
                end = n.end;
            }
            add(end);
            continue;
        }

        //  ?[\\p{P}\\p{S}]+[\\r\\n]*
        size_t punctuation = pos;
        if (cur.value == ' ' && cur.end < size && punct_symbol(at(text, size, cur.end, flags).flags))
            punctuation = cur.end;
        if (punct_symbol(at(text, size, punctuation, flags).flags)) {
            size_t end = punctuation;
            while (end < size) {
                const CodePoint n = at(text, size, end, flags);
                if (!punct_symbol(n.flags)) break;
                end = n.end;
            }
            while (end < size) {
                const CodePoint n = at(text, size, end, flags);
                if (n.value != '\r' && n.value != '\n') break;
                end = n.end;
            }
            add(end);
            continue;
        }

        if (cur.flags & kWhitespace) {
            size_t end = pos, last_newline_end = 0, count = 0;
            while (end < size) {
                const CodePoint n = at(text, size, end, flags);
                if (!(n.flags & kWhitespace)) break;
                end = n.end;
                ++count;
                if (n.value == '\r' || n.value == '\n') last_newline_end = end;
            }
            // \\s*[\\r\\n]+, followed by the remaining regex whitespace alternatives.
            if (last_newline_end) {
                add(last_newline_end);
                continue;
            }
            // \\s+(?!\\S) leaves one final space before non-whitespace text when possible.
            if (end == size) {
                add(end);
                continue;
            }
            if (count > 1) {
                size_t prefix_end = pos;
                for (size_t i = 1; i < count; ++i) prefix_end = at(text, size, prefix_end, flags).end;
                add(prefix_end);
                continue;
            }
            add(end);
            continue;
        }

        add(cur.end);
    }
}

}  // namespace basalt::models::deepseek4
