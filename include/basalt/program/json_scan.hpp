// include/basalt/program/json_scan.hpp - a small forward scanner over JSON text, and UTF-8 helpers.
// Reads the flat shapes the pack's tokenizer files use (an object of string -> integer, an array of integers).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace basalt::program::json {

inline void put_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char) c;
    else if (c < 0x800) { s += (char) (0xC0 | (c >> 6)); s += (char) (0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { s += (char) (0xE0 | (c >> 12)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F)); }
    else { s += (char) (0xF0 | (c >> 18)); s += (char) (0x80 | ((c >> 12) & 0x3F)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F)); }
}

// the next code point of valid UTF-8 at s[i] (a stray byte is its own value)
inline uint32_t next_cp(std::string_view s, size_t& i) {
    const uint8_t c = (uint8_t) s[i++];
    int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t v = extra == 3 ? c & 0x07 : extra == 2 ? c & 0x0F : extra == 1 ? c & 0x1F : c;
    while (extra-- > 0 && i < s.size()) v = (v << 6) | ((uint8_t) s[i++] & 0x3F);
    return v;
}

struct Scan {
    std::string_view s;
    size_t i = 0;
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
    bool str(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] != '\\') { out += s[i++]; continue; }
            if (++i >= s.size()) return false;
            const char e = s[i++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    auto hex4 = [&](uint32_t& v) {
                        if (i + 4 > s.size()) return false;
                        v = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = s[i + (size_t) k];
                            const int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                            if (d < 0) return false;
                            v = v * 16 + (uint32_t) d;
                        }
                        i += 4;
                        return true;
                    };
                    uint32_t v = 0;
                    if (!hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, v);
                    break;
                }
                default: out += e; break;   // " \ /
            }
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    }
    bool integer(int64_t& v) {
        ws();
        const size_t b = i;
        v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        return i > b;
    }
};

}  // namespace basalt::program::json
