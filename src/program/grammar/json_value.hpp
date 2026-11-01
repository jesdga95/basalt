// src/program/grammar/json_value.hpp - a small ordered JSON value for reading schemas (member order is kept; the
// engine's own Json sorts members and keeps only the head of an array).
#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace basalt::program::grammar {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    std::string s;                                          ///< String text, or the Number's source text
    std::vector<JsonValue> items;                           ///< Array
    std::vector<std::pair<std::string, JsonValue>> members; ///< Object, in source order

    bool is(Type t) const { return type == t; }
    const JsonValue* get(const std::string& key) const {
        for (const auto& m : members) if (m.first == key) return &m.second;
        return nullptr;
    }
    bool contains(const std::string& key) const { return get(key) != nullptr; }
    /// Replaces the member, or appends it.
    void set(const std::string& key, JsonValue v) {
        for (auto& m : members) {
            if (m.first == key) { m.second = std::move(v); return; }
        }
        members.emplace_back(key, std::move(v));
    }
    /// A number written without a fraction or an exponent.
    bool is_integer() const {
        return type == Type::Number && s.find_first_of(".eE") == std::string::npos;
    }
    int64_t as_int() const { return (int64_t) std::strtoll(s.c_str(), nullptr, 10); }
    double as_double() const { return std::strtod(s.c_str(), nullptr); }
    bool is_true() const { return type == Type::Bool && b; }
    /// Compact JSON text, as a grammar constant for this value (numbers are written back the way a JSON library
    /// prints them: integers as integers, others in the shortest form that reads back the same).
    std::string dump() const {
        switch (type) {
            case Type::Null: return "null";
            case Type::Bool: return b ? "true" : "false";
            case Type::Number: return number_text();
            case Type::String: return quote(s);
            case Type::Array: {
                std::string o = "[";
                for (size_t i = 0; i < items.size(); ++i) o += (i ? "," : "") + items[i].dump();
                return o + "]";
            }
            case Type::Object: {
                std::string o = "{";
                for (size_t i = 0; i < members.size(); ++i)
                    o += (i ? "," : "") + quote(members[i].first) + ":" + members[i].second.dump();
                return o + "}";
            }
        }
        return "null";
    }
    std::string number_text() const {
        if (is_integer()) {
            if (s.size() < 19) return std::to_string(std::strtoll(s.c_str(), nullptr, 10));
            if (s[0] != '-') {
                unsigned long long u = std::strtoull(s.c_str(), nullptr, 10);
                if (u != ULLONG_MAX) return std::to_string(u);
            } else {
                errno = 0;
                long long i = std::strtoll(s.c_str(), nullptr, 10);
                if (errno == 0) return std::to_string(i);
            }
        }
        double d = std::strtod(s.c_str(), nullptr);
        if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
        char buf[64];
        int prec = 0;
        for (prec = 0; prec < 17; ++prec) {
            std::snprintf(buf, sizeof buf, "%.*e", prec, std::fabs(d));
            if (std::strtod(buf, nullptr) == std::fabs(d)) break;
        }
        // digits and decimal exponent from d.ddde+XX
        std::string mant(buf), digits;
        size_t epos = mant.find('e');
        int exp10 = std::atoi(mant.c_str() + epos + 1);
        for (size_t i = 0; i < epos; ++i) if (mant[i] != '.') digits += mant[i];
        while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
        int k = (int) digits.size();
        int dec = exp10 - (k - 1);   // value = digits * 10^dec
        int n = k + dec;
        std::string o = d < 0 ? "-" : "";
        if (dec >= 0 && n <= 15) {
            o += digits + std::string((size_t) dec, '0') + ".0";
        } else if (0 < n && n <= 15) {
            o += digits.substr(0, (size_t) n) + "." + digits.substr((size_t) n);
        } else if (-4 < n && n <= 0) {
            o += "0." + std::string((size_t) -n, '0') + digits;
        } else {
            o += digits.substr(0, 1);
            if (k > 1) o += "." + digits.substr(1);
            int e = n - 1;
            o += e < 0 ? "e-" : "e+";
            int ae = e < 0 ? -e : e;
            if (ae < 10) o += "0";
            o += std::to_string(ae);
        }
        return o;
    }
    static std::string quote(const std::string& in) {
        std::string o = "\"";
        for (unsigned char c : in) {
            switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                case '\b': o += "\\b"; break;
                case '\f': o += "\\f"; break;
                default:
                    if (c < 0x20) {
                        static const char* hex = "0123456789abcdef";
                        o += "\\u00";
                        o += hex[c >> 4];
                        o += hex[c & 15];
                    } else {
                        o += (char) c;
                    }
            }
        }
        return o + "\"";
    }
};

class JsonReader {
public:
    static bool parse(const std::string& text, JsonValue& out, std::string& err) {
        JsonReader r(text.c_str(), text.c_str() + text.size());
        r.ws();
        if (!r.value(out, 0)) { err = "json: " + r.err; return false; }
        r.ws();
        if (r.p != r.end) { err = "json: trailing data"; return false; }
        return true;
    }

private:
    JsonReader(const char* b, const char* e) : p(b), end(e), begin(b) {}
    const char* p;
    const char* end;
    const char* begin;
    std::string err;

    bool fail(const char* m) { err = std::string(m) + " at byte " + std::to_string((size_t) (p - begin)); return false; }
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool lit(const char* w) {
        size_t n = std::char_traits<char>::length(w);
        if ((size_t) (end - p) >= n && std::string(p, n) == w) { p += n; return true; }
        return false;
    }
    static void utf8(std::string& o, uint32_t c) {
        if (c < 0x80) { o += (char) c; }
        else if (c < 0x800) { o += (char) (0xC0 | (c >> 6)); o += (char) (0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += (char) (0xE0 | (c >> 12)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
        else { o += (char) (0xF0 | (c >> 18)); o += (char) (0x80 | ((c >> 12) & 0x3F)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
    }
    bool hex4(uint32_t& v) {
        if (end - p < 4) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i, ++p) {
            char c = *p;
            v <<= 4;
            if (c >= '0' && c <= '9') v += (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v += (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v += (uint32_t) (c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool str(std::string& o) {
        ++p;
        while (p < end && *p != '"') {
            if (*p != '\\') { o += *p++; continue; }
            if (++p >= end) break;
            char c = *p++;
            switch (c) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t v = 0;
                    if (!hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        const char* save = p;
                        p += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        if (lo >= 0xDC00 && lo < 0xE000) v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        else p = save;
                    }
                    utf8(o, v);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        if (p >= end) return fail("unterminated string");
        ++p;
        return true;
    }
    bool value(JsonValue& v, int depth) {
        if (depth > 200) return fail("nesting too deep");
        if (p >= end) return fail("unexpected end");
        char c = *p;
        if (c == '{') {
            ++p;
            v.type = JsonValue::Type::Object;
            ws();
            if (p < end && *p == '}') { ++p; return true; }
            for (;;) {
                ws();
                if (p >= end || *p != '"') return fail("expected a key");
                std::string key;
                if (!str(key)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                ++p;
                ws();
                JsonValue child;
                if (!value(child, depth + 1)) return false;
                v.members.emplace_back(std::move(key), std::move(child));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            v.type = JsonValue::Type::Array;
            ws();
            if (p < end && *p == ']') { ++p; return true; }
            for (;;) {
                ws();
                JsonValue child;
                if (!value(child, depth + 1)) return false;
                v.items.push_back(std::move(child));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = JsonValue::Type::String; return str(v.s); }
        if (lit("true")) { v.type = JsonValue::Type::Bool; v.b = true; return true; }
        if (lit("false")) { v.type = JsonValue::Type::Bool; v.b = false; return true; }
        if (lit("null")) { v.type = JsonValue::Type::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* s = p;
            ++p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
            v.type = JsonValue::Type::Number;
            v.s.assign(s, p);
            return true;
        }
        return fail("unexpected character");
    }
};

}  // namespace basalt::program::grammar
