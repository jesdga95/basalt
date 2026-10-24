// src/program/grammar/schema_convert.cpp - SchemaNode tree -> GBNF. See schema_internal.hpp.
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "schema_internal.hpp"

namespace basalt::program::grammar {

namespace {

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) o += sep;
        o += v[i];
    }
    return o;
}

std::string repeat(const std::string& s, size_t n) {
    std::string r;
    for (size_t i = 0; i < n; ++i) r += s;
    return r;
}

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

// Every run of characters that cannot be in a rule name becomes one '-'.
std::string sanitize(const std::string& in) {
    std::string o;
    bool gap = false;
    for (char c : in) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
        if (ok) {
            if (gap) o += '-';
            gap = false;
            o += c;
        } else {
            gap = true;
        }
    }
    if (gap) o += '-';
    return o;
}

std::string build_repetition(const std::string& item_rule, int min_items, int max_items,
                             const std::string& separator_rule = "") {
    auto has_max = max_items != (std::numeric_limits<int>::max)();

    if (max_items == 0) return "";
    if (min_items == 0 && max_items == 1) return item_rule + "?";

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) return item_rule + "+";
        if (min_items == 0 && !has_max) return item_rule + "*";
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " +
                  build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1,
                                   has_max ? max_items - 1 : max_items);
    if (min_items == 0) result = "(" + result + ")?";
    return result;
}

void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream& out, int decimals_left = 16,
                       bool top_level = true) {
    auto has_min = min_value != (std::numeric_limits<int64_t>::min)();
    auto has_max = max_value != (std::numeric_limits<int64_t>::max)();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) return;
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != (std::numeric_limits<int>::max)()) out << max_digits;
        }
        out << "}";
    };
    std::function<void(const std::string_view&, const std::string_view&)> uniform_range =
        [&](const std::string_view& from, const std::string_view& to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) i++;
            if (i > 0) out << "\"" << from.substr(0, i) << "\"";
            if (i < from.length() && i < to.length()) {
                if (i > 0) out << " ";
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = repeat("0", sub_len);
                    auto sub_nines = repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], (char) (to[i] - 1));
                        out << " ";
                        more_digits((int) sub_len, (int) sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range((char) (from[i] + 1), to[i]);
                                to_reached = true;
                            } else {
                                digit_range((char) (from[i] + 1), (char) (to[i] - 1));
                            }
                            out << " ";
                            more_digits((int) sub_len, (int) sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, repeat("9", digits));
            min_s = "1" + repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = (std::max)(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int((std::numeric_limits<int64_t>::min)(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = (char) ('0' + min_value);
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, (char) (c - 1));
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', (char) (c - 1));
                out << " ";
                more_digits((int) len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), (std::numeric_limits<int64_t>::max)(), out, less_decimals,
                              /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range((char) (c + 1), '9');
                out << " ";
                more_digits((int) len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, (std::numeric_limits<int64_t>::max)(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const char* const kSpaceRule = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

const std::unordered_map<std::string, BuiltinRule>& primitive_rules() {
    static const std::unordered_map<std::string, BuiltinRule> t = {
        {"boolean", {"(\"true\" | \"false\")", {}}},
        {"decimal-part", {"[0-9]{1,16}", {}}},
        {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
        {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
        {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
        {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
        {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
        {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
        {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
        {"char", {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
        {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
        {"null", {"\"null\"", {}}},
    };
    return t;
}

const std::unordered_map<std::string, BuiltinRule>& string_format_rules() {
    static const std::unordered_map<std::string, BuiltinRule> t = {
        {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
        {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
        {"date-time", {"date \"T\" time", {"date", "time"}}},
        {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
        {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
        {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}},
    };
    return t;
}

bool is_reserved_name(const std::string& name) {
    static const std::unordered_set<std::string> reserved = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto& p : primitive_rules()) s.insert(p.first);
        for (const auto& p : string_format_rules()) s.insert(p.first);
        return s;
    }();
    return reserved.find(name) != reserved.end();
}

const int kMaxPatternDepth = 100;

bool non_literal(char c) {
    switch (c) {
        case '|': case '.': case '(': case ')': case '[': case ']': case '{': case '}': case '*': case '+': case '?':
        case '^': case '$':
            return true;
        default:
            return false;
    }
}

bool escaped_in_regexps_but_not_in_literals(char c) {
    switch (c) {
        case '^': case '$': case '.': case '[': case ']': case '(': case ')': case '|': case '{': case '}': case '*':
        case '+': case '?':
            return true;
        default:
            return false;
    }
}

// The length of a backslash escape the GBNF parser understands at `pos`, 0 if it is not one.
size_t gbnf_escape_length(const std::string& pattern, size_t pos) {
    if (pos + 1 >= pattern.length() || pattern[pos] != '\\') return 0;
    size_t n_hex = 0;
    switch (pattern[pos + 1]) {
        case 'x': n_hex = 2; break;
        case 'u': n_hex = 4; break;
        case 'U': n_hex = 8; break;
        case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']': case '-':
            return 2;
        default:
            return 0;
    }
    if (pos + 2 + n_hex > pattern.length()) return 0;
    for (size_t i = pos + 2; i < pos + 2 + n_hex; i++) {
        char h = pattern[i];
        if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) return 0;
    }
    return 2 + n_hex;
}

// Bytes below 0x20 and 0x7F as GBNF \xNN escapes; every other byte passes through untouched. A JSON schema can
// carry a raw control byte (a decoded \u0000 escape in a pattern or an enum string), and the grammar parser reads its
// text as a C string: a raw NUL would end the text mid-rule ("unexpected end of input"). The escape names the
// same code point and keeps the grammar text free of control bytes.
std::string gbnf_escape_ctl(const std::string& s) {
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char v = (unsigned char) c;
        if (v < 0x20 || v == 0x7F) {
            out += "\\x";
            out += hexd[v >> 4];
            out += hexd[v & 0xF];
        } else {
            out += c;
        }
    }
    return out;
}

// UTF-8 code points of `s`; a malformed byte ends the walk.
std::vector<uint32_t> utf8_points(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char) s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (len == 0 || i + (size_t) len > s.size()) break;
        uint32_t v = len == 1 ? c : c & (0xFFu >> (len + 1));
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            unsigned char d = (unsigned char) s[i + (size_t) k];
            if ((d & 0xC0) != 0x80) { ok = false; break; }
            v = (v << 6) | (d & 0x3F);
        }
        if (!ok) break;
        out.push_back(v);
        i += (size_t) len;
    }
    return out;
}

std::string cpt_to_utf8(uint32_t c) {
    std::string o;
    if (c < 0x80) {
        o += (char) c;
    } else if (c < 0x800) {
        o += (char) (0xC0 | (c >> 6));
        o += (char) (0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        o += (char) (0xE0 | (c >> 12));
        o += (char) (0x80 | ((c >> 6) & 0x3F));
        o += (char) (0x80 | (c & 0x3F));
    } else {
        o += (char) (0xF0 | (c >> 18));
        o += (char) (0x80 | ((c >> 12) & 0x3F));
        o += (char) (0x80 | ((c >> 6) & 0x3F));
        o += (char) (0x80 | (c & 0x3F));
    }
    return o;
}

// A trie over the code points of a set of strings.
struct Trie {
    struct Node {
        std::map<uint32_t, size_t> children;
        int32_t pattern = -1;
    };
    std::vector<Node> nodes;
    int32_t n_patterns = 0;

    explicit Trie(const std::vector<std::string>& words) {
        nodes.emplace_back();
        for (const auto& w : words) {
            size_t cur = 0;
            for (uint32_t ch : utf8_points(w)) {
                auto it = nodes[cur].children.find(ch);
                if (it == nodes[cur].children.end()) {
                    nodes.emplace_back();
                    size_t child = nodes.size() - 1;
                    nodes[cur].children[ch] = child;
                    cur = child;
                } else {
                    cur = it->second;
                }
            }
            if (nodes[cur].pattern < 0) nodes[cur].pattern = n_patterns++;
        }
    }
};

// thrown when the pattern is a valid regex with no grammar equivalent
struct unsupported_pattern : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// thrown when the pattern is not a valid regex
struct invalid_pattern : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace

std::string gbnf_format_literal(const std::string& literal) {
    std::string escaped;
    for (char c : literal) {
        switch (c) {
            case '\r': escaped += "\\r"; break;
            case '\n': escaped += "\\n"; break;
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            default: escaped += gbnf_escape_ctl(std::string(1, c));
        }
    }
    return "\"" + escaped + "\"";
}

struct SchemaConverter::Impl {
    bool dotall;
    std::map<std::string, std::string> rules;
    std::unordered_set<std::string> refs_being_resolved;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    explicit Impl(bool d) : dotall(d) { rules["space"] = kSpaceRule; }

    std::string add_rule(const std::string& name, const std::string& rule) {
        std::string esc_name = sanitize(name);
        if (rules.find(esc_name) == rules.end() || rules[esc_name] == rule) {
            rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (rules.find(esc_name + std::to_string(i)) != rules.end() && rules[esc_name + std::to_string(i)] != rule) i++;
        std::string key = esc_name + std::to_string(i);
        rules[key] = rule;
        return key;
    }

    std::string add_primitive(const std::string& name, const BuiltinRule& rule) {
        auto n = add_rule(name, rule.content);
        for (const auto& dep : rule.deps) {
            auto it = primitive_rules().find(dep);
            if (it == primitive_rules().end()) {
                it = string_format_rules().find(dep);
                if (it == string_format_rules().end()) {
                    errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (rules.find(dep) == rules.end()) add_primitive(dep, it->second);
        }
        return n;
    }

    std::string prim(const std::string& name) { return add_primitive(name, primitive_rules().at(name)); }

    std::string generate_union_rule(const std::string& name, const std::vector<SchemaPtr>& alt_schemas) {
        std::vector<std::string> rs;
        rs.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rs.push_back(visit(*alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return join(rs, " | ");
    }

    std::string visit_pattern(const std::string& pattern, const std::string& name) {
        auto rules_snapshot = rules;
        try {
            return pattern_to_rule(pattern, name);
        } catch (const unsupported_pattern& err) {
            rules = std::move(rules_snapshot);
            warnings.push_back("pattern " + pattern + " is not supported (" + err.what() + "), accepting any string");
            return add_rule(name, add_primitive("string", primitive_rules().at("string")));
        } catch (const invalid_pattern& err) {
            rules = std::move(rules_snapshot);
            errors.push_back("Invalid pattern " + pattern + ": " + err.what());
            return "";
        }
    }

    std::string pattern_to_rule(const std::string& pattern, const std::string& name) {
        if (pattern.length() < 2 || pattern.front() != '^' || pattern.back() != '$') {
            throw unsupported_pattern("not anchored with '^' and '$'");
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();
        int paren_depth = 0;

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule& ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) return false;
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto& item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto& item : ret) results.push_back(to_rule(item));
                return std::make_pair(join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2;   // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookaround, named group, inline flags, ...
                            throw unsupported_pattern("unsupported group syntax");
                        }
                    }
                    paren_depth++;
                    if (paren_depth > kMaxPatternDepth) throw unsupported_pattern("pattern nesting too deep");
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) throw invalid_pattern("unbalanced parentheses");
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') {
                    throw unsupported_pattern("anchor inside the pattern");
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            auto escape_length = gbnf_escape_length(sub_pattern, i);
                            if (escape_length == 0) {
                                throw unsupported_pattern("unsupported escape in character class: " + sub_pattern.substr(i, 2));
                            }
                            square_brackets += sub_pattern.substr(i, escape_length);
                            i += escape_length;
                        } else {
                            square_brackets += gbnf_escape_ctl(std::string(1, sub_pattern[i]));
                            i++;
                        }
                    }
                    if (i >= length) throw invalid_pattern("unterminated character class");
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) throw invalid_pattern("nothing to repeat");
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) throw unsupported_pattern("unterminated curly brackets");
                    curly_brackets += '}';
                    i++;
                    auto nums = split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = (std::numeric_limits<int>::max)();
                    if (nums.size() != 1 && nums.size() != 2) {
                        throw unsupported_pattern("wrong number of values in curly brackets");
                    }
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else {
                            if (!nums[0].empty()) min_times = std::stoi(nums[0]);
                            if (!nums[1].empty()) max_times = std::stoi(nums[1]);
                        }
                    } catch (const std::logic_error&) {
                        throw unsupported_pattern("invalid number in curly brackets");
                    }
                    if (seq.empty()) throw invalid_pattern("nothing to repeat");
                    auto& last = seq.back();
                    auto& sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string& sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(sub_is_literal ? "\"" + sub + "\"" : sub, min_times, max_times, "");
                    seq.back().second = false;
                } else {
                    std::string literal;
                    while (i < length) {
                        if (sub_pattern[i] == '\\') {
                            if (i == length - 1) throw invalid_pattern("trailing backslash");
                            char next = sub_pattern[i + 1];
                            if (escaped_in_regexps_but_not_in_literals(next)) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                auto escape_length = gbnf_escape_length(sub_pattern, i);
                                if (escape_length == 0) {
                                    throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                                }
                                literal += sub_pattern.substr(i, escape_length);
                                i += escape_length;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!non_literal(sub_pattern[i]) &&
                                   (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' ||
                                    !non_literal(sub_pattern[i + 1]))) {
                            literal += gbnf_escape_ctl(std::string(1, sub_pattern[i]));
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (literal.empty()) {   // nothing was consumed, ex. a stray ']' or '}'
                        throw unsupported_pattern(std::string("unsupported character: ") + c);
                    }
                    seq.emplace_back(literal, true);
                }
            }
            return join_seq();
        };

        auto rule = to_rule(transform());
        if (paren_depth != 0) throw invalid_pattern("unbalanced parentheses");

        return add_rule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
    }

    // A rule that matches a JSON string that is none of the provided strings.
    std::string not_strings(const std::vector<std::string>& strings) {
        Trie trie(strings);

        std::string char_rule = prim("char");
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(size_t)> walk = [&](size_t idx) {
            const auto& node = trie.nodes[idx];
            std::string rejects;
            auto first = true;
            for (const auto& kv : node.children) {
                std::string c = gbnf_escape_ctl(cpt_to_utf8(kv.first));
                rejects += c;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << c << "]";
                if (!trie.nodes[kv.second].children.empty()) {
                    out << " (";
                    walk(kv.second);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) out << " | [^\"" << rejects << "] " << char_rule << "*";
        };
        walk(0);

        out << " )";
        if (trie.nodes[0].pattern < 0) out << "?";
        out << " [\"]";
        return out.str();
    }

    std::string resolve_ref(const SchemaNode& schema) {
        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        std::string ref_name = "ref" + sanitize(ref_fragment);
        if (rules.find(ref_name) == rules.end() && refs_being_resolved.find(schema.ref) == refs_being_resolved.end()) {
            if (!schema.target) {
                errors.push_back("Unresolved $ref " + schema.ref);
                return "";
            }
            refs_being_resolved.insert(schema.ref);
            ref_name = visit(*schema.target, ref_name);
            refs_being_resolved.erase(schema.ref);
        }
        return ref_name;
    }

    std::string build_object_rule(const std::vector<std::pair<std::string, const SchemaNode*>>& properties,
                                  const std::unordered_set<std::string>& required, const std::string& name,
                                  const SchemaNode* additional_properties) {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto& kv : properties) {
            const auto& prop_name = kv.first;
            const auto& prop_schema = kv.second;

            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = add_rule(name + (name.empty() ? "" : "-") + prop_name + "-kv",
                                                     gbnf_format_literal(JsonValue::quote(prop_name)) + " space \":\" space " + prop_rule_name);
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule = additional_properties->kind != SchemaNode::Any
                                         ? visit(*additional_properties, sub_name + "-value")
                                         : prim("value");

            auto key_rule = prop_names.empty() ? prim("string") : add_rule(sub_name + "-k", not_strings(prop_names));
            std::string kv_rule = add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) return "\"{\" space \"}\"";

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) rule += " \",\" space ";
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) rule += " \",\" space ( ";

            std::function<std::string(const std::vector<std::string>&, bool)> get_recursive_refs =
                [&](const std::vector<std::string>& ks, bool first_is_optional) {
                    std::string res;
                    if (ks.empty()) return res;
                    const std::string& k = ks[0];
                    std::string kv_rule_name = prop_kv_rule_names[k];
                    std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                    if (first_is_optional) {
                        res = comma_ref + (k == "*" ? "*" : "?");
                    } else {
                        res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                    }
                    if (ks.size() > 1) {
                        res += " " + add_rule(name + (name.empty() ? "" : "-") + k + "-rest",
                                              get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true));
                    }
                    return res;
                };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) rule += " | ";
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + (std::ptrdiff_t) i, optional_props.end()), false);
            }
            if (!required_props.empty()) rule += " )";
            rule += " )?";
        }

        rule += " space \"}\"";
        return rule;
    }

    static std::string generate_constant_rule(const JsonValue& value) { return gbnf_format_literal(value.dump()); }

    std::string visit_primitive(const std::string& rule_name, const std::string& type) {
        return add_primitive(rule_name == "root" ? "root" : type, primitive_rules().at(type));
    }

    std::string visit_all_of(const SchemaNode& schema, const std::string& name, const std::string& rule_name) {
        std::unordered_set<std::string> required;
        std::vector<std::pair<std::string, const SchemaNode*>> properties;
        std::map<std::string, size_t> enum_values;
        std::function<void(const SchemaNode&, bool)> add_component = [&](const SchemaNode& comp, bool is_required) {
            if (comp.kind == SchemaNode::Ref) {
                if (const auto* target = comp.target) add_component(*target, is_required);
            } else if (comp.kind == SchemaNode::Object) {
                for (const auto& prop : comp.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (is_required) required.insert(prop.name);
                }
            } else if (comp.kind == SchemaNode::Enum) {
                for (const auto& v : comp.values) enum_values[generate_constant_rule(v)] += 1;
            }
        };
        for (const auto& child : schema.children) {
            if (child->kind == SchemaNode::AnyOf) {
                for (const auto& alt : child->children) add_component(*alt, false);
            } else {
                add_component(*child, true);
            }
        }
        if (!enum_values.empty()) {
            std::vector<std::string> enum_intersection;
            for (const auto& p : enum_values) {
                if (p.second == schema.children.size()) enum_intersection.push_back(p.first);
            }
            if (!enum_intersection.empty()) {
                return add_rule(rule_name, "(" + join(enum_intersection, " | ") + ")");
            }
        }
        return add_rule(rule_name, build_object_rule(properties, required, name, nullptr));
    }

    std::string visit(const SchemaNode& schema, const std::string& name) {
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name = name + (name.empty() ? "" : "-");

        switch (schema.kind) {
            case SchemaNode::Ref:
                return add_rule(rule_name, resolve_ref(schema));
            case SchemaNode::AnyOf:
                return add_rule(rule_name, generate_union_rule(name, schema.children));
            case SchemaNode::AllOf:
                return visit_all_of(schema, name, rule_name);
            case SchemaNode::Const:
                return add_rule(rule_name, generate_constant_rule(schema.values.at(0)));
            case SchemaNode::Enum: {
                std::vector<std::string> enum_values;
                for (const auto& v : schema.values) enum_values.push_back(generate_constant_rule(v));
                return add_rule(rule_name, "(" + join(enum_values, " | ") + ")");
            }
            case SchemaNode::Object: {
                if (schema.properties.empty() && schema.additional_properties &&
                    schema.additional_properties->kind == SchemaNode::Any) {
                    return add_rule(rule_name, prim("object"));
                }
                std::vector<std::pair<std::string, const SchemaNode*>> properties;
                std::unordered_set<std::string> required;
                for (const auto& prop : schema.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (prop.required) required.insert(prop.name);
                }
                return add_rule(rule_name, build_object_rule(properties, required, name, schema.additional_properties.get()));
            }
            case SchemaNode::Tuple: {
                const auto& items = schema.children;
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) rule += " \",\" space ";
                    rule += visit(*items[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return add_rule(rule_name, rule);
            }
            case SchemaNode::Array: {
                if (schema.items->kind == SchemaNode::Any && schema.min_items == 0 && schema.max_items < 0) {
                    return visit_primitive(rule_name, "array");
                }
                std::string item_rule_name = visit(*schema.items, sub_name + "item");
                int max_items = schema.max_items < 0 ? (std::numeric_limits<int>::max)() : schema.max_items;
                return add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, schema.min_items, max_items, "\",\" space") + " space \"]\"");
            }
            case SchemaNode::String: {
                if (!schema.pattern.empty()) return visit_pattern(schema.pattern, rule_name);
                if (schema.format == SchemaNode::FormatUuid) return visit_primitive(rule_name, "uuid");
                if (schema.format != SchemaNode::FormatNone) {
                    std::string prim_name = std::string(schema.format == SchemaNode::FormatDate   ? "date"
                                                        : schema.format == SchemaNode::FormatTime ? "time"
                                                                                                  : "date-time") + "-string";
                    return add_rule(rule_name, add_primitive(prim_name, string_format_rules().at(prim_name)));
                }
                if (schema.min_length > 0 || schema.max_length >= 0) {
                    std::string char_rule = prim("char");
                    int max_len = schema.max_length < 0 ? (std::numeric_limits<int>::max)() : schema.max_length;
                    return add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, schema.min_length, max_len) + " \"\\\"\"");
                }
                return visit_primitive(rule_name, "string");
            }
            case SchemaNode::Integer: {
                if (schema.minimum == (std::numeric_limits<int64_t>::min)() && schema.maximum == (std::numeric_limits<int64_t>::max)()) {
                    return visit_primitive(rule_name, "integer");
                }
                std::stringstream out;
                out << "(";
                build_min_max_int(schema.minimum, schema.maximum, out);
                out << ")";
                return add_rule(rule_name, out.str());
            }
            case SchemaNode::Number:
                return visit_primitive(rule_name, "number");
            case SchemaNode::Boolean:
                return visit_primitive(rule_name, "boolean");
            case SchemaNode::Null:
                return visit_primitive(rule_name, "null");
            case SchemaNode::Any:
                return add_rule(rule_name, prim("value"));
        }
        return "";
    }
};

SchemaConverter::SchemaConverter(bool dotall) : impl_(std::make_shared<Impl>(dotall)) {}

std::string SchemaConverter::add_rule(const std::string& name, const std::string& rule) {
    return impl_->add_rule(name, rule);
}

std::string SchemaConverter::add_schema(const std::string& name, const SchemaNode& schema) {
    return impl_->visit(schema, name == "root" ? "" : name);
}

void SchemaConverter::check_errors() {
    if (!impl_->errors.empty()) {
        throw std::invalid_argument("JSON schema conversion failed:\n" + join(impl_->errors, "\n"));
    }
}

std::string SchemaConverter::format_grammar() const {
    std::stringstream ss;
    for (const auto& kv : impl_->rules) ss << kv.first << " ::= " << kv.second << '\n';
    return ss.str();
}

std::string schema_document_to_gbnf(const SchemaDocument& doc) {
    SchemaConverter converter(false);
    converter.add_schema("", *doc.root);
    converter.check_errors();
    return converter.format_grammar();
}

}  // namespace basalt::program::grammar
