// src/program/grammar/schema_convert.cpp - SchemaNode tree -> GBNF. See schema_internal.hpp.
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
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
    if (min_items < 0 || max_items < 0 || min_items > max_items) {
        throw std::runtime_error("invalid repetition bounds");
    }
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

uint64_t neg_magnitude(int64_t v) {
    return (uint64_t) (-(v + 1)) + 1u;
}

void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream& out, int decimals_left = 16) {
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

    auto build_bounded_uint = [&](uint64_t lo, uint64_t hi) {
        auto min_s = std::to_string(lo);
        auto max_s = std::to_string(hi);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, repeat("9", digits));
            min_s = "1" + repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
    };

    std::function<void(std::string_view, int, bool)> build_min_digits =
        [&](std::string_view min_s, int rem_decimals, bool top_level) {
            const size_t len = min_s.length();
            const int less_decimals = (std::max)(rem_decimals - 1, 1);
            bool all_zero = true;
            for (char ch : min_s) {
                if (ch != '0') { all_zero = false; break; }
            }
            if (all_zero) {
                if (top_level) {
                    out << "[0] | [1-9] ";
                    more_digits(0, less_decimals);
                } else {
                    more_digits((int) len, rem_decimals);
                }
                return;
            }
            const char c = min_s[0];
            const char range_start = top_level ? '1' : '0';
            if (len == 1) {
                if (c > range_start) {
                    digit_range(range_start, (char) (c - 1));
                    out << " ";
                    more_digits(1, less_decimals);
                    out << " | ";
                }
                digit_range(c, '9');
                out << " ";
                more_digits(0, less_decimals);
                return;
            }
            if (c > range_start) {
                digit_range(range_start, (char) (c - 1));
                out << " ";
                more_digits((int) len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_digits(min_s.substr(1), less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range((char) (c + 1), '9');
                out << " ";
                more_digits((int) len - 1, less_decimals);
            }
        };

    if (has_min && has_max) {
        if (min_value > max_value) throw std::runtime_error("minimum exceeds maximum");
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_bounded_uint(neg_magnitude(max_value), neg_magnitude(min_value));
            out << ")";
            return;
        }
        if (min_value < 0) {
            out << "\"-\" (";
            build_bounded_uint(0, neg_magnitude(min_value));
            out << ") | ";
            build_bounded_uint(0, (uint64_t) max_value);
            return;
        }
        build_bounded_uint((uint64_t) min_value, (uint64_t) max_value);
        return;
    }

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_bounded_uint(0, neg_magnitude(min_value));
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else {
            const std::string ms = std::to_string((uint64_t) min_value);
            const int dl = (int) ms.length() >= decimals_left ? (int) ms.length() + 1 : decimals_left;
            build_min_digits(ms, dl, /* top_level= */ true);
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            out << "\"-\" [1-9] ";
            more_digits(0, (std::max)(decimals_left - 1, 1));
            out << " | ";
            build_bounded_uint(0, (uint64_t) max_value);
        } else {
            const std::string ms = std::to_string(neg_magnitude(max_value));
            const int dl = (int) ms.length() >= decimals_left ? (int) ms.length() + 1 : decimals_left;
            out << "\"-\" (";
            build_min_digits(ms, dl, /* top_level= */ false);
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
    std::unordered_map<const SchemaNode*, std::string> ref_rule_names;
    std::unordered_set<std::string> reserved_ref_names;
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

            bool last_quantified = false;
            auto parse_nonneg_int = [](const std::string& s, int& out) -> bool {
                if (s.empty()) return false;
                for (char ch : s) {
                    if (ch < '0' || ch > '9') return false;
                }
                try {
                    size_t pos = 0;
                    int v = std::stoi(s, &pos);
                    if (pos != s.size() || v < 0) return false;
                    out = v;
                    return true;
                } catch (const std::logic_error&) {
                    return false;
                }
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    last_quantified = false;
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
                    last_quantified = false;
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
                    last_quantified = false;
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    last_quantified = false;
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty() || (!seq.back().second && seq.back().first == "|") || last_quantified) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                    // Full-string language is identical for lazy quantifiers (*?, +?, ??); consume the modifier.
                    if (i < length && sub_pattern[i] == '?') i++;
                    last_quantified = true;
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
                    if (nums.size() == 1) {
                        if (!parse_nonneg_int(nums[0], min_times)) {
                            throw unsupported_pattern("invalid number in curly brackets");
                        }
                        max_times = min_times;
                    } else {
                        if (!parse_nonneg_int(nums[0], min_times)) {
                            throw unsupported_pattern("invalid number in curly brackets");
                        }
                        if (!nums[1].empty() && !parse_nonneg_int(nums[1], max_times)) {
                            throw unsupported_pattern("invalid number in curly brackets");
                        }
                    }
                    if (min_times > max_times) {
                        throw invalid_pattern("repetition minimum exceeds maximum");
                    }
                    if (seq.empty() || (!seq.back().second && seq.back().first == "|") || last_quantified) {
                        throw invalid_pattern("nothing to repeat");
                    }
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
                    if (i < length && sub_pattern[i] == '?') i++;
                    last_quantified = true;
                } else {
                    // Emit one complete atom (escape, quote, or UTF-8 code point) per entry so any following
                    // quantifier binds only to it; join_seq() merges consecutive unquantified literals.
                    std::string atom;
                    if (c == '\\') {
                        if (i == length - 1) throw invalid_pattern("trailing backslash");
                        char next = sub_pattern[i + 1];
                        if (escaped_in_regexps_but_not_in_literals(next)) {
                            atom += next;
                            i += 2;
                        } else {
                            auto escape_length = gbnf_escape_length(sub_pattern, i);
                            if (escape_length == 0) {
                                throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                            }
                            atom = sub_pattern.substr(i, escape_length);
                            i += escape_length;
                        }
                    } else if (c == '"') {
                        atom = "\\\"";
                        i++;
                    } else if (non_literal(c)) {
                        throw unsupported_pattern(std::string("unsupported character: ") + c);
                    } else {
                        unsigned char uc = (unsigned char) c;
                        size_t clen = uc < 0x80 ? 1 : (uc >> 5) == 6 ? 2 : (uc >> 4) == 14 ? 3 : (uc >> 3) == 30 ? 4 : 0;
                        if (clen == 0 || i + clen > length) throw invalid_pattern("invalid UTF-8 in pattern");
                        uint32_t cp = clen == 1 ? uc : uc & (0xFFu >> (clen + 1));
                        for (size_t k = 1; k < clen; ++k) {
                            unsigned char d = (unsigned char) sub_pattern[i + k];
                            if ((d & 0xC0) != 0x80) throw invalid_pattern("invalid UTF-8 in pattern");
                            cp = (cp << 6) | (d & 0x3F);
                        }
                        static constexpr uint32_t kMinCp[5] = {0, 0, 0x80, 0x800, 0x10000};
                        if (cp < kMinCp[clen] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
                            throw invalid_pattern("invalid UTF-8 in pattern");
                        }
                        atom = clen == 1 ? gbnf_escape_ctl(std::string(1, c)) : sub_pattern.substr(i, clen);
                        i += clen;
                    }
                    seq.emplace_back(std::move(atom), true);
                    last_quantified = false;
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
        std::vector<std::string> encoded;
        encoded.reserve(strings.size());
        for (const auto& s : strings) {
            std::string q = JsonValue::quote(s);
            encoded.push_back(q.substr(1, q.size() - 2));
        }
        Trie trie(encoded);

        auto class_char = [](uint32_t cp) -> std::string {
            switch (cp) {
                case '\\': return "\\\\";
                case '"': return "\\\"";
                case '[': return "\\[";
                case ']': return "\\]";
                case '-': return "\\-";
                case '^': return "\\x5e";
                default: return gbnf_escape_ctl(cpt_to_utf8(cp));
            }
        };

        std::string char_rule = prim("char");
        std::ostringstream out;
        if (trie.nodes[0].children.empty()) {
            out << "[\"] " << char_rule << (trie.nodes[0].pattern >= 0 ? "+" : "*") << " [\"]";
            return out.str();
        }

        out << "[\"] ( ";
        std::function<void(size_t, bool, int)> walk = [&](size_t idx, bool after_bs, int hex_rem) {
            const auto& node = trie.nodes[idx];
            std::string rejects;
            bool has_backslash_child = false;
            bool first = true;
            for (const auto& kv : node.children) {
                const uint32_t cp = kv.first;
                const size_t child = kv.second;
                const std::string c = class_char(cp);
                if (!after_bs && hex_rem == 0) {
                    if (cp == '\\') has_backslash_child = true;
                    else if (cp != '"' && cp >= 0x20 && cp != 0x7F) rejects += c;
                }
                if (first) first = false;
                else out << " | ";
                out << "[" << c << "]";
                if (!after_bs && hex_rem == 0 && cp == '\\') {
                    out << " (";
                    walk(child, true, 0);
                    out << ")";
                } else if (after_bs && cp == 'u') {
                    out << " (";
                    walk(child, false, 4);
                    out << ")";
                } else if (hex_rem > 1) {
                    out << " (";
                    walk(child, false, hex_rem - 1);
                    out << ")";
                } else if (!trie.nodes[child].children.empty()) {
                    out << " (";
                    walk(child, false, 0);
                    out << ")";
                    if (trie.nodes[child].pattern < 0) out << "?";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (after_bs) {
                std::string rem_esc;
                for (char ec : std::string_view("\"\\bfnrt")) {
                    if (node.children.find((uint32_t) (unsigned char) ec) == node.children.end()) {
                        rem_esc += class_char((uint32_t) (unsigned char) ec);
                    }
                }
                if (!rem_esc.empty()) out << " | [" << rem_esc << "] " << char_rule << "*";
                if (node.children.find((uint32_t) 'u') == node.children.end()) {
                    out << " | \"u\" [0-9a-fA-F]{4} " << char_rule << "*";
                }
            } else if (hex_rem > 0) {
                std::string rem_hex;
                for (char hc : std::string_view("0123456789abcdefABCDEF")) {
                    if (node.children.find((uint32_t) (unsigned char) hc) == node.children.end()) {
                        rem_hex += hc;
                    }
                }
                if (!rem_hex.empty()) {
                    out << " | [" << rem_hex << "]";
                    if (hex_rem > 1) out << " [0-9a-fA-F]{" << (hex_rem - 1) << "}";
                    out << " " << char_rule << "*";
                }
            } else if (!node.children.empty()) {
                if (has_backslash_child) {
                    out << " | [^\"\\\\\\x7F\\x00-\\x1F" << rejects << "] " << char_rule << "*";
                } else {
                    out << " | ([^\"\\\\\\x7F\\x00-\\x1F" << rejects << "] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})) "
                        << char_rule << "*";
                }
            }
        };
        walk(0, false, 0);

        out << " )";
        if (trie.nodes[0].pattern < 0) out << "?";
        out << " [\"]";
        return out.str();
    }

    std::string resolve_ref(const SchemaNode& schema) {
        if (!schema.target) {
            errors.push_back("Unresolved $ref " + schema.ref);
            return "";
        }
        auto existing = ref_rule_names.find(schema.target);
        if (existing != ref_rule_names.end()) return existing->second;

        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        std::string base = "ref" + sanitize(ref_fragment);
        std::string chosen = base;
        if (rules.find(chosen) != rules.end() || reserved_ref_names.count(chosen)) {
            int suffix = 0;
            while (rules.find(base + std::to_string(suffix)) != rules.end() ||
                   reserved_ref_names.count(base + std::to_string(suffix))) {
                ++suffix;
            }
            chosen = base + std::to_string(suffix);
        }
        ref_rule_names[schema.target] = chosen;
        reserved_ref_names.insert(chosen);
        std::string actual = visit(*schema.target, chosen);
        ref_rule_names[schema.target] = actual;
        if (actual != chosen) reserved_ref_names.erase(chosen);
        return actual;
    }

    std::string build_object_rule(const std::vector<std::pair<std::string, const SchemaNode*>>& properties,
                                  const std::unordered_set<std::string>& required, const std::string& name,
                                  const SchemaNode* additional_properties) {
        struct OptProp {
            std::string name;
            std::string kv_rule;
            bool is_additional;
        };
        std::vector<std::string> required_kv_rules;
        std::vector<OptProp> optional_props;
        std::vector<std::string> prop_names;
        for (const auto& kv : properties) {
            const auto& prop_name = kv.first;
            const auto& prop_schema = kv.second;

            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            std::string kv_rule_name = add_rule(name + (name.empty() ? "" : "-") + prop_name + "-kv",
                                                gbnf_format_literal(JsonValue::quote(prop_name)) + " space \":\" space " + prop_rule_name);
            if (required.find(prop_name) != required.end()) {
                required_kv_rules.push_back(kv_rule_name);
            } else {
                optional_props.push_back({prop_name, kv_rule_name, false});
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
            optional_props.push_back({"*", kv_rule, true});
        }

        if (required_kv_rules.empty() && optional_props.empty()) return "\"{\" space \"}\"";

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_kv_rules.size(); i++) {
            if (i > 0) rule += " \",\" space ";
            rule += required_kv_rules[i];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_kv_rules.empty()) rule += " \",\" space ( ";

            std::function<std::string(size_t, bool)> get_recursive_refs = [&](size_t idx, bool first_is_optional) {
                std::string res;
                if (idx >= optional_props.size()) return res;
                const OptProp& p = optional_props[idx];
                std::string comma_ref = "( \",\" space " + p.kv_rule + " )";
                if (first_is_optional) {
                    res = comma_ref + (p.is_additional ? "*" : "?");
                } else {
                    res = p.kv_rule + (p.is_additional ? " " + comma_ref + "*" : "");
                }
                if (idx + 1 < optional_props.size()) {
                    res += " " + add_rule(name + (name.empty() ? "" : "-") + p.name + "-rest",
                                          get_recursive_refs(idx + 1, true));
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) rule += " | ";
                rule += get_recursive_refs(i, false);
            }
            if (!required_kv_rules.empty()) rule += " )";
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
        if (schema.children.size() == 1) {
            std::unordered_set<const SchemaNode*> seen{&schema};
            const SchemaNode* cur = schema.children[0].get();
            while (cur) {
                if (!seen.insert(cur).second) {
                    errors.push_back("Cyclic $ref in allOf");
                    return "";
                }
                if (cur->kind == SchemaNode::Ref) {
                    if (!cur->target) {
                        errors.push_back("Unresolved $ref " + cur->ref);
                        return "";
                    }
                    cur = cur->target;
                } else if (cur->kind == SchemaNode::AllOf && cur->children.size() == 1) {
                    cur = cur->children[0].get();
                } else {
                    break;
                }
            }
            if (cur && cur->kind == SchemaNode::Enum) {
                std::vector<std::string> vals;
                std::set<std::string> uniq;
                for (const auto& v : cur->values) uniq.insert(generate_constant_rule(v));
                for (const auto& s : uniq) vals.push_back(s);
                return add_rule(rule_name, "(" + join(vals, " | ") + ")");
            }
            return visit(*schema.children[0], name);
        }

        std::unordered_set<const SchemaNode*> active{&schema};
        std::unordered_set<std::string> required;
        std::vector<std::pair<std::string, const SchemaNode*>> properties;
        std::unordered_map<std::string, size_t> prop_idx;
        std::map<std::string, size_t> enum_values;
        size_t enum_branches = 0;
        bool saw_object = false;
        bool saw_cycle = false;
        std::vector<const SchemaNode*> other_branches;

        std::function<void(const SchemaNode&, bool)> add_component = [&](const SchemaNode& comp, bool is_required) {
            if (!active.insert(&comp).second) {
                errors.push_back("Cyclic $ref in allOf");
                saw_cycle = true;
                return;
            }
            struct EraseGuard {
                std::unordered_set<const SchemaNode*>& s;
                const SchemaNode* p;
                ~EraseGuard() { s.erase(p); }
            } guard{active, &comp};

            if (comp.kind == SchemaNode::Ref) {
                if (!comp.target) {
                    errors.push_back("Unresolved $ref " + comp.ref);
                    return;
                }
                add_component(*comp.target, is_required);
            } else if (comp.kind == SchemaNode::AllOf) {
                for (const auto& c : comp.children) {
                    if (c->kind == SchemaNode::AnyOf) {
                        for (const auto& alt : c->children) add_component(*alt, false);
                    } else {
                        add_component(*c, is_required);
                    }
                }
            } else if (comp.kind == SchemaNode::Object) {
                saw_object = true;
                for (const auto& prop : comp.properties) {
                    auto it = prop_idx.find(prop.name);
                    if (it == prop_idx.end()) {
                        prop_idx[prop.name] = properties.size();
                        properties.emplace_back(prop.name, prop.schema.get());
                    } else if (properties[it->second].second->kind == SchemaNode::Any &&
                               prop.schema->kind != SchemaNode::Any) {
                        properties[it->second].second = prop.schema.get();
                    }
                    if (is_required && prop.required) required.insert(prop.name);
                }
            } else if (comp.kind == SchemaNode::Enum) {
                ++enum_branches;
                std::unordered_set<std::string> branch_vals;
                for (const auto& v : comp.values) {
                    std::string r = generate_constant_rule(v);
                    if (branch_vals.insert(r).second) enum_values[r] += 1;
                }
            } else if (comp.kind != SchemaNode::Any) {
                other_branches.push_back(&comp);
            }
        };
        for (const auto& child : schema.children) {
            if (child->kind == SchemaNode::AnyOf) {
                for (const auto& alt : child->children) add_component(*alt, false);
            } else {
                add_component(*child, true);
            }
        }
        if (saw_cycle) return "";
        if (enum_branches > 0) {
            std::vector<std::string> enum_intersection;
            for (const auto& p : enum_values) {
                if (p.second == enum_branches) enum_intersection.push_back(p.first);
            }
            if (!enum_intersection.empty()) {
                return add_rule(rule_name, "(" + join(enum_intersection, " | ") + ")");
            }
            if (!saw_object) {
                errors.push_back("allOf enum branches have empty intersection");
                return "";
            }
        }
        if (saw_object) {
            return add_rule(rule_name, build_object_rule(properties, required, name, nullptr));
        }
        if (other_branches.empty()) {
            return add_rule(rule_name, prim("value"));
        }
        if (other_branches.size() == 1) {
            return visit(*other_branches[0], name);
        }
        for (const SchemaNode* b : other_branches) {
            if (b->kind == SchemaNode::Const) return visit(*b, name);
        }
        const SchemaNode::Kind first_kind = other_branches[0]->kind;
        bool same_kind = true;
        for (const SchemaNode* b : other_branches) {
            if (b->kind != first_kind) { same_kind = false; break; }
        }
        if (same_kind) {
            if (first_kind == SchemaNode::String) {
                SchemaNode merged;
                merged.kind = SchemaNode::String;
                for (const SchemaNode* b : other_branches) {
                    if (b->min_length > merged.min_length) merged.min_length = b->min_length;
                    if (b->max_length >= 0 && (merged.max_length < 0 || b->max_length < merged.max_length)) {
                        merged.max_length = b->max_length;
                    }
                    if (!b->pattern.empty() && merged.pattern.empty()) merged.pattern = b->pattern;
                    if (b->format != SchemaNode::FormatNone && merged.format == SchemaNode::FormatNone) {
                        merged.format = b->format;
                    }
                }
                return visit(merged, name);
            }
            if (first_kind == SchemaNode::Integer) {
                SchemaNode merged;
                merged.kind = SchemaNode::Integer;
                for (const SchemaNode* b : other_branches) {
                    if (b->minimum > merged.minimum) merged.minimum = b->minimum;
                    if (b->maximum < merged.maximum) merged.maximum = b->maximum;
                }
                return visit(merged, name);
            }
            if (first_kind == SchemaNode::Array) {
                int min_it = 0;
                int max_it = -1;
                const SchemaNode* item_schema = other_branches[0]->items.get();
                for (const SchemaNode* b : other_branches) {
                    if (b->min_items > min_it) min_it = b->min_items;
                    if (b->max_items >= 0 && (max_it < 0 || b->max_items < max_it)) max_it = b->max_items;
                    if (b->items && b->items->kind != SchemaNode::Any &&
                        (!item_schema || item_schema->kind == SchemaNode::Any)) {
                        item_schema = b->items.get();
                    }
                }
                if (max_it >= 0 && min_it > max_it) {
                    errors.push_back("minItems exceeds maxItems");
                    return "";
                }
                if ((!item_schema || item_schema->kind == SchemaNode::Any) && min_it == 0 && max_it < 0) {
                    return visit_primitive(rule_name, "array");
                }
                const std::string sub_name = name + (name.empty() ? "" : "-");
                std::string item_rule_name = item_schema ? visit(*item_schema, sub_name + "item") : prim("value");
                int eff_max = max_it < 0 ? (std::numeric_limits<int>::max)() : max_it;
                return add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, min_it, eff_max, "\",\" space") + " space \"]\"");
            }
            if (first_kind == SchemaNode::Number || first_kind == SchemaNode::Boolean || first_kind == SchemaNode::Null) {
                return visit(*other_branches[0], name);
            }
        }
        errors.push_back("Unsupported non-object allOf combination");
        return "";
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
                if (schema.max_items >= 0 && schema.min_items > schema.max_items) {
                    errors.push_back("minItems exceeds maxItems");
                    return "";
                }
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
                    if (schema.max_length >= 0 && schema.min_length > schema.max_length) {
                        errors.push_back("minLength exceeds maxLength");
                        return "";
                    }
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
                if (schema.minimum != (std::numeric_limits<int64_t>::min)() &&
                    schema.maximum != (std::numeric_limits<int64_t>::max)() &&
                    schema.minimum > schema.maximum) {
                    errors.push_back("minimum exceeds maximum");
                    return "";
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
    struct ScopeClear {
        Impl& impl;
        ScopeClear(Impl& i) : impl(i) {
            impl.ref_rule_names.clear();
            impl.reserved_ref_names.clear();
        }
        ~ScopeClear() {
            impl.ref_rule_names.clear();
            impl.reserved_ref_names.clear();
        }
    } scope(*impl_);
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
