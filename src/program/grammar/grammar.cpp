// src/program/grammar/grammar.cpp - see grammar.hpp. The GBNF parser, the pushdown stacks (interned, with memoized
// closure and character steps) and the vocabulary trie walk.
#include "basalt/program/grammar/grammar.hpp"

#include <algorithm>
#include <cstdio>
#include <atomic>
#include <deque>
#include <map>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace basalt::program::grammar {

namespace {

constexpr uint32_t kMaxRepetition = 2000;

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_word(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || is_digit(c); }
bool end_of_seq(const Element& e) { return e.type == ElemType::End || e.type == ElemType::Alt; }

using Rule = std::vector<Element>;

// --------------------------------------------------------------------------------------------- parser

// A rule that can reach itself as its leftmost symbol would never leave the closure.
bool left_recursive(const std::vector<Rule>& rules, size_t idx, std::vector<bool>& visited,
                    std::vector<bool>& in_progress, std::vector<bool>& may_be_empty) {
    if (in_progress[idx]) return true;
    in_progress[idx] = true;
    const Rule& rule = rules[idx];
    bool at_start = true;
    for (const Element& e : rule) {
        if (end_of_seq(e)) {
            if (at_start) {
                may_be_empty[idx] = true;
                break;
            }
            at_start = true;
        } else {
            at_start = false;
        }
    }
    bool descend = true;
    for (const Element& e : rule) {
        if (e.type == ElemType::RuleRef && descend) {
            if (left_recursive(rules, e.value, visited, in_progress, may_be_empty)) return true;
            if (!may_be_empty[e.value]) descend = false;
        } else if (end_of_seq(e)) {
            descend = true;
        } else {
            descend = false;
        }
    }
    in_progress[idx] = false;
    visited[idx] = true;
    return false;
}

// Throws if a rule refers to a rule that is not defined or if any rule is left-recursive.
void check_rules(const std::vector<Rule>& rules, const std::function<std::string(uint32_t)>& name_of) {
    for (const Rule& r : rules) {
        if (r.empty() || r.back().type != ElemType::End) throw std::runtime_error("undefined rule");
    }
    for (const Rule& r : rules) {
        for (const Element& e : r) {
            if (e.type == ElemType::RuleRef && (e.value >= rules.size() || rules[e.value].empty()))
                throw std::runtime_error("undefined rule identifier '" + name_of(e.value) + "'");
        }
    }
    std::vector<bool> visited(rules.size()), in_progress(rules.size()), may_be_empty(rules.size());
    for (size_t i = 0; i < rules.size(); ++i) {
        if (!visited[i] && left_recursive(rules, i, visited, in_progress, may_be_empty))
            throw std::runtime_error("left recursion in rule '" + name_of((uint32_t) i) + "'");
    }
}

}  // namespace

class GrammarParser {
public:
    std::vector<Rule> rules;
    std::map<std::string, uint32_t> ids;
    ParseOptions opt;

    void run(const std::string& text) {
        src_ = text.c_str();
        const char* pos = space(src_, true);
        while (*pos) pos = parse_rule(pos);
        if (!ids.count(opt.root)) throw std::runtime_error("no " + opt.root + " rule");
        check_rules(rules, [this](uint32_t id) { return name_of(id); });
    }

private:
    const char* src_ = nullptr;

    std::string name_of(uint32_t id) const {
        for (const auto& kv : ids) if (kv.second == id) return kv.first;
        return "?";
    }
    static std::runtime_error fail(const std::string& what, const char* at) {
        std::string ctx(at, (std::min<size_t>)(std::char_traits<char>::length(at), 24));
        return std::runtime_error(what + " at '" + ctx + "'");
    }

    uint32_t symbol(const std::string& name) {
        uint32_t next = (uint32_t) ids.size();
        return ids.emplace(name, next).first->second;
    }
    uint32_t fresh(const std::string& base) {
        uint32_t next = (uint32_t) ids.size();
        ids[base + "_" + std::to_string(next)] = next;
        return next;
    }
    void set_rule(uint32_t id, const Rule& r) {
        if (rules.size() <= id) rules.resize(id + 1);
        rules[id] = r;
    }

    static const char* space(const char* p, bool newline_ok) {
        while (*p == ' ' || *p == '\t' || *p == '#' || (newline_ok && (*p == '\r' || *p == '\n'))) {
            if (*p == '#') {
                while (*p && *p != '\r' && *p != '\n') ++p;
            } else {
                ++p;
            }
        }
        return p;
    }
    static const char* name(const char* p) {
        const char* s = p;
        while (is_word(*p)) ++p;
        if (p == s) throw fail("expecting name", s);
        return p;
    }
    static const char* integer(const char* p) {
        const char* s = p;
        while (is_digit(*p)) ++p;
        if (p == s) throw fail("expecting integer", s);
        return p;
    }
    static uint64_t to_u64(const char* b, const char* e) {
        uint64_t v = 0;
        for (; b < e; ++b) {
            v = v * 10 + (uint64_t) (*b - '0');
            if (v > (1ull << 40)) throw fail("integer too large", b);
        }
        return v;
    }
    static std::pair<uint32_t, const char*> hex(const char* p, int n) {
        uint32_t v = 0;
        const char* s = p;
        for (int i = 0; i < n; ++i, ++p) {
            char c = *p;
            v <<= 4;
            if (c >= 'a' && c <= 'f') v += (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v += (uint32_t) (c - 'A' + 10);
            else if (is_digit(c)) v += (uint32_t) (c - '0');
            else throw fail("expecting hex digits", s);
        }
        return {v, p};
    }
    // One (possibly escaped) character; plain text is read as UTF-8.
    static std::pair<uint32_t, const char*> character(const char* p) {
        if (*p == '\\') {
            switch (p[1]) {
                case 'x': return hex(p + 2, 2);
                case 'u': return hex(p + 2, 4);
                case 'U': return hex(p + 2, 8);
                case 't': return {'\t', p + 2};
                case 'r': return {'\r', p + 2};
                case 'n': return {'\n', p + 2};
                case '\\': case '"': case '[': case ']': case '-': return {(uint32_t) (unsigned char) p[1], p + 2};
                default: throw fail("unknown escape", p);
            }
        }
        if (!*p) throw std::runtime_error("unexpected end of input");
        unsigned char c = (unsigned char) *p;
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (len == 0) throw fail("invalid UTF-8", p);
        uint32_t v = len == 1 ? c : c & (0xFFu >> (len + 1));
        for (int i = 1; i < len; ++i) {
            if (((unsigned char) p[i] & 0xC0) != 0x80) throw fail("invalid UTF-8", p);
            v = (v << 6) | ((unsigned char) p[i] & 0x3F);
        }
        return {v, p + len};
    }

    // <[id]> or <name>; a name is looked up through the resolver.
    std::pair<uint32_t, const char*> token(const char* src) {
        const char* p = src;
        if (*p != '<') throw fail("expecting '<'", p);
        ++p;
        if (*p == '[') {
            ++p;
            const char* ie = integer(p);
            uint64_t id = to_u64(p, ie);
            if (id > UINT32_MAX) throw fail("token id is too big", p);
            p = ie;
            if (*p != ']') throw fail("expecting ']'", p);
            ++p;
            if (*p != '>') throw fail("expecting '>'", p);
            return {(uint32_t) id, p + 1};
        }
        if (!opt.token) throw fail("no vocabulary to look up the token", src);
        while (*p && *p != '>') ++p;
        if (*p != '>') throw fail("expecting '>'", p);
        ++p;
        uint32_t id = 0;
        if (!opt.token(std::string_view(src, (size_t) (p - src)), id))
            throw std::runtime_error("invalid token '" + std::string(src, (size_t) (p - src)) + "'");
        return {id, p};
    }

    const char* parse_rule(const char* p) {
        const char* name_end = name(p);
        std::string nm(p, name_end);
        const char* pos = space(name_end, false);
        uint32_t id = symbol(nm);
        if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) throw fail("expecting ::=", pos);
        pos = space(pos + 3, true);
        pos = alternates(pos, nm, id, false);
        if (*pos == '\r') pos += pos[1] == '\n' ? 2 : 1;
        else if (*pos == '\n') ++pos;
        else if (*pos) throw fail("expecting newline or end", pos);
        return space(pos, true);
    }

    const char* alternates(const char* p, const std::string& rule_name, uint32_t id, bool nested) {
        Rule rule;
        const char* pos = sequence(p, rule_name, rule, nested);
        while (*pos == '|') {
            rule.push_back({ElemType::Alt, 0});
            pos = space(pos + 1, true);
            pos = sequence(pos, rule_name, rule, nested);
        }
        rule.push_back({ElemType::End, 0});
        set_rule(id, rule);
        return pos;
    }

    const char* sequence(const char* p, const std::string& rule_name, Rule& rule, bool nested) {
        size_t last_start = rule.size();
        const char* pos = p;
        uint64_t n_prev = 1;

        // S{m,n} becomes m copies of S followed by a chain of n-m optional S rules; S* and S+ use one recursive rule.
        auto repeat = [&](uint64_t min_t, uint64_t max_t) {
            bool no_max = max_t == UINT64_MAX;
            if (last_start == rule.size()) throw fail("expecting an item before a repetition", pos);
            Rule prev(rule.begin() + (std::ptrdiff_t) last_start, rule.end());
            uint64_t total = 1;
            if (!no_max && max_t > 0) total = max_t;
            else if (min_t > 0) total = min_t;
            if (n_prev * total > kMaxRepetition) throw std::runtime_error("repetition is too large");
            if (min_t == 0) {
                rule.resize(last_start);
            } else {
                for (uint64_t i = 1; i < min_t; ++i) rule.insert(rule.end(), prev.begin(), prev.end());
            }
            uint32_t last_rec = 0;
            uint64_t n_opt = no_max ? 1 : max_t - min_t;
            Rule rec(prev);
            for (uint64_t i = 0; i < n_opt; ++i) {
                rec.resize(prev.size());
                uint32_t rec_id = fresh(rule_name);
                if (i > 0 || no_max) rec.push_back({ElemType::RuleRef, no_max ? rec_id : last_rec});
                rec.push_back({ElemType::Alt, 0});
                rec.push_back({ElemType::End, 0});
                set_rule(rec_id, rec);
                last_rec = rec_id;
            }
            if (n_opt > 0) rule.push_back({ElemType::RuleRef, last_rec});
            n_prev *= total;
        };

        while (*pos) {
            if (*pos == '"') {
                ++pos;
                last_start = rule.size();
                n_prev = 1;
                while (*pos != '"') {
                    if (!*pos) throw std::runtime_error("unexpected end of input");
                    auto c = character(pos);
                    pos = c.second;
                    rule.push_back({ElemType::Char, c.first});
                }
                pos = space(pos + 1, nested);
            } else if (*pos == '[') {
                ++pos;
                ElemType start = ElemType::Char;
                if (*pos == '^') {
                    ++pos;
                    start = ElemType::CharNot;
                }
                last_start = rule.size();
                n_prev = 1;
                while (*pos != ']') {
                    if (!*pos) throw std::runtime_error("unexpected end of input");
                    auto c = character(pos);
                    pos = c.second;
                    rule.push_back({last_start < rule.size() ? ElemType::CharAlt : start, c.first});
                    if (pos[0] == '-' && pos[1] != ']') {
                        if (!pos[1]) throw std::runtime_error("unexpected end of input");
                        auto hi = character(pos + 1);
                        pos = hi.second;
                        rule.push_back({ElemType::CharRngUpper, hi.first});
                    }
                }
                pos = space(pos + 1, nested);
            } else if (*pos == '<' || *pos == '!') {
                ElemType type = ElemType::Token;
                if (*pos == '!') {
                    type = ElemType::TokenNot;
                    ++pos;
                }
                auto t = token(pos);
                last_start = rule.size();
                n_prev = 1;
                rule.push_back({type, t.first});
                pos = space(t.second, nested);
            } else if (is_word(*pos)) {
                const char* e = name(pos);
                uint32_t ref = symbol(std::string(pos, e));
                pos = space(e, nested);
                last_start = rule.size();
                n_prev = 1;
                rule.push_back({ElemType::RuleRef, ref});
            } else if (*pos == '(') {
                pos = space(pos + 1, true);
                size_t before = ids.size();
                uint32_t sub = fresh(rule_name);
                pos = alternates(pos, rule_name, sub, true);
                n_prev = (std::max<uint64_t>)(1, ids.size() - before);
                last_start = rule.size();
                rule.push_back({ElemType::RuleRef, sub});
                if (*pos != ')') throw fail("expecting ')'", pos);
                pos = space(pos + 1, nested);
            } else if (*pos == '.') {
                last_start = rule.size();
                n_prev = 1;
                rule.push_back({ElemType::CharAny, 0});
                pos = space(pos + 1, nested);
            } else if (*pos == '*') {
                pos = space(pos + 1, nested);
                repeat(0, UINT64_MAX);
            } else if (*pos == '+') {
                pos = space(pos + 1, nested);
                repeat(1, UINT64_MAX);
            } else if (*pos == '?') {
                pos = space(pos + 1, nested);
                repeat(0, 1);
            } else if (*pos == '{') {
                pos = space(pos + 1, nested);
                if (!is_digit(*pos)) throw fail("expecting an integer", pos);
                const char* ie = integer(pos);
                uint64_t min_t = to_u64(pos, ie);
                pos = space(ie, nested);
                uint64_t max_t = UINT64_MAX;
                if (*pos == '}') {
                    max_t = min_t;
                    pos = space(pos + 1, nested);
                } else if (*pos == ',') {
                    pos = space(pos + 1, nested);
                    if (is_digit(*pos)) {
                        const char* je = integer(pos);
                        max_t = to_u64(pos, je);
                        pos = space(je, nested);
                    }
                    if (*pos != '}') throw fail("expecting '}'", pos);
                    pos = space(pos + 1, nested);
                } else {
                    throw fail("expecting ','", pos);
                }
                if (min_t > kMaxRepetition) throw std::runtime_error("repetition is too large");
                if (max_t != UINT64_MAX && max_t > kMaxRepetition) max_t = UINT64_MAX;
                repeat(min_t, max_t);
            } else {
                break;
            }
        }
        return pos;
    }
};

std::shared_ptr<const Grammar> Grammar::make(const std::vector<std::vector<Element>>& rules,
                                             std::map<std::string, uint32_t> ids, uint32_t root) {
    auto g = std::make_shared<Grammar>();
    g->root_ = root;
    g->symbols_ = std::move(ids);
    for (const Rule& r : rules) {
        g->rule_start_.push_back((uint32_t) g->elems_.size());
        g->elems_.insert(g->elems_.end(), r.begin(), r.end());
    }
    // Partition [0, 0x10FFFF] into equivalence classes whose code points match the same grammar elements.
    std::vector<uint32_t> bp = {0u, 0x80u, 0x110000u};
    for (size_t i = 0; i < g->elems_.size(); ++i) {
        const ElemType ty = g->elems_[i].type;
        if (ty == ElemType::Char || ty == ElemType::CharNot || ty == ElemType::CharAlt) {
            const uint32_t lo = g->elems_[i].value;
            uint32_t hi = lo;
            if (i + 1 < g->elems_.size() && g->elems_[i + 1].type == ElemType::CharRngUpper) hi = g->elems_[++i].value;
            if (lo <= hi && lo <= 0x10FFFFu) {
                bp.push_back(lo);
                if (hi < 0x10FFFFu) bp.push_back(hi + 1);
            }
        }
    }
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end()), bp.end());
    g->class_rep_.reserve(bp.size() - 1);
    for (size_t k = 0; k + 1 < bp.size(); ++k) {
        const uint32_t lo = bp[k], hi = bp[k + 1];
        g->class_rep_.push_back(lo);
        if (lo < 0x80u) {
            const uint32_t end = (std::min)(hi, 0x80u);
            for (uint32_t c = lo; c < end; ++c) g->ascii_cls_[c] = (uint16_t) k;
        } else if (lo == 0x80u) {
            g->wide_base_ = (uint32_t) k;
        } else {
            g->wide_bp_.push_back(lo);
        }
    }
    g->n_classes_ = (uint32_t) g->class_rep_.size();
    return g;
}

std::shared_ptr<const Grammar> Grammar::parse(const std::string& gbnf, std::string& err) {
    return parse(gbnf, err, ParseOptions{});
}

std::shared_ptr<const Grammar> Grammar::parse(const std::string& gbnf, std::string& err, const ParseOptions& opt) {
    // The parser walks the text as a C string; a raw NUL (a generator that forgot to escape, e.g. a decoded \u0000
    // from a JSON pattern) would otherwise surface as a cryptic "unexpected end of input" at the byte.
    if (gbnf.find('\0') != std::string::npos) {
        err = "grammar: the grammar text contains a raw NUL byte";
        return nullptr;
    }
    GrammarParser p;
    p.opt = opt;
    try {
        p.run(gbnf);
    } catch (const std::exception& e) {
        err = std::string("grammar: ") + e.what();
        return nullptr;
    }
    return make(p.rules, p.ids, p.ids.at(opt.root));
}

std::shared_ptr<const Grammar> Grammar::from_rules(const std::vector<std::vector<Element>>& rules, uint32_t root,
                                                   std::string& err) {
    try {
        if (root >= rules.size()) throw std::runtime_error("no root rule");
        check_rules(rules, [](uint32_t id) { return std::to_string(id); });
    } catch (const std::exception& e) {
        err = std::string("grammar: ") + e.what();
        return nullptr;
    }
    return make(rules, {}, root);
}

// --------------------------------------------------------------------------------------------- stacks

namespace {

// Valid non-surrogate Unicode intervals reachable by completing a partial UTF-8 sequence `(value, n_remain)`.
// Returns 0 (no valid completion: overlong lead, surrogate block, or > U+10FFFF), 1, or 2 disjoint intervals.
int utf8_partial_ranges(uint32_t value, int n_remain, uint32_t out_lo[2], uint32_t out_hi[2]) {
    if (n_remain <= 0 || n_remain > 3 || (n_remain == 1 && value < 2)) return 0;
    uint32_t low = value << (n_remain * 6);
    uint32_t high = low | ((1u << (n_remain * 6)) - 1);
    if (low == 0) {
        if (n_remain == 2) low = 1u << 11;
        else if (n_remain == 3) low = 1u << 16;
    }
    if (high > 0x10FFFFu) high = 0x10FFFFu;
    if (low > high) return 0;
    if (high < 0xD800u || low > 0xDFFFu) {
        out_lo[0] = low;
        out_hi[0] = high;
        return 1;
    }
    if (low >= 0xD800u && high <= 0xDFFFu) return 0;
    int n = 0;
    if (low < 0xD800u) {
        out_lo[n] = low;
        out_hi[n] = 0xD7FFu;
        ++n;
    }
    if (high > 0xDFFFu) {
        out_lo[n] = 0xE000u;
        out_hi[n] = high;
        ++n;
    }
    return n;
}

uint64_t hash_u32_vec(const std::vector<uint32_t>& v) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ ((uint64_t) v.size() * 0xFF51AFD7ED558CCDull);
    for (uint32_t x : v) {
        h ^= (uint64_t) x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ull;
    }
    return h ^ (h >> 31);
}

}  // namespace

// A stack is a chain of interned (element, parent) nodes; the element is the next one to match in its rule
// sequence. Node 0 is the empty stack. Closures, single-stack steps, and stack-set DFA transitions are memoized
// by grammar character equivalence class.
struct GrammarState::Arena {
    struct Node {
        uint32_t elem;
        uint32_t parent;
    };
    static constexpr size_t kMaxFlatEntries = 4u * 1024u * 1024u;
    const Grammar* g;
    uint32_t n_cls = 1;
    size_t max_flat_nodes = 0;
    size_t max_flat_sets = 0;
    std::vector<Node> nodes;
    std::unordered_map<uint64_t, uint32_t> interned;
    std::deque<std::vector<uint32_t>> sets;       // stable references; set 0 is always {}
    std::vector<uint8_t> set_flags;               // bit 0: has token element; bit 1: accepts (front == 0); bit 2: only {0}
    std::unordered_map<uint64_t, std::vector<int32_t>> set_buckets;
    std::vector<int32_t> closure_set;             // per node, -1 = not computed
    std::vector<int32_t> node_step;               // node * n_cls + cls -> set_id, -1 = not computed
    std::unordered_map<uint64_t, int32_t> node_step_overflow;
    std::vector<int32_t> set_trans;               // set_id * n_cls + cls -> next set_id, -1 = not computed
    std::unordered_map<uint64_t, int32_t> set_trans_overflow;
    std::vector<uint32_t> merge_scratch;
    struct MaskEntry {
        uint64_t vocab;
        int32_t set_id;
        std::vector<uint8_t> mask;
        uint64_t used;
    };
    static constexpr size_t kMaskCache = 32;
    std::vector<MaskEntry> masks;
    uint64_t tick = 0;

    explicit Arena(const Grammar* gr) : g(gr), n_cls(gr ? (std::max)(1u, gr->class_count()) : 1u) {
        max_flat_nodes = n_cls <= 256 ? kMaxFlatEntries / n_cls : 0;
        max_flat_sets = n_cls <= 256 ? kMaxFlatEntries / n_cls : 0;
        nodes.push_back({0, 0});
        closure_set.push_back(-1);
        if (max_flat_nodes > 0) node_step.assign(n_cls, -1);
        // set 0 is the empty stack set {}
        sets.emplace_back();
        set_flags.push_back(0);
        if (max_flat_sets > 0) set_trans.assign(n_cls, 0);
    }

    const Element& el(uint32_t i) const { return g->elements()[i]; }
    bool is_token(uint32_t e) const { return el(e).type == ElemType::Token || el(e).type == ElemType::TokenNot; }
    bool token_matches(uint32_t e, uint32_t id) const {
        return el(e).type == ElemType::Token ? el(e).value == id : el(e).value != id;
    }
    bool set_has_token(int32_t set_id) const { return (set_flags[(size_t) set_id] & 1u) != 0; }
    bool set_accepts(int32_t set_id) const { return (set_flags[(size_t) set_id] & 2u) != 0; }
    bool set_only_empty(int32_t set_id) const { return (set_flags[(size_t) set_id] & 4u) != 0; }

    uint32_t push(uint32_t parent, uint32_t elem) {
        uint64_t key = ((uint64_t) elem << 32) | parent;
        auto it = interned.find(key);
        if (it != interned.end()) return it->second;
        uint32_t id = (uint32_t) nodes.size();
        nodes.push_back({elem, parent});
        closure_set.push_back(-1);
        if (nodes.size() <= max_flat_nodes) node_step.resize(nodes.size() * n_cls, -1);
        interned.emplace(key, id);
        return id;
    }

    int32_t intern_set(std::vector<uint32_t>&& s) {
        if (s.empty()) return 0;
        const uint64_t h = hash_u32_vec(s);
        auto& bucket = set_buckets[h];
        for (int32_t id : bucket) {
            if (sets[(size_t) id] == s) return id;
        }
        const int32_t id = (int32_t) sets.size();
        uint8_t fl = 0;
        if (s.front() == 0) {
            fl |= 2u;
            if (s.size() == 1) fl |= 4u;
        }
        for (uint32_t st : s) {
            if (st != 0 && is_token(nodes[st].elem)) { fl |= 1u; break; }
        }
        sets.push_back(std::move(s));
        set_flags.push_back(fl);
        if (sets.size() <= max_flat_sets) set_trans.resize(sets.size() * n_cls, -1);
        bucket.push_back(id);
        return id;
    }

    // The interned stack set (each stack ending at a character/token element, or empty) a stack expands into.
    int32_t closure_id(uint32_t s) {
        if (closure_set[s] >= 0) return closure_set[s];
        std::vector<uint32_t> todo{s}, out;
        std::unordered_set<uint32_t> seen;
        while (!todo.empty()) {
            uint32_t c = todo.back();
            todo.pop_back();
            if (!seen.insert(c).second) continue;
            if (c == 0) {
                out.push_back(0);
                continue;
            }
            uint32_t e = nodes[c].elem;
            if (el(e).type == ElemType::RuleRef) {
                uint32_t sub = g->rule_start(el(e).value);
                for (;;) {
                    uint32_t next = nodes[c].parent;
                    if (!end_of_seq(el(e + 1))) next = push(next, e + 1);
                    if (!end_of_seq(el(sub))) next = push(next, sub);
                    todo.push_back(next);
                    while (!end_of_seq(el(sub))) ++sub;
                    if (el(sub).type == ElemType::Alt) ++sub;
                    else break;
                }
            } else {
                out.push_back(c);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        const int32_t id = intern_set(std::move(out));
        closure_set[s] = id;
        return id;
    }

    // Whether `cp` matches the class at element `e`, and the element after the class.
    bool match(uint32_t e, uint32_t cp, uint32_t& next) const {
        const Element* p = &g->elements()[e];
        bool positive = p->type == ElemType::Char || p->type == ElemType::CharAny;
        bool found = false;
        do {
            if (p[1].type == ElemType::CharRngUpper) {
                found = found || (p->value <= cp && cp <= p[1].value);
                p += 2;
            } else if (p->type == ElemType::CharAny) {
                found = true;
                p += 1;
            } else {
                found = found || p->value == cp;
                p += 1;
            }
        } while (p->type == ElemType::CharAlt);
        next = (uint32_t) (p - g->elements().data());
        return found == positive;
    }

    // Whether some valid completion in the disjoint intervals [r_lo[r], r_hi[r]] matches the class at element `e`.
    bool match_partial_ranges(uint32_t e, const uint32_t r_lo[2], const uint32_t r_hi[2], int n_ranges) const {
        const Element* p = &g->elements()[e];
        if (p->type == ElemType::CharAny) return true;
        const bool positive = p->type == ElemType::Char;
        if (positive) {
            do {
                uint32_t lo = p->value, hi = lo;
                if (p[1].type == ElemType::CharRngUpper) { hi = p[1].value; p += 2; }
                else { p += 1; }
                if (lo <= hi) {
                    for (int r = 0; r < n_ranges; ++r) {
                        if (lo <= r_hi[r] && r_lo[r] <= hi) return true;
                    }
                }
            } while (p->type == ElemType::CharAlt);
            return false;
        }
        // Negated class: matches some valid completion iff the excluded ranges do not completely cover [r_lo, r_hi].
        const uint32_t span_lo = r_lo[0], span_hi = r_hi[n_ranges - 1];
        std::pair<uint32_t, uint32_t> stack_buf[16];
        size_t n_excl = 0;
        std::vector<std::pair<uint32_t, uint32_t>> heap_buf;
        do {
            uint32_t lo = p->value, hi = lo;
            if (p[1].type == ElemType::CharRngUpper) { hi = p[1].value; p += 2; }
            else { p += 1; }
            if (lo <= hi && lo <= span_hi && span_lo <= hi) {
                if (n_excl < 16) {
                    stack_buf[n_excl++] = {lo, hi};
                } else {
                    if (heap_buf.empty()) heap_buf.assign(stack_buf, stack_buf + 16);
                    heap_buf.emplace_back(lo, hi);
                    ++n_excl;
                }
            }
        } while (p->type == ElemType::CharAlt);
        if (n_excl == 0) return true;
        std::pair<uint32_t, uint32_t>* excl = heap_buf.empty() ? stack_buf : heap_buf.data();
        if (n_excl > 1) {
            std::sort(excl, excl + n_excl,
                      [](const auto& a, const auto& b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
        }
        for (int r = 0; r < n_ranges; ++r) {
            uint32_t cur = r_lo[r];
            const uint32_t end = r_hi[r];
            for (size_t i = 0; i < n_excl; ++i) {
                if (excl[i].second < cur) continue;
                if (excl[i].first > cur) return true;
                if (excl[i].second >= end) { cur = end + 1; break; }
                cur = excl[i].second + 1;
            }
            if (cur <= end) return true;
        }
        return false;
    }

    // The interned stack set after single stack `s` consumes character class `cls`; 0 when it does not match.
    int32_t step_cls(uint32_t s, uint32_t cls) {
        const bool flat = s < max_flat_nodes;
        if (flat) {
            const int32_t cached = node_step[(size_t) s * n_cls + cls];
            if (cached >= 0) return cached;
        } else {
            const auto it = node_step_overflow.find(((uint64_t) s << 32) | cls);
            if (it != node_step_overflow.end()) return it->second;
        }
        int32_t id = 0;
        uint32_t next;
        const uint32_t cp = g->class_rep(cls);
        if (s != 0 && !is_token(nodes[s].elem) && match(nodes[s].elem, cp, next)) {
            uint32_t ns = nodes[s].parent;
            if (!end_of_seq(el(next))) ns = push(ns, next);
            id = closure_id(ns);
        }
        if (flat) node_step[(size_t) s * n_cls + cls] = id;
        else node_step_overflow[((uint64_t) s << 32) | cls] = id;
        return id;
    }

    // The interned stack set after stack set `set_id` consumes character class `cls`; 0 when no stack survives.
    int32_t set_step_cls(int32_t set_id, uint32_t cls) {
        if (set_id == 0) return 0;
        const bool flat = (size_t) set_id < max_flat_sets;
        if (flat) {
            const int32_t cached = set_trans[(size_t) set_id * n_cls + cls];
            if (cached >= 0) return cached;
        } else {
            const auto it = set_trans_overflow.find(((uint64_t) (uint32_t) set_id << 32) | cls);
            if (it != set_trans_overflow.end()) return it->second;
        }
        int32_t next_id = 0;
        const std::vector<uint32_t>& st = sets[(size_t) set_id];
        if (st.size() == 1) {
            next_id = step_cls(st[0], cls);
        } else {
            merge_scratch.clear();
            for (uint32_t s : st) {
                const int32_t sub = step_cls(s, cls);
                if (sub == 0) continue;
                const std::vector<uint32_t>& r = sets[(size_t) sub];
                merge_scratch.insert(merge_scratch.end(), r.begin(), r.end());
            }
            if (!merge_scratch.empty()) {
                std::sort(merge_scratch.begin(), merge_scratch.end());
                merge_scratch.erase(std::unique(merge_scratch.begin(), merge_scratch.end()), merge_scratch.end());
                std::vector<uint32_t> tmp(merge_scratch);
                next_id = intern_set(std::move(tmp));
            }
        }
        if (flat) set_trans[(size_t) set_id * n_cls + cls] = next_id;
        else set_trans_overflow[((uint64_t) (uint32_t) set_id << 32) | cls] = next_id;
        return next_id;
    }

    bool partial_ok_set(int32_t set_id, uint32_t value, int n_remain) {
        if (set_id == 0) return false;
        uint32_t r_lo[2], r_hi[2];
        const int n_ranges = utf8_partial_ranges(value, n_remain, r_lo, r_hi);
        if (n_ranges == 0) return false;
        if (g->wide_uniform()) return set_step_cls(set_id, g->wide_base()) != 0;
        for (uint32_t s : sets[(size_t) set_id]) {
            if (s != 0 && !is_token(nodes[s].elem) && match_partial_ranges(nodes[s].elem, r_lo, r_hi, n_ranges))
                return true;
        }
        return false;
    }
};

// --------------------------------------------------------------------------------------------- vocabulary

Vocab::Vocab(std::vector<std::string> bytes, std::vector<int32_t> end_ids)
    : bytes_(std::move(bytes)), end_ids_(std::move(end_ids)) {
    static std::atomic<uint64_t> next_id{1};
    id_ = next_id.fetch_add(1);
    struct BuildNode {
        uint32_t first_child = 0;
        uint32_t next_sibling = 0;
        int32_t first_token = -1;
        uint8_t byte = 0;
    };
    std::vector<BuildNode> build(1);
    next_token_.assign(bytes_.size(), -1);
    std::vector<uint8_t> is_end(bytes_.size(), 0);
    for (int32_t e : end_ids_) if (e >= 0 && (size_t) e < bytes_.size()) is_end[(size_t) e] = 1;
    for (size_t t = 0; t < bytes_.size(); ++t) {
        if (bytes_[t].empty() || is_end[t]) continue;
        uint32_t cur = 0;
        for (unsigned char b : bytes_[t]) {
            uint32_t c = build[cur].first_child;
            while (c != 0 && build[c].byte != b) c = build[c].next_sibling;
            if (c == 0) {
                c = (uint32_t) build.size();
                BuildNode n;
                n.byte = b;
                n.next_sibling = build[cur].first_child;
                build.push_back(n);
                build[cur].first_child = c;
            }
            cur = c;
        }
        next_token_[t] = build[cur].first_token;
        build[cur].first_token = (int32_t) t;
    }
    // Compact the trie into DFS preorder with children ordered by byte value so walks scan linearly and skip dead
    // subtrees in O(1).
    nodes_.reserve(build.size());
    nodes_.push_back({(uint32_t) build.size(), -1, 0});
    struct Frame {
        uint32_t idx;
        uint32_t depth;
        bool exit;
    };
    std::vector<Frame> stk;
    std::vector<uint32_t> kids;
    auto push_children = [&](uint32_t parent, uint32_t depth) {
        kids.clear();
        for (uint32_t c = build[parent].first_child; c != 0; c = build[c].next_sibling) kids.push_back(c);
        std::sort(kids.begin(), kids.end(), [&](uint32_t a, uint32_t b) { return build[a].byte > build[b].byte; });
        for (uint32_t c : kids) stk.push_back({c, depth, false});
    };
    push_children(0, 1);
    while (!stk.empty()) {
        const Frame f = stk.back();
        stk.pop_back();
        if (f.exit) {
            nodes_[f.idx].skip = (uint32_t) nodes_.size();
            continue;
        }
        const uint32_t flat_idx = (uint32_t) nodes_.size();
        nodes_.push_back({0, build[f.idx].first_token, (f.depth << 8) | build[f.idx].byte});
        if (f.depth > max_depth_) max_depth_ = f.depth;
        stk.push_back({flat_idx, 0, true});
        push_children(f.idx, f.depth + 1);
    }
    nodes_[0].skip = (uint32_t) nodes_.size();
}

// --------------------------------------------------------------------------------------------- state

// Static upper bound on the byte length of any match of an unanchored ECMAScript regex.
// Returns 0 (unbounded) for anchors, word boundaries, lookarounds, backreferences, open-ended quantifiers, or
// anything exceeding kMaxBound, so the caller falls back to scanning the full tail.
static size_t pattern_max_match_bytes(std::string_view p) {
    constexpr size_t kMaxBound = 4096;
    struct Parser {
        std::string_view s;
        size_t i = 0;

        bool disj(size_t& out) {
            size_t best = 0;
            for (;;) {
                size_t cur = 0;
                if (!alt(cur)) return false;
                best = (std::max)(best, cur);
                if (i < s.size() && s[i] == '|') { ++i; continue; }
                break;
            }
            out = best;
            return true;
        }

        bool alt(size_t& out) {
            size_t total = 0;
            while (i < s.size() && s[i] != '|' && s[i] != ')') {
                size_t atom_len = 0;
                if (!atom(atom_len)) return false;
                if (!quant(atom_len)) return false;
                if (atom_len > kMaxBound - total) return false;
                total += atom_len;
            }
            out = total;
            return true;
        }

        bool atom(size_t& out) {
            const char c = s[i++];
            if (c == '^' || c == '$' || c == '*' || c == '+' || c == '?' || c == '{') return false;
            if (c == '.') { out = 1; return true; }
            if (c == '\\') {
                if (i >= s.size()) return false;
                const char e = s[i++];
                if (e == 'b' || e == 'B' || e == 'k' || (e >= '1' && e <= '9')) return false;
                if (e == '0') {
                    if (i < s.size() && s[i] >= '0' && s[i] <= '9') return false;
                    out = 1;
                    return true;
                }
                if (e == 'x') {
                    if (i + 2 > s.size()) return false;
                    for (int k = 0; k < 2; ++k) {
                        const char h = s[i++];
                        const bool hex = (h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F');
                        if (!hex) return false;
                    }
                    out = 1;
                    return true;
                }
                if (e == 'u') {
                    if (i + 4 > s.size()) return false;
                    for (int k = 0; k < 4; ++k) {
                        const char h = s[i++];
                        const bool hex = (h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F');
                        if (!hex) return false;
                    }
                    out = 1;
                    return true;
                }
                if (e == 'c') {
                    if (i >= s.size()) return false;
                    const char ch = s[i++];
                    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))) return false;
                    out = 1;
                    return true;
                }
                out = 1;
                return true;
            }
            if (c == '[') {
                if (i < s.size() && s[i] == '^') ++i;
                while (i < s.size() && s[i] != ']') {
                    if (s[i] == '\\') {
                        if (i + 1 >= s.size()) return false;
                        i += 2;
                        continue;
                    }
                    if (s[i] == '[' && i + 1 < s.size() && (s[i + 1] == ':' || s[i + 1] == '.' || s[i + 1] == '='))
                        return false;
                    ++i;
                }
                if (i >= s.size()) return false;
                ++i;
                out = 1;
                return true;
            }
            if (c == '(') {
                if (i < s.size() && s[i] == '?') {
                    if (i + 1 >= s.size() || s[i + 1] != ':') return false;
                    i += 2;
                }
                if (!disj(out)) return false;
                if (i >= s.size() || s[i] != ')') return false;
                ++i;
                return true;
            }
            out = 1;
            return true;
        }

        bool parse_uint(size_t& v) {
            if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
            size_t n = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                const size_t d = (size_t) (s[i++] - '0');
                if (n > (kMaxBound - d) / 10) return false;
                n = n * 10 + d;
            }
            v = n;
            return true;
        }

        bool quant(size_t& len) {
            if (i >= s.size()) return true;
            const char q = s[i];
            if (q == '*' || q == '+') return false;
            if (q == '?') {
                ++i;
                if (i < s.size() && s[i] == '?') ++i;
            } else if (q == '{') {
                ++i;
                size_t lo = 0, hi = 0;
                if (!parse_uint(lo)) return false;
                hi = lo;
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    if (i >= s.size() || s[i] == '}') return false;
                    if (!parse_uint(hi) || hi < lo) return false;
                }
                if (i >= s.size() || s[i] != '}') return false;
                ++i;
                if (i < s.size() && s[i] == '?') ++i;
                if (hi > 0 && len > kMaxBound / hi) return false;
                len *= hi;
            } else {
                return true;
            }
            if (i < s.size() && (s[i] == '*' || s[i] == '+' || s[i] == '?' || s[i] == '{')) return false;
            return true;
        }
    };
    Parser parser{p, 0};
    size_t bound = 0;
    if (!parser.disj(bound) || parser.i != p.size()) return 0;
    return bound;
}

struct GrammarState::PatternSet {
    std::vector<std::regex> re;
    std::vector<std::string> src;
    std::vector<size_t> max_bytes;   ///< 0: unbounded; else any match has length <= max_bytes[i]
    bool all_bounded = true;
    size_t max_keep = 0;             ///< max(max_bytes[i] - 1) across bounded patterns

    // Where the grammar starts in `input` for pattern i: the first non-empty capture group, else the match itself.
    size_t find(size_t i, const std::string& input, size_t lo) const {
        auto start_of = [lo](const std::smatch& m) {
            for (size_t g = 1; g < m.size(); ++g) {
                if (m.length(g) > 0) return lo + (size_t) m.position(g);
            }
            return lo + (size_t) m.position(0);
        };
        const std::string& p = src[i];
        std::smatch m;
        if (lo == 0 && !p.empty() && p.front() == '^' && p.back() == '$' && std::regex_match(input, m, re[i]))
            return start_of(m);
        if (std::regex_search(input.cbegin() + (std::ptrdiff_t) lo, input.cend(), m, re[i]))
            return start_of(m);
        return std::string::npos;
    }
};

bool Triggers::valid(std::string& err) const {
    for (const std::string& p : patterns) {
        try {
            std::regex re(p);
        } catch (const std::regex_error& e) {
            err = "trigger pattern '" + p + "': " + e.what();
            return false;
        }
    }
    return true;
}

GrammarState::GrammarState(std::shared_ptr<const Grammar> g, Triggers triggers)
    : g_(std::move(g)), arena_(std::make_shared<Arena>(g_.get())), triggers_(std::move(triggers)) {
    size_t longest = 0;
    for (const std::string& w : triggers_.words) longest = (std::max)(longest, w.size());
    tail_keep_ = longest > 0 ? longest - 1 : 0;
    if (!triggers_.patterns.empty()) {
        std::string err;
        if (!triggers_.valid(err)) throw std::invalid_argument(err);
        auto set = std::make_shared<PatternSet>();
        for (const std::string& p : triggers_.patterns) {
            set->re.emplace_back(p);
            set->src.push_back(p);
            const size_t mb = pattern_max_match_bytes(p);
            set->max_bytes.push_back(mb);
            if (mb == 0) set->all_bounded = false;
            else set->max_keep = (std::max)(set->max_keep, mb - 1);
        }
        patterns_ = std::move(set);
    }
    reset();
}

int32_t GrammarState::root_set() const {
    std::vector<uint32_t> out;
    Arena& a = *arena_;
    uint32_t sub = g_->rule_start(g_->root());
    for (;;) {
        uint32_t s = end_of_seq(a.el(sub)) ? 0 : a.push(0, sub);
        const int32_t cid = a.closure_id(s);
        const std::vector<uint32_t>& c = a.sets[(size_t) cid];
        out.insert(out.end(), c.begin(), c.end());
        while (!end_of_seq(a.el(sub))) ++sub;
        if (a.el(sub).type == ElemType::Alt) ++sub;
        else break;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return a.intern_set(std::move(out));
}

void GrammarState::reset() {
    active_ = !triggers_.lazy();
    utf8_ = Utf8{};
    tail_.clear();
    pieces_.clear();
    set_id_ = root_set();
}

bool GrammarState::accepting() const {
    if (!active_) return true;
    return utf8_.n_remain == 0 && arena_ != nullptr && arena_->set_accepts(set_id_);
}

// Feeds bytes through the interned stack set. False on a byte that no stack accepts.
bool GrammarState::feed(const std::string_view& bytes, int32_t& set_id, Utf8& u) const {
    Arena& a = *arena_;
    const Grammar& g = *g_;
    const uint16_t* ascii_cls = g.ascii_classes();
    int32_t cur = set_id;
    for (unsigned char b : bytes) {
        if (u.n_remain == 0 && b < 0x80) {
            cur = a.set_step_cls(cur, ascii_cls[b]);
            if (cur == 0) return false;
            continue;
        }
        uint32_t cp;
        if (u.n_remain == 0) {
            if (b >= 0xC2 && b <= 0xDF) {
                u = {(uint32_t) (b & 0x1F), 1};
                continue;
            } else if (b >= 0xE0 && b <= 0xEF) {
                u = {(uint32_t) (b & 0x0F), 2};
                continue;
            } else if (b >= 0xF0 && b <= 0xF4) {
                u = {(uint32_t) (b & 0x07), 3};
                continue;
            } else {
                return false;
            }
        } else {
            if ((b & 0xC0) != 0x80) return false;
            u.value = (u.value << 6) | (b & 0x3F);
            if (--u.n_remain > 0) continue;
            cp = u.value;
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        }
        cur = a.set_step_cls(cur, g.char_class(cp));
        if (cur == 0) return false;
    }
    if (u.n_remain > 0 && !a.partial_ok_set(cur, u.value, u.n_remain)) return false;
    set_id = cur;
    return true;
}

// Advances `set_id` over one token: a stack waiting for a token reference takes the token by id, the others read its
// bytes. False if no stack survives.
bool GrammarState::advance(int32_t& set_id, Utf8& u, std::string_view bytes, int32_t token_id) const {
    Arena& a = *arena_;
    if (set_id == 0) return false;
    if (!a.set_has_token(set_id)) return feed(bytes, set_id, u);
    std::vector<uint32_t> text, out;
    for (uint32_t s : a.sets[(size_t) set_id]) {
        if (s == 0 || !a.is_token(a.nodes[s].elem)) {
            text.push_back(s);
        } else if (token_id >= 0 && a.token_matches(a.nodes[s].elem, (uint32_t) token_id)) {
            uint32_t e = a.nodes[s].elem;
            uint32_t ns = a.nodes[s].parent;
            if (!end_of_seq(a.el(e + 1))) ns = a.push(ns, e + 1);
            const int32_t cid = a.closure_id(ns);
            const std::vector<uint32_t>& c = a.sets[(size_t) cid];
            out.insert(out.end(), c.begin(), c.end());
        }
    }
    Utf8 nu = u;
    if (!text.empty()) {
        int32_t tid = a.intern_set(std::move(text));
        if (feed(bytes, tid, nu)) {
            const std::vector<uint32_t>& survived = a.sets[(size_t) tid];
            out.insert(out.end(), survived.begin(), survived.end());
            u = nu;
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) return false;
    set_id = a.intern_set(std::move(out));
    return true;
}

// Starts the grammar from the root and feeds it the output from `from` on (the replay of the buffered tokens when
// patterns are in use); on failure the state is unchanged.
bool GrammarState::fire(size_t from, bool by_token, int32_t token_id, std::string_view bytes) {
    int32_t st = root_set();
    Utf8 u;
    if (by_token) {
        if (!advance(st, u, bytes, token_id)) return false;
    } else if (patterns_) {
        for (const Piece& p : pieces_) {
            if (p.end <= from) continue;
            size_t b = (std::max)(p.begin, from);
            if (!advance(st, u, std::string_view(tail_).substr(b, p.end - b), p.token)) return false;
        }
    } else if (!advance(st, u, std::string_view(tail_).substr(from), -1)) {
        return false;
    }
    set_id_ = st;
    utf8_ = u;
    active_ = true;
    tail_.clear();
    pieces_.clear();
    if (arena_->set_only_empty(set_id_) && utf8_.n_remain == 0) reset();
    return true;
}

bool GrammarState::accept(std::string_view bytes, int32_t token_id) {
    if (!g_) return true;
    if (active_) {
        int32_t st = set_id_;
        Utf8 u = utf8_;
        if (!advance(st, u, bytes, token_id)) return false;
        set_id_ = st;
        utf8_ = u;
        if (triggers_.lazy() && arena_->set_only_empty(set_id_) && utf8_.n_remain == 0) reset();
        return true;
    }
    for (int32_t t : triggers_.tokens) {
        if (t == token_id) return fire(0, true, token_id, bytes);
    }
    // text trigger: look for a word or a pattern in the output so far plus this token
    const size_t prev_tail = tail_.size();
    const size_t prev_pieces = pieces_.size();
    tail_.append(bytes.data(), bytes.size());
    size_t from = std::string::npos;
    for (const std::string& w : triggers_.words) {
        if (w.empty()) continue;
        size_t lo = prev_tail >= w.size() ? prev_tail - w.size() + 1 : 0;
        size_t p = tail_.find(w, lo);
        if (p != std::string::npos) from = (std::min)(from, p);
    }
    if (patterns_) {
        pieces_.push_back({token_id, prev_tail, tail_.size()});
        for (size_t i = 0; i < patterns_->src.size(); ++i) {
            const size_t mb = patterns_->max_bytes[i];
            const size_t lo = mb > 0 && prev_tail >= mb - 1 ? prev_tail - (mb - 1) : 0;
            size_t p = patterns_->find(i, tail_, lo);
            if (p != std::string::npos) {
                from = (std::min)(from, p);
                break;
            }
        }
    }
    if (from == std::string::npos) {
        if (!patterns_) {
            if (tail_.size() > tail_keep_) tail_.erase(0, tail_.size() - tail_keep_);
        } else if (patterns_->all_bounded) {
            const size_t keep = (std::max)(tail_keep_, patterns_->max_keep);
            if (tail_.size() > keep * 2 + 256) {
                const size_t drop = tail_.size() - keep;
                tail_.erase(0, drop);
                size_t first = 0;
                while (first < pieces_.size() && pieces_[first].end <= drop) ++first;
                if (first > 0) pieces_.erase(pieces_.begin(), pieces_.begin() + (std::ptrdiff_t) first);
                for (Piece& pc : pieces_) {
                    pc.begin = pc.begin > drop ? pc.begin - drop : 0;
                    pc.end -= drop;
                }
            }
        }
        return true;
    }
    if (fire(from, false, token_id, bytes)) return true;
    tail_.resize(prev_tail);
    if (patterns_) pieces_.resize(prev_pieces);
    return false;
}

bool GrammarState::allows(std::string_view bytes, int32_t token_id) const {
    if (!g_) return true;
    if (active_) {
        int32_t st = set_id_;
        Utf8 u = utf8_;
        return advance(st, u, bytes, token_id);
    }
    GrammarState copy = *this;
    return copy.accept(bytes, token_id);
}

void GrammarState::walk(const Vocab& v, int32_t root_set_id, const Utf8& u, std::vector<uint8_t>& out) const {
    const size_t n_nodes = v.nodes_.size();
    if (n_nodes <= 1) return;
    Arena& a = *arena_;
    const Grammar& g = *g_;
    const uint16_t* ascii_cls = g.ascii_classes();
    const Vocab::Node* nodes = v.nodes_.data();
    const int32_t* next_tok = v.next_token_.data();
    const size_t depth_cap = (size_t) v.max_depth_ + 1;
    int32_t stack_set_buf[256];
    Utf8 stack_u_buf[256];
    std::vector<int32_t> heap_set;
    std::vector<Utf8> heap_u;
    int32_t* set_stk = stack_set_buf;
    Utf8* u_stk = stack_u_buf;
    if (depth_cap > 256) {
        heap_set.resize(depth_cap);
        heap_u.resize(depth_cap);
        set_stk = heap_set.data();
        u_stk = heap_u.data();
    }
    set_stk[0] = root_set_id;
    u_stk[0] = u;
    const uint32_t n_cls = a.n_cls;
    const size_t max_flat = a.max_flat_sets;
    uint8_t* out_data = out.data();
    for (uint32_t i = 1; i < n_nodes;) {
        const Vocab::Node& nd = nodes[i];
        const uint32_t d = nd.depth_byte >> 8;
        const unsigned char b = (unsigned char) (nd.depth_byte & 0xFFu);
        const int32_t parent_set = set_stk[d - 1];
        Utf8 nu = u_stk[d - 1];
        int32_t next_set = parent_set;
        if (nu.n_remain == 0 && b < 0x80) {
            const uint32_t cls = ascii_cls[b];
            if ((size_t) parent_set < max_flat) {
                next_set = a.set_trans[(size_t) parent_set * n_cls + cls];
                if (next_set < 0) next_set = a.set_step_cls(parent_set, cls);
            } else {
                next_set = a.set_step_cls(parent_set, cls);
            }
            if (next_set == 0) {
                i = nd.skip;
                continue;
            }
        } else {
            uint32_t cp = 0;
            bool complete = false;
            if (nu.n_remain == 0) {
                if (b >= 0xC2 && b <= 0xDF) nu = {(uint32_t) (b & 0x1F), 1};
                else if (b >= 0xE0 && b <= 0xEF) nu = {(uint32_t) (b & 0x0F), 2};
                else if (b >= 0xF0 && b <= 0xF4) nu = {(uint32_t) (b & 0x07), 3};
                else { i = nd.skip; continue; }
            } else {
                if ((b & 0xC0) != 0x80) { i = nd.skip; continue; }
                nu.value = (nu.value << 6) | (b & 0x3F);
                if (--nu.n_remain == 0) {
                    cp = nu.value;
                    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { i = nd.skip; continue; }
                    complete = true;
                }
            }
            if (complete) {
                next_set = a.set_step_cls(parent_set, g.char_class(cp));
                if (next_set == 0) { i = nd.skip; continue; }
            } else if (!a.partial_ok_set(parent_set, nu.value, nu.n_remain)) {
                i = nd.skip;
                continue;
            }
        }
        for (int32_t t = nd.first_token; t >= 0; t = next_tok[(size_t) t]) out_data[(size_t) t] = 1;
        set_stk[d] = next_set;
        u_stk[d] = nu;
        ++i;
    }
}

void GrammarState::fill_mask(const Vocab& vocab, std::vector<uint8_t>& out, bool use_cache) const {
    if (!active_ || !g_) {
        out.assign(vocab.size(), 1);
        return;
    }
    Arena& a = *arena_;
    use_cache = use_cache && utf8_.n_remain == 0;
    if (use_cache) {
        for (Arena::MaskEntry& m : a.masks) {
            if (m.vocab == vocab.id() && m.set_id == set_id_) {
                m.used = ++a.tick;
                out = m.mask;
                return;
            }
        }
    }
    out.assign(vocab.size(), 0);
    if (set_id_ != 0) walk(vocab, set_id_, utf8_, out);
    if (utf8_.n_remain == 0 && set_id_ != 0 && a.set_has_token(set_id_)) {
        for (uint32_t s : a.sets[(size_t) set_id_]) {
            if (s == 0 || !a.is_token(a.nodes[s].elem)) continue;
            const Element& t = a.el(a.nodes[s].elem);
            if (t.type == ElemType::Token) {
                if (t.value < out.size()) out[t.value] = 1;
            } else {
                for (size_t id = 0; id < out.size(); ++id) {
                    if (id != t.value && !vocab.bytes((int32_t) id).empty()) out[id] = 1;
                }
            }
        }
    }
    bool end_ok = accepting();
    for (int32_t e : vocab.end_ids()) {
        if (e >= 0 && (size_t) e < out.size()) out[(size_t) e] = end_ok ? 1 : 0;
    }
    if (!use_cache) return;
    if (a.masks.size() < Arena::kMaskCache) {
        a.masks.push_back({vocab.id(), set_id_, out, ++a.tick});
    } else {
        auto oldest = std::min_element(a.masks.begin(), a.masks.end(),
                                       [](const Arena::MaskEntry& x, const Arena::MaskEntry& y) { return x.used < y.used; });
        *oldest = {vocab.id(), set_id_, out, ++a.tick};
    }
}

std::vector<std::vector<Element>> GrammarState::stack_elements() const {
    std::vector<std::vector<Element>> out;
    if (!g_ || !arena_ || set_id_ == 0) return out;
    const Arena& a = *arena_;
    for (uint32_t s : a.sets[(size_t) set_id_]) {
        std::vector<Element> st;
        for (uint32_t n = s; n != 0; n = a.nodes[n].parent) st.push_back(a.el(a.nodes[n].elem));
        std::reverse(st.begin(), st.end());
        out.push_back(std::move(st));
    }
    return out;
}

}  // namespace basalt::program::grammar
