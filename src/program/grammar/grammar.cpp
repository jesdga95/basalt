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

// A stack is a chain of interned (element, parent) nodes; the element is the next one to match in its rule
// sequence. Node 0 is the empty stack. Closures and character steps are memoized per node.
struct GrammarState::Arena {
    struct Node {
        uint32_t elem;
        uint32_t parent;
    };
    const Grammar* g;
    std::vector<Node> nodes;
    std::unordered_map<uint64_t, uint32_t> interned;
    std::deque<std::vector<uint32_t>> sets;       // stable references
    std::vector<int32_t> closure_set;             // per node, -1 = not computed
    std::vector<int32_t> ascii_step;              // node * 128 + cp, -1 = not computed
    std::unordered_map<uint64_t, int32_t> wide_step;
    struct MaskEntry {
        uint64_t vocab;
        std::vector<uint32_t> stacks;
        std::vector<uint8_t> mask;
        uint64_t used;
    };
    static constexpr size_t kMaskCache = 32;
    std::vector<MaskEntry> masks;
    uint64_t tick = 0;

    explicit Arena(const Grammar* gr) : g(gr) {
        nodes.push_back({0, 0});
        closure_set.push_back(-1);
        ascii_step.assign(128, -1);
    }

    const Element& el(uint32_t i) const { return g->elements()[i]; }
    bool is_token(uint32_t e) const { return el(e).type == ElemType::Token || el(e).type == ElemType::TokenNot; }
    bool token_matches(uint32_t e, uint32_t id) const {
        return el(e).type == ElemType::Token ? el(e).value == id : el(e).value != id;
    }

    uint32_t push(uint32_t parent, uint32_t elem) {
        uint64_t key = ((uint64_t) elem << 32) | parent;
        auto it = interned.find(key);
        if (it != interned.end()) return it->second;
        uint32_t id = (uint32_t) nodes.size();
        nodes.push_back({elem, parent});
        closure_set.push_back(-1);
        ascii_step.resize(ascii_step.size() + 128, -1);
        interned.emplace(key, id);
        return id;
    }

    // The stacks (each ending at a character element, or empty) a stack can be expanded into.
    const std::vector<uint32_t>& closure(uint32_t s) {
        if (closure_set[s] >= 0) return sets[(size_t) closure_set[s]];
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
        sets.push_back(std::move(out));
        closure_set[s] = (int32_t) sets.size() - 1;
        return sets.back();
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

    // Whether some completion of a partial UTF-8 sequence could match the class at element `e`.
    bool match_partial(uint32_t e, uint32_t value, int n_remain) const {
        const Element* p = &g->elements()[e];
        bool positive = p->type == ElemType::Char || p->type == ElemType::CharAny;
        if (n_remain < 0 || (n_remain == 1 && value < 2)) return false;
        uint32_t low = value << (n_remain * 6);
        uint32_t high = low | ((1u << (n_remain * 6)) - 1);
        if (low == 0) {
            if (n_remain == 2) low = 1u << 11;
            else if (n_remain == 3) low = 1u << 16;
        }
        do {
            if (p[1].type == ElemType::CharRngUpper) {
                if (p->value <= high && low <= p[1].value) return positive;
                p += 2;
            } else if (p->type == ElemType::CharAny) {
                return true;
            } else {
                if (low <= p->value && p->value <= high) return positive;
                p += 1;
            }
        } while (p->type == ElemType::CharAlt);
        return !positive;
    }

    // The stacks after `s` consumes `cp`; empty when it does not match.
    const std::vector<uint32_t>& step(uint32_t s, uint32_t cp) {
        int32_t* slot;
        if (cp < 128) {
            slot = &ascii_step[(size_t) s * 128 + cp];
        } else {
            slot = &wide_step.emplace(((uint64_t) s << 32) | cp, -1).first->second;
        }
        if (*slot >= 0) return sets[(size_t) *slot];
        std::vector<uint32_t> out;
        uint32_t next;
        if (s != 0 && !is_token(nodes[s].elem) && match(nodes[s].elem, cp, next)) {
            uint32_t ns = nodes[s].parent;
            if (!end_of_seq(el(next))) ns = push(ns, next);
            out = closure(ns);
        }
        sets.push_back(std::move(out));
        // push() may have grown ascii_step, so the slot is looked up again
        int32_t idx = (int32_t) sets.size() - 1;
        if (cp < 128) ascii_step[(size_t) s * 128 + cp] = idx;
        else wide_step[((uint64_t) s << 32) | cp] = idx;
        return sets.back();
    }

    bool partial_ok(const std::vector<uint32_t>& stacks, uint32_t value, int n_remain) const {
        for (uint32_t s : stacks) {
            if (s != 0 && !is_token(nodes[s].elem) && match_partial(nodes[s].elem, value, n_remain)) return true;
        }
        return false;
    }
};

// --------------------------------------------------------------------------------------------- vocabulary

Vocab::Vocab(std::vector<std::string> bytes, std::vector<int32_t> end_ids)
    : bytes_(std::move(bytes)), end_ids_(std::move(end_ids)) {
    static std::atomic<uint64_t> next_id{1};
    id_ = next_id.fetch_add(1);
    nodes_.emplace_back();
    next_token_.assign(bytes_.size(), -1);
    std::vector<int32_t> is_end(bytes_.size(), 0);
    for (int32_t e : end_ids_) if (e >= 0 && (size_t) e < bytes_.size()) is_end[(size_t) e] = 1;
    for (size_t t = 0; t < bytes_.size(); ++t) {
        if (bytes_[t].empty() || is_end[t]) continue;
        uint32_t cur = 0;
        for (unsigned char b : bytes_[t]) {
            uint32_t c = nodes_[cur].first_child;
            while (c != 0 && nodes_[c].byte != b) c = nodes_[c].next_sibling;
            if (c == 0) {
                c = (uint32_t) nodes_.size();
                Node n;
                n.byte = b;
                n.next_sibling = nodes_[cur].first_child;
                nodes_.push_back(n);
                nodes_[cur].first_child = c;
            }
            cur = c;
        }
        next_token_[t] = nodes_[cur].first_token;
        nodes_[cur].first_token = (int32_t) t;
    }
}

// --------------------------------------------------------------------------------------------- state

struct GrammarState::PatternSet {
    std::vector<std::regex> re;
    std::vector<std::string> src;

    // Where the grammar starts in `input` for pattern i: the first non-empty capture group, else the match itself.
    size_t find(size_t i, const std::string& input) const {
        auto start_of = [](const std::smatch& m) {
            for (size_t g = 1; g < m.size(); ++g) {
                if (m.length(g) > 0) return (size_t) m.position(g);
            }
            return (size_t) m.position(0);
        };
        const std::string& p = src[i];
        std::smatch m;
        if (!p.empty() && p.front() == '^' && p.back() == '$' && std::regex_match(input, m, re[i])) return start_of(m);
        if (std::regex_search(input, m, re[i])) return start_of(m);
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
        }
        patterns_ = std::move(set);
    }
    reset();
}

void GrammarState::reset() {
    active_ = !triggers_.lazy();
    utf8_ = Utf8{};
    tail_.clear();
    pieces_.clear();
    stacks_.clear();
    // enter the root rule: one stack per alternative
    Arena& a = *arena_;
    uint32_t sub = g_->rule_start(g_->root());
    for (;;) {
        uint32_t s = end_of_seq(a.el(sub)) ? 0 : a.push(0, sub);
        const std::vector<uint32_t>& c = a.closure(s);
        stacks_.insert(stacks_.end(), c.begin(), c.end());
        while (!end_of_seq(a.el(sub))) ++sub;
        if (a.el(sub).type == ElemType::Alt) ++sub;
        else break;
    }
    std::sort(stacks_.begin(), stacks_.end());
    stacks_.erase(std::unique(stacks_.begin(), stacks_.end()), stacks_.end());
}

bool GrammarState::accepting() const {
    if (!active_) return true;
    return utf8_.n_remain == 0 && !stacks_.empty() && stacks_.front() == 0;
}

// Feeds bytes through the stacks. False on a byte that no stack accepts.
bool GrammarState::feed(const std::string_view& bytes, std::vector<uint32_t>& stacks, Utf8& u) const {
    Arena& a = *arena_;
    std::vector<uint32_t> tmp;
    for (unsigned char b : bytes) {
        uint32_t cp;
        if (u.n_remain == 0) {
            if (b < 0x80) {
                cp = b;
            } else if (b >= 0xC2 && b <= 0xDF) {
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
        tmp.clear();
        for (uint32_t s : stacks) {
            const std::vector<uint32_t>& r = a.step(s, cp);
            tmp.insert(tmp.end(), r.begin(), r.end());
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        if (tmp.empty()) return false;
        stacks.swap(tmp);
    }
    if (u.n_remain > 0 && !a.partial_ok(stacks, u.value, u.n_remain)) return false;
    return true;
}

// Advances `stacks` over one token: a stack waiting for a token reference takes the token by id, the others read its
// bytes. False if no stack survives.
bool GrammarState::advance(std::vector<uint32_t>& stacks, Utf8& u, std::string_view bytes, int32_t token_id) const {
    Arena& a = *arena_;
    bool any_token = false;
    for (uint32_t s : stacks) {
        if (s != 0 && a.is_token(a.nodes[s].elem)) any_token = true;
    }
    if (!any_token) return feed(bytes, stacks, u);
    std::vector<uint32_t> text, out;
    for (uint32_t s : stacks) {
        if (s == 0 || !a.is_token(a.nodes[s].elem)) {
            text.push_back(s);
        } else if (token_id >= 0 && a.token_matches(a.nodes[s].elem, (uint32_t) token_id)) {
            uint32_t e = a.nodes[s].elem;
            uint32_t ns = a.nodes[s].parent;
            if (!end_of_seq(a.el(e + 1))) ns = a.push(ns, e + 1);
            const std::vector<uint32_t>& c = a.closure(ns);
            out.insert(out.end(), c.begin(), c.end());
        }
    }
    Utf8 nu = u;
    if (!text.empty() && feed(bytes, text, nu)) {
        out.insert(out.end(), text.begin(), text.end());
        u = nu;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) return false;
    stacks.swap(out);
    return true;
}

// Starts the grammar from the root and feeds it the output from `from` on (the replay of the buffered tokens when
// patterns are in use); on failure the state is unchanged.
bool GrammarState::fire(size_t from, bool by_token, int32_t token_id, std::string_view bytes) {
    GrammarState fresh = *this;
    fresh.reset();
    std::vector<uint32_t> st = fresh.stacks_;
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
    stacks_.swap(st);
    utf8_ = u;
    active_ = true;
    tail_.clear();
    pieces_.clear();
    if (stacks_.size() == 1 && stacks_[0] == 0 && utf8_.n_remain == 0) reset();
    return true;
}

bool GrammarState::accept(std::string_view bytes, int32_t token_id) {
    if (!g_) return true;
    if (active_) {
        std::vector<uint32_t> st = stacks_;
        Utf8 u = utf8_;
        if (!advance(st, u, bytes, token_id)) return false;
        stacks_.swap(st);
        utf8_ = u;
        if (triggers_.lazy() && stacks_.size() == 1 && stacks_[0] == 0 && utf8_.n_remain == 0) reset();
        return true;
    }
    for (int32_t t : triggers_.tokens) {
        if (t == token_id) return fire(0, true, token_id, bytes);
    }
    // text trigger: look for a word or a pattern in the output so far plus this token
    std::string buf = tail_;
    buf.append(bytes.data(), bytes.size());
    size_t from = std::string::npos;
    for (const std::string& w : triggers_.words) {
        if (w.empty()) continue;
        size_t lo = tail_.size() >= w.size() ? tail_.size() - w.size() + 1 : 0;
        size_t p = buf.find(w, lo);
        if (p != std::string::npos) from = (std::min)(from, p);
    }
    std::vector<Piece> pieces;
    if (patterns_) {
        pieces = pieces_;
        pieces.push_back({token_id, tail_.size(), buf.size()});
        for (size_t i = 0; i < patterns_->src.size(); ++i) {
            size_t p = patterns_->find(i, buf);
            if (p != std::string::npos) {
                from = (std::min)(from, p);
                break;
            }
        }
    }
    if (from == std::string::npos) {
        if (patterns_) {
            tail_ = std::move(buf);
            pieces_ = std::move(pieces);
        } else {
            tail_ = buf.size() > tail_keep_ ? buf.substr(buf.size() - tail_keep_) : buf;
        }
        return true;
    }
    GrammarState keep = *this;
    tail_ = std::move(buf);
    pieces_ = std::move(pieces);
    if (fire(from, false, token_id, bytes)) return true;
    *this = std::move(keep);
    return false;
}

bool GrammarState::allows(std::string_view bytes, int32_t token_id) const {
    GrammarState copy = *this;
    return copy.accept(bytes, token_id);
}

void GrammarState::walk(const Vocab& v, uint32_t node, const std::vector<uint32_t>& stacks, const Utf8& u,
                        std::vector<uint8_t>& out) const {
    Arena& a = *arena_;
    std::vector<uint32_t> tmp;
    for (uint32_t c = v.nodes_[node].first_child; c != 0; c = v.nodes_[c].next_sibling) {
        unsigned char b = v.nodes_[c].byte;
        Utf8 nu = u;
        const std::vector<uint32_t>* next = &stacks;
        uint32_t cp = 0;
        bool complete = false;
        if (nu.n_remain == 0) {
            if (b < 0x80) { cp = b; complete = true; }
            else if (b >= 0xC2 && b <= 0xDF) nu = {(uint32_t) (b & 0x1F), 1};
            else if (b >= 0xE0 && b <= 0xEF) nu = {(uint32_t) (b & 0x0F), 2};
            else if (b >= 0xF0 && b <= 0xF4) nu = {(uint32_t) (b & 0x07), 3};
            else continue;
        } else {
            if ((b & 0xC0) != 0x80) continue;
            nu.value = (nu.value << 6) | (b & 0x3F);
            if (--nu.n_remain == 0) {
                cp = nu.value;
                if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) continue;
                complete = true;
            }
        }
        if (complete) {
            if (stacks.size() == 1) {
                next = &a.step(stacks[0], cp);
            } else {
                tmp.clear();
                for (uint32_t s : stacks) {
                    const std::vector<uint32_t>& r = a.step(s, cp);
                    tmp.insert(tmp.end(), r.begin(), r.end());
                }
                std::sort(tmp.begin(), tmp.end());
                tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
                next = &tmp;
            }
            if (next->empty()) continue;
        } else if (!a.partial_ok(stacks, nu.value, nu.n_remain)) {
            continue;
        }
        for (int32_t t = v.nodes_[c].first_token; t >= 0; t = v.next_token_[(size_t) t]) out[(size_t) t] = 1;
        if (v.nodes_[c].first_child != 0) {
            walk(v, c, *next, nu, out);
        }
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
            if (m.vocab == vocab.id() && m.stacks == stacks_) {
                m.used = ++a.tick;
                out = m.mask;
                return;
            }
        }
    }
    out.assign(vocab.size(), 0);
    if (!stacks_.empty()) walk(vocab, 0, stacks_, utf8_, out);
    if (utf8_.n_remain == 0) {
        for (uint32_t s : stacks_) {
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
        a.masks.push_back({vocab.id(), stacks_, out, ++a.tick});
    } else {
        auto oldest = std::min_element(a.masks.begin(), a.masks.end(),
                                       [](const Arena::MaskEntry& x, const Arena::MaskEntry& y) { return x.used < y.used; });
        *oldest = {vocab.id(), stacks_, out, ++a.tick};
    }
}

std::vector<std::vector<Element>> GrammarState::stack_elements() const {
    std::vector<std::vector<Element>> out;
    if (!g_) return out;
    const Arena& a = *arena_;
    for (uint32_t s : stacks_) {
        std::vector<Element> st;
        for (uint32_t n = s; n != 0; n = a.nodes[n].parent) st.push_back(a.el(a.nodes[n].elem));
        std::reverse(st.begin(), st.end());
        out.push_back(std::move(st));
    }
    return out;
}

}  // namespace basalt::program::grammar
