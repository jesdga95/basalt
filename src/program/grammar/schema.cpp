// src/program/grammar/schema.cpp - see schema.hpp. The JSON-schema conversion is a port of llama.cpp's
// json-schema-to-grammar (schema_convert.cpp, json_schema.cpp); this file adds the tool-call and request grammars.
#include "basalt/program/grammar/schema.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "schema_internal.hpp"

namespace basalt::program::grammar {

namespace {

using Type = JsonValue::Type;

std::string literal(const std::string& s) { return gbnf_format_literal(s); }

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
    return o;
}

bool read_json(const std::string& text, JsonValue& v, std::string& err) { return JsonReader::parse(text, v, err); }

// Follows local $ref links ("#/a/b") to the schema they name; null if one does not resolve.
const JsonValue* deref(const JsonValue* root, const JsonValue* s) {
    for (int guard = 0; s && s->is(Type::Object) && guard < 32; ++guard) {
        const JsonValue* r = s->get("$ref");
        if (!r || !r->is(Type::String)) break;
        if (r->s.compare(0, 2, "#/") != 0 || !root) return nullptr;
        const JsonValue* cur = root;
        size_t pos = 2;
        while (cur && pos <= r->s.size()) {
            size_t e = r->s.find('/', pos);
            std::string part = r->s.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
            if (cur->is(Type::Object)) {
                cur = cur->get(part);
            } else if (cur->is(Type::Array) && !part.empty() && part.find_first_not_of("0123456789") == std::string::npos &&
                       std::stoul(part) < cur->items.size()) {
                cur = &cur->items[std::stoul(part)];
            } else {
                cur = nullptr;
            }
            if (e == std::string::npos) break;
            pos = e + 1;
        }
        s = cur;
    }
    return s;
}

}  // namespace

std::string json_schema_to_gbnf(const std::string& schema_json, std::string& err) {
    JsonValue schema;
    if (!read_json(schema_json, schema, err)) return "";
    try {
        return schema_document_to_gbnf(build_schema_document(schema));
    } catch (const std::invalid_argument& e) {
        err = std::string("schema: ") + e.what();
    } catch (const std::runtime_error& e) {
        err = std::string("schema: JSON schema conversion failed:\n") + e.what();
    }
    return "";
}

// ------------------------------------------------------------------------------------------ tool calls

namespace {

// True when the schema is a type that is raw text in a parameter: "string", or a type list naming "string".
bool is_text_type(const JsonValue& t) {
    if (t.is(Type::String)) return t.s == "string";
    return t.is(Type::Array) && std::any_of(t.items.begin(), t.items.end(),
                                            [](const JsonValue& v) { return v.is(Type::String) && v.s == "string"; });
}

// One branch of an anyOf/oneOf/allOf: a plain string (type "string" or a list of string and null, no enum/const/
// structure) or null. `has_text` is set when it can be text.
bool plain_text_branch(const JsonValue* b, bool& has_text) {
    if (!b || !b->is(Type::Object)) return false;
    for (const char* k : {"enum", "const", "anyOf", "oneOf", "allOf", "properties", "items", "$ref"})
        if (b->get(k)) return false;
    const JsonValue* t = b->get("type");
    if (!t) return false;
    if (t->is(Type::String)) {
        if (t->s == "string") has_text = true;
        return t->s == "string" || t->s == "null";
    }
    if (!t->is(Type::Array)) return false;
    for (const JsonValue& v : t->items) {
        if (!v.is(Type::String) || (v.s != "string" && v.s != "null")) return false;
        if (v.s == "string") has_text = true;
    }
    return true;
}

// A parameter written as anyOf/oneOf (or a one-branch allOf) whose every branch is a plain string or null, with at
// least one string: it takes raw text like a string, and `null` is just text it may write.
bool text_union(const JsonValue* ps, const JsonValue* root) {
    if (!ps || !ps->is(Type::Object) || ps->get("type") || ps->get("enum") || ps->get("const") ||
        ps->get("properties") || ps->get("items"))
        return false;
    const JsonValue* u = nullptr;
    for (const char* k : {"anyOf", "oneOf", "allOf"}) {
        if (const JsonValue* v = ps->get(k)) {
            if (u || !v->is(Type::Array) || v->items.empty()) return false;
            if (k[0] == 'a' && k[1] == 'l' && v->items.size() != 1) return false;
            u = v;
        }
    }
    if (!u) return false;
    bool has_text = false;
    for (const JsonValue& b : u->items)
        if (!plain_text_branch(deref(root, &b), has_text)) return false;
    return has_text;
}

std::string hex_escape(unsigned char c) {
    static const char* h = "0123456789abcdef";
    return std::string("\\x") + h[c >> 4] + h[c & 15];
}

// Text that never contains the closing tag: a KMP automaton over the tag, one rule per matched length. The tag's last
// character is the one move the automaton refuses.
std::string add_free_text(SchemaConverter& c, const std::string& tag) {
    size_t n = tag.size();
    std::set<unsigned char> alpha(tag.begin(), tag.end());
    std::string excluded;
    for (unsigned char a : alpha) excluded += hex_escape(a);
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::string> alts;
        for (unsigned char a : alpha) {
            std::string seen = tag.substr(0, i) + (char) a;
            size_t k = (std::min)(seen.size(), n);
            while (k > 0 && tag.compare(0, k, seen, seen.size() - k, k) != 0) --k;
            if (k == n) continue;
            alts.push_back(literal(std::string(1, (char) a)) + " free-text-" + std::to_string(k));
        }
        alts.push_back("[^" + excluded + "] free-text-0");
        alts.push_back("\"\"");
        c.add_rule("free-text-" + std::to_string(i), join(alts, " | "));
    }
    return "free-text-0";
}

}  // namespace

std::string tool_call_gbnf(const std::string& tools_json, const ToolGrammarOptions& opt, std::string& err) {
    JsonValue tools;
    if (!read_json(tools_json, tools, err)) return "";
    if (!tools.is(Type::Array) || tools.items.empty()) {
        err = "tools: expected a non-empty array";
        return "";
    }
    SchemaConverter c;
    std::string text_rule;
    std::vector<std::string> fns;
    for (size_t ti = 0; ti < tools.items.size(); ++ti) {
        const JsonValue* tool = &tools.items[ti];
        if (const JsonValue* f = tool->is(Type::Object) ? tool->get("function") : nullptr) tool = f;
        const JsonValue* nm = tool->is(Type::Object) ? tool->get("name") : nullptr;
        if (!nm || !nm->is(Type::String) || nm->s.empty()) {
            err = "tools: every tool needs a name";
            return "";
        }
        const JsonValue* params = tool->get("parameters");
        if (!params) params = tool->get("input_schema");
        const JsonValue* props = params && params->is(Type::Object) ? params->get("properties") : nullptr;
        std::set<std::string> required;
        if (const JsonValue* r = params && params->is(Type::Object) ? params->get("required") : nullptr) {
            for (const JsonValue& n : r->items) if (n.is(Type::String)) required.insert(n.s);
        }
        std::string fn_id = "fn" + std::to_string(ti);
        std::vector<std::string> req_items, opt_items;
        if (props && props->is(Type::Object)) {
            for (const auto& m : props->members) {
                const JsonValue* ps = deref(params, &m.second);
                JsonValue any_schema;   // `true` and a missing target are the empty schema
                if (!ps || ps->is(Type::Bool)) {
                    any_schema.type = Type::Object;
                    ps = &any_schema;
                }
                const JsonValue* ty = ps && ps->is(Type::Object) ? ps->get("type") : nullptr;
                const JsonValue* en = ps && ps->is(Type::Object) ? ps->get("enum") : nullptr;
                bool raw = false;
                std::string value;
                bool typed = ps && ps->is(Type::Object) &&
                             (ps->get("anyOf") || ps->get("oneOf") || ps->get("allOf") || ps->get("const") ||
                              ps->get("properties") || ps->get("items"));
                if (en && en->is(Type::Array) && !en->items.empty()) {
                    bool all_str = std::all_of(en->items.begin(), en->items.end(),
                                               [](const JsonValue& v) { return v.is(Type::String); });
                    if (all_str) {
                        std::vector<std::string> vals;
                        for (const JsonValue& v : en->items) vals.push_back(literal(v.s));
                        value = c.add_rule(fn_id + "-" + m.first + "-enum", "(" + join(vals, " | ") + ")");
                        raw = true;
                    }
                } else if (ty && is_text_type(*ty)) {
                    raw = true;
                } else if (text_union(ps, params)) {
                    raw = true;
                } else if (!ty && !typed) {
                    raw = true;
                }
                if (raw && value.empty()) {
                    if (text_rule.empty()) text_rule = add_free_text(c, "</parameter>");
                    value = text_rule;
                }
                if (!raw) {
                    try {
                        SchemaDocument d = build_schema_document(*ps, params);
                        value = c.add_schema(fn_id + "-" + m.first, *d.root);
                    } catch (const std::exception& e) {
                        err = "tools: " + nm->s + "." + m.first + ": " + e.what();
                        return "";
                    }
                }
                if (value.empty()) {
                    err = "tools: " + nm->s + "." + m.first + ": no rule";
                    return "";
                }
                (required.count(m.first) ? req_items : opt_items)
                    .push_back(literal("<parameter=" + m.first + ">\n") + " " + value + " " + literal("\n</parameter>\n"));
            }
        }
        // As llama.cpp's Qwen3-Coder parser (common/chat-peg-parser.cpp permute, parsers/qwen3-coder.cpp): the required
        // parameters in any order, each once (one rule per set still to write; past max_permute they keep the declared
        // order), then the optional ones in any order, repeatable.
        std::string body;
        const size_t n = req_items.size();
        if (n == 1 || n > opt.max_permute) {
            for (const std::string& it : req_items) body += it + " ";
        } else if (n > 1) {
            auto rule_of = [&](size_t remaining) { return fn_id + "-req" + std::to_string(remaining); };
            for (size_t remaining = 1; remaining < ((size_t) 1 << n); ++remaining) {
                std::vector<std::string> alts;
                for (size_t i = 0; i < n; ++i) {
                    const size_t bit = (size_t) 1 << i;
                    if (!(remaining & bit)) continue;
                    alts.push_back(req_items[i] + (remaining & ~bit ? " " + rule_of(remaining & ~bit) : ""));
                }
                c.add_rule(rule_of(remaining), join(alts, " | "));
            }
            body = rule_of(((size_t) 1 << n) - 1) + " ";
        }
        if (!opt_items.empty()) body += "(" + c.add_rule(fn_id + "-opt", join(opt_items, " | ")) + ")* ";
        fns.push_back(c.add_rule(fn_id, literal("<function=" + nm->s + ">\n") + " " + body + literal("</function>")));
    }
    c.add_rule("call", literal("<tool_call>\n") + " (" + join(fns, " | ") + ") " + literal("\n</tool_call>"));
    c.add_rule("root", opt.multiple_calls ? "call ([ \\t\\n]* call)*" : "call");
    try {
        c.check_errors();
    } catch (const std::exception& e) {
        err = std::string("tools: ") + e.what();
        return "";
    }
    return c.format_grammar();
}

RequestGrammar grammar_from_request(const std::string& grammar_json, std::string& err) {
    RequestGrammar out;
    JsonValue req;
    if (!read_json(grammar_json, req, err)) return out;
    const JsonValue* kind = req.is(Type::Object) ? req.get("kind") : nullptr;
    if (!kind || !kind->is(Type::String)) {
        err = "grammar: missing kind";
        return out;
    }
    std::string gbnf;
    if (kind->s == "json_object") {
        gbnf = json_schema_to_gbnf(R"({"type":"object"})", err);
    } else if (kind->s == "json_schema") {
        const JsonValue* sch = req.get("schema");
        if (!sch) {
            err = "grammar: json_schema needs a schema";
            return out;
        }
        gbnf = json_schema_to_gbnf(sch->dump(), err);
    } else if (kind->s == "tools") {
        const JsonValue* tools = req.get("tools");
        if (!tools || !tools->is(Type::Array)) {
            err = "grammar: tools needs an array";
            return out;
        }
        std::string only;   // a named choice
        if (const JsonValue* tc = req.get("tool_choice")) {
            if (tc->is(Type::String) && tc->s == "none") return out;
            if (tc->is(Type::Object)) {
                const JsonValue* fn = tc->get("function");
                const JsonValue* nm = fn && fn->is(Type::Object) ? fn->get("name") : tc->get("name");
                if (nm && nm->is(Type::String)) only = nm->s;
            }
        }
        JsonValue list = *tools;
        if (!only.empty()) {
            list.items.clear();
            for (const JsonValue& t : tools->items) {
                const JsonValue* f = t.is(Type::Object) && t.get("function") ? t.get("function") : &t;
                const JsonValue* nm = f->is(Type::Object) ? f->get("name") : nullptr;
                if (nm && nm->is(Type::String) && nm->s == only) list.items.push_back(t);
            }
        }
        gbnf = tool_call_gbnf(list.dump(), ToolGrammarOptions{}, err);
        out.triggers.words = {tool_call_trigger()};
    } else {
        err = "grammar: unknown kind '" + kind->s + "'";
        return out;
    }
    if (gbnf.empty()) return out;
    out.grammar = Grammar::parse(gbnf, err);
    return out;
}

}  // namespace basalt::program::grammar
